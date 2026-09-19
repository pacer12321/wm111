#!/usr/bin/env python3
"""One-shot, fail-closed extraction of the verified private H3 environment.

No repair/resume/delete path exists. An interrupted new environment is left
with its ownership marker and an external failed status for manual review.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import socket
import stat
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone

ROOT = Path("/cache/zhonghao/h3")
ENV = ROOT / "env"
ARCHIVE = ROOT / "env.tar"
SOURCE_ENV = "/cache/yunfeng/envs/minimax-h3-npu"
TARGET_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
DEPLOYMENT = "zhonghao_h3_31731_20260913_v2"
OWNER_NAME = ".h3_extraction_owner.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def canonical_directory(path):
    if not path.is_absolute() or not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError(f"Missing/noncanonical directory: {path}")


def read_regular_json(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Missing/nonregular JSON file: {path}")
    with path.open() as stream:
        return json.load(stream)


def verified_entry(root):
    canonical_directory(root)
    owner = read_regular_json(root / "deployment_owner.json")
    if owner.get("deployment") != DEPLOYMENT:
        raise RuntimeError("Private deployment ownership mismatch")
    entry = read_regular_json(root / "transfer_verified.json").get("env.tar")
    if not isinstance(entry, dict) or entry.get("kind") != "file" or entry.get("path") != "env.tar":
        raise RuntimeError("No verified env.tar entry")
    if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[a-fA-F0-9]{64}", entry["sha256"]):
        raise RuntimeError("Invalid env.tar SHA256")
    if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
        raise RuntimeError("Invalid env.tar size")
    return entry


def normalized_name(member):
    name = member.name
    if member.isdir() and name.endswith("/"):
        name = name[:-1]
    if (not name or name.startswith("/") or "\\" in name or
            any(ord(c) < 32 or ord(c) == 127 for c in name) or
            re.match(r"^[A-Za-z]:", name) or
            any(part in ("", ".", "..") for part in name.split("/"))):
        raise RuntimeError(f"Unsafe/noncanonical archive member: {member.name!r}")
    if name == OWNER_NAME or name.startswith(OWNER_NAME + "/"):
        raise RuntimeError("Archive collides with extraction ownership marker")
    return name


def map_symlink(name, link, destination):
    if not link or "\\" in link or any(ord(c) < 32 or ord(c) == 127 for c in link):
        raise RuntimeError(f"Unsafe symlink: {name} -> {link!r}")
    if link.startswith("/"):
        matched = None
        for prefix in (SOURCE_ENV, destination.as_posix()):
            if link == prefix or link.startswith(prefix + "/"):
                matched = link[len(prefix):].lstrip("/")
                break
        if matched is None:
            raise RuntimeError(f"External absolute symlink: {name} -> {link}")
        # Reject traversal in an absolute prefix suffix rather than interpreting
        # /source-env/../other as if it were still beneath source-env.
        if any(p in (".", "..", "") for p in matched.split("/")) and matched:
            raise RuntimeError(f"Noncanonical absolute symlink: {name} -> {link}")
        link = posixpath.relpath(matched or ".", posixpath.dirname(name) or ".")
    # First enforce lexical containment, then preflight checks full link chains.
    stack = list(PurePosixPath(name).parent.parts)
    for part in link.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not stack:
                raise RuntimeError(f"Escaping relative symlink: {name} -> {link}")
            stack.pop()
        else:
            stack.append(part)
    return link


def resolve_virtual_link(name, links, types):
    """Resolve archive symlink semantics without touching the real filesystem."""
    pending, stack, link_hops = deque(name.split("/")), [], 0
    while pending:
        part = pending.popleft()
        if part in ("", "."):
            continue
        if part == "..":
            if not stack:
                raise RuntimeError(f"Symlink chain escapes environment: {name}")
            stack.pop()
            continue
        candidate = "/".join(stack + [part])
        if candidate in links:
            link_hops += 1
            if link_hops > 40:
                raise RuntimeError(f"Symlink cycle/excessive depth: {name}")
            pending.extendleft(reversed(links[candidate].split("/")))
        else:
            if types.get(candidate) == "file" and pending:
                raise RuntimeError(f"Symlink traverses a regular file: {name}")
            stack.append(part)
    return "/".join(stack)


def preflight(tar, destination):
    plan, types, links = [], {}, {}
    counts = {"regular_files": 0, "directories": 0, "symlinks": 0,
              "hardlinks": 0, "expanded_file_bytes": 0, "mapped_absolute_symlinks": 0}
    for original in tar:
        member = copy.copy(original)
        name = normalized_name(member)
        if name in types:
            raise RuntimeError(f"Duplicate archive member: {name}")
        member.name = name
        if member.islnk():
            counts["hardlinks"] += 1
            raise RuntimeError(f"Hardlink requires manual review and is refused: {name} -> {member.linkname}")
        if member.isfile():
            if member.size < 0:
                raise RuntimeError(f"Negative file size: {name}")
            types[name] = "file"
            counts["regular_files"] += 1
            counts["expanded_file_bytes"] += member.size
        elif member.isdir():
            types[name] = "directory"
            counts["directories"] += 1
        elif member.issym():
            types[name] = "symlink"
            if member.linkname.startswith("/"):
                counts["mapped_absolute_symlinks"] += 1
            member.linkname = map_symlink(name, member.linkname, destination)
            links[name] = member.linkname
            counts["symlinks"] += 1
        else:
            raise RuntimeError(f"Unsupported special archive member: {name} type={member.type!r}")
        plan.append(member)
        if len(plan) > 1_000_000:
            raise RuntimeError("Archive member-count guard exceeded")
    for name in types:
        for parent in PurePosixPath(name).parents:
            parent_name = parent.as_posix()
            if parent_name != "." and types.get(parent_name) in ("file", "symlink"):
                raise RuntimeError(f"Archive writes below non-directory/link parent: {name}")
    for name in links:
        resolve_virtual_link(name, links, types)
    if not plan or not counts["regular_files"]:
        raise RuntimeError("Empty/non-environment archive")
    # Links are last, avoiding dependence on archive order during extraction.
    plan.sort(key=lambda m: bool(m.issym()))
    return plan, counts


def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def extract_once(root=ROOT, inspect_only=False):
    entry = verified_entry(root)
    destination, archive_path = root / "env", root / "env.tar"
    if os.path.lexists(destination):
        raise RuntimeError("env already exists; automatic overwrite/resume is forbidden")
    if archive_path.is_symlink() or archive_path.resolve() != archive_path:
        raise RuntimeError("Archive is symlinked or noncanonical")
    run_id = uuid.uuid4().hex
    status_path = root / ("env-extraction-" + run_id + ".json")
    status = {"deployment": DEPLOYMENT, "run_id": run_id, "pid": os.getpid(),
              "hostname": socket.gethostname(), "started_at": now(), "phase": "hashing",
              "archive": str(archive_path), "destination": str(destination), "target_created": False}

    def update(**fields):
        status.update(fields, updated_at=now())
        # --inspect-only performs no filesystem writes.
        if not inspect_only:
            write_json(status_path, status)
        print(json.dumps(status), flush=True)

    try:
        update()
        fd = os.open(archive_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            initial = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial.st_mode) or initial.st_size != entry["size"]:
                raise RuntimeError("Archive file type/size differs from verified transfer")
            digest, hashed, last = hashlib.sha256(), 0, time.monotonic()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                hashed += len(block)
                if time.monotonic() - last >= 30:
                    update(hashed_bytes=hashed)
                    last = time.monotonic()
            if digest.hexdigest() != entry["sha256"].lower():
                raise RuntimeError("Archive SHA256 differs from verified transfer")
            update(phase="preflighting", verified_sha256=digest.hexdigest(), hashed_bytes=hashed)
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:") as tar:
                tar.errorlevel = 2
                plan, counts = preflight(tar, destination)
                update(phase="preflight_passed", archive_members=counts)
                if fingerprint(initial) != fingerprint(os.fstat(stream.fileno())):
                    raise RuntimeError("Archive changed during hashing/preflight")
                if inspect_only:
                    return status
                if shutil.disk_usage(root).free < counts["expanded_file_bytes"] + 1024**3:
                    raise RuntimeError("Insufficient free space for archive contents plus 1 GiB reserve")
                # Atomic mkdir is the final concurrency and no-overwrite gate.
                canonical_directory(root)
                destination.mkdir(mode=0o700, exist_ok=False)
                status["target_created"] = True
                marker = destination / OWNER_NAME
                with marker.open("x") as owner:
                    json.dump({"deployment": DEPLOYMENT, "run_id": run_id,
                               "archive_sha256": digest.hexdigest(), "created_at": now()}, owner)
                    owner.flush()
                    os.fsync(owner.fileno())
                update(phase="extracting")
                tar.extractall(path=destination, members=plan, filter=tarfile.data_filter)
                if fingerprint(initial) != fingerprint(os.fstat(stream.fileno())):
                    raise RuntimeError("Archive changed during extraction")
                canonical_directory(destination)
                update(phase="extracted_runtime_validation_required", completed_at=now(),
                       runtime_verified=False, automatic_resume_permitted=False)
                return status
    except BaseException as exc:
        update(phase="failed", error_type=type(exc).__name__, error=str(exc),
               manual_review_required=True, automatic_resume_permitted=False)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 12) or not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Python 3.12+ with tarfile.data_filter is required")
    if socket.gethostname() != TARGET_HOST:
        raise RuntimeError("Extraction is only permitted on the approved 31731 node")
    extract_once(inspect_only=args.inspect_only)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
