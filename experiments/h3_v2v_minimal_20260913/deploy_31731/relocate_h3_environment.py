#!/usr/bin/env python3
"""Audit/repair a PRIVATE H3 copy after conda-pack --dest-prefix.

The CLI has no root override: it only accepts the four production roots below.
Default is a read-only JSON plan; --apply is explicit and rejects any blockers.
Do not run against an environment currently used by a service. Each replacement
is atomic, not the entire plan; rerunning after interruption is idempotent.

Only UTF-8 text of declared types, Python entry-point scripts, and symlinks are
eligible. ELF files, bytecode, archives, logs, and Conda package provenance are
not rewritten. Source edits only repair generated/runtime text, not binaries.
This does not claim to relocate arbitrary third-party binary prefixes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from urllib.parse import urlsplit
import uuid


OLD_ENV = "/cache/yunfeng/envs/minimax-h3-npu"
OLD_SRC = "/cache/yunfeng/minimax_h3_npu/src"
SOURCE_NAMES = ("vllm", "vllm-ascend", "vllm-omni")
PRODUCTION_ENV = Path("/cache/zhonghao/h3/env")
PRODUCTION_SRC = Path("/cache/zhonghao/h3/src")
MAX_TEXT_BYTES = 16 * 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024
ENV_TEXT_SUFFIXES = frozenset({
    ".py", ".sh", ".pth", ".egg-link", ".info", ".cfg", ".conf", ".ini",
    ".json", ".pc", ".cmake", ".properties", ".yaml", ".yml", ".xml",
})
SOURCE_TEXT_SUFFIXES = frozenset({".py", ".sh", ".json", ".cfg", ".conf", ".ini"})
SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache"})
LEGACY_MARKERS = (OLD_ENV, OLD_SRC)
LIMITATIONS = [
    "Binary artifacts, Python bytecode, arbitrary logs and Conda provenance are not rewritten/audited for embedded prefixes.",
    "Unrelated external symlinks are preserved, never followed for writes; validate runtime library resolution separately.",
    "Source CMake caches/build logs are not relocated; no rebuild is attempted.",
    "Oversize JSON is audited with bounded-memory raw-prefix scanning and SHA256, never rewritten or fully parsed.",
    "Noneditable local-wheel direct_url.json archive provenance is preserved; editable directory bindings remain subject to relocation.",
    "This is an offline deployment operation; do not use concurrently with a service or another relocator.",
]


@dataclass(frozen=True)
class Layout:
    env: Path
    src: Path

    @property
    def roots(self):
        return (self.env,) + tuple(self.src / name for name in SOURCE_NAMES)

    @property
    def replacements(self):
        return ((OLD_ENV, self.env.as_posix()),) + tuple(
            (f"{OLD_SRC}/{name}", (self.src / name).as_posix()) for name in SOURCE_NAMES
        )


def production_layout():
    return Layout(PRODUCTION_ENV, PRODUCTION_SRC)


def _inside(path: Path, root: Path):
    return path == root or root in path.parents


def validate_roots(layout: Layout):
    for root in layout.roots:
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError(f"Root must be an absolute normalized path: {root}")
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            raise ValueError(f"Missing/noncanonical/symlink root is forbidden: {root}")
        for ancestor in root.parents:
            if ancestor.is_symlink():
                raise ValueError(f"Symlink ancestor is forbidden: {ancestor}")


def authorized_path(path: Path, layout: Layout, *, leaf_symlink=False):
    """Validate the writable entry, NEVER dereference a symlink leaf for a write."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Noncanonical target: {path}")
    root = next((root for root in layout.roots if _inside(path, root) and path != root), None)
    if root is None:
        raise ValueError(f"Target is outside the exact private roots: {path}")
    cursor = path.parent
    while True:
        if cursor.is_symlink() or not cursor.is_dir():
            raise ValueError(f"Symlink/missing parent is forbidden: {cursor}")
        if cursor == root:
            break
        cursor = cursor.parent
    if path.parent.resolve(strict=True) != path.parent:
        raise ValueError(f"Noncanonical parent: {path}")
    if path.is_symlink():
        if not leaf_symlink:
            raise ValueError(f"Refusing to write through a symlink: {path}")
    elif path.exists() and (not path.is_file() or path.resolve(strict=True) != path):
        raise ValueError(f"Not a canonical regular file: {path}")
    return root


def rewrite_text(text: str, layout: Layout):
    # Match a real path boundary: never turn vllm-extra into vllm's destination.
    for old, new in sorted(layout.replacements, key=lambda pair: -len(pair[0])):
        pattern = re.escape(old) + r"(?=$|[/\s\x00'\";,:\)\]\}])"
        text = re.sub(pattern, lambda _: new, text)
    return text


def remaining_markers(text: str):
    return [marker for marker in LEGACY_MARKERS if marker in text]


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def _file_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def scan_large_json(path: Path, chunk_size=None):
    """Read every byte with bounded memory; detect prefixes spanning chunks."""
    chunk_size = STREAM_CHUNK_BYTES if chunk_size is None else chunk_size
    if chunk_size < 1:
        raise ValueError("Streaming chunk size must be positive")
    markers = {marker: marker.encode("utf-8") for marker in LEGACY_MARKERS}
    overlap = max(len(marker) for marker in markers.values()) - 1
    digest, seen, tail, size = hashlib.sha256(), set(), b"", 0
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular streaming input: {path}")
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
            size += len(block)
            window = tail + block
            seen.update(name for name, marker in markers.items() if marker in window)
            tail = window[-overlap:] if overlap else b""
        after = os.fstat(stream.fileno())
    if (_file_identity(before) != _file_identity(after) or size != before.st_size
            or _file_identity(path.lstat()) != _file_identity(after)):
        raise ValueError(f"Streaming input changed during audit: {path}")
    return {"path": str(path), "size_bytes": size, "sha256": digest.hexdigest(),
            "markers": sorted(seen), "scan": "bounded_memory_raw_prefix_bytes",
            "action": "preserved_no_write" if not seen else "blocked_no_write"}


def local_wheel_provenance(path: Path, root: Path, layout: Layout, text: str):
    """Recognize archive provenance, never exempt an editable/runtime binding."""
    if (root != layout.env or path.name != "direct_url.json"
            or not path.parent.name.endswith(".dist-info")):
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if (not isinstance(data, dict) or "dir_info" in data or "vcs_info" in data
            or not isinstance(data.get("archive_info"), dict)
            or not isinstance(data.get("url"), str)):
        return None
    url = urlsplit(data["url"])
    if url.scheme != "file" or url.netloc not in ("", "localhost") or not url.path.endswith(".whl"):
        return None
    hashes = data["archive_info"].get("hashes", {})
    digest = hashes.get("sha256") if isinstance(hashes, dict) else None
    old_hash = data["archive_info"].get("hash", "")
    if digest is None and isinstance(old_hash, str) and old_hash.startswith("sha256="):
        digest = old_hash[len("sha256="):]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return None
    if old_hash and old_hash != "sha256=" + digest:
        return None
    return {"kind": "noneditable_local_wheel_archive_provenance", "archive_sha256": digest,
            "editable": False, "action": "preserved_no_write"}


def _eligible(path: Path, root: Path, layout: Layout):
    relative = path.relative_to(root)
    if root == layout.env:
        if relative.parts[0] == "conda-meta":
            return False
        return path.suffix.lower() in ENV_TEXT_SUFFIXES or relative.parts[0] == "bin"
    # Explicit text types only; no CMakeCache, build .txt, binary, or blanket sed.
    return path.suffix.lower() in SOURCE_TEXT_SUFFIXES


def _iter_entries(root: Path):
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        keep = []
        for name in sorted(directories):
            child = parent / name
            if child.is_symlink():
                yield child
            elif name not in SKIP_DIRS:
                keep.append(name)
        directories[:] = keep
        for name in sorted(files):
            yield parent / name


def make_plan(layout: Layout):
    validate_roots(layout)
    report = {"mode": "dry_run", "roots": [str(root) for root in layout.roots],
              "changes": [], "blockers": [], "external_links_preserved": [],
              "skipped_oversize": [], "skipped_native_elf": [],
              "streamed_large_json": [], "preserved_archive_provenance": [],
              "limitations": LIMITATIONS.copy()}
    for root in layout.roots:
        for path in _iter_entries(root):
            if path.is_symlink():
                authorized_path(path, layout, leaf_symlink=True)
                original = os.readlink(path)
                rewritten = rewrite_text(original, layout)
                if remaining_markers(rewritten):
                    report["blockers"].append({"path": str(path), "reason": "Unmapped legacy symlink target", "target": original})
                elif rewritten != original:
                    target = Path(rewritten)
                    if (not target.is_absolute() or ".." in target.parts
                            or not any(_inside(target, allowed) for allowed in layout.roots)):
                        report["blockers"].append({"path": str(path), "reason": "Rewritten link target escaped private roots"})
                    else:
                        report["changes"].append({"kind": "symlink", "path": str(path),
                                                  "before": original, "after": rewritten})
                elif not any(_inside(path.resolve(strict=False), allowed) for allowed in layout.roots):
                    report["external_links_preserved"].append({"path": str(path), "target": original})
                continue
            if not _eligible(path, root, layout):
                continue
            authorized_path(path, layout)
            # env/bin includes native executables. Recognize ELF before the
            # text size gate; never read/replace a complete native executable.
            # Embedded ELF prefixes require separate binary/runtime auditing.
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic == b"\x7fELF":
                report["skipped_native_elf"].append(str(path))
                continue
            if path.stat().st_size > MAX_TEXT_BYTES:
                if path.suffix.lower() == ".json":
                    audit = scan_large_json(path)
                    report["streamed_large_json"].append(audit)
                    if audit["markers"]:
                        report["blockers"].append({"path": str(path),
                            "reason": "Legacy prefix in oversize JSON; streaming audit only, manual review required",
                            "markers": audit["markers"], "sha256": audit["sha256"], "size_bytes": audit["size_bytes"]})
                    continue
                report["skipped_oversize"].append(str(path))
                report["blockers"].append({"path": str(path), "reason": "Eligible text/entry point exceeds audit size limit"})
                continue
            raw = path.read_bytes()
            # bin contains native executables too: inspect as bytes, never replace.
            if b"\x00" in raw:
                if any(marker.encode() in raw for marker in LEGACY_MARKERS):
                    report["blockers"].append({"path": str(path), "reason": "Legacy prefix in a binary-like eligible file; manual review required"})
                continue
            try:
                original = raw.decode("utf-8")
            except UnicodeDecodeError:
                if any(marker.encode() in raw for marker in LEGACY_MARKERS):
                    report["blockers"].append({"path": str(path), "reason": "Legacy prefix in non-UTF-8 text; manual review required"})
                continue
            provenance = local_wheel_provenance(path, root, layout, original)
            if provenance is not None:
                report["preserved_archive_provenance"].append({"path": str(path),
                    "size_bytes": len(raw), "sha256": sha256(raw), **provenance})
                continue
            rewritten = rewrite_text(original, layout)
            if remaining_markers(rewritten):
                report["blockers"].append({"path": str(path), "reason": "Unmapped legacy text prefix", "markers": remaining_markers(rewritten)})
                continue
            if rewritten != original:
                if path.stat().st_nlink > 1:
                    # Atomic replacement would not mutate another hardlink, but
                    # unknown shared identity deserves an explicit review first.
                    report["blockers"].append({"path": str(path), "reason": "Multiple hardlinks require review"})
                    continue
                report["changes"].append({"kind": "text", "path": str(path),
                                          "before_sha256": sha256(raw),
                                          "after_sha256": sha256(rewritten.encode("utf-8")),
                                          "replacements": sum(original.count(old) for old, _ in layout.replacements)})
    report["change_count"] = len(report["changes"])
    report["status"] = "blocked" if report["blockers"] else "ready"
    return report


def _verify_change(change, layout):
    path = Path(change["path"])
    root = authorized_path(path, layout, leaf_symlink=change["kind"] == "symlink")
    if change["kind"] == "symlink":
        if not path.is_symlink() or os.readlink(path) != change["before"]:
            raise ValueError(f"Symlink changed since planning: {path}")
        if rewrite_text(change["before"], layout) != change["after"]:
            raise ValueError(f"Invalid link rewrite: {path}")
        target = Path(change["after"])
        if (not target.is_absolute() or ".." in target.parts or remaining_markers(change["after"])
                or not any(_inside(target, allowed) for allowed in layout.roots)):
            raise ValueError(f"Invalid rewritten link target: {path}")
        return None
    if change["kind"] != "text" or path.is_symlink():
        raise ValueError(f"Invalid change kind/entry: {path}")
    raw = path.read_bytes()
    if (not _eligible(path, root, layout) or len(raw) > MAX_TEXT_BYTES or b"\x00" in raw
            or sha256(raw) != change["before_sha256"] or path.stat().st_nlink != 1):
        raise ValueError(f"File identity/content changed since planning: {path}")
    rewritten = rewrite_text(raw.decode("utf-8"), layout).encode("utf-8")
    if remaining_markers(rewritten.decode("utf-8")) or sha256(rewritten) != change["after_sha256"]:
        raise ValueError(f"Rewrite hash mismatch: {path}")
    return rewritten


def _replace_posix(change, payload, path):
    """Anchor every pathname operation to no-follow parent directory handles."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, flags)
    temporary_name = None
    temporary_created = False
    try:
        for component in path.parent.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if change["kind"] == "symlink":
            if (not stat.S_ISLNK(current.st_mode)
                    or os.readlink(path.name, dir_fd=directory_fd) != change["before"]):
                raise ValueError(f"Symlink changed before atomic replacement: {path}")
            temporary_name = ".h3-relocate-" + uuid.uuid4().hex
            os.symlink(change["after"], temporary_name, dir_fd=directory_fd)
            temporary_created = True
        else:
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise ValueError(f"Unsafe text entry before replacement: {path}")
            old_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            with os.fdopen(old_fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_ino, opened.st_dev) != (current.st_ino, current.st_dev):
                    raise ValueError(f"File identity changed before replacement: {path}")
                if sha256(stream.read(MAX_TEXT_BYTES + 1)) != change["before_sha256"]:
                    raise ValueError(f"File contents changed before replacement: {path}")
            temporary_name = ".h3-relocate-" + uuid.uuid4().hex
            fd = os.open(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         stat.S_IMODE(current.st_mode), dir_fd=directory_fd)
            temporary_created = True
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fchmod(stream.fileno(), stat.S_IMODE(current.st_mode))
                os.fsync(stream.fileno())
        os.replace(temporary_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        temporary_name = None
        temporary_created = False
        os.fsync(directory_fd)
    finally:
        if temporary_name is not None and temporary_created:
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def apply_plan(report, layout):
    validate_roots(layout)
    if report["blockers"] or report["status"] != "ready":
        raise ValueError("Refusing to apply a blocked/unreviewed plan")
    if report["roots"] != [str(root) for root in layout.roots]:
        raise ValueError("Plan roots do not match authorized layout")
    # Preserved audit inputs are still part of the gate. A file becoming a
    # runtime binding or gaining an old prefix must fail before any mutation.
    for item in report.get("streamed_large_json", []):
        path = Path(item["path"])
        authorized_path(path, layout)
        current = scan_large_json(path)
        if (current["markers"] or current["sha256"] != item["sha256"]
                or current["size_bytes"] != item["size_bytes"]):
            raise ValueError(f"Preserved large JSON changed since planning: {path}")
    for item in report.get("preserved_archive_provenance", []):
        path = Path(item["path"])
        root = authorized_path(path, layout)
        if path.stat().st_size > MAX_TEXT_BYTES:
            raise ValueError(f"Provenance grew beyond size gate: {path}")
        raw = path.read_bytes()
        if (sha256(raw) != item["sha256"] or len(raw) != item["size_bytes"]
                or local_wheel_provenance(path, root, layout, raw.decode("utf-8")) is None):
            raise ValueError(f"Preserved provenance changed since planning: {path}")
    # Check every input before making even the first mutation.
    for change in report["changes"]:
        _verify_change(change, layout)
    applied = []
    for change in report["changes"]:
        path = Path(change["path"])
        payload = _verify_change(change, layout)
        if os.name == "posix":
            _replace_posix(change, payload, path)
            applied.append(str(path))
            continue
        # Windows branch exists only for local temporary-fixture CPU tests.
        # Production CLI explicitly requires POSIX no-follow directory handles.
        temporary = None
        try:
            fd, temp_name = tempfile.mkstemp(prefix=".h3-relocate-", dir=path.parent)
            temporary = Path(temp_name)
            if change["kind"] == "symlink":
                os.close(fd)
                temporary.unlink()
                os.symlink(change["after"], temporary, target_is_directory=path.is_dir())
            else:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            _verify_change(change, layout)
            authorized_path(temporary, layout, leaf_symlink=change["kind"] == "symlink")
            os.replace(temporary, path)
            temporary = None
            applied.append(str(path))
        finally:
            if temporary is not None and (temporary.exists() or temporary.is_symlink()):
                authorized_path(temporary, layout, leaf_symlink=True)
                temporary.unlink()
    after = make_plan(layout)
    after.update(mode="apply", applied=applied, applied_count=len(applied))
    if after["blockers"] or after["changes"]:
        after["status"] = "incomplete"
    else:
        after["status"] = "relocated"
    return after


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the printed plan only inside the four fixed private roots")
    args = parser.parse_args(argv)
    layout = production_layout()
    try:
        if os.name != "posix":
            raise ValueError("Production relocation requires POSIX directory-handle/no-follow writes")
        report = make_plan(layout)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.apply:
            report = apply_plan(report, layout)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 2 if report["status"] in {"blocked", "incomplete"} else 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(json.dumps({"status": "failed_closed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
