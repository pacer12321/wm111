#!/usr/bin/env python3
"""Pack H3 without restoring stale setuptools/wheel files from conda's cache.

Only the in-memory archive manifest is changed. The source environment and
package cache are never repaired or written. Failed immutable output objects
are deliberately retained and must not be published by the transfer process.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import socket
import tarfile
import time

import audit_pack_environment as audit

SOURCE_ENV = audit.PREFIX
TARGET_ENV = Path("/cache/zhonghao/h3/env")
STAGE = Path("/temp/zhonghao/h3_31731_20260913_v2")
OBJECTS = STAGE / "objects"
DEPLOYMENT = "zhonghao_h3_31731_20260913_v2"
SOURCE_HOST = "ma-job-e03cb638-a995-4fcf-aad0-ebcf50a7c3ff-worker-0"
LEGACY_VERSIONS = {"setuptools": "83.0.0", "wheel": "0.47.0"}
CURRENT_VERSIONS = {"setuptools": "80.10.2", "wheel": "0.48.0"}


def run_audit():
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        audit.main()
    result = json.loads(captured.getvalue())
    if result.get("status") != "passed":
        raise RuntimeError("Source audit did not pass")
    return result


def regular_source(prefix, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe package target: {relative}")
    path = prefix / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(prefix):
        raise RuntimeError(f"Missing, symlinked, or escaping package file: {relative}")
    return path


def legacy_targets(prefix):
    targets = set()
    for name, version in LEGACY_VERSIONS.items():
        records = list((prefix / "conda-meta").glob(name + "-*.json"))
        if len(records) != 1:
            raise RuntimeError(f"Expected one conda record for {name}")
        metadata = json.loads(records[0].read_text())
        if metadata.get("name") != name or metadata.get("version") != version:
            raise RuntimeError(f"Legacy conda version changed: {name}")
        targets.update(metadata["files"])
    return targets


def redirect_legacy_files(env, targets, file_class, prefix=SOURCE_ENV):
    """Do not mutate the conda File objects or any on-disk package metadata."""
    files, replaced, seen = [], [], set()
    for item in env.files:
        if item.target in seen:
            raise RuntimeError(f"Duplicate conda-pack target: {item.target}")
        seen.add(item.target)
        if item.target in targets:
            source = regular_source(prefix, item.target)
            files.append(file_class(str(source), item.target, is_conda=False,
                                    prefix_placeholder=None, file_mode="unknown"))
            replaced.append(item.target)
        else:
            files.append(item)
    expected = targets - audit.ALLOWED_MISSING
    if set(replaced) != expected:
        raise RuntimeError("Legacy package manifest coverage changed: " +
                           repr(sorted(set(replaced) ^ expected)))
    env.files = files
    return replaced


def record_snapshots(prefix=SOURCE_ENV, dest_prefix=TARGET_ENV):
    """Snapshot every hash-bearing current RECORD entry, plus both __init__.py."""
    snapshots = {}
    for name, version in CURRENT_VERSIONS.items():
        dist = importlib.metadata.distribution(name)
        if dist.version != version or not dist.files:
            raise RuntimeError(f"Current distribution changed or lacks RECORD: {name}")
        for item in dist.files:
            if not item.hash:
                continue
            path = Path(dist.locate_file(item)).resolve()
            if not path.is_relative_to(prefix) or not path.is_file():
                raise RuntimeError(f"Current RECORD escapes environment: {name}/{item}")
            relative = path.relative_to(prefix).as_posix()
            data = path.read_bytes()
            allowed = {hashlib.sha256(data).hexdigest()}
            # Conda-pack intentionally relocates installed entry-point shebangs.
            if relative.startswith("bin/") and data.startswith(b"#!"):
                lines = data.split(b"\n", 1)
                relocated = lines[0].replace(str(prefix).encode(), str(dest_prefix).encode())
                if len(lines) == 2:
                    relocated += b"\n" + lines[1]
                allowed.add(hashlib.sha256(relocated).hexdigest())
            snapshots[relative] = allowed
        init_name = audit.SITE + name + "/__init__.py"
        if init_name not in snapshots:
            raise RuntimeError(f"Current RECORD does not hash {init_name}")
    return snapshots


def verify_archive(path, snapshots):
    found = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name
            if name not in snapshots:
                continue
            if name in found or not member.isfile():
                raise RuntimeError(f"Duplicate/nonregular verified archive member: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unreadable archive member: {name}")
            digest = hashlib.sha256()
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() not in snapshots[name]:
                raise RuntimeError(f"Archive differs from actual environment: {name}")
            found.add(name)
    if found != set(snapshots):
        raise RuntimeError("Archive missing current RECORD files: " + repr(sorted(set(snapshots) - found)))
    return len(found)


def checked_output(value):
    path = Path(value)
    if path.parent != OBJECTS or not re.fullmatch(r"env-[A-Za-z0-9_-]+\.tar", path.name):
        raise RuntimeError("Output must be a new immutable env-*.tar in the approved v2 objects directory")
    if not OBJECTS.is_dir() or OBJECTS.resolve() != OBJECTS:
        raise RuntimeError("Output directory is missing or contains a symlink")
    if os.path.lexists(path):
        raise RuntimeError("Output already exists; it will not be overwritten")
    ready = sorted(STAGE.glob("deployment_owner.json.version-*.ready"))
    if not ready or ready[-1].is_symlink():
        raise RuntimeError("Missing safe deployment owner commit")
    record = ready[-1].with_name(ready[-1].name[:-6])
    if record.is_symlink() or json.loads(record.read_text()).get("deployment") != DEPLOYMENT:
        raise RuntimeError("Output stage belongs to another deployment")
    return path


def build_manifest():
    import conda_pack
    from conda_pack.core import File
    if conda_pack.__version__ != "0.9.2":
        raise RuntimeError(f"Unreviewed conda-pack version: {conda_pack.__version__}")
    env = conda_pack.CondaEnv.from_prefix(str(SOURCE_ENV), ignore_missing_files=True,
                                        ignore_editable_packages=True)
    replaced = redirect_legacy_files(env, legacy_targets(SOURCE_ENV), File)
    # Copy runtime contents only, never incidental credentials in an environment.
    excluded_parts = {".ssh", ".aws", ".azure"}
    excluded_names = {".env", ".netrc", ".pypirc", "credentials", "credentials.json",
                      "id_rsa", "id_ed25519"}
    env.files = [item for item in env.files
                 if not (set(Path(item.target).parts) & excluded_parts)
                 and Path(item.target).name not in excluded_names]
    return env, replaced


def write_archive(env, output):
    # CondaEnv.pack uses a large tempfile and shutil.move. The archival NFS
    # forbids rename, so use its reviewed 0.9.2 Packer API on an exclusive object.
    from conda_pack.core import Packer
    from conda_pack.formats import archive
    history = SOURCE_ENV / "conda-meta/history"
    mtime = history.lstat().st_mtime if history.exists() else None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        with archive(stream, str(output), "", "tar", compress_level=0, zip_symlinks=False,
                     zip_64=True, n_threads=1, verbose=False, output=str(output), mtime=mtime) as arc:
            packer = Packer(str(SOURCE_ENV), arc, str(TARGET_ENV), None)
            last = time.monotonic()
            for index, item in enumerate(env.files, 1):
                packer.add(item)
                if time.monotonic() - last >= 30:
                    print(json.dumps({"phase": "packing", "files": index, "total": len(env.files)}), flush=True)
                    last = time.monotonic()
            packer.finish()
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != SOURCE_HOST:
        raise RuntimeError("Packing is only permitted on the approved source node")
    if not args.audit_only and not args.output:
        parser.error("--output is required unless --audit-only")
    before = run_audit()
    snapshots = record_snapshots()
    env, replaced = build_manifest()
    print(json.dumps({"phase": "manifest_verified", "source_audit": before,
                      "redirected_legacy_files": len(replaced), "archive_files": len(env.files),
                      "record_hashes_to_verify": len(snapshots)}), flush=True)
    if args.audit_only:
        return
    output = checked_output(args.output)
    write_archive(env, output)
    after = run_audit()
    if before != after:
        raise RuntimeError("Source package audit changed during packing")
    verified = verify_archive(output, snapshots)
    print(json.dumps({"phase": "archive_verified", "output": str(output),
                      "size": output.stat().st_size, "verified_current_record_files": verified,
                      "source_environment_modified": False}), flush=True)


if __name__ == "__main__":
    main()
