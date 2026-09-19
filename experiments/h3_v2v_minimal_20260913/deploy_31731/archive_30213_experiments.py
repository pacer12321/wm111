"""Seal completed 30213 experiment evidence; copy to a NEW 31731 archive.

No SSH/client credentials, subprocesses, torch, NPU, deletes, model weights or
environment copying. CLI host/path bindings are intentionally not configurable.
Producer holds existing A/B/tiny run locks SHARED/nonblocking for the whole seal.
NFS publication uses a unique immutable object + exclusive completion record;
consumer ignores objects until the complete record exists and verifies hashes.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import tarfile
import uuid

SOURCE_HOST = "ma-job-e03cb638-a995-4fcf-aad0-ebcf50a7c3ff-worker-0"
DEST_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
SOURCE_CODE = Path("/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913")
SOURCE_RESULTS = Path("/cache/zhonghao/h3_v2v_minimal_20260913")
STAGE = Path("/temp/zhonghao/h3_30213_archive_20260913_v1")
DEST_PARENT = Path("/cache/zhonghao/h3_30213_archives")
REQUIRED_RUN = "b_smoke_formal_20260913T122957_063593Z"
TERMINAL = {"completed", "failed", "smoke_completed_review_required", "formal_completed_review_required"}
GROUPS = (("runs", "a_status.json"), ("b_model/runs", "b_status.json"),
          ("b_validation/runs", "validation_status.json"))
LOCKS = ("run.lock", "b_model/run.lock", "b_validation/run.lock")
CODE_FILES = ("README.md", "experiment.json", "run_a_supervised.py", "run_b_supervised.py",
              "launch_a_server.sh", "launch_b_server.sh")
CODE_TREES = ("b_adaptation", "cpu_regression_20260913_1828")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "kernel_meta", ".cache", "node_modules"}
FORBIDDEN_DIRS = {".ssh", ".aws", ".azure", ".kube", ".config", "envs", "weights", "checkpoints"}
FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".so", ".whl", ".tar", ".gz", ".zip"}
SECRET_NAMES = {"id_rsa", "id_ed25519", "authorized_keys", "known_hosts", ".env", "credentials", "credentials.json", "token"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".txt", ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv", ".html", ".js", ".ts", ".rst"}
SECRET_PATTERN = re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|\b(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,})\b|https?://[^/\s:@]+:[^/\s@]+@")
MAX_FILES = 15000
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
CHUNK = 1024 * 1024


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("Archive path must be a nonempty POSIX relative path")
    parts = name.split("/")
    if name.startswith("/") or any(p in ("", ".", "..") or ":" in p for p in parts):
        raise ValueError("Archive path traversal/noncanonical name rejected")
    return PurePosixPath(name)


def no_links(path, *, owner=False):
    """Check every existing ancestor, not just the leaf."""
    path = Path(os.path.abspath(path))
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            raise ValueError(f"Symlink path rejected: {part}")
    if owner and path.exists() and hasattr(os, "geteuid") and path.stat().st_uid != os.geteuid():
        raise ValueError(f"Path is not owned by the executing account: {path}")
    return path


def identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


@contextlib.contextmanager
def open_source(root, relative):
    """Linux openat walk with O_NOFOLLOW; refuse symlink parent replacement."""
    relative = safe_relative(relative)
    root = no_links(root, owner=True)
    if os.name == "nt":  # CPU toy tests only; production CLI requires Linux.
        path = no_links(root.joinpath(*relative.parts), owner=True)
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Not a regular source file")
            yield stream
        return
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in relative.parts[:-1]:
            next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        file_fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise ValueError("Source must be a regular account-owned file")
            yield stream
    finally:
        os.close(fd)


def proc_identity(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return dict(pid=int(pid), state=fields[0], pgrp=int(fields[2]), session=int(fields[3]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def completed_gate(record, run_path, process_reader=proc_identity):
    if record.get("phase") not in TERMINAL or not record.get("finished_at"):
        raise ValueError("Run is not terminal and finished")
    dt.datetime.fromisoformat(record["finished_at"])
    if record.get("cleanup_completed") is not True or record.get("remaining_owned_process_groups") != {} or record.get("needs_attention"):
        raise ValueError("Run cleanup is not conclusively complete")
    if record.get("run_directory") != str(run_path):
        raise ValueError("Run status path does not identify the selected directory")
    supervisor = record.get("supervisor_proc_identity")
    if not isinstance(supervisor, dict) or type(supervisor.get("pid")) is not int or type(supervisor.get("start_ticks")) is not int:
        raise ValueError("Run lacks supervisor process identity")
    for name in ("supervisor_proc_identity", "server_proc_identity", "worker_proc_identity"):
        old = record.get(name)
        if old is None:
            continue
        if not isinstance(old, dict) or type(old.get("pid")) is not int or type(old.get("start_ticks")) is not int:
            raise ValueError("Malformed recorded process identity")
        current = process_reader(old["pid"])
        if current is not None and current.get("start_ticks") == old["start_ticks"]:
            raise ValueError("Recorded run process has not exited (including zombies)")
    return {"run_id": record.get("run_id"), "phase": record["phase"], "finished_at": record["finished_at"],
            "cleanup_completed": True, "error": record.get("error")}


@contextlib.contextmanager
def quiescence_locks(results_root):
    import fcntl
    streams = []
    try:
        for relative in LOCKS:
            path = no_links(results_root / relative, owner=True)
            stream = path.open("rb")  # existing lock only; never create/touch/truncate
            streams.append(stream)
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield
    finally:
        for stream in reversed(streams):
            stream.close()


def source_policy(relative):
    parts = safe_relative(relative).parts
    path = PurePosixPath(relative)
    if any(part.lower() in FORBIDDEN_DIRS or part.lower() in SECRET_NAMES or part.lower().startswith(".env") for part in parts):
        raise ValueError(f"Credential/environment/weight path rejected: {relative}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Credential/weight/archive binary rejected: {relative}")


def collect_tree(root, relative, label, *, exclusions=None):
    result = []
    base = no_links(root / relative, owner=True)
    if not base.is_dir():
        raise ValueError(f"Expected source directory missing: {base}")
    for directory, dirs, files in os.walk(base, followlinks=False):
        for name in list(dirs):
            p = Path(directory) / name
            if label == "results" and Path(directory) == base and name == "cache":
                # Explicit root authorization: only this run-root runtime cache,
                # never code/vendor cache modules or logs/requests/videos.
                if exclusions is None:
                    raise ValueError("Runtime-cache exclusion requires an explicit evidence list")
                exclusions.append({"type": "runtime_cache", "path": str(p),
                                   "reason": "Explicitly excluded run-root compiled runtime cache; source retained"})
                dirs.remove(name)
            elif name in SKIP_DIRS:
                dirs.remove(name)
            elif p.is_symlink():
                raise ValueError(f"Source directory symlink rejected: {p}")
            elif name.lower() in FORBIDDEN_DIRS:
                raise ValueError(f"Unexpected environment/weight directory: {p}")
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix == ".pyc":
                continue
            rel = path.relative_to(root).as_posix()
            source_policy(rel)
            no_links(path, owner=True)
            if not path.is_file():
                raise ValueError(f"Special source file rejected: {path}")
            result.append((root, rel, f"{label}/{rel}"))
    return result


def select_sources(code_root, results_root, required_run=REQUIRED_RUN, process_reader=proc_identity):
    selected, accepted, excluded = [], [], []
    for name in CODE_FILES:
        source_policy(name)
        no_links(code_root / name, owner=True)
        selected.append((code_root, name, "code/" + name))
    for tree in CODE_TREES:
        selected.extend(collect_tree(code_root, tree, "code"))
    required_found = False
    for group, status_name in GROUPS:
        for run in sorted(no_links(results_root / group).iterdir()):
            if not run.is_dir() or run.is_symlink():
                raise ValueError("Unexpected run container member")
            status_path = no_links(run / status_name, owner=True)
            if not status_path.is_file():
                excluded.append({"type": "unsealed_run", "path": str(run), "reason": "no supervisor completion status"})
                continue
            try:
                with open_source(results_root, status_path.relative_to(results_root).as_posix()) as stream:
                    status_record = json.load(stream)
                gate = completed_gate(status_record, run, process_reader)
            except (ValueError, KeyError, TypeError) as error:
                excluded.append({"type": "unsealed_run", "path": str(run), "reason": str(error)})
                if run.name == required_run:
                    raise ValueError(f"Required latest B run cannot be sealed: {error}") from error
                continue
            if run.name == required_run:
                if group != "b_model/runs":
                    raise ValueError("Required B name is in wrong group")
                required_found = True
            accepted.append({"path": str(run), **gate})
            selected.extend(collect_tree(results_root, run.relative_to(results_root).as_posix(), "results", exclusions=excluded))
    if not required_found:
        raise ValueError("Required latest B run missing or not safely completed")
    if len(selected) > MAX_FILES or len({item[2] for item in selected}) != len(selected):
        raise ValueError("Too many or duplicate archive source paths")
    return sorted(selected, key=lambda item: item[2]), accepted, excluded


def scan_stream(stream, *, check_credentials):
    digest, total, overlap = hashlib.sha256(), 0, b""
    while chunk := stream.read(CHUNK):
        if check_credentials and SECRET_PATTERN.search(overlap + chunk):
            raise ValueError("Credential-like content detected; no bytes or values disclosed")
        overlap = (overlap + chunk)[-512:]
        digest.update(chunk)
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise ValueError("Individual source exceeds bounded experiment-archive size")
    return digest.hexdigest(), total


def inspect_files(selected):
    entries, total = [], 0
    for root, relative, archive_name in selected:
        with open_source(root, relative) as stream:
            before = os.fstat(stream.fileno())
            checksum, size = scan_stream(stream, check_credentials=Path(relative).suffix.lower() in TEXT_SUFFIXES)
            if identity(before) != identity(os.fstat(stream.fileno())) or size != before.st_size:
                raise ValueError(f"Source changed during hashing: {archive_name}")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Archive exceeds bounded experiment-only size")
        entries.append(dict(name=archive_name, source_root=str(root), source_relative=relative,
                            sha256=checksum, size=size, identity=identity(before),
                            mode=0o755 if before.st_mode & stat.S_IXUSR else 0o644))
    return entries


def fsync_json_exclusive(path, value):
    no_links(path.parent, owner=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())


def ensure_own_dir(path):
    no_links(path, owner=True)
    path.mkdir(parents=False, exist_ok=True)
    no_links(path, owner=True)
    if not path.is_dir():
        raise ValueError("Expected own directory")


def publish_record(stage, archive_id, value, suffix):
    # One immutable filename for each unique producer attempt. Readers accept
    # only valid JSON containing complete status + matching hashes, never .tar
    # existence. Partial records fail closed and are never overwritten/reused.
    fsync_json_exclusive(stage / "records" / f"{archive_id}.{suffix}.json", value)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()
        self.size = 0

    def read(self, count=-1):
        data = self.stream.read(count)
        self.sha.update(data)
        self.size += len(data)
        return data


def write_archive(path, selected, entries, manifest):
    by_name = {entry["name"]: entry for entry in entries}
    with path.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
            payload = canonical_json(manifest)
            info = tarfile.TarInfo("archive_manifest.json")
            info.size, info.mode = len(payload), 0o644
            archive.addfile(info, io.BytesIO(payload))
            for root, relative, name in selected:
                entry = by_name[name]
                with open_source(root, relative) as source:
                    if identity(os.fstat(source.fileno())) != entry["identity"]:
                        raise ValueError(f"Source identity changed before copying: {name}")
                    reader = HashingReader(source)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = entry["size"], entry["mode"]
                    archive.addfile(info, reader)
                    if reader.sha.hexdigest() != entry["sha256"] or reader.size != entry["size"] or identity(os.fstat(source.fileno())) != entry["identity"]:
                        raise ValueError(f"Source mutated during archive: {name}")
        output.flush()
        os.fsync(output.fileno())


def hash_file(path):
    with open_source(path.parent, path.name) as stream:
        before = identity(os.fstat(stream.fileno()))
        if before[2] > MAX_TOTAL_BYTES + 64 * 1024 * 1024:
            raise ValueError("Archive exceeds bounded total size before reading")
        hasher = hashlib.sha256()
        while chunk := stream.read(CHUNK):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        if before != identity(os.fstat(stream.fileno())):
            raise ValueError("File changed while hashing")
        return digest, before


def produce(*, inspect_only=False):
    require_host(SOURCE_HOST)
    # Holding all three read-only shared locks prevents a new matching
    # supervisor starting while this snapshot is selected, hashed and sealed.
    with quiescence_locks(SOURCE_RESULTS):
        selected, accepted, excluded = select_sources(SOURCE_CODE, SOURCE_RESULTS)
        entries = inspect_files(selected)
        plan = dict(status="ready_to_seal", host=SOURCE_HOST, completed_runs=accepted,
                    excluded_unsealed_runs=[e for e in excluded if e["type"] == "unsealed_run"],
                    excluded_runtime_caches=[e for e in excluded if e["type"] == "runtime_cache"],
                    file_count=len(entries), bytes=sum(e["size"] for e in entries))
        if inspect_only:
            return plan
        no_links(STAGE.parent, owner=True)
        ensure_own_dir(STAGE)
        ensure_own_dir(STAGE / "objects")
        ensure_own_dir(STAGE / "records")
        archive_id = uuid.uuid4().hex
        path = STAGE / "objects" / f"experiments-{archive_id}.tar"
        manifest = {"schema": 1, "archive_id": archive_id, "source_host": SOURCE_HOST,
                    "created_at": now(), "completed_runs": accepted,
                    "excluded_unsealed_runs": plan["excluded_unsealed_runs"],
                    "excluded_runtime_caches": plan["excluded_runtime_caches"],
                    "files": entries, "scope": "completed run evidence and bounded experiment code only; no weights/env/credentials"}
        try:
            write_archive(path, selected, entries, manifest)
            # Re-select and re-hash *every* file: changes/new names during
            # snapshot invalidate the whole unpublished object, never resume.
            selected_after, accepted_after, excluded_after = select_sources(SOURCE_CODE, SOURCE_RESULTS)
            if selected_after != selected or accepted_after != accepted or excluded_after != excluded or inspect_files(selected_after) != entries:
                raise ValueError("Source set/content changed before sealing")
            checksum, archive_stat = hash_file(path)
            record = {**plan, "status": "complete", "archive_id": archive_id, "sealed_at": now(),
                      "archive_name": path.name, "archive_sha256": checksum, "archive_bytes": archive_stat[2],
                      "manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest()}
            publish_record(STAGE, archive_id, record, "complete")
            return record
        except BaseException as error:
            publish_record(STAGE, archive_id, {"status": "failed", "archive_id": archive_id,
                           "failed_at": now(), "error": str(error), "source_preserved": True}, "failed")
            raise


def require_host(expected):
    if sys.platform != "linux" or socket.gethostname() != expected:
        raise ValueError(f"This operation is bound to Linux host {expected}")


def validate_archive_id(archive_id):
    if not isinstance(archive_id, str) or re.fullmatch(r"[0-9a-f]{32}", archive_id) is None:
        raise ValueError("Archive ID must be one generated lowercase UUID hex")


def inspect_tar(path, record):
    with open_source(path.parent, path.name) as stream, tarfile.open(fileobj=stream, mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > MAX_FILES + 1 or len({m.name for m in members}) != len(members):
            raise ValueError("Too many/duplicate archive members")
        for member in members:
            safe_relative(member.name)
            if not member.isfile() or member.size < 0 or member.size > MAX_FILE_BYTES:
                raise ValueError("Archive accepts regular bounded files only; no links/devices/directories")
        by_name = {member.name: member for member in members}
        top = by_name.get("archive_manifest.json")
        if top is None or top.size > 16 * 1024 * 1024:
            raise ValueError("Missing/oversized archive manifest")
        payload = archive.extractfile(top).read()
        if hashlib.sha256(payload).hexdigest() != record["manifest_sha256"]:
            raise ValueError("Manifest hash mismatch")
        manifest = json.loads(payload)
        if manifest.get("schema") != 1 or manifest.get("archive_id") != record["archive_id"] or manifest.get("source_host") != SOURCE_HOST:
            raise ValueError("Manifest identity mismatch")
        entries = manifest.get("files")
        if not isinstance(entries, list) or len(entries) != record["file_count"]:
            raise ValueError("Manifest file count mismatch")
        names = set()
        for entry in entries:
            name = entry["name"]
            parts = safe_relative(name).parts
            if parts[0] not in ("code", "results") or len(parts) < 2 or name in names:
                raise ValueError("Unexpected/duplicate destination scope")
            source_policy("/".join(parts[1:]))
            names.add(name)
            member = by_name.get(name)
            if member is None or type(entry["size"]) is not int or entry["size"] != member.size or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
                raise ValueError("Manifest/header file mismatch")
        if names | {"archive_manifest.json"} != set(by_name) or sum(e["size"] for e in entries) > MAX_TOTAL_BYTES:
            raise ValueError("Extra/missing entries or excessive archive size")
        for name in names:
            if any(str(parent) in names for parent in PurePosixPath(name).parents):
                raise ValueError("Archive file collides with another file's parent directory")
        return manifest


def atomic_rename_new(source, dest):
    """No-overwrite directory publication, even if another process races us."""
    if os.name == "nt":  # Toy platform; Windows rename never replaces target.
        os.rename(source, dest)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    func = getattr(libc, "renameat2", None)
    if func is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) unavailable; refusing unsafe fallback")
    func.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    func.restype = ctypes.c_int
    if func(-100, os.fsencode(source), -100, os.fsencode(dest), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


@contextlib.contextmanager
def create_destination_file(destination, relative, mode):
    relative = safe_relative(relative)
    if os.name == "nt":  # Windows fixture branch; production is Linux-only.
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        no_links(target.parent, owner=True)
        with target.open("xb") as output:
            yield output
        os.chmod(target, mode & 0o755)
        return
    fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in relative.parts[:-1]:
            try:
                os.mkdir(name, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        file_fd = os.open(relative.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(file_fd, "wb") as output:
            yield output
            os.fchmod(output.fileno(), mode & 0o755)
    finally:
        os.close(fd)


def extract_verified(path, destination, manifest):
    destination.mkdir(exist_ok=False)
    fsync_json_exclusive(destination / "ARCHIVE_OWNER.json", {"owner": "zhonghao", "archive_id": manifest["archive_id"], "created_at": now()})
    with open_source(path.parent, path.name) as stream, tarfile.open(fileobj=stream, mode="r:") as archive:
        for entry in manifest["files"]:
            digest, size = hashlib.sha256(), 0
            with archive.extractfile(entry["name"]) as source, create_destination_file(destination, entry["name"], entry["mode"]) as output:
                while chunk := source.read(CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Extracted hash/size mismatch: {entry['name']}")
    fsync_json_exclusive(destination / "archive_manifest.json", manifest)


def receive(archive_id):
    require_host(DEST_HOST)
    validate_archive_id(archive_id)
    for parent in (STAGE, STAGE / "objects", STAGE / "records"):
        no_links(parent, owner=True)
    record_path = no_links(STAGE / "records" / f"{archive_id}.complete.json", owner=True)
    # No polling/background wait and no inference from a partial tar's size.
    record = json.loads(record_path.read_text())
    if record.get("status") != "complete" or record.get("archive_id") != archive_id:
        raise ValueError("Archive lacks matching complete producer record")
    expected_name = f"experiments-{archive_id}.tar"
    if record.get("archive_name") != expected_name:
        raise ValueError("Record archive path mismatch")
    source = no_links(STAGE / "objects" / expected_name, owner=True)
    checksum, source_stat = hash_file(source)
    if checksum != record["archive_sha256"] or source_stat[2] != record["archive_bytes"]:
        raise ValueError("Transferred immutable archive hash/size mismatch")
    manifest = inspect_tar(source, record)
    no_links(DEST_PARENT.parent, owner=True)
    ensure_own_dir(DEST_PARENT)
    final = DEST_PARENT / f"30213-{archive_id}"
    pending = DEST_PARENT / f".partial-{archive_id}"
    if final.exists() or final.is_symlink() or pending.exists() or pending.is_symlink():
        raise ValueError("Destination already exists; no overwrite, resume or automatic deletion")
    try:
        extract_verified(source, pending, manifest)
        checksum_after, stat_after = hash_file(source)
        if source_stat != stat_after or checksum_after != checksum:
            raise ValueError("Archive changed during receiving")
        fsync_json_exclusive(pending / "ARCHIVE_COMPLETE.json", {**record, "received_at": now(), "destination_host": DEST_HOST,
                                                               "source_preserved": True, "result_not_recomputed": True})
        atomic_rename_new(pending, final)
        return {"status": "complete", "archive_id": archive_id, "destination": str(final), "files": len(manifest["files"]), "source_preserved": True}
    except BaseException as error:
        if pending.is_dir() and not (pending / "ARCHIVE_FAILED.json").exists():
            fsync_json_exclusive(pending / "ARCHIVE_FAILED.json", {"status": "failed", "archive_id": archive_id,
                                                                  "error": str(error), "at": now()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "produce", "receive"))
    parser.add_argument("--archive-id")
    args = parser.parse_args()
    if (args.mode == "receive") != bool(args.archive_id):
        parser.error("--archive-id is required only for receive")
    result = receive(args.archive_id) if args.mode == "receive" else produce(inspect_only=args.mode == "inspect")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
