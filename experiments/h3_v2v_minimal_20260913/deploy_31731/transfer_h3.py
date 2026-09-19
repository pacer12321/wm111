#!/usr/bin/env python3
"""Copy the approved H3 runtime through shared NFS, without modifying sources.

Run produce on 30213 and consume on 31731. This transfers files only; it never
extracts archives, changes drivers, starts a model, or claims any NPU device.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

DEPLOYMENT = "zhonghao_h3_31731_20260913_v2"
STAGE = Path("/temp/zhonghao/h3_31731_20260913_v2")
TARGET = Path("/cache/zhonghao/h3")
SOURCE_ENV = Path("/cache/yunfeng/envs/minimax-h3-npu")
SOURCE_SRC = Path("/cache/yunfeng/minimax_h3_npu/src")
SOURCE_HOST = "ma-job-e03cb638-a995-4fcf-aad0-ebcf50a7c3ff-worker-0"
TARGET_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
PACK_TOOL = Path("/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/deploy_31731/pack_tool")
CHUNK = 8 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ssh", ".aws", ".azure"}
SKIP_FILES = {".env", ".netrc", ".pypirc", "credentials", "credentials.json", "id_rsa", "id_ed25519"}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    canonical(path.parent)
    if path.is_relative_to(STAGE):
        # This shared archival NFS permits creation but rejects rename. Respect
        # that policy with immutable records and a separately-created commit
        # marker. Readers never consume an uncommitted record.
        record = path.with_name(path.name + f".version-{time.time_ns():020d}-" + uuid.uuid4().hex)
        with record.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        with record.with_name(record.name + ".ready").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        return
    tmp = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    if path.is_relative_to(STAGE):
        committed = sorted(path.parent.glob(path.name + ".version-*.ready"))
        if not committed:
            return default
        path = committed[-1].with_name(committed[-1].name[:-6])
    if not path.exists():
        return default
    if path.is_symlink():
        raise RuntimeError(f"Symlink metadata refused: {path}")
    return json.loads(path.read_text())


def canonical(path):
    if path.resolve() != path:
        raise RuntimeError(f"Noncanonical/symlink destination refused: {path}")


def claim(root):
    canonical(root)
    marker = root / "deployment_owner.json"
    if root.exists():
        owner = read_json(marker)
        if not owner or owner.get("deployment") != DEPLOYMENT:
            raise RuntimeError(f"Refusing existing unowned destination: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(marker, {"deployment": DEPLOYMENT, "created_at": now()})
    canonical(root)


def within(root, relative, allow_existing_symlink=False):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe relative path: {relative}")
    result = root / relative
    if not result.parent.resolve().is_relative_to(root):
        raise RuntimeError(f"Destination parent escapes root: {result}")
    if result.is_symlink() and not allow_existing_symlink:
        raise RuntimeError(f"Refusing to overwrite symlink: {result}")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Transfer:
    def __init__(self, mode):
        self.mode = mode
        self.root = STAGE if mode == "produce" else TARGET
        expected = SOURCE_HOST if mode == "produce" else TARGET_HOST
        if socket.gethostname() != expected:
            raise RuntimeError(f"Wrong node for {mode}: {socket.gethostname()}")
        claim(self.root)
        lock_path = self.root / f"{mode}.lock"
        self.lock = os.fdopen(os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600), "w")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = {"deployment": DEPLOYMENT, "role": mode, "pid": os.getpid(),
                      "hostname": socket.gethostname(), "started_at": now(),
                      "bytes_copied_this_process": 0, "files_completed": 0}
        self.last_status = 0.0
        self.update("starting")
        if shutil.disk_usage(self.root).free < 200 * 1024**3:
            self.update("failed", error="Less than 200 GiB free; refusing migration")
            raise RuntimeError("Less than 200 GiB free; refusing migration")

    def update(self, phase=None, **kwargs):
        self.state.update(kwargs)
        if phase:
            self.state["phase"] = phase
        self.state["updated_at"] = now()
        atomic_json(self.root / f"{self.mode}_status.json", self.state)
        self.last_status = time.monotonic()

    def copy_file(self, source, dest, expected=None):
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Not a regular source file: {source}")
        before = source.stat()
        dest.parent.mkdir(parents=True, exist_ok=True)
        canonical(dest.parent)
        if dest.exists():
            digest = sha256(dest)
            src_digest = expected or sha256(source)
            if dest.stat().st_size == before.st_size and digest == src_digest:
                return digest, before.st_size
            raise RuntimeError(f"Existing destination differs; no overwrite: {dest}")
        partial = dest if self.mode == "produce" else dest.with_name(dest.name + ".h3copy-partial-" + uuid.uuid4().hex)
        digest = hashlib.sha256()
        copied = 0
        # Never truncate a pre-existing partial; retain failed partials for review.
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as src, partial.open("xb") as out:
            for chunk in iter(lambda: src.read(CHUNK), b""):
                out.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                self.state["bytes_copied_this_process"] += len(chunk)
                if time.monotonic() - self.last_status >= 5:
                    self.update(current_file=str(dest.relative_to(self.root)),
                                current_file_bytes=copied, current_file_size=before.st_size)
            out.flush()
            os.fsync(out.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"Source changed while reading: {source}")
        digest = digest.hexdigest()
        if expected is not None and digest != expected:
            raise RuntimeError(f"Transport SHA256 mismatch: {source}")
        if copied != before.st_size or partial.stat().st_size != copied:
            raise RuntimeError(f"Copy size mismatch: {source}")
        if self.mode == "consume":
            os.chmod(partial, stat.S_IMODE(before.st_mode) & 0o777)
            # Local /cache supports no-overwrite atomic link publication.
            os.link(partial, dest, follow_symlinks=False)
            partial.unlink()
        self.state["files_completed"] += 1
        return digest, copied

    def produce(self):
        manifest_path = STAGE / "manifest.json"
        manifest = {"deployment": DEPLOYMENT, "source_host": SOURCE_HOST,
                    "target_host": TARGET_HOST, "complete": False, "entries": []}
        previous_manifest = read_json(manifest_path)
        if previous_manifest:
            manifest = previous_manifest
            if manifest["deployment"] != DEPLOYMENT:
                raise RuntimeError("Manifest belongs to another deployment")
        done = {entry["path"] for entry in manifest["entries"]}

        def publish(source, relative):
            relative = str(relative)
            if relative in done:
                return
            if source.is_symlink():
                entry = {"path": relative, "kind": "symlink", "link": os.readlink(source)}
            else:
                storage_path = "objects/" + uuid.uuid4().hex + ".blob"
                target = within(STAGE, storage_path)
                digest, size = self.copy_file(source, target)
                entry = {"path": relative, "kind": "file", "size": size, "sha256": digest,
                         "storage_path": storage_path, "mode": stat.S_IMODE(source.stat().st_mode) & 0o777}
            manifest["entries"].append(entry)
            done.add(relative)
            if len(manifest["entries"]) % 100 == 0 or time.monotonic() - published[0] >= 3:
                manifest["updated_at"] = now()
                atomic_json(manifest_path, manifest)
                published[0] = time.monotonic()

        published = [0.0]
        archive = STAGE / "objects" / ("env-" + uuid.uuid4().hex + ".tar")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if "env.tar" not in done:
            self.update("packing_environment", target_prefix=str(TARGET / "env"))
            audit = subprocess.run([str(SOURCE_ENV / "bin/python"), "-B",
                                    str(Path(__file__).with_name("audit_pack_environment.py"))],
                                   capture_output=True, text=True, check=True)
            audit_record = json.loads(audit.stdout)
            if audit_record.get("status") != "passed":
                raise RuntimeError("Source environment metadata audit failed")
            atomic_json(STAGE / "source_environment_audit.json", audit_record)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PACK_TOOL)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with (STAGE / ("pack-" + uuid.uuid4().hex + ".log")).open("x") as log:
                subprocess.run([str(SOURCE_ENV / "bin/python"), "-B",
                                str(Path(__file__).with_name("pack_h3_actual_environment.py")),
                                "--output", str(archive)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest["entries"].append({"path": "env.tar", "kind": "file", "size": archive.stat().st_size,
                                        "sha256": sha256(archive), "storage_path": str(archive.relative_to(STAGE)),
                                        "mode": 0o640})
            done.add("env.tar")
            atomic_json(manifest_path, manifest)
        self.update("copying_sources_and_weights")
        trees = [(SOURCE_SRC / name, Path("src") / name) for name in ("vllm", "vllm-ascend", "vllm-omni")]
        trees += [(Path("/cache/yunfeng/models/MiniMax-H3/Ref2VA"), Path("models/MiniMax-H3/Ref2VA")),
                  (Path("/cache/yunfeng/models/OpenVDN-vdn-minimax-h3/stage-b-step-2000"),
                   Path("models/OpenVDN-vdn-minimax-h3/stage-b-step-2000"))]
        for source_root, relative_root in trees:
            if not source_root.is_dir():
                raise RuntimeError(f"Missing source tree: {source_root}")
            print(f"Copying approved tree {source_root}", flush=True)
            for root, dirs, files in os.walk(source_root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for directory in list(dirs):
                    path = Path(root) / directory
                    if path.is_symlink():
                        publish(path, relative_root / path.relative_to(source_root))
                        dirs.remove(directory)
                for name in sorted(files):
                    if name in SKIP_FILES:
                        continue
                    path = Path(root) / name
                    publish(path, relative_root / path.relative_to(source_root))
            atomic_json(manifest_path, manifest)
        publish(Path("/cache/yunfeng/minimax_h3_npu/output/minimax_h3_t2va_50step.mp4"),
                Path("data/minimax_h3_t2va_50step.mp4"))
        manifest.update(complete=True, completed_at=now())
        atomic_json(manifest_path, manifest)
        self.update("completed", manifest_entries=len(manifest["entries"]))

    def consume(self):
        completed_path = TARGET / "transfer_verified.json"
        completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}
        self.update("waiting_for_producer")
        started = time.monotonic()
        while time.monotonic() - started < 12 * 3600:
            status_path = STAGE / "produce_status.json"
            producer_status = read_json(status_path, {})
            if producer_status.get("phase") == "failed":
                raise RuntimeError("Source producer failed; inspect its status and log")
            manifest_path = STAGE / "manifest.json"
            manifest = read_json(manifest_path)
            if not manifest:
                if time.monotonic() - self.last_status >= 30:
                    self.update("waiting_for_producer")
                time.sleep(5)
                continue
            if manifest["deployment"] != DEPLOYMENT or manifest["target_host"] != TARGET_HOST:
                raise RuntimeError("Wrong transfer manifest")
            self.update("copying_from_shared_staging", published_entries=len(manifest["entries"]))
            for entry in manifest["entries"]:
                name = entry["path"]
                if name in completed:
                    continue
                dest = within(TARGET, name, allow_existing_symlink=entry["kind"] == "symlink")
                if entry["kind"] == "file":
                    storage = Path(entry["storage_path"])
                    if len(storage.parts) != 2 or storage.parts[0] != "objects":
                        raise RuntimeError("Invalid immutable object path")
                    self.copy_file(within(STAGE, storage), dest, entry["sha256"])
                    os.chmod(dest, entry["mode"] & 0o777)
                elif entry["kind"] == "symlink":
                    link = entry["link"].replace(str(SOURCE_ENV), str(TARGET / "env"))
                    link = link.replace(str(SOURCE_SRC), str(TARGET / "src"))
                    link_path = Path(link) if os.path.isabs(link) else dest.parent / link
                    if not link_path.resolve().is_relative_to(TARGET):
                        raise RuntimeError(f"Source symlink requires manual review: {name} -> {link}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.is_symlink():
                        if os.readlink(dest) != link:
                            raise RuntimeError(f"Existing symlink differs: {dest}")
                    elif dest.exists():
                        raise RuntimeError(f"Refusing existing link destination: {dest}")
                    else:
                        os.symlink(link, dest)
                else:
                    raise RuntimeError(f"Unknown manifest entry type: {entry}")
                completed[name] = entry
                if len(completed) % 100 == 0:
                    atomic_json(completed_path, completed)
            atomic_json(completed_path, completed)
            if manifest["complete"] and len(completed) == len(manifest["entries"]):
                self.update("transfer_completed_extraction_and_validation_required", verified_entries=len(completed))
                return
            status_path = STAGE / "produce_status.json"
            if read_json(status_path, {}).get("phase") == "failed":
                raise RuntimeError("Source producer failed; inspect its status and log")
            time.sleep(5)
        raise TimeoutError("Transfer exceeded 12-hour guard")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("produce", "consume"))
    args = parser.parse_args()
    transfer = None
    try:
        transfer = Transfer(args.mode)
        getattr(transfer, args.mode)()
    except BaseException as exc:
        if transfer is not None:
            transfer.update("failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
