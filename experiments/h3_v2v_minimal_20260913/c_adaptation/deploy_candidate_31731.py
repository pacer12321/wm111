#!/usr/bin/env python3
"""Deploy one NEW private C source tree from B plus six pinned C overlay files.

Default is read-only inspect. --apply creates c_v1 exclusively, copies (never
hardlinks), verifies all bytes and source stability, then publishes the vendor
directory atomically. No NPU, model, shell, network, environment, or launcher is
invoked. Failed staging is retained and blocks automatic retry.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import stat
import uuid

INSTALL = Path("/cache/zhonghao/h3")
EXPECTED_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
EXPECTED_BOOT = "8e904236-bb23-44b5-b7ba-af73ba5c1f77"
MODEL_REL = Path("vllm_omni/diffusion/models/minimax_h3")
MAX_FILE = 20 * 1024**2
MAX_TREE = 128 * 1024**2
COPY_CHUNK = 512 * 1024
IGNORED_NAMES = {".git", "__pycache__"}
B_PINS = {
    "openvdn_npu.py": "4808ef3017343904d8b52cda9cfebc94798f4d609f8ec19e63c59b92abf3381f",
    "openvdn_checkpoint.py": "fbe011bae524bea16f54a14032e61e82f2c68aa4da6d426b98a13e097fb8f19f",
    "minimax_h3_transformer.py": "c1b6b89a1be2a0cac3a2bf3add15ed73d19340086ede91e4a085a218581a0c83",
    "pipeline_minimax_h3.py": "de86c6d435254db31e4c094f13d06f71aa87c5fe26d5c344b9c97c610febd690",
}
C_PINS = {
    "openvdn_npu.py": B_PINS["openvdn_npu.py"],
    "openvdn_checkpoint.py": B_PINS["openvdn_checkpoint.py"],
    "minimax_h3_transformer.py": "750056c7166238dc6275e9163f751034ecf2f8e3b1c03af400ea6fe9c83043df",
    "pipeline_minimax_h3.py": "9a124e4bef1421c2f0a3e7c45afa4260f975eee09a4565e7343e07daa8648414",
    "strict_source_layout.py": "0986978dbd005ec97af88613073465b5eada55434b0293b8098a5f40bfe81028",
    "strict_source_attention.py": "b96d20631f4cf9b5a06ea562e63c127c3e1e7c22fef0eb234aac9b1929941158",
}


@dataclass(frozen=True)
class Paths:
    private_root: Path = INSTALL
    source_b: Path = INSTALL / "candidates/b_v1/vllm-omni"
    source_c: Path = INSTALL / "c_adaptation/candidate"
    destination_root: Path = INSTALL / "candidates/c_v1"

    @property
    def destination_vendor(self):
        return self.destination_root / "vllm-omni"

    @property
    def staging_vendor(self):
        return self.destination_root / "vllm-omni.incomplete"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require_host(expected=None):
    actual = dict(hostname=socket.gethostname(), machine=os.uname().machine,
                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    if actual != dict(hostname=EXPECTED_HOST, machine="aarch64", boot_id=EXPECTED_BOOT):
        raise RuntimeError("Only the reviewed 31731 host and boot are allowed; no automatic rebind")
    if expected is not None and expected != actual:
        raise RuntimeError("Host/boot identity changed during deployment")
    return actual


def canonical(path, private_root):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or not path.is_relative_to(private_root)
            or path.resolve() != path):
        raise RuntimeError(f"Noncanonical or out-of-scope private path: {path}")
    # resolve() equality catches symlink ancestors on both existing and new paths.
    return path


def validate_paths(paths):
    root = paths.private_root
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise RuntimeError("Private installation root is unavailable/noncanonical")
    for path in (paths.source_b, paths.source_c, paths.destination_root,
                 paths.destination_vendor, paths.staging_vendor):
        canonical(path, root)
        if path == root:
            raise RuntimeError("A broad installation root is not a deployment target")
    if not paths.source_b.is_dir() or not paths.source_c.is_dir():
        raise RuntimeError("Both existing source directories are required")
    sources = (paths.source_b, paths.source_c)
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((sources[0], sources[1]),
            (paths.destination_root, sources[0]), (paths.destination_root, sources[1]))):
        raise RuntimeError("Source and destination trees must be disjoint")
    if paths.destination_root.exists() or paths.destination_root.is_symlink():
        raise FileExistsError("c_v1 already exists: no overwrite, resume or automatic cleanup")
    if not paths.destination_root.parent.is_dir():
        raise RuntimeError("Existing private candidates parent is required")


def stamp(info):
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, mode=stat.S_IMODE(info.st_mode), links=info.st_nlink)


def regular_record(path, max_file=MAX_FILE):
    if path.resolve() != path:
        raise RuntimeError(f"Symlink ancestor in source path: {path}")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Symlink or nonregular source file: {path}")
    if before.st_size > max_file:
        raise RuntimeError(f"Source file exceeds {max_file} bytes: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    h, total = hashlib.sha256(), 0
    with os.fdopen(fd, "rb") as stream:
        if stamp(os.fstat(stream.fileno())) != stamp(before):
            raise RuntimeError(f"Source identity changed before hashing: {path}")
        for block in iter(lambda: stream.read(COPY_CHUNK), b""):
            total += len(block)
            if total > max_file:
                raise RuntimeError(f"Source grew beyond per-file cap: {path}")
            h.update(block)
        if stamp(os.fstat(stream.fileno())) != stamp(before):
            raise RuntimeError(f"Source changed during hashing: {path}")
    if path.resolve() != path or stamp(path.lstat()) != stamp(before) or total != before.st_size:
        raise RuntimeError(f"Source changed after hashing: {path}")
    return dict(size=total, sha256=h.hexdigest(), identity=stamp(before))


def inventory(root, *, max_file=MAX_FILE, max_tree=MAX_TREE, ignore_runtime=True):
    """Reject symlinks/special entries even when their names would be ignored."""
    if not stat.S_ISDIR(root.lstat().st_mode) or root.resolve() != root:
        raise RuntimeError("Inventory root is not a canonical real directory")
    files, directories, excluded = {}, [], []
    total = 0

    def visit(directory):
        nonlocal total
        if directory.resolve() != directory or not stat.S_ISDIR(directory.lstat().st_mode):
            raise RuntimeError("Source directory changed into a link/special entry")
        for path in sorted(directory.iterdir(), key=lambda x: x.name):
            info = path.lstat()
            rel = path.relative_to(root).as_posix()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise RuntimeError(f"Symlink or special entry in source tree: {rel}")
            if ignore_runtime and (path.name in IGNORED_NAMES or (path.is_file() and path.suffix == ".pyc")):
                excluded.append(dict(path=rel, reason="git_or_python_runtime_cache"))
                continue
            if stat.S_ISDIR(info.st_mode):
                directories.append(rel)
                visit(path)
            else:
                record = regular_record(path, max_file)
                total += record["size"]
                if total > max_tree:
                    raise RuntimeError(f"Source tree exceeds {max_tree} cumulative bytes")
                files[rel] = record
    visit(root)
    return dict(files=files, directories=sorted(directories), excluded=excluded,
                file_count=len(files), total_bytes=total)


def same_included(a, b):
    # Excluded cache contents and ignored-entry presence are intentionally not
    # a code-stability claim. Included files and directories must be identical.
    return a["files"] == b["files"] and a["directories"] == b["directories"]


def overlay_inventory(root, pins, max_file=MAX_FILE):
    records = {}
    for name, expected in pins.items():
        if Path(name).name != name or name in ("", ".", ".."):
            raise RuntimeError("Overlay names must be plain filenames")
        record = regular_record(root / name, max_file)
        if record["sha256"] != expected:
            raise RuntimeError(f"Pinned C overlay hash mismatch: {name}")
        records[name] = record
    return records


def expected_files(b, c):
    expected = {name: dict(size=record["size"], sha256=record["sha256"], origin="B")
                for name, record in b["files"].items()}
    for name, record in c.items():
        expected[(MODEL_REL / name).as_posix()] = dict(size=record["size"], sha256=record["sha256"], origin="C_overlay")
    return expected


def inspect(paths, *, b_pins=B_PINS, c_pins=C_PINS, max_file=MAX_FILE, max_tree=MAX_TREE):
    validate_paths(paths)
    b = inventory(paths.source_b, max_file=max_file, max_tree=max_tree)
    for name, expected in b_pins.items():
        if b["files"].get((MODEL_REL / name).as_posix(), {}).get("sha256") != expected:
            raise RuntimeError(f"Pinned B source hash mismatch: {name}")
    c = overlay_inventory(paths.source_c, c_pins, max_file)
    expected = expected_files(b, c)
    if sum(record["size"] for record in expected.values()) > max_tree:
        raise RuntimeError("Combined C deployment would exceed cumulative size cap")
    return dict(b=b, c=c, expected=expected)


def write_json_atomic(path, payload):
    if path.parent.resolve() != path.parent or not path.parent.is_dir():
        raise RuntimeError("Manifest parent changed/noncanonical")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def copy_new(source, destination, expected, *, max_file=MAX_FILE):
    """Copy verified source bytes into an exclusive new file; never a link."""
    if source.resolve() != source or destination.parent.resolve() != destination.parent:
        raise RuntimeError("Copy source/destination parent changed into a symlink")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    h, total = hashlib.sha256(), 0
    with os.fdopen(fd, "rb") as inp:
        if stamp(os.fstat(inp.fileno())) != expected["identity"]:
            raise RuntimeError(f"Source changed before copying: {source}")
        with destination.open("xb") as out:
            for block in iter(lambda: inp.read(COPY_CHUNK), b""):
                total += len(block)
                if total > max_file:
                    raise RuntimeError("Source exceeded per-file cap while copying")
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if stamp(os.fstat(inp.fileno())) != expected["identity"]:
            raise RuntimeError(f"Source changed during copying: {source}")
    if (source.resolve() != source or destination.parent.resolve() != destination.parent
            or total != expected["size"] or h.hexdigest() != expected["sha256"] or stamp(source.lstat()) != expected["identity"]):
        raise RuntimeError(f"Copied source content/identity drifted: {source}")
    # Preserve executability, never propagate setuid/setgid/sticky permissions.
    os.chmod(destination, 0o755 if expected["identity"]["mode"] & 0o111 else 0o644)
    current = regular_record(destination, max_file)
    if (current["size"], current["sha256"]) != (expected["size"], expected["sha256"]):
        raise RuntimeError(f"Destination copy verification failed: {destination}")
    if (current["identity"]["device"], current["identity"]["inode"]) == (
            expected["identity"]["device"], expected["identity"]["inode"]):
        raise RuntimeError("Destination is unexpectedly the same inode as source")


def rename_new_dir(source, destination):
    """Atomic publication which cannot replace even an empty raced-in folder."""
    if os.name == "nt":
        os.rename(source, destination)  # Windows rename is no-replace.
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace renameat2 unavailable; refuse unsafe publication")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def verify_destination(root, expected, expected_dirs, *, max_file=MAX_FILE, max_tree=MAX_TREE):
    actual = inventory(root, max_file=max_file, max_tree=max_tree, ignore_runtime=False)
    if any(record["identity"]["links"] != 1 for record in actual["files"].values()):
        raise RuntimeError("Destination contains a hardlinked file")
    contents = {name: {key: record[key] for key in ("size", "sha256")} for name, record in actual["files"].items()}
    wanted = {name: {key: record[key] for key in ("size", "sha256")} for name, record in expected.items()}
    if contents != wanted or actual["directories"] != expected_dirs:
        raise RuntimeError("Destination contains missing/extra/changed files or directories")
    return actual


def deploy(paths, host_check, *, b_pins=B_PINS, c_pins=C_PINS,
           max_file=MAX_FILE, max_tree=MAX_TREE, copy_fn=copy_new):
    """Engine is path-injectable only for CPU tests; CLI has no path overrides."""
    host = host_check()
    proof = inspect(paths, b_pins=b_pins, c_pins=c_pins, max_file=max_file, max_tree=max_tree)
    host_check(host)
    # Revalidate immediately before the exclusive reservation. A concurrent or
    # previously failed deployment is not resumed or overwritten.
    validate_paths(paths)
    paths.destination_root.mkdir(mode=0o700, exist_ok=False)
    manifest = dict(schema=1, run_id=uuid.uuid4().hex, state="staging", host=host,
        started_at=now(), source_b=str(paths.source_b), source_c=str(paths.source_c),
        destination=str(paths.destination_vendor), source_b_before=proof["b"],
        source_c_before=proof["c"], expected_files=proof["expected"],
        file_summary=dict(file_count=len(proof["expected"]), total_bytes=sum(x["size"] for x in proof["expected"].values())),
        limits=dict(max_file_bytes=max_file, max_tree_bytes=max_tree),
        source_preserved=True, copied_without_hardlinks=True, npu_invoked=False,
        note="Only included source files/directories are stability-checked; .git and Python caches are excluded.")
    manifest_path = paths.destination_root / "deploy_manifest.json"
    published = False
    try:
        write_json_atomic(manifest_path, manifest)
        paths.staging_vendor.mkdir(exist_ok=False)
        for name in sorted(proof["b"]["directories"], key=lambda s: (len(Path(s).parts), s)):
            (paths.staging_vendor / name).mkdir(exist_ok=False)
        for name, record in proof["b"]["files"].items():
            copy_fn(paths.source_b / name, paths.staging_vendor / name, record, max_file=max_file)
        for name, record in proof["c"].items():
            destination = paths.staging_vendor / MODEL_REL / name
            old = proof["b"]["files"].get((MODEL_REL / name).as_posix())
            if old is not None:
                actual = regular_record(destination, max_file)
                if (actual["size"], actual["sha256"]) != (old["size"], old["sha256"]):
                    raise RuntimeError("Own staged B file changed before authorized overlay")
                temporary = destination.with_name(destination.name + ".overlay_" + manifest["run_id"])
                copy_fn(paths.source_c / name, temporary, record, max_file=max_file)
                os.replace(temporary, destination)  # only our verified new B copy
            else:
                copy_fn(paths.source_c / name, destination, record, max_file=max_file)
        verified = verify_destination(paths.staging_vendor, proof["expected"], proof["b"]["directories"],
                                      max_file=max_file, max_tree=max_tree)
        b_after = inventory(paths.source_b, max_file=max_file, max_tree=max_tree)
        c_after = overlay_inventory(paths.source_c, c_pins, max_file)
        if not same_included(proof["b"], b_after) or proof["c"] != c_after:
            raise RuntimeError("Source changed during deployment; refuse publication")
        host_check(host)
        if paths.destination_vendor.exists() or paths.destination_vendor.is_symlink():
            raise FileExistsError("Destination appeared during staging; refuse publication")
        canonical(paths.destination_root, paths.private_root)
        rename_new_dir(paths.staging_vendor, paths.destination_vendor)
        published = True
        # Verify the published path, then repeat source stability before claiming
        # completion. Failure after publication leaves an explicit failed record.
        verified = verify_destination(paths.destination_vendor, proof["expected"], proof["b"]["directories"],
                                      max_file=max_file, max_tree=max_tree)
        b_final = inventory(paths.source_b, max_file=max_file, max_tree=max_tree)
        c_final = overlay_inventory(paths.source_c, c_pins, max_file)
        if not same_included(proof["b"], b_final) or proof["c"] != c_final:
            raise RuntimeError("Source changed before final proof; deployment is not accepted")
        host_check(host)
        manifest.update(state="completed", completed_at=now(), source_copy_before_after_unchanged=True,
            source_b_after=b_final, source_c_after=c_final,
            destination_files=verified["files"], destination_directories=verified["directories"],
            verified_all_destination_size_sha256=True, published=True)
        write_json_atomic(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest.update(state="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}",
                        published=published, source_copy_before_after_unchanged=False,
                        note="Retained own staging/output for inspection; no automatic deletion or retry. Do not use failed vendor.")
        write_json_atomic(manifest_path, manifest)
        raise


def interrupted(signum, _frame):
    raise InterruptedError(f"Deployment interrupted by signal {signum}; retain incomplete state")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Opt in to one new, verified C source copy")
    args = parser.parse_args(argv)
    host = require_host()
    paths = Paths()
    if args.apply:
        old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            result = deploy(paths, require_host)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        print(json.dumps(dict(state=result["state"], host=result["host"], destination=result["destination"],
                             file_summary=result["file_summary"], manifest=str(paths.destination_root / "deploy_manifest.json"))))
    else:
        result = inspect(paths)
        require_host(host)
        print(json.dumps(dict(state="inspected_readonly", host=host, source_b=result["b"], source_c=result["c"],
                             file_count=len(result["expected"]), total_bytes=sum(x["size"] for x in result["expected"].values())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
