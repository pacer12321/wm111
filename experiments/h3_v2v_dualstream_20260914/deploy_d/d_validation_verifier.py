"""Read-only admission for an explicit completed D synthetic SP8 run.

Deploy THIS file into dualstream_v1/validation_d, not deploy_d. Verification
imports only pinned stdlib validation readers; it never invokes worker/main,
torch, npu-smi, locks, cleanup, or another experiment. Historical cleanup is
checked without requiring the currently running full-model trial to be idle.
"""
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

INSTALL = Path("/cache/zhonghao/h3")
EXPERIMENT = INSTALL / "dualstream_v1"
CODE_ROOT = EXPERIMENT / "validation_d"
RUNS_ROOT = EXPERIMENT / "validation_results/01234567/runs"
VENDOR = EXPERIMENT / "candidate/vllm-omni"
CASE, MODE = "D_tiny_SP8", "D_source_hybrid_same_frame_v1"
MODEL_RELATIVE = "vllm_omni/diffusion/models/minimax_h3/"
CANDIDATES = {"openvdn_npu.py", "openvdn_checkpoint.py", "minimax_h3_transformer.py",
              "pipeline_minimax_h3.py", "strict_source_layout.py", "strict_source_attention.py",
              "dual_stream_attention.py", "dual_stream_adapter.py"}
VALIDATORS = {"d_profiles.py", "d_cases.py", "d_cpu_regression.py", "d_npu_regression.py",
              "run_d_validation.py", "launch_d_validation.sh", "d_validation_verifier.py"}
ATTENTION = dict(endpoint_policy="vdn_anchors", auxiliary_policy="preserve_existing",
                 source_linear_text="text", target_linear_text="text", chunk_frames=5,
                 chunk_radius=1, share_parameters=True, independent_stream_states=True)
IMPORTS = ("d_profiles", "d_cases", "d_cpu_regression", "d_npu_regression")


def sha_ok(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_file(path, *, limit=8 * 1024**2):
    path = Path(path)
    info = path.lstat()
    if (not path.is_absolute() or path.resolve(strict=True) != path
            or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or not 0 < info.st_size <= limit
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
        raise RuntimeError("Expected canonical owned bounded single-link D evidence")
    raw = path.read_bytes()
    after = path.lstat()
    if ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != info.st_size):
        raise RuntimeError("D evidence changed while reading")
    return raw, dict(path=str(path), size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read_json(path):
    raw, record = read_file(path)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("Duplicate D evidence JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(RuntimeError("Nonfinite JSON")))
    except (ValueError, TypeError, RecursionError) as exc:
        raise RuntimeError("Malformed D evidence JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Expected a D evidence JSON object")
    return value, record


def require_sha(record, expected):
    if not sha_ok(expected) or record["sha256"] != expected:
        raise RuntimeError("Pinned D evidence SHA mismatch: " + record["path"])


def load_checked(name, path, expected):
    require_sha(read_file(path)[1], expected)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing pinned D reader import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("D reader imported from another deployment")
    require_sha(read_file(path)[1], expected)
    return module


@contextmanager
def verified_readers(pins, pins_sha256):
    """Isolate fixed reader names and pins env; never change the caller's policy."""
    old_modules = {name: sys.modules.get(name) for name in IMPORTS}
    old_env = os.environ.get("D_VALIDATION_PINS_SHA256")
    old_path = list(sys.path)
    try:
        os.environ["D_VALIDATION_PINS_SHA256"] = pins_sha256
        sys.path.insert(0, str(CODE_ROOT))
        modules = {}
        for name in IMPORTS:
            modules[name] = load_checked(name, CODE_ROOT / (name + ".py"), pins["validation"][name + ".py"])
        yield modules["d_profiles"], modules["d_npu_regression"]
    finally:
        sys.path[:] = old_path
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if old_env is None:
            os.environ.pop("D_VALIDATION_PINS_SHA256", None)
        else:
            os.environ["D_VALIDATION_PINS_SHA256"] = old_env


def check_identity_exited(base, identity, expected_pid, *, leader=False):
    keys = ("pid", "start_ticks", "pgrp", "session")
    if (not isinstance(identity, dict)
            or any(type(identity.get(key)) is not int or identity[key] <= 0 for key in keys)
            or type(expected_pid) is not int or identity["pid"] != expected_pid
            or (leader and (identity["pgrp"] != expected_pid or identity["session"] != expected_pid))):
        raise RuntimeError("D original process identity is incomplete or inconsistent")
    current = base.proc_identity(expected_pid)
    if current is not None and current.get("start_ticks") == identity["start_ticks"]:
        raise RuntimeError("Original D validation process identity has not exited")
    return {key: identity[key] for key in keys}


def completed_status(status, *, host, run_id, run, result_sha):
    expected = dict(case=CASE, mode=MODE, phase="completed", host=host, group="01234567",
                    run_id=run_id, run_directory=str(run), allocated_physical_npu_ids=list(range(8)),
                    validation_passed=True, cleanup_completed=True,
                    selected_cards_verified_idle_after_cleanup=True, needs_attention=False,
                    remaining_owned_process_groups={}, error=None,
                    result_path=str(run / "d_tiny.json"), result_sha256=result_sha)
    if any(key not in status or status[key] != value for key, value in expected.items()):
        raise RuntimeError("D explicit validation is not successfully completed and released")
    for key in ("validation_passed", "cleanup_completed", "selected_cards_verified_idle_after_cleanup", "needs_attention"):
        if type(status[key]) is not bool:
            raise RuntimeError("Nonboolean D terminal evidence")
    try:
        start, end = (dt.datetime.fromisoformat(status[key]) for key in ("started_at", "finished_at"))
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("Invalid completed D chronology")
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("Missing valid D start/end chronology") from exc


def verify_completed(*, run_directory, run_id, host, policy):
    """Read one approved per-run proof; never 'latest', retry, execute or clean."""
    if (Path(__file__).resolve() != CODE_ROOT / "d_validation_verifier.py"
            or not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
            or not isinstance(policy, dict) or policy.get("case") != "D" or policy.get("mode") != MODE
            or policy.get("review_status") != "approved_for_inference" or policy.get("training_performed") is not False
            or policy.get("attention") != ATTENTION
            or policy.get("candidate_vendor") != VENDOR.as_posix()):
        raise RuntimeError("D verifier deployment or final inference policy mismatch")
    run = Path(run_directory)
    tiny = policy.get("tiny_proof", {})
    if (run.parent != RUNS_ROOT or run.resolve(strict=True) != run
            or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z_" + run_id, run.name) is None
            or tiny.get("case") != CASE or tiny.get("mode") != MODE
            or tiny.get("run_id") != run_id or tiny.get("run_directory") != str(run)):
        raise RuntimeError("D validator must use the explicitly approved immutable run")
    if policy.get("validation_pins_path") != (CODE_ROOT / "pins.json").as_posix():
        raise RuntimeError("D validation pins path differs")
    pins, pins_record = read_json(CODE_ROOT / "pins.json")
    require_sha(pins_record, policy.get("validation_pins_sha256"))
    if (set(pins) != {"schema_version", "case", "mode", "candidate", "validation"}
            or type(pins["schema_version"]) is not int or pins["schema_version"] != 1
            or pins["case"] != CASE or pins["mode"] != MODE
            or not isinstance(pins["candidate"], dict) or set(pins["candidate"]) != CANDIDATES
            or not isinstance(pins["validation"], dict) or set(pins["validation"]) != VALIDATORS
            or any(not sha_ok(sha) for section in ("candidate", "validation") for sha in pins[section].values())):
        raise RuntimeError("Incomplete or incompatible D validation pin set")
    files = {}
    for name, sha in pins["candidate"].items():
        if policy.get("candidate_files", {}).get(MODEL_RELATIVE + name) != sha:
            raise RuntimeError("D tiny and full inference do not use identical candidate code")
        files["candidate/" + name] = read_file(VENDOR / MODEL_RELATIVE / name, limit=20 * 1024**2)[1]
        require_sha(files["candidate/" + name], sha)
    for name, sha in pins["validation"].items():
        files["validation/" + name] = read_file(CODE_ROOT / name)[1]
        require_sha(files["validation/" + name], sha)
    require_sha(files["validation/d_validation_verifier.py"], policy.get("tiny_verifier_sha256"))
    status, status_record = read_json(run / "d_validation_status.json")
    manifest, manifest_record = read_json(run / "source_manifest.json")
    raw_result, result_record = read_json(run / "d_tiny.json")
    for record, key in ((status_record, "status_sha256"), (manifest_record, "source_manifest_sha256"),
                        (result_record, "result_sha256")):
        require_sha(record, tiny.get(key))
    completed_status(status, host=host, run_id=run_id, run=run, result_sha=result_record["sha256"])
    if read_json(run / "host_identity.json")[0] != host:
        raise RuntimeError("D validation recorded host/boot differs")
    fixture, fixture_record = read_json(run / "validation_policy.json")
    expected_fixture = dict(schema_version=1, mode=MODE, review_status="approved_for_inference",
                            scope="synthetic_validation_only", attention=ATTENTION)
    if fixture != expected_fixture or type(fixture.get("schema_version")) is not int:
        raise RuntimeError("D synthetic policy differs from approved attention semantics")
    if (status.get("validation_policy_sha256") != fixture_record["sha256"]
            or raw_result.get("fixture_policy_sha256") != fixture_record["sha256"]):
        raise RuntimeError("Actual synthetic policy was not bound by both controller and numeric result")
    owner, owner_record = read_json(run / "owner_proof.json")
    with verified_readers(pins, pins_record["sha256"]) as (p, checker):
        if (p.CODE_ROOT != CODE_ROOT or p.VENDOR != VENDOR or p.CASE != CASE or p.MODE != MODE
                or p.WORLD != 8 or tuple(p.CARDS) != tuple(range(8))
                or p.Profile().output_root / "runs" != RUNS_ROOT):
            raise RuntimeError("D reader fixed deployment/profile differs")
        p.require_host(host)
        current_manifest = p.source_manifest()
        if current_manifest != manifest or status.get("source_sha256") != manifest:
            raise RuntimeError("D current/frozen/controller validation dependencies differ")
        for label, record in files.items():
            if manifest.get(label) != {"path": record["path"], "sha256": record["sha256"]}:
                raise RuntimeError("D archived manifest is missing a reviewed pinned file")
        report = checker.verify_results(run / "d_tiny.json", host, run_id, manifest)
        if report != raw_result:
            raise RuntimeError("D numeric verifier did not return the pinned actual result")
        base = p.load_helper("supervision_base.py")
        original_supervisor = check_identity_exited(base, status.get("supervisor_proc_identity"), status.get("supervisor_pid"))
        original_worker = check_identity_exited(base, status.get("worker_proc_identity"), status.get("worker_pid"), leader=True)
        if original_supervisor["pid"] == original_worker["pid"]:
            raise RuntimeError("D controller and worker cannot have one process identity")
        expected_leases = [str(p.Profile().output_root / "run.lock")] + [
            str(p.Profile().lease_root / f"device{card}.lock") for card in range(8)]
        if (owner.get("case") != CASE or owner.get("run_id") != run_id or owner.get("host") != host
                or owner.get("supervisor_pid") != original_supervisor["pid"]
                or any(owner.get("proc_identity", {}).get(key) != value for key, value in original_supervisor.items())
                or owner.get("cards") != list(range(8)) or owner.get("resources_checked") is not True
                or owner.get("lease_paths") != expected_leases):
            raise RuntimeError("D original owner/lease/resource proof differs")
        fds = owner.get("lease_fds", [])
        if len(fds) != 9 or len(set(fds)) != 9 or any(type(fd) is not int or fd < 3 for fd in fds):
            raise RuntimeError("D original owner did not hold nine inherited mutex FDs")
        # Construct only to invoke the frozen read-only /proc group reader.
        observer = object.__new__(base.ProcessSupervisor)
        observer.run_id = run_id
        if observer.owned_group_members(original_worker["pid"]):
            raise RuntimeError("D marked owned process group still contains live processes")
        npu_raw, npu_record = read_file(run / "npu_after.txt")
        release = base.selected_idle(npu_raw.decode("utf-8"), tuple(range(8)))
        if release != status.get("resource_release_check"):
            raise RuntimeError("D recorded historical eight-card cleanup differs from raw evidence")
        # Catch edited dependencies across the entire read without model hashes.
        if p.source_manifest() != manifest:
            raise RuntimeError("D validation dependencies changed during admission")
    for record in (pins_record, status_record, manifest_record, result_record, fixture_record, owner_record, npu_record):
        if read_file(Path(record["path"]))[1] != record:
            raise RuntimeError("D proof record changed during admission")
    return dict(verified=True, case=CASE, mode=MODE, run_id=run_id, host=host,
                status_record=status_record, result_record=result_record, source_manifest_record=manifest_record,
                pins_record=pins_record, fixture_policy_record=fixture_record, owner_record=owner_record,
                numeric_rank_count=len(report["rank_results"]), candidate_and_validation_files=files,
                original_supervisor=original_supervisor, original_worker=original_worker,
                all_original_identities_exited=True, remaining_owned_process_groups={},
                historical_release_record=npu_record, historical_release=release,
                current_card_idle_checked=False,
                scope="Pinned actual D CPU/numeric/SP8 evidence and historical owned cleanup; not full-model quality or speed.")
