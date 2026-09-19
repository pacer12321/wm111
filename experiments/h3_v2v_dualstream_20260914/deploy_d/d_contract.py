"""D-only immutable admission and one-submission ledger; standard library only.

No default policy exists. A reviewed policy must be explicitly SHA-pinned at
invocation. A claim is durable even if startup later fails; never delete/retry it.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

CASE = "D"
MODE = "D_source_hybrid_same_frame_v1"
INSTALL = Path("/cache/zhonghao/h3")
EXPERIMENT = INSTALL / "dualstream_v1"
CODE = EXPERIMENT / "deploy_d"
VENDOR = EXPERIMENT / "candidate/vllm-omni"
POLICY = CODE / "reviewed_policy.json"
VERIFIER = CODE / "d_execution_verifier.py"
SAMPLE = "shirt_red_couple_124"
GROUP = "01234567"
LEDGER_NAME = "submission_20260914.json"
REQUIRED_CANDIDATE_FILES = {
    "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
    "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py",
    "vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py",
    "vllm_omni/diffusion/models/minimax_h3/openvdn_checkpoint.py",
    "vllm_omni/diffusion/models/minimax_h3/dual_stream_attention.py",
    "vllm_omni/diffusion/models/minimax_h3/dual_stream_adapter.py",
    "vllm_omni/diffusion/models/minimax_h3/strict_source_layout.py",
    "vllm_omni/diffusion/models/minimax_h3/strict_source_attention.py",
}
POLICY_KEYS = {"schema_version", "case", "mode", "review_status", "policy_revision",
               "semantic_decisions_complete", "training_performed", "sample_id", "group",
               "candidate_vendor", "candidate_files", "execution_verifier_sha256",
               "tiny_proof", "attention", "execution_contract", "tiny_verifier_sha256",
               "validation_pins_path", "validation_pins_sha256"}
TINY_KEYS = {"case", "mode", "run_id", "run_directory", "status_sha256",
             "result_sha256", "source_manifest_sha256"}


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("Duplicate JSON object key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(
            RuntimeError("Non-finite JSON constant")))
    except (ValueError, TypeError, RecursionError) as exc:
        raise RuntimeError("Malformed bounded JSON record") from exc


def owned_regular(path, *, max_bytes=4 * 1024**2):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or not 0 < info.st_size <= max_bytes
            or path.resolve(strict=True) != path
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
        raise RuntimeError("Expected canonical owned single-link bounded regular record")
    return path


def read_record(path, *, max_bytes=4 * 1024**2):
    path = owned_regular(path, max_bytes=max_bytes)
    raw = path.read_bytes()
    return unique_json(raw), {"path": str(path), "size_bytes": len(raw), "sha256": digest_bytes(raw)}


def relative_vendor_file(value):
    if (not isinstance(value, str) or "\\" in value or value.startswith("/")
            or not value.startswith("vllm_omni/") or value.endswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise RuntimeError("Candidate dependency must be a normalized vllm_omni-relative file")
    return value


def validate_policy(row):
    if not isinstance(row, dict) or set(row) != POLICY_KEYS:
        raise RuntimeError("D policy schema incomplete or unknown fields")
    expected = dict(schema_version=1, case=CASE, mode=MODE, review_status="approved_for_inference",
                    semantic_decisions_complete=True, training_performed=False,
                    sample_id=SAMPLE, group=GROUP, candidate_vendor=VENDOR.as_posix())
    if (type(row["schema_version"]) is not int
            or any(type(row[key]) is not bool for key in ("semantic_decisions_complete", "training_performed"))
            or any(row.get(key) != value for key, value in expected.items())
            or not isinstance(row["policy_revision"], str)
            or re.fullmatch("[a-zA-Z0-9_.-]{1,80}", row["policy_revision"]) is None):
        raise RuntimeError("D policy is not explicitly finalized/reviewed training-free D")
    files = row["candidate_files"]
    if (not isinstance(files, dict) or not REQUIRED_CANDIDATE_FILES.issubset(files)
            or len(files) > 100 or any(not valid_sha(value) for value in files.values())):
        raise RuntimeError("Incomplete candidate dependency SHA closure")
    for name in files:
        relative_vendor_file(name)
    if not valid_sha(row["execution_verifier_sha256"]):
        raise RuntimeError("Missing reviewed D execution verifier SHA")
    tiny = row["tiny_proof"]
    if (not isinstance(tiny, dict) or set(tiny) != TINY_KEYS or tiny.get("case") != "D_tiny_SP8"
            or tiny.get("mode") != MODE or not isinstance(tiny.get("run_id"), str)
            or re.fullmatch("[0-9a-f]{32}", tiny["run_id"]) is None
            or not isinstance(tiny.get("run_directory"), str)
            or not all(valid_sha(tiny[key]) for key in ("status_sha256", "result_sha256", "source_manifest_sha256"))):
        raise RuntimeError("D needs an explicit own SP8 tiny proof, never old C/B latest")
    directory = Path(tiny["run_directory"])
    if (directory.parent != EXPERIMENT / "validation_results/01234567/runs"
            or re.fullmatch("[0-9]{8}T[0-9]{6}Z_" + tiny["run_id"], directory.name) is None):
        raise RuntimeError("Noncanonical fixed D tiny run directory")
    for key in ("attention", "execution_contract"):
        if not isinstance(row[key], dict) or not row[key]:
            raise RuntimeError("Explicit attention policy and actual execution contract are required")
    attention = row["attention"]
    expected_attention_keys = {"endpoint_policy", "auxiliary_policy", "source_linear_text",
                               "target_linear_text", "chunk_frames", "chunk_radius",
                               "share_parameters", "independent_stream_states"}
    if (set(attention) != expected_attention_keys
            or attention["endpoint_policy"] != "vdn_anchors"
            or attention["auxiliary_policy"] != "preserve_existing"
            or attention["source_linear_text"] != "text" or attention["target_linear_text"] != "text"
            or type(attention["chunk_frames"]) is not int or attention["chunk_frames"] != 5
            or type(attention["chunk_radius"]) is not int or attention["chunk_radius"] != 1
            or attention["share_parameters"] is not True or attention["independent_stream_states"] is not True
            or not valid_sha(row["tiny_verifier_sha256"])
            or not valid_sha(row["validation_pins_sha256"])
            or row["validation_pins_path"] != (EXPERIMENT / "validation_d/pins.json").as_posix()):
        raise RuntimeError("D attention decisions or own tiny verifier are not explicitly approved")
    return row


def policy_gate(expected_sha256):
    if not valid_sha(expected_sha256):
        raise RuntimeError("Explicit reviewed-policy SHA256 is required")
    row, record = read_record(POLICY)
    if record["sha256"] != expected_sha256:
        raise RuntimeError("D policy file differs from explicit invocation SHA")
    return {"record": record, "value": validate_policy(row)}


def exclusive_record(path, row):
    """Create-only, no rename-overwrite. Partial crash records stay fail-closed."""
    path = Path(path)
    if path.parent.resolve(strict=True) != path.parent:
        raise RuntimeError("Noncanonical ledger parent")
    raw = (json.dumps(row, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return read_record(path)[1]


def claim_submission(output_root, run_id, host, policy_sha256, supervisor_identity, created_at):
    expected = EXPERIMENT / "results" / GROUP / SAMPLE / CASE
    output_root = Path(output_root)
    if (output_root != expected or output_root.resolve(strict=True) != output_root
            or not isinstance(run_id, str) or re.fullmatch("[0-9a-f]{32}", run_id) is None
            or not valid_sha(policy_sha256) or not isinstance(host, dict)
            or set(host) != {"hostname", "boot_id", "machine"}
            or not isinstance(supervisor_identity, dict)
            or any(type(supervisor_identity.get(key)) is not int or supervisor_identity[key] <= 0
                   for key in ("pid", "start_ticks", "pgrp", "session"))
            or not isinstance(created_at, str) or not created_at):
        raise RuntimeError("Invalid fixed D submission identity")
    row = dict(schema_version=1, case=CASE, mode=MODE, group=GROUP, sample_id=SAMPLE,
               run_id=run_id, host=host, policy_sha256=policy_sha256,
               supervisor_identity=supervisor_identity, created_at=created_at,
               allowed_requests=[{"name": "smoke_2step", "steps": 2}, {"name": "d_50step", "steps": 50}],
               retry_policy="single submission consumed even on failure; no automatic retry")
    return {"record": exclusive_record(output_root / LEDGER_NAME, row), "value": row}


def verify_submission(claim, output_root, run_id, host, policy_sha256):
    path = Path(output_root) / LEDGER_NAME
    row, record = read_record(path)
    if (claim != {"record": record, "value": row} or row.get("run_id") != run_id
            or row.get("host") != host or row.get("case") != CASE or row.get("mode") != MODE
            or row.get("policy_sha256") != policy_sha256):
        raise RuntimeError("D immutable submission claim changed or belongs to another run")
    return claim
