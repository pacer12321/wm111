"""Verify real D first-forward execution records; no model imports or writes."""
import re
from d_contract import MODE, VENDOR, validate_policy, unique_json, valid_sha

MARKER = "D_EXECUTION_RECORD "
MODULE_FILE = "vllm_omni/diffusion/models/minimax_h3/dual_stream_attention.py"
EXPECTED_EXECUTION_CONTRACT = {
    "marker": "D_EXECUTION_RECORD", "format": "json", "first_forward_index": 1,
    "layer_indices": list(range(50)), "world": 8, "heads_per_rank": 7,
    "module_file": MODULE_FILE, "dtype": "torch.bfloat16", "device_type": "npu",
    "source_frames": 37, "target_frames": 37,
    "source_tokens_per_frame": 1008, "target_tokens_per_frame": 1008,
}
KEYS = {"mode", "request_index", "forward_index", "layer_index", "rank", "world",
        "endpoint_policy", "auxiliary_policy", "policy_sha256", "module_sha256", "module_path",
        "source_frames", "target_frames", "source_tokens_per_frame", "target_tokens_per_frame",
        "heads", "device", "dtype", "softmax_source_completed", "softmax_target_completed",
        "linear_source_completed", "linear_target_completed"}
INT_KEYS = {"request_index", "forward_index", "layer_index", "rank", "world",
            "source_frames", "target_frames", "source_tokens_per_frame",
            "target_tokens_per_frame", "heads"}
BOOL_KEYS = {"softmax_source_completed", "softmax_target_completed",
             "linear_source_completed", "linear_target_completed"}


def parse_execution_records(log_text, worker_pids, *, run_id, policy, policy_sha256, requests=1):
    validate_policy(policy)
    if (not isinstance(log_text, str) or type(requests) is not int or requests not in (1, 2)
            or not isinstance(run_id, str) or re.fullmatch("[0-9a-f]{32}", run_id) is None
            or not valid_sha(policy_sha256) or len(set(worker_pids)) != 8
            or any(type(pid) is not int or pid <= 1 for pid in worker_pids)
            or policy["execution_contract"] != EXPECTED_EXECUTION_CONTRACT):
        raise RuntimeError("Invalid fixed D execution evidence contract")
    records = {pid: {} for pid in worker_pids}
    ranks_by_pid = {}
    module_sha = policy["candidate_files"][MODULE_FILE]
    for line in log_text.splitlines():
        if MARKER not in line:
            continue
        match = re.fullmatch(r"\.{0,64}H3D pid=(\d+) [^\n]*?" + re.escape(MARKER) + r"(\{[^\n]*\})", line)
        if match is None or len(match.group(2)) > 8192:
            raise RuntimeError("Malformed/unlabelled/oversized actual D execution record")
        pid = int(match.group(1))
        if pid not in records:
            raise RuntimeError("D kernel execution came from an unknown worker PID")
        row = unique_json(match.group(2))
        if (not isinstance(row, dict) or set(row) != KEYS
                or any(type(row[k]) is not int for k in INT_KEYS)
                or any(row[k] is not True for k in BOOL_KEYS)
                or row["mode"] != MODE or row["request_index"] not in range(1, requests + 1)
                or row["forward_index"] != 1 or row["layer_index"] not in range(50)
                or row["rank"] not in range(8) or row["world"] != 8 or row["heads"] != 7
                or row["policy_sha256"] != policy_sha256 or row["module_sha256"] != module_sha
                or row["module_path"] != (VENDOR / MODULE_FILE).as_posix()
                or row["endpoint_policy"] != policy["attention"]["endpoint_policy"]
                or row["auxiliary_policy"] != policy["attention"]["auxiliary_policy"]
                or row["source_frames"] != 37 or row["target_frames"] != 37
                or row["source_tokens_per_frame"] != 1008 or row["target_tokens_per_frame"] != 1008
                or row["dtype"] != "torch.bfloat16" or row["device"] != "npu:" + str(row["rank"])):
            raise RuntimeError("D kernel execution violated mode/policy/geometry/branches/device/head contract")
        if pid in ranks_by_pid and ranks_by_pid[pid] != row["rank"]:
            raise RuntimeError("Worker PID changed D rank between records")
        ranks_by_pid[pid] = row["rank"]
        key = (row["request_index"], row["layer_index"])
        if key in records[pid]:
            raise RuntimeError("Duplicate D request/layer execution proof")
        records[pid][key] = row
    if set(ranks_by_pid.values()) != set(range(8)) or len(ranks_by_pid) != 8:
        raise RuntimeError("D execution does not have eight distinct complete ranks/PIDs")
    expected_keys = {(request, layer) for request in range(1, requests + 1) for layer in range(50)}
    if any(set(rows) != expected_keys for rows in records.values()):
        raise RuntimeError("Missing one or more actual D layer/worker/request execution records")
    return {
        "case": "D", "mode": MODE, "run_id": run_id, "policy_sha256": policy_sha256,
        "request_count": requests, "record_count": requests * 8 * 50,
        "rank_by_pid": ranks_by_pid,
        "records_by_pid": {pid: [rows[key] for key in sorted(rows)] for pid, rows in records.items()},
        "evidence_source": "Actual completed source/target Softmax and linear calls on first forward of each request, 50 DiT layers and 8 ranks.",
        "limitation": "First-forward execution evidence, not a complete per-step mask trace, NFE count, equality proof, speed or edit-quality evaluation.",
    }
