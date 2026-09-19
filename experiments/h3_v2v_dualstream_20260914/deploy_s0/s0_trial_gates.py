"""New S0-only gates; old A/B are frozen read-only safety/asset dependencies.

No old C proof or latest is accepted. No NPU work at import or verification.
The independently reviewed policy and S0 tiny verifier must exist before launch.
"""
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import s0_contract as contract

B_CODE = Path("/cache/zhonghao/h3/color_trial_v1/b")
sys.path.insert(0, str(B_CODE))
import b_trial_gates as b_gates

a_gates = b_gates.a_gates
profiles, base = a_gates.profiles, a_gates.base
ROOT, VALIDATION_CODE = a_gates.ROOT, a_gates.VALIDATION_CODE
CODE, VENDOR = contract.CODE, contract.VENDOR
S0_VALIDATION_CODE = contract.EXPERIMENT / "validation_code"
S0_VALIDATION_VERIFIER = S0_VALIDATION_CODE / "s0_validation_verifier.py"
CHECKPOINT, OFFICIAL_COMMIT = b_gates.CHECKPOINT, b_gates.OFFICIAL_COMMIT
CASE, MODE = contract.CASE, contract.MODE
HTTP_PORTS = {"01234567": 19101}
DEFAULT_SAMPLE, SAMPLE_IDS = a_gates.DEFAULT_SAMPLE, a_gates.SAMPLE_IDS
GENERATION = dict(a_gates.GENERATION)
MIN_RAM, MIN_HBM_MIB = a_gates.MIN_RAM, a_gates.MIN_HBM_MIB
B_GATES_SHA256 = "b784b9d7b131230ebbf669a81b9b017fdc7479bab952dc47dd10d520e222dd56"
A_GATES_SHA256 = "f786cb3c3ccead8b6de365ca62d2f9dea8aaece1493cd9be0b01e1a4fd4cefa0"
SUPPORT_SHA256 = {
    'time_request.py': '9bf880418209cc91e7131bd44595d140db5e9b213f0c3585da4b41e25b4691fc',
    'reference_video.py': 'b9b3f6a767cecb934e4abd4f2a94146bcdc54255528656b7b0664f7702d9c6f5',
    'vae.py': '2d70c68f1131665bff87788b3256e76270699d142b1e0a86e960cd5e3d9fc3ad',
    'packed_sequence.py': '5743e124a609f6e4794854ec17e777dc7143e62d775c9db82dbcbd19d47d4d49',
    'packed_tokens.py': '20994e723054071e34acc41398e9d1636c226802e49894ac9162f5a808feb06f',
    'denoise_loop.py': 'c0156b9c977770ed0c0bb4b7e2a721737c547c3d758892f7e3247ef7d2582246',
}

sample_profile = a_gates.sample_profile
sample_gate = a_gates.sample_gate
transfer_gate = a_gates.transfer_gate
runtime_gate = a_gates.runtime_gate
regular = a_gates.regular
file_record = a_gates.file_record
json_file = a_gates.json_file
video_probe = a_gates.video_probe
weight_manifest = b_gates.weight_manifest
policy_gate = contract.policy_gate


def selected_profile(group, sample_id=DEFAULT_SAMPLE):
    sample = sample_profile(sample_id)
    if group != contract.GROUP or sample_id != contract.SAMPLE:
        raise RuntimeError("Only the fixed S0 eight-card color trial is allowed")
    return replace(profiles.profile(group),
                   output_root=contract.EXPERIMENT / "results" / group / sample.sample_id / CASE,
                   code_root=CODE, vendor_root=VENDOR, master_port=HTTP_PORTS[group])


def parallelism(selected):
    return b_gates.parallelism(selected)


def checked_module(path, sha256, module_name):
    record = file_record(path)
    if record["sha256"] != sha256:
        raise RuntimeError("Reviewed S0 verifier dependency SHA changed")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the fixed reviewed S0 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("S0 verifier imported from unexpected path")
    return module


def source_code_manifest(policy):
    contract.validate_policy(policy)
    for module, root, name, expected in (
            (b_gates, B_CODE, "b_trial_gates.py", B_GATES_SHA256),
            (a_gates, b_gates.A_CODE, "trial_gates.py", A_GATES_SHA256)):
        if Path(module.__file__).resolve() != root / name or file_record(root/name)["sha256"] != expected:
            raise RuntimeError("Frozen A/B read-only gate dependency changed")
    result = a_gates.source_code_manifest()
    result["dependency/b_trial_gates.py"] = file_record(B_CODE / "b_trial_gates.py")
    for name, expected in policy["candidate_files"].items():
        path = VENDOR / contract.relative_vendor_file(name)
        record = file_record(path)
        if record["sha256"] != expected:
            raise RuntimeError("Approved S0 candidate differs: " + name)
        result["s0_candidate/" + name] = record
    for name, expected in SUPPORT_SHA256.items():
        relative = "vllm_omni/diffusion/models/minimax_h3/" + name
        record = file_record(VENDOR / relative)
        if record["sha256"] != policy["candidate_files"].get(relative, expected):
            raise RuntimeError("S0 actual shape/reference/packing support differs: " + name)
        result["s0_model_support/" + name] = record
    for relative in ("vllm_omni/__init__.py", "vllm_omni/logger.py",
                     "vllm_omni/diffusion/attention/parallel/ulysses.py",
                     "vllm_omni/diffusion/attention/parallel/base.py",
                     "vllm_omni/diffusion/attention/parallel/factory.py",
                     "vllm_omni/diffusion/attention/layer.py"):
        result["s0_vendor/" + relative] = file_record(VENDOR / relative)
    for name in ("s0_trial_gates.py", "run_s0_trial.py", "launch_s0_trial.sh", "s0_contract.py",
                 "s0_execution_verifier.py", "reviewed_policy.json"):
        result["s0_trial/" + name] = file_record(CODE / name)
    if (result["s0_trial/s0_execution_verifier.py"]["sha256"] != policy["execution_verifier_sha256"]
            or Path(contract.__file__).resolve() != CODE / "s0_contract.py"):
        raise RuntimeError("Actual S0 policy/parser module does not match its reviewed dependency")
    tiny_record = file_record(S0_VALIDATION_VERIFIER)
    if tiny_record["sha256"] != policy["tiny_verifier_sha256"]:
        raise RuntimeError("S0 own tiny verifier differs from approved policy")
    result["s0_validation/s0_validation_verifier.py"] = tiny_record
    return result


def tiny_gate(group, host, policy):
    if group != contract.GROUP:
        raise RuntimeError("S0 requires its own complete SP8 tiny proof")
    contract.validate_policy(policy)
    tiny = policy["tiny_proof"]
    run = Path(tiny["run_directory"])
    profiles.canonical_private(run)
    for name, key in (("s0_validation_status.json", "status_sha256"),
                      ("s0_tiny.json", "result_sha256"),
                      ("source_manifest.json", "source_manifest_sha256")):
        if file_record(run/name)["sha256"] != tiny[key]:
            raise RuntimeError("S0 own tiny per-run proof file changed: " + name)
    status = json_file(run/"s0_validation_status.json")
    expected = dict(case="S0_tiny_SP8", mode=MODE, phase="completed", host=host,
                    group=group, run_id=tiny["run_id"], run_directory=str(run),
                    allocated_physical_npu_ids=list(range(8)), validation_passed=True,
                    cleanup_completed=True, selected_cards_verified_idle_after_cleanup=True,
                    needs_attention=False, remaining_owned_process_groups={}, error=None,
                    result_path=str(run/"s0_tiny.json"), result_sha256=tiny["result_sha256"])
    if any(status.get(key) != value for key, value in expected.items()):
        raise RuntimeError("S0 needs a completed, cleaned-up own current-host tiny; C/B evidence forbidden")
    if json_file(run/"host_identity.json") != host:
        raise RuntimeError("S0 tiny host/boot changed")
    checker = checked_module(S0_VALIDATION_VERIFIER, policy["tiny_verifier_sha256"],
                             "_reviewed_s0_validation_verifier")
    verified = checker.verify_completed(run_directory=str(run), run_id=tiny["run_id"],
                                         host=host, policy=policy)
    if not isinstance(verified, dict) or verified.get("verified") is not True:
        raise RuntimeError("Reviewed S0 actual numeric/SP8 result verifier did not approve proof")
    return {"status_record": file_record(run/"s0_validation_status.json"), "status": status,
            "result_record": file_record(run/"s0_tiny.json"),
            "source_manifest_record": file_record(run/"source_manifest.json"), "verification": verified,
            "scope": "S0 own actual SP8 correctness and source/policy proof, never old C/B latest."}


def execution_gate(log_text, worker_pids, sample_id, *, run_id, policy, requests=1):
    sample_profile(sample_id)
    checker = checked_module(contract.VERIFIER, policy["execution_verifier_sha256"],
                             "_reviewed_s0_execution_verifier")
    return checker.parse_execution_records(log_text, worker_pids, run_id=run_id,
                                             policy=policy, requests=requests)


def logging_configuration():
    return {"version": 1, "disable_existing_loggers": False,
            "formatters": {"h3s0": {"format": "H3S0 pid=%(process)d %(asctime)s %(levelname)s %(name)s %(message)s"}},
            "handlers": {"h3s0": {"class": "logging.StreamHandler", "formatter": "h3s0", "stream": "ext://sys.stderr"}},
            "loggers": {name: {"handlers": ["h3s0"], "level": "INFO", "propagate": False}
                        for name in ("vllm", "vllm_omni")}}


def parse_full_load_evidence(log_text, weights):
    records = {}
    for marker, field in (('OPENVDN_B_LOAD_RECORD', 'load'), ('OPENVDN_B_CPU_LOADING_END', 'cpu_loading')):
        pattern = re.compile(r'H3S0 pid=(\d+) [^\n]*?' + marker + r' (\{[^\n]*\})')
        for match in pattern.finditer(log_text):
            pid = int(match.group(1))
            row = records.setdefault(pid, {})
            if pid <= 0 or field in row:
                raise RuntimeError(f'Invalid PID or duplicate {marker} for worker PID {pid}')
            row[field] = json.loads(match.group(2))
    if len(records) != 8 or any(set(row) != {'load', 'cpu_loading'} for row in records.values()):
        raise RuntimeError('Need complete load and restored-CPU-loader records from eight distinct worker PIDs')
    expected = {'checkpoint': str(CHECKPOINT), 'base_partition': 'ref2va', 'base_tensor_count': 535,
                'branch_tensor_count': 800, 'lora_pairs_merged': 208, 'lora_rank': 64, 'lora_alpha': 64,
                'lora_scale': 1.0, 'official_scale_source_commit': OFFICIAL_COMMIT,
                'merge_dtype': 'FP32 delta, cast to parameter dtype, then add',
                'qkv_merge_layout': 'post-base-loader contiguous Q/K/V thirds'}
    for pid, row in records.items():
        load, cpu = row['load'], row['cpu_loading']
        if not isinstance(load, dict) or any(load.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'Incomplete/mismatched original weight coverage for PID {pid}')
        for kind in ('branch', 'lora'):
            if any(load.get(kind, {}).get(key) != weights[kind][key]
                   for key in ('path', 'size_bytes', 'header_sha256', 'tensor_count')):
                raise RuntimeError(f'Loaded {kind} differs from this run checkpoint for PID {pid}')
        if (not isinstance(cpu, dict) or cpu.get('status') != 'completed'
                or cpu.get('requested_loading_intraop') != 4 or cpu.get('intraop_loading') != 4
                or type(cpu.get('intraop_before')) is not int or cpu['intraop_before'] <= 0
                or cpu.get('intraop_after') != cpu['intraop_before']
                or type(cpu.get('interop_before')) is not int or cpu['interop_before'] <= 0
                or cpu.get('interop_after') != cpu['interop_before']
                or cpu.get('interop_loading') != cpu['interop_before']):
            raise RuntimeError(f'CPU loader did not complete and restore inference threads for PID {pid}')
        timings = cpu.get('phase_seconds', {})
        if any(type(timings.get(key)) not in (int, float) or not math.isfinite(timings[key]) or timings[key] < 0
               for key in ('base', 'branch', 'lora')):
            raise RuntimeError(f'Missing/invalid base/branch/LoRA load timing for PID {pid}')
    return records
