"""Independent C8 full-model gates. No B tiny substitution or device operations.

Frozen A/B helpers are read-only dependencies for the identical assets/runtime
and loading checks. C's six-file candidate and its own real SP8 regression are
required independently. Strict metadata comes from real pipeline log records;
it is not described as a per-layer tensor-coordinate or mask trace.
"""
import ast
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import re
import sys

B_CODE = Path('/cache/zhonghao/h3/color_trial_v1/b')
sys.path.insert(0, str(B_CODE))
import b_trial_gates as b_gates

a_gates = b_gates.a_gates
profiles, base = a_gates.profiles, a_gates.base
ROOT, VALIDATION_CODE = a_gates.ROOT, a_gates.VALIDATION_CODE
CODE = ROOT / 'color_trial_v1/c'
C_VALIDATION_CODE = ROOT / 'c_validation_code'
CHECKPOINT = b_gates.CHECKPOINT
OFFICIAL_COMMIT = b_gates.OFFICIAL_COMMIT
VENDOR = ROOT / 'candidates/c_v1/vllm-omni'
HTTP_PORTS = {'01234567': 19100}
DEFAULT_SAMPLE, SAMPLE_IDS = a_gates.DEFAULT_SAMPLE, a_gates.SAMPLE_IDS
GENERATION = dict(a_gates.GENERATION)
MIN_RAM, MIN_HBM_MIB = a_gates.MIN_RAM, a_gates.MIN_HBM_MIB
B_GATES_SHA256 = 'b784b9d7b131230ebbf669a81b9b017fdc7479bab952dc47dd10d520e222dd56'
A_GATES_SHA256 = 'f786cb3c3ccead8b6de365ca62d2f9dea8aaece1493cd9be0b01e1a4fd4cefa0'
C_VALIDATION_SHA256 = {
    'c_profiles.py': '1d609b5542f5765a0d8b3612ddf9d715ecf1714b72c982a32bfceb584cd11e34',
    'c_npu_regression.py': '5f7306c903e0c7ba2b01a96a5a7cb0902a669090b172aa75357a76f9dac69d32',
    'c_cases.py': '76c5a482c8f8ce896a4661d7f052f89ad31a1c06a9c6a020a7cead48be7c5e3f',
    'run_c_validation.py': '3c99090d3af0944dd04d412bf4451701500a2644d510e8d65642b814dca93516',
    'launch_c_validation.sh': 'e783b7dda5a73481e7c687b8e0dc9bda1928ebde88b0d1de59fe3f33164eda8e',
}
SUPPORT_SHA256 = {
    'time_request.py': '9bf880418209cc91e7131bd44595d140db5e9b213f0c3585da4b41e25b4691fc',
    'reference_video.py': 'b9b3f6a767cecb934e4abd4f2a94146bcdc54255528656b7b0664f7702d9c6f5',
    'vae.py': '2d70c68f1131665bff87788b3256e76270699d142b1e0a86e960cd5e3d9fc3ad',
    'packed_sequence.py': '5743e124a609f6e4794854ec17e777dc7143e62d775c9db82dbcbd19d47d4d49',
    'packed_tokens.py': '20994e723054071e34acc41398e9d1636c226802e49894ac9162f5a808feb06f',
    'denoise_loop.py': 'c0156b9c977770ed0c0bb4b7e2a721737c547c3d758892f7e3247ef7d2582246',
}
STRICT_MODE = 'C_strict_visual_source_same_latent_frame_v1'
STRICT_MARKER = 'C strict same-frame source metadata: '
STRICT_KEYS = {'mode', 'reference_count', 'reference_kind', 'patch_size', 'source_shape',
               'target_shape', 'text_len', 'source_audio_t', 'target_audio_t'}

sample_profile = a_gates.sample_profile
sample_gate = a_gates.sample_gate
transfer_gate = a_gates.transfer_gate
runtime_gate = a_gates.runtime_gate
regular = a_gates.regular
file_record = a_gates.file_record
json_file = a_gates.json_file
video_probe = a_gates.video_probe
weight_manifest = b_gates.weight_manifest


def selected_profile(group, sample_id=DEFAULT_SAMPLE):
    sample = sample_profile(sample_id)
    return replace(profiles.profile(group), output_root=ROOT / 'color_trial_v1/results' / group / sample.sample_id / 'C',
                   code_root=CODE, vendor_root=VENDOR, master_port=HTTP_PORTS[group])


def parallelism(selected):
    return b_gates.parallelism(selected)  # Same reviewed USP8/textTP8/VAE8, C HTTP only.


def c_validation_modules():
    # Verify before importing: these are CPU report readers at module scope.
    for name, expected in C_VALIDATION_SHA256.items():
        if file_record(C_VALIDATION_CODE / name)['sha256'] != expected:
            raise RuntimeError(f'C tiny verifier source changed: {name}')
    sys.path.insert(0, str(C_VALIDATION_CODE))
    modules = [importlib.import_module(name) for name in ('c_profiles', 'c_cases', 'c_npu_regression')]
    for module in modules:
        if Path(module.__file__).resolve() != C_VALIDATION_CODE / (module.__name__ + '.py'):
            raise RuntimeError('C tiny verifier imported outside its fixed private root')
    return modules[0], modules[2]


def source_code_manifest():
    for module, root, name, expected in (
            (b_gates, B_CODE, 'b_trial_gates.py', B_GATES_SHA256),
            (a_gates, b_gates.A_CODE, 'trial_gates.py', A_GATES_SHA256)):
        if Path(module.__file__).resolve() != root / name or file_record(root / name)['sha256'] != expected:
            raise RuntimeError('Frozen read-only A/B gate dependency changed; review C before launch')
    result = a_gates.source_code_manifest()
    result['dependency/b_trial_gates.py'] = file_record(B_CODE / 'b_trial_gates.py')
    cp, _ = c_validation_modules()
    manifest = cp.source_manifest()  # Exact six C files, helpers, C verifier, strategy and env.
    if cp.VENDOR != VENDOR or cp.CARDS != tuple(range(8)):
        raise RuntimeError('C tiny is not bound to the same C8 vendor/cards')
    for name, record in manifest.items():
        result['c_tiny_source/' + name] = record
    for name, expected in SUPPORT_SHA256.items():
        record = file_record(VENDOR / 'vllm_omni/diffusion/models/minimax_h3' / name)
        if record['sha256'] != expected:
            raise RuntimeError(f'C actual shape/reference/packing support differs from the reviewed source: {name}')
        result['c_model_support/' + name] = record
    for relative in ('vllm_omni/__init__.py', 'vllm_omni/logger.py',
                     'vllm_omni/diffusion/attention/layer.py'):
        result['c_vendor/' + relative] = file_record(VENDOR / relative)
    for name in ('c_trial_gates.py', 'run_c_trial.py', 'launch_c_trial.sh'):
        result['c_trial/' + name] = file_record(CODE / name)
    return result


def tiny_gate(group, host):
    if group != '01234567':
        raise RuntimeError('C requires the complete eight-card tiny proof')
    cp, checker = c_validation_modules()
    cp.require_host(host)
    selected = cp.Profile()
    latest = selected.output_root / 'c_validation_status.json'
    status = json_file(latest)  # Missing proof is a hard failure, never replaced by B's proof.
    if (status.get('case') != cp.CASE or status.get('phase') != 'completed' or status.get('host') != host
            or status.get('group') != group or status.get('allocated_physical_npu_ids') != list(cp.CARDS)
            or status.get('validation_passed') is not True or status.get('cleanup_completed') is not True
            or status.get('selected_cards_verified_idle_after_cleanup') is not True
            or status.get('needs_attention') is not False or status.get('error') is not None
            or status.get('remaining_owned_process_groups') != {}):
        raise RuntimeError('C requires its own current-host SP8 pass and completed cleanup, not B evidence')
    run = Path(status.get('run_directory', ''))
    profiles.canonical_private(run)
    run_id = status.get('run_id', '')
    if (re.fullmatch(r'[0-9a-f]{32}', run_id) is None or run.parent != selected.output_root / 'runs'
            or re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_' + run_id, run.name) is None):
        raise RuntimeError('Noncanonical C tiny run identity/path')
    if json_file(run / 'c_validation_status.json') != status or json_file(run / 'host_identity.json') != host:
        raise RuntimeError('C latest state differs from its completed per-run evidence')
    result_path = run / 'c_tiny.json'
    if status.get('result_path') != str(result_path) or status.get('result_sha256') != profiles.digest(regular(result_path)):
        raise RuntimeError('C tiny result path/hash changed')
    manifest = cp.source_manifest()
    if status.get('source_sha256') != manifest or json_file(run / 'source_manifest.json') != manifest:
        raise RuntimeError('C candidate/strategy/env/verifier changed after the actual C tiny run')
    result = checker.verify_results(result_path, host, run_id, manifest)
    return {'status_record': file_record(latest), 'status': status, 'result_record': file_record(result_path),
            'source_manifest_record': file_record(run / 'source_manifest.json'), 'result': result,
            'scope': 'Real C SP8 synthetic correctness only; full checkpoint and pipeline require this fresh smoke.'}


def logging_configuration():
    return {'version': 1, 'disable_existing_loggers': False,
            'formatters': {'h3c': {'format': 'H3C pid=%(process)d %(asctime)s %(levelname)s %(name)s %(message)s'}},
            'handlers': {'h3c': {'class': 'logging.StreamHandler', 'formatter': 'h3c', 'stream': 'ext://sys.stderr'}},
            'loggers': {name: {'handlers': ['h3c'], 'level': 'INFO', 'propagate': False}
                        for name in ('vllm', 'vllm_omni')}}


def parse_strict_metadata(log_text, worker_pids, sample_id, *, requests=1):
    """Read real per-diffuse PID metadata; do not infer a logged per-layer mask.

    requests=1 gates formal on the smoke. requests=2 additionally checks that
    the completed formal request used the same strict shapes and worker set.
    """
    sample_profile(sample_id)
    if (type(requests) is not int or requests not in (1, 2) or len(set(worker_pids)) != 8
            or any(type(pid) is not int or pid <= 0 for pid in worker_pids)):
        raise RuntimeError('Strict metadata requires exactly eight known worker PIDs and one/two requests')
    records = {pid: [] for pid in worker_pids}
    for line in log_text.splitlines():
        if STRICT_MARKER not in line:
            continue
        # Shared stderr can join progress dots to an otherwise complete record.
        # Permit only bounded ASCII dots, not arbitrary text/whitespace prefixes.
        match = re.fullmatch(r'\.{0,64}H3C pid=(\d+) [^\n]*?' + re.escape(STRICT_MARKER) + r'(\{[^\n]*\})', line)
        if match is None or len(match.group(2)) > 4096:
            raise RuntimeError('Unlabelled, oversized or malformed real C strict metadata record')
        pid = int(match.group(1))
        if pid not in records or len(records[pid]) >= requests:
            raise RuntimeError('C strict metadata contains an unknown PID or excess request records')
        try:
            row = ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
            raise RuntimeError('C strict metadata is not a safe literal dictionary') from exc
        if (not isinstance(row, dict) or set(row) != STRICT_KEYS or row.get('mode') != STRICT_MODE
                or type(row.get('reference_count')) is not int or row['reference_count'] != 1
                or row.get('reference_kind') != 'video' or row.get('patch_size') != (1, 2, 2)
                or any(type(v) is not int for v in row.get('patch_size', ()))):
            raise RuntimeError('C strict mode/reference/patch metadata differs from the actual fixed candidate')
        for key in ('source_shape', 'target_shape'):
            value = row[key]
            if (not isinstance(value, tuple) or len(value) != 3
                    or any(type(v) is not int or v <= 0 for v in value)
                    or value[0] != 37 or value[1] % 2 or value[2] % 2):
                raise RuntimeError('C source/target latent frame count or spatial patch geometry is invalid')
        if (row['target_shape'] != (37, 48, 84) or type(row['text_len']) is not int or row['text_len'] <= 0
                or type(row['source_audio_t']) is not int or row['source_audio_t'] < 0
                or type(row['target_audio_t']) is not int or row['target_audio_t'] != 207
                or row['source_audio_t'] != 0):
            raise RuntimeError('C metadata violates the frozen 124-frame/24fps target or explicit source audio contract')
        records[pid].append(row)
    if any(len(rows) != requests for rows in records.values()):
        raise RuntimeError('Missing real strict metadata for one or more of the eight workers/requests')
    first = next(iter(records.values()))[0]
    if any(row != first for rows in records.values() for row in rows):
        raise RuntimeError('Strict metadata differs across workers or between smoke and formal')
    return {'records_by_pid': records, 'request_count_per_pid': requests,
            'evidence_source': 'Actual pipeline make_metadata from VAE/ref blocks, logged once per diffuse request.',
            'limitation': 'Not a per-layer mask/tensor-coordinate trace. Successful requests plus pinned C source and own C tiny establish the strict infer/layout path.'}


def parse_full_load_evidence(log_text, weights):
    records = {}
    for marker, field in (('OPENVDN_B_LOAD_RECORD', 'load'), ('OPENVDN_B_CPU_LOADING_END', 'cpu_loading')):
        pattern = re.compile(r'H3C pid=(\d+) [^\n]*?' + marker + r' (\{[^\n]*\})')
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
