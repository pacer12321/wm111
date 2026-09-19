"""31731 B8 gates: read-only reuse of frozen A/validation, no global mutation.

The original Ref2VA weights plus the released complete Stage-B branch/LoRA
are used. B retains the original linear branch and globally visible source;
this entry never implements the C corresponding-frame restriction.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

A_CODE = Path('/cache/zhonghao/h3/model_trial_code')
sys.path.insert(0, str(A_CODE))
import trial_gates as a_gates

profiles, base = a_gates.profiles, a_gates.base
ROOT, VALIDATION_CODE = a_gates.ROOT, a_gates.VALIDATION_CODE
CODE = ROOT / 'b_model_trial_code'
CHECKPOINT = ROOT / 'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000'
VENDOR = ROOT / 'candidates/b_v1/vllm-omni'
HTTP_PORTS = {'01234567': 19099}
DEFAULT_SAMPLE, SAMPLE_IDS = a_gates.DEFAULT_SAMPLE, a_gates.SAMPLE_IDS
GENERATION = dict(a_gates.GENERATION)
MIN_RAM, MIN_HBM_MIB = a_gates.MIN_RAM, a_gates.MIN_HBM_MIB
OFFICIAL_COMMIT = '2f740c9291431d89d4f2330743b093fac4390d09'
A_GATES_SHA256 = 'b9bfed7e602c9aa030d6838fc8188cfcea2707114a78acc783791978138d029f'

# These are read-only function aliases, not writes to the frozen module globals.
sample_profile = a_gates.sample_profile
sample_gate = a_gates.sample_gate
transfer_gate = a_gates.transfer_gate
runtime_gate = a_gates.runtime_gate
tiny_gate = a_gates.tiny_gate
regular = a_gates.regular
file_record = a_gates.file_record
json_file = a_gates.json_file
video_probe = a_gates.video_probe


def selected_profile(group, sample_id=DEFAULT_SAMPLE):
    sample = sample_profile(sample_id)
    tiny = profiles.profile(group)
    return replace(tiny, output_root=ROOT / 'b_model_trial' / group / sample.sample_id / 'B',
                   code_root=CODE, vendor_root=VENDOR, master_port=HTTP_PORTS[group])


def parallelism(selected):
    values = a_gates.parallelism(selected)  # A8 compute settings, isolated B HTTP port.
    expected = {'num_gpus': 8, 'usp': 8, 'ring': 1, 'dit_tensor_parallel_size': 1,
                'text_encoder_tp_size': 8, 'vae_patch_parallel_size': 8,
                'vae_parallel_mode': 'tile', 'vae_use_tiling': True, 'layerwise_offload': True}
    if any(values.get(key) != value for key, value in expected.items()):
        raise RuntimeError('B8 requires the reviewed same-A full-world textTP8/VAE8 configuration')
    return values


def source_code_manifest():
    if Path(a_gates.__file__).resolve() != A_CODE / 'trial_gates.py':
        raise RuntimeError('B must import the fixed read-only frozen A gate module')
    if profiles.digest(regular(A_CODE / 'trial_gates.py')) != A_GATES_SHA256:
        raise RuntimeError('The frozen A gate dependency changed; review B before launch')
    result = a_gates.source_code_manifest()
    model = VENDOR / 'vllm_omni/diffusion/models/minimax_h3'
    for name, expected in profiles.CANDIDATE_SHA256.items():
        record = file_record(model / name)
        if record['sha256'] != expected:
            raise RuntimeError(f'B candidate differs from the tiny-validated source: {name}')
        result[f'b_candidate/{name}'] = record
    for relative in ('vllm_omni/__init__.py', 'vllm_omni/logger.py',
                     'vllm_omni/diffusion/attention/parallel/ulysses.py',
                     'vllm_omni/diffusion/attention/parallel/base.py',
                     'vllm_omni/diffusion/attention/parallel/factory.py',
                     'vllm_omni/diffusion/attention/layer.py'):
        result[f'b_vendor/{relative}'] = file_record(VENDOR / relative)
    for name in ('b_trial_gates.py', 'run_b_trial.py', 'launch_b_trial.sh'):
        result[f'b_trial/{name}'] = file_record(CODE / name)
    return result


def safetensors_record(path):
    regular(path)
    info = path.stat()
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError('Truncated Stage-B safetensors length')
        size = struct.unpack('<Q', prefix)[0]
        if not 2 <= size <= 16 * 1024**2:
            raise RuntimeError('Invalid Stage-B safetensors header length')
        raw = stream.read(size)
    if len(raw) != size:
        raise RuntimeError('Truncated Stage-B safetensors header')
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise RuntimeError('Stage-B safetensors header must be an object')
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    if not tensors or any(not isinstance(row, dict) or 'dtype' not in row for row in tensors.values()):
        raise RuntimeError('Invalid Stage-B tensor descriptors')
    return {'path': str(path), 'size_bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'device': info.st_dev, 'inode': info.st_ino,
            'header_sha256': hashlib.sha256(raw).hexdigest(), 'tensor_count': len(tensors),
            'tensor_dtypes': sorted({row['dtype'] for row in tensors.values()})}


def stage_b_weight_manifest(verified):
    prefix = 'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000/'
    expected = {name: entry for name, entry in verified.items() if name.startswith(prefix)}
    if not expected:
        raise RuntimeError('Completed transfer does not contain Stage-B weights')
    files = {}
    for name, entry in sorted(expected.items()):
        path = ROOT / name
        regular(path)
        info = path.stat()
        if (entry.get('kind') != 'file' or entry.get('size') != info.st_size
                or re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')) is None):
            raise RuntimeError(f'Stage-B asset differs from the verified transfer: {name}')
        record = {'path': str(path), 'size_bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                  'device': info.st_dev, 'inode': info.st_ino}
        if path.suffix == '.safetensors':
            record = safetensors_record(path)
        elif info.st_size <= 1024**2:
            record['sha256'] = profiles.digest(path)
            if record['sha256'] != entry['sha256']:
                raise RuntimeError(f'Stage-B configuration changed: {name}')
            if path.suffix == '.json':
                record['json'] = json_file(path)
        record['transfer_payload_sha256'] = entry['sha256']
        files[name[len(prefix):]] = record
    branch = files.get('linear_branch/model.safetensors')
    lora = files.get('adapters/default/adapter_model.safetensors')
    if (branch is None or branch['tensor_count'] != 800 or branch['tensor_dtypes'] != ['BF16']
            or lora is None or lora['tensor_count'] != 416):
        raise RuntimeError('Expected full 800-BF16-tensor branch and 416-tensor / 208-pair Stage-B LoRA')
    return {'files': files, 'branch': branch, 'lora': lora,
            'payload_hash_scope': 'Full transfer SHA was verified; current size/identity/header and small config SHA only. No large payload rehash.'}


def weight_manifest(verified):
    stage_b = stage_b_weight_manifest(verified)
    return {'ref2va': a_gates.weight_manifest(verified), 'stage_b': stage_b,
            'branch': stage_b['branch'], 'lora': stage_b['lora']}


def logging_configuration():
    # Process IDs come from Python's LogRecord.process, never a rank guessed
    # from eight unlabelled messages. Match the real 30213 B loader format.
    return {'version': 1, 'disable_existing_loggers': False,
            'formatters': {'h3b': {'format': 'H3B pid=%(process)d %(asctime)s %(levelname)s %(name)s %(message)s'}},
            'handlers': {'h3b': {'class': 'logging.StreamHandler', 'formatter': 'h3b', 'stream': 'ext://sys.stderr'}},
            'loggers': {name: {'handlers': ['h3b'], 'level': 'INFO', 'propagate': False}
                        for name in ('vllm', 'vllm_omni')}}


def parse_full_load_evidence(log_text, weights):
    records = {}
    for marker, field in (('OPENVDN_B_LOAD_RECORD', 'load'), ('OPENVDN_B_CPU_LOADING_END', 'cpu_loading')):
        pattern = re.compile(r'H3B pid=(\d+) [^\n]*?' + marker + r' (\{[^\n]*\})')
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
