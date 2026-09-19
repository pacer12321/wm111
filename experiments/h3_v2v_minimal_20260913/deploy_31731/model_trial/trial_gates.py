"""Fail-closed CPU gates for two frozen A inputs on the single 31731 USP8 group."""
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

VALIDATION_CODE = Path('/cache/zhonghao/h3/validation_code')
sys.path.insert(0, str(VALIDATION_CODE))
import profiles
import supervision_base as base
import run_validation as tiny_validation

ROOT = profiles.INSTALL
CODE = ROOT / 'model_trial_code'
REF2VA = ROOT / 'models/MiniMax-H3/Ref2VA'
ORIGINAL_VENDOR = ROOT / 'src/vllm-omni'
MEDIA_FFMPEG = ROOT / 'bin/ffmpeg'
MEDIA_FFMPEG_BINARY = ROOT / 'env/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2'
DEPLOYMENT = 'zhonghao_h3_31731_20260913_v2'
HTTP_PORTS = {'01234567': 19098}
GENERATION = {'width': 1344, 'height': 768, 'fps': 24, 'duration_seconds': 5.0,
              'seed': 4101, 'num_inference_steps': 50, 'flow_shift': 12.0, 'audio_flow_shift': 3.0}
ORIGINAL_HASHES = {
    'minimax_h3_transformer.py': '5adad3862b401cf94e1383a11b8bedd7d9d47ca7c8877dc41859eb4e25d4be5f',
    'pipeline_minimax_h3.py': '05e5631964444c93ac58a92d2d772b6b7278908b9f37ea7bc122761a94e55561',
}
LAKE_PROMPT = ('Transform the reference lake video into a photorealistic winter snow scene. '
               'Preserve the original camera movement, framing, shot timing, lake shoreline, mountains, and overall composition. '
               'Cover the landscape with fresh snow, freeze parts of the lake, add gentle falling snow, '
               'and keep synchronized natural winter ambience.')
LAKE_SHA256 = 'e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9'
LAKE_SIZE_BYTES = 12754317
REVERSE_PROMPT = ('Reverse the temporal order of the entire reference clip: play all of its actions backward in time. '
                  'Keep the same two people, their appearance, lighting, camera framing, and background.')
REVERSE_SHA256 = '4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2'
DEFAULT_SAMPLE = 'lake_snow'
SAMPLE_IDS = (DEFAULT_SAMPLE, 'explicit_reverse_couple_124')
MIN_RAM = 600 * 1024**3
MIN_HBM_MIB = 55 * 1024


@dataclass(frozen=True)
class SampleProfile:
    sample_id: str
    source: Path
    prompt: str
    manifest_path: Path | None


def sample_profile(sample_id=DEFAULT_SAMPLE):
    if sample_id == 'lake_snow':
        return SampleProfile(sample_id, ROOT / 'data/minimax_h3_t2va_50step.mp4', LAKE_PROMPT, None)
    if sample_id == 'explicit_reverse_couple_124':
        directory = ROOT / 'data/explicit_reverse_couple_124'
        return SampleProfile(sample_id, directory / 'source.mp4', REVERSE_PROMPT, directory / 'manifest.json')
    raise ValueError('Only the frozen lake_snow or explicit_reverse_couple_124 sample is allowed')


def selected_profile(group, sample_id=DEFAULT_SAMPLE):
    sample = sample_profile(sample_id)
    tiny = profiles.profile(group)
    return replace(tiny, output_root=ROOT / 'model_trial' / group / sample.sample_id / 'A', code_root=CODE,
                   vendor_root=ORIGINAL_VENDOR, master_port=HTTP_PORTS[group])


def parallelism(selected):
    return {'num_gpus': 8, 'usp': 8, 'ring': 1, 'dit_tensor_parallel_size': 1,
            'text_encoder_tp_size': 8, 'layerwise_offload': True, 'vae_use_tiling': True,
            'vae_parallel_mode': 'tile', 'vae_patch_parallel_size': 8,
            'attention_backend': 'FLASH_ATTN', 'listen_host': '127.0.0.1',
            'listen_port': selected.master_port, 'dtype_cli_override': None,
            'hccl_internal_ports': 'Unchanged original vLLM automatic allocation'}


def regular(path):
    profiles.canonical_private(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise RuntimeError(f'Missing, empty or nonregular private file: {path}')
    return path


def json_file(path):
    regular(path)
    return json.loads(path.read_text(encoding='utf-8'))


def file_record(path):
    regular(path)
    return {'path': str(path), 'sha256': profiles.digest(path), 'size_bytes': path.stat().st_size}


def source_code_manifest():
    model = ORIGINAL_VENDOR / 'vllm_omni/diffusion/models/minimax_h3'
    files = {f'original/{name}': model / name for name in ORIGINAL_HASHES}
    files.update({f'original/{name}': ORIGINAL_VENDOR / relative for name, relative in {
        'ulysses.py': 'vllm_omni/diffusion/attention/parallel/ulysses.py',
        'layer.py': 'vllm_omni/diffusion/attention/layer.py',
        'base.py': 'vllm_omni/diffusion/attention/parallel/base.py',
        'factory.py': 'vllm_omni/diffusion/attention/parallel/factory.py',
    }.items()})
    files.update({f'trial/{name}': CODE / name for name in ('trial_gates.py', 'run_a_trial.py', 'launch_a_trial.sh')})
    files.update({f'validation/{name}': VALIDATION_CODE / name for name in
                  ('profiles.py', 'supervision_base.py', 'run_validation.py')})
    files['runtime/env_h3_31731.sh'] = ROOT / 'env_h3_31731.sh'
    # The unchanged bootstrap already places private bin first. Pin the actual
    # executable wrapper and its binary so A/B use one reproducible libx264 path.
    for label, path in (('media/ffmpeg_wrapper', MEDIA_FFMPEG), ('media/ffmpeg_binary', MEDIA_FFMPEG_BINARY)):
        if not os.access(regular(path), os.X_OK):
            raise RuntimeError(f'Private media encoder is not executable: {path}')
        files[label] = path
    result = {label: file_record(path) for label, path in files.items()}
    for name, expected in ORIGINAL_HASHES.items():
        if result[f'original/{name}']['sha256'] != expected:
            raise RuntimeError(f'A original {name} differs from the reviewed unchanged source')
    return result


def transfer_gate(host):
    status_path, verified_path = ROOT / 'consume_status.json', ROOT / 'transfer_verified.json'
    status, verified = json_file(status_path), json_file(verified_path)
    if (status.get('phase') != 'transfer_completed_extraction_and_validation_required'
            or status.get('deployment') != DEPLOYMENT or status.get('role') != 'consume'
            or status.get('hostname') != host['hostname']
            or type(status.get('verified_entries')) is not int or not verified
            or status['verified_entries'] != len(verified)):
        raise RuntimeError('The current complete H3 migration is not verified; do not launch during transfer')
    for prefix in ('src/vllm/', 'src/vllm-ascend/', 'src/vllm-omni/', 'models/MiniMax-H3/Ref2VA/',
                   'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000/'):
        if not any(name.startswith(prefix) for name in verified):
            raise RuntimeError(f'Completed transfer lacks required tree {prefix}')
    if 'env.tar' not in verified:
        raise RuntimeError('Completed transfer lacks the environment archive')
    return {'status_record': file_record(status_path), 'verified_record': file_record(verified_path),
            'status': status, 'verified': verified}


def runtime_gate(host, sources):
    path = ROOT / 'runtime_validation.json'
    report = json_file(path)
    if (report.get('status') != 'passed_cpu_runtime_checks_only' or report.get('host') != host
            or report.get('npu_inference_verified') is not False
            or report.get('env_script_sha256') != sources['runtime/env_h3_31731.sh']['sha256']):
        raise RuntimeError('Missing/current-host CPU runtime pass bound to the active environment script')
    guard = report.get('device_guard', {})
    if (guard.get('python_npu_initialized') is not False or guard.get('device_operations_requested') != 0
            or guard.get('npu_lazy_init_and_c_init_guarded') is not True):
        raise RuntimeError('CPU runtime pass lacks the no-device-initialization evidence')
    expected_original = {name: sources[f'original/{name}']['sha256'] for name in ORIGINAL_HASHES}
    if report.get('source_sha256') != expected_original:
        raise RuntimeError('CPU runtime proof is stale for the original A pipeline/transformer')
    return {'record': file_record(path), 'report': report}


def tiny_gate(group, host):
    selected = profiles.profile(group)
    latest = selected.output_root / 'validation_status.json'
    status = json_file(latest)
    if (status.get('phase') != 'completed' or status.get('host') != host or status.get('group') != group
            or status.get('allocated_physical_npu_ids') != list(selected.cards)
            or status.get('validation_passed') is not True or status.get('cleanup_completed') is not True
            or status.get('selected_cards_verified_idle_after_cleanup') is not True
            or status.get('needs_attention') is not False):
        raise RuntimeError('This group requires its own current-host tiny pass and verified cleanup')
    run = Path(status.get('run_directory', ''))
    profiles.canonical_private(run)
    if run.parent != selected.output_root / 'runs':
        raise RuntimeError('Tiny run directory is outside this group')
    if json_file(run / 'validation_status.json') != status:
        raise RuntimeError('Latest tiny state differs from its completed per-run proof')
    result_path = run / 'npu_wrapper.json'
    if status.get('result_path') != str(result_path) or status.get('result_sha256') != profiles.digest(regular(result_path)):
        raise RuntimeError('Tiny result path/hash differs from the completed status')
    manifest = profiles.source_manifest(selected)
    if status.get('source_sha256') != manifest:
        raise RuntimeError('Tiny source/candidate/runtime files changed after its run')
    result = tiny_validation.verify_results(result_path, selected, host, status.get('run_id'), manifest)
    return {'status_record': file_record(latest), 'status': status, 'result_record': file_record(result_path),
            'result': result}


def video_probe(path, *, target=False):
    regular(path)
    executable = ROOT / 'bin/ffprobe'
    if not executable.is_file():
        executable = Path('/usr/local/ffmpeg/bin/ffprobe')
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('ffprobe is mandatory; no unverified source/smoke/formal output')
    run = subprocess.run([str(executable), '-v', 'error', '-count_frames', '-show_entries',
                          'stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames',
                          '-show_entries', 'format=duration,size', '-of', 'json', str(path)],
                         capture_output=True, text=True, timeout=90, check=False)
    if run.returncode:
        raise RuntimeError(f'ffprobe failed: {run.stderr}')
    data = json.loads(run.stdout)
    videos = [row for row in data.get('streams', []) if row.get('codec_type') == 'video']
    if len(videos) != 1:
        raise RuntimeError('Exactly one video stream is required')
    row = videos[0]
    fps = float(Fraction(row['r_frame_rate']))
    frames = int(row['nb_read_frames'])
    duration = float(data['format']['duration'])
    shape = [row['width'], row['height']]
    if (fps != 24 or frames != 124 or not math.isfinite(duration) or abs(duration-5.0) > 0.5
            or any(type(size) is not int or size <= 0 for size in shape)
            or (target and shape != [1344, 768])):
        raise RuntimeError('Video shape/FPS/decoded-frame count/duration violates the fixed trial')
    return {'verified': True, 'metadata': data, 'ffprobe': str(executable),
            'checks': {'dimensions': shape, 'fps': fps, 'frame_count': frames,
                       'container_duration_seconds': duration}}


def sample_gate(sample_id=DEFAULT_SAMPLE, *, transfer=None):
    selected = sample_profile(sample_id)
    if selected.sample_id == 'lake_snow':
        return lake_sample_gate(selected, transfer)
    manifest = json_file(selected.manifest_path)
    # This schema is intentionally fixed: a prompt/source cannot be swapped
    # through command-line arguments or an unrelated experiment JSON.
    metadata, provenance, router = (manifest.get(key, {}) for key in ('source_metadata', 'provenance', 'router_control'))
    if (manifest.get('schema_version') != 1 or manifest.get('sample_id') != 'explicit_reverse_couple_124'
            or manifest.get('source_video') != str(selected.source)
            or metadata.get('frames') != 124 or metadata.get('fps') != 24 or metadata.get('has_audio') is not False
            or provenance.get('start_frame') != 104 or provenance.get('end_frame_exclusive') != 228
            or provenance.get('original_fps') != 24 or provenance.get('official_source_id') != 'Te_Temporal_reordering_04'
            or provenance.get('original_sha256') != 'a18b5f2dcabe5a97d30350de9e402039a7e6c0886442162b164887698d75d638'
            or provenance.get('instruction_is_self_authored') is not True
            or provenance.get('official_paired_target_available') is not False
            or provenance.get('gt_target_generated') is not False
            or router != {'temporal_change_required': True, 'text_only_should_identify_temporal_edit': True,
                          'vlm_necessity_evidence': False}
            or manifest.get('requested_generation') != GENERATION
            or manifest.get('edit_prompt') != selected.prompt):
        raise RuntimeError('Sample manifest does not match the fixed approved temporal-edit trial')
    source = file_record(selected.source)
    if manifest.get('source_sha256') != source['sha256'] or source['sha256'] != REVERSE_SHA256:
        raise RuntimeError('Source SHA256 differs from the approved sample manifest')
    probe = video_probe(selected.source)
    if (probe['checks']['dimensions'] != [metadata.get('width'), metadata.get('height')]
            or abs(probe['checks']['container_duration_seconds'] - metadata.get('duration_seconds', 0)) > 0.01
            or any(stream.get('codec_type') == 'audio' for stream in probe['metadata'].get('streams', []))):
        raise RuntimeError('Decoded source metadata differs from its frozen source manifest')
    return {'manifest_record': file_record(selected.manifest_path), 'manifest': manifest,
            'source': source, 'probe': probe}


def lake_sample_gate(selected, transfer):
    """Original A source, not a derived OmniEdit sample; no reverse schema reuse."""
    if selected != sample_profile('lake_snow') or not isinstance(transfer, dict):
        raise RuntimeError('Lake profile needs the completed migration evidence')
    name = 'data/minimax_h3_t2va_50step.mp4'
    entry = transfer.get('verified', {}).get(name, {})
    source = file_record(selected.source)
    if (entry.get('kind') != 'file' or entry.get('size') != source['size_bytes']
            or entry.get('sha256') != source['sha256']
            or source['sha256'] != LAKE_SHA256 or source['size_bytes'] != LAKE_SIZE_BYTES):
        raise RuntimeError('Lake source differs from its fully verified original-A transfer entry')
    probe = video_probe(selected.source, target=True)
    manifest = {
        'schema_version': 1, 'sample_id': selected.sample_id, 'source_video': str(selected.source),
        'source_sha256': source['sha256'], 'edit_prompt': selected.prompt,
        'requested_generation': GENERATION.copy(),
        'source_metadata': probe['checks'],
        'provenance': {'type': 'unchanged_original_A_source', 'transfer_entry': name,
                       'original_A_run_id': 'a_20260913T095139_917734Z',
                       'source_cropped_or_reencoded': False, 'gt_target_generated': False},
        'router_control': {'temporal_change_required': False, 'vlm_necessity_evidence': False},
    }
    return {'manifest_record': None, 'manifest_origin': 'frozen_code_profile_and_verified_transfer',
            'transfer_manifest_record': transfer['verified_record'], 'transfer_entry': entry,
            'manifest': manifest, 'source': source, 'probe': probe}


def weight_manifest(verified):
    """Use transfer full SHA plus current file identity/header; no 135GB rehash."""
    prefix = 'models/MiniMax-H3/Ref2VA/'
    expected = {name: row for name, row in verified.items() if name.startswith(prefix)}
    if not expected:
        raise RuntimeError('No transferred original Ref2VA assets')
    files = {}
    for name, row in sorted(expected.items()):
        path = ROOT / name
        regular(path)
        info = path.stat()
        if row.get('kind') != 'file' or row.get('size') != info.st_size or not isinstance(row.get('sha256'), str):
            raise RuntimeError(f'Ref2VA asset differs from its fully checked transfer entry: {name}')
        entry = {'path': str(path), 'size_bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                 'device': info.st_dev, 'inode': info.st_ino, 'transfer_payload_sha256': row['sha256']}
        if path.suffix == '.safetensors':
            with path.open('rb') as stream:
                size = struct.unpack('<Q', stream.read(8))[0]
                if not 2 <= size <= 16 * 1024**2:
                    raise RuntimeError(f'Invalid safetensors header: {name}')
                raw = stream.read(size)
            header = json.loads(raw)
            if len(raw) != size:
                raise RuntimeError(f'Truncated safetensors header: {name}')
            entry.update(header_sha256=hashlib.sha256(raw).hexdigest(),
                         tensor_count=len([key for key in header if key != '__metadata__']),
                         tensor_dtypes=sorted({value['dtype'] for key, value in header.items() if key != '__metadata__'}))
        elif info.st_size <= 1024**2:
            entry['sha256'] = profiles.digest(path)
            if entry['sha256'] != row['sha256']:
                raise RuntimeError(f'Model configuration differs from its transfer SHA: {name}')
            if path.suffix == '.json':
                entry['json'] = json_file(path)
        files[name[len(prefix):]] = entry
    index = json_file(REF2VA / 'transformer/model.safetensors.index.json')['weight_map']
    if len(index) != 535 or len(set(index.values())) != 13:
        raise RuntimeError('Expected original Ref2VA 535 tensors / 13 transformer shards')
    for shard in set(index.values()):
        if f'transformer/{shard}' not in files:
            raise RuntimeError('A required Ref2VA transformer shard is missing')
    return {'files': files, 'payload_hash_scope':
            'Full SHA verified during transfer; now verify size/file identity, tensor headers and small config SHA. No full payload rehash.'}
