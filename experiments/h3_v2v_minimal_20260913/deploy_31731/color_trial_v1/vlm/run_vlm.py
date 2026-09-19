#!/usr/bin/env python3
"""One real VLM request under all eight personal leases; actual use cards0,1."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

import vlm_contract as contract

base, profiles = contract.base, contract.profiles


def verify_result(directory, frozen, expected_worker):
    result = contract.json_read(directory/'worker_result.json')
    expected = dict(status='passed', case=contract.CASE, host=frozen['host'], run_id=frozen['run_id'],
                    sample_id=contract.SAMPLE, source=frozen['source'], edit_prompt=contract.PROMPT,
                    leased_physical_cards=list(range(8)), used_physical_cards=[0, 1],
                    inference_executed=True, video_tensor_present=True, human_router_labels_sent_to_model=False)
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError('VLM result does not belong to this real host/run/source/inference')
    actual_identity = result.get('worker_proc_identity', {})
    if any(actual_identity.get(key) != expected_worker.get(key) for key in ('pid', 'start_ticks', 'pgrp', 'session')):
        raise RuntimeError('VLM result worker identity differs from the actual spawned worker')
    for name in ('preprocessing.json', 'model_load.json', 'model_output.json'):
        if result.get('records', {}).get(name) != contract.record(directory/name):
            raise RuntimeError('VLM subordinate evidence changed: '+name)
    prep = contract.json_read(directory/'preprocessing.json')
    raw = contract.json_read(directory/'model_output.json')
    load = contract.json_read(directory/'model_load.json')
    if (prep.get('host') != frozen['host'] or prep.get('run_id') != frozen['run_id']
            or prep.get('source') != frozen['source'] or prep.get('edit_prompt') != contract.PROMPT
            or prep.get('frame_indices') != list(contract.FRAME_INDICES)
            or prep.get('messages') != contract.messages() or prep.get('video_tensor_present') is not True
            or prep.get('video_token_count', 0) <= 0 or len(prep.get('sampled_frames', [])) != 16
            or prep.get('rgb_video_shape') != [16, 720, 1280, 3]
            or prep.get('tensors', {}).get('pixel_values_videos', {}).get('numel', 0) <= 0):
        raise RuntimeError('VLM result lacks the actual frozen 16-frame multimodal preprocessing proof')
    if (raw.get('host') != frozen['host'] or raw.get('run_id') != frozen['run_id'] or raw.get('source') != frozen['source']
            or raw.get('edit_prompt') != contract.PROMPT or raw.get('input_token_count') != prep.get('input_token_count')
            or raw.get('generation') != dict(do_sample=False, num_beams=1, max_new_tokens=512, use_cache=True)
            or not isinstance(raw.get('generated_ids'), list) or not 1 <= len(raw['generated_ids']) < 512
            or raw.get('generated_token_count') != len(raw['generated_ids'])
            or any(type(value) is not int or value < 0 for value in raw['generated_ids'])
            or result.get('classification') != contract.parse_classification(raw.get('raw_text'))):
        raise RuntimeError('VLM raw generation and parsed classification do not match')
    if (load.get('host') != frozen['host'] or load.get('run_id') != frozen['run_id']
            or load.get('device_map') != contract.DEVICE_MAP or load.get('parameter_bytes') != 66714780128
            or load.get('model_class') != 'Qwen3VLForConditionalGeneration'
            or load.get('dtype') != 'bfloat16' or load.get('attention') != 'eager'
            or any(load.get('loading_info', {}).get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))):
        raise RuntimeError('Actual full VLM loading evidence is incomplete')
    return result


def verify_completed(run_dir, run_id):
    """Read-only admission reader. No NPU imports, initialization or writes.

    Caller binds this exact per-run directory/status/result SHA in admission;
    do not chase latest. An uncertain/change class is returned unchanged.
    """
    directory = contract.run_directory(run_dir, run_id)
    host = profiles.require_host()
    status = contract.json_read(directory/'vlm_status.json')
    expected = dict(case=contract.CASE, sample_id=contract.SAMPLE, host=host, run_id=run_id,
        run_directory=str(directory), phase='completed', inference_attempts=1, inference_completed=True,
        leased_physical_cards=list(range(8)), used_physical_cards=[0, 1], cleanup_completed=True,
        selected_cards_verified_idle_after_cleanup=True, needs_attention=False, remaining_owned_process_groups={}, error=None)
    if any(status.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Exact VLM run has not completed once with verified cleanup')
    frozen = contract.json_read(directory/'frozen_evidence.json')
    if (status.get('frozen_evidence_sha256') != profiles.digest(contract.regular(directory/'frozen_evidence.json'))
            or frozen.get('host') != host or frozen.get('run_id') != run_id
            or frozen.get('source') != contract.source_record() or frozen.get('edit_prompt') != contract.PROMPT
            or frozen.get('sources') != contract.source_manifest() or frozen.get('model') != contract.model_identity()
            or frozen.get('device_map') != contract.DEVICE_MAP or frozen.get('frame_indices') != list(contract.FRAME_INDICES)
            or frozen.get('runtime') != contract.record(contract.ROOT/'runtime_validation.json')
            or frozen.get('transfer') != contract.record(contract.ROOT/'consume_status.json')
            or contract.json_read(directory/'host_identity.json') != host):
        raise RuntimeError('VLM frozen/current source, host, model, API or runtime proof differs')
    worker = status.get('worker_proc_identity', {})
    supervisor = status.get('supervisor_proc_identity', {})
    for identity in (worker, supervisor):
        if (not isinstance(identity, dict) or type(identity.get('pid')) is not int or identity['pid'] <= 1
                or type(identity.get('start_ticks')) is not int):
            raise RuntimeError('Missing actual VLM process identity')
        current = base.proc_identity(identity['pid'])
        if current is not None and current.get('start_ticks') == identity['start_ticks']:
            raise RuntimeError('Original VLM process identity still exists; admission must wait for exit')
    if status.get('worker_pid') != worker['pid'] or status.get('supervisor_pid') != supervisor['pid']:
        raise RuntimeError('VLM PID fields do not match their recorded identities')
    reader = object.__new__(base.ProcessSupervisor); reader.run_id = run_id
    if reader.owned_group_members(worker['pid']): raise RuntimeError('VLM-owned group members remain')
    release_text = contract.regular(directory/'npu_after.txt').read_text()
    release = base.selected_idle(release_text, tuple(range(8)))
    if status.get('resource_release_check') != release: raise RuntimeError('VLM recorded eight-card release proof differs')
    result_path = directory/'worker_result.json'
    if status.get('result_path') != str(result_path) or status.get('result_sha256') != profiles.digest(contract.regular(result_path)):
        raise RuntimeError('Completed VLM result hash/path changed')
    result = verify_result(directory, frozen, worker)
    if status.get('classification') != result['classification']: raise RuntimeError('VLM terminal classification changed')
    return {'status': status, 'result': result, 'frozen': frozen,
            'records': {name: contract.record(directory/name) for name in (
                'vlm_status.json', 'worker_result.json', 'frozen_evidence.json', 'preprocessing.json',
                'model_output.json', 'model_load.json', 'host_identity.json', 'npu_after.txt')}}


class VLMSupervisor(base.ProcessSupervisor):
    def __init__(self):
        self.host = profiles.require_host()
        selected = replace(profiles.profile('01234567'), output_root=contract.OUTPUT, code_root=contract.CODE)
        super().__init__(selected, uuid.uuid4().hex)
        self.frozen = None
        self.status = dict(case=contract.CASE, sample_id=contract.SAMPLE, host=self.host, run_id=self.run_id,
                           supervisor_pid=os.getpid(), supervisor_proc_identity=base.proc_identity(os.getpid()),
                           started_at=base.utc_now(), used_physical_cards=[0, 1], leased_physical_cards=list(range(8)),
                           inference_attempts=0, inference_completed=False, outer_timeout_seconds=contract.TIMEOUT)

    def update(self, phase, **values):
        self.status.update(values, phase=phase, updated_at=base.utc_now())
        if self.run_dir is not None:
            base.atomic_json(self.run_dir/'vlm_status.json', self.status)
            base.atomic_json(self.selected.output_root/'vlm_status.json', self.status)
        print(json.dumps(dict(phase=phase, run_id=self.run_id, **values)), flush=True)

    def acquire(self):
        for path in (self.selected.output_root, self.selected.lease_root):
            profiles.canonical_private(path); path.mkdir(parents=True, exist_ok=True); profiles.canonical_private(path)
        self.take_lock(self.selected.output_root/'run.lock')
        directory = self.selected.output_root/'runs'/f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{self.run_id}'
        profiles.canonical_private(directory); directory.mkdir(parents=True, exist_ok=False)
        self.run_dir = directory
        self.status['run_directory'] = str(directory)
        base.atomic_json(directory/'host_identity.json', self.host)
        temporary = contract.ROOT/'tmp'/f'vlm_{self.run_id}'
        profiles.canonical_private(temporary); temporary.mkdir(parents=True, exist_ok=False)
        self.update('acquiring_personal_device_leases')
        for card in range(8): self.take_lock(self.selected.lease_root/f'device{card}.lock')

    def evidence_snapshot(self):
        profiles.require_host(self.host)
        sources = contract.source_manifest()
        runtime = contract.json_read(contract.ROOT/'runtime_validation.json')
        transfer = contract.json_read(contract.ROOT/'consume_status.json')
        if (runtime.get('status') != 'passed_cpu_runtime_checks_only' or runtime.get('host') != self.host
                or runtime.get('env_script_sha256') != sources['runtime/env_h3_31731.sh']['sha256']
                or runtime.get('npu_inference_verified') is not False
                or transfer.get('phase') != 'transfer_completed_extraction_and_validation_required'
                or transfer.get('hostname') != self.host['hostname']):
            raise RuntimeError('Current-host runtime/complete transfer prerequisite is absent')
        return dict(host=self.host, run_id=self.run_id, source=contract.source_record(), edit_prompt=contract.PROMPT,
                    sources=sources, runtime=contract.record(contract.ROOT/'runtime_validation.json'),
                    transfer=contract.record(contract.ROOT/'consume_status.json'), model=contract.model_identity(),
                    device_map=contract.DEVICE_MAP.copy(), frame_indices=list(contract.FRAME_INDICES))

    def preflight(self):
        if (Path(__file__).resolve().parent != contract.CODE or Path(base.__file__).resolve().parent != contract.VALIDATION
                or Path(profiles.__file__).resolve().parent != contract.VALIDATION):
            raise RuntimeError('Unexpected VLM or supervision dependency root')
        if len(self.locks) != 9 or any(handle.closed for handle in self.locks): raise RuntimeError('Missing eight leases or group lock')
        for name in ('npu-smi', 'bash'):
            if shutil.which(name) is None: raise RuntimeError('Missing executable '+name)
        self.frozen = self.evidence_snapshot()
        base.atomic_json(self.run_dir/'frozen_evidence.json', self.frozen)
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=False)
        (self.run_dir/'npu_before.txt').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode: raise RuntimeError('npu-smi failed')
        idle = base.selected_idle(result.stdout, self.selected.cards)
        health = base.selected_health_memory(result.stdout, self.selected.cards, 55*1024)
        memory = base.host_memory_snapshot(128*1024**3)
        self.update('preflight_passed', npu_before=idle, card_health_memory=health, host_memory=memory,
                    frozen_evidence_sha256=profiles.digest(self.run_dir/'frozen_evidence.json'))

    def env(self):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith('ZHONGHAO_H3_OPENVDN') or key in ('PYTHONPATH', 'VLLM_LOGGING_CONFIG_PATH'):
                env.pop(key)
        env.update(H3_ROOT=str(self.run_dir), H3_MINIMAL_RUN_ID=self.run_id, H3_VLM_SUPERVISOR_PID=str(os.getpid()),
                   H3_VLM_LEASE_FDS=','.join(str(handle.fileno()) for handle in self.locks),
                   H3_GROUP_ID=f'zhonghao_vlm_{self.run_id}', ASCEND_RT_VISIBLE_DEVICES='0,1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   TORCH_DEVICE_BACKEND_AUTOLOAD='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        return env

    def launch(self):
        if self.status['inference_attempts'] != 0: raise RuntimeError('No VLM retries or repeated generation')
        if self.evidence_snapshot() != self.frozen: raise RuntimeError('VLM evidence changed before launch')
        self.update('launching_single_inference', inference_attempts=1)
        log = (self.run_dir/'worker.log').open('ab', buffering=0); self.handles.append(log)
        self.server = self.spawn(['bash', str(contract.CODE/'launch_vlm.sh')], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        identity = base.proc_identity(self.server.pid)
        if identity is None: raise RuntimeError('Spawned VLM worker identity unavailable')
        self.update('vlm_running', worker_pid=self.server.pid, worker_proc_identity=identity, worker_started_at=base.utc_now())
        started = time.monotonic()
        while self.server.poll() is None:
            if time.monotonic()-started > contract.TIMEOUT: raise TimeoutError('VLM exceeded 1800 seconds; no retry')
            time.sleep(1)
        if self.server.returncode != 0: raise RuntimeError('VLM worker failed; retain raw logs/reply, no classification assumed')
        if self.evidence_snapshot() != self.frozen: raise RuntimeError('VLM evidence changed during inference')
        result = verify_result(self.run_dir, self.frozen, identity)
        self.update('inference_verified_before_cleanup', inference_completed=True,
                    result_path=str(self.run_dir/'worker_result.json'), result_sha256=profiles.digest(self.run_dir/'worker_result.json'),
                    classification=result['classification'], inference_wall_seconds=time.monotonic()-started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args(argv)
    if not args.allow_npu: parser.error('No inference without explicit --allow-npu; use worker --prepare-only for CPU decode')
    owner = VLMSupervisor()
    signal.signal(signal.SIGTERM, base.interrupted); signal.signal(signal.SIGINT, base.interrupted)
    error = None
    try:
        owner.acquire(); owner.preflight(); owner.launch()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'; owner.update('failed_before_cleanup', error=error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean = owner.cleanup()
            if not clean or owner.status.get('needs_attention'): error = error or 'VLM owned process/device release failed'
            if (owner.status.get('inference_completed') is not True
                    or owner.status.get('selected_cards_verified_idle_after_cleanup') is not True):
                error = error or 'No completed inference and verified eight-card idle cleanup'
            owner.update('needs_attention' if owner.status.get('needs_attention') else ('failed' if error else 'completed'),
                         error=error, finished_at=base.utc_now(),
                         note='Classification is VLM evidence only, not ground truth or generated-video quality.')
        finally:
            owner.release_locks()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
