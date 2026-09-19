#!/usr/bin/env python3
"""31731 fixed USP8 A: frozen lake (default) or reverse, smoke/formal/cleanup.

Opt-in only. Imports independent reviewed validation safety helpers, never
mutates their globals. No B/C candidate execution, training, retries or choices
of arbitrary source/model paths. All evidence gates precede the model launch.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import trial_gates as gates

profiles, base = gates.profiles, gates.base
HEALTH_TIMEOUT = 2700
REQUEST_TIMEOUT = 1800


def selected_workers(text, cards):
    result = {}
    pattern = r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*[^|]+\|\s*\d+\s*\|$'
    for match in re.finditer(pattern, text, re.MULTILINE):
        card, _chip, pid = map(int, match.groups())
        if card in cards:
            if card in result or pid in result.values():
                raise RuntimeError('Expected one distinct owned DiT worker per selected physical card')
            result[card] = pid
    if set(result) != set(cards):
        raise RuntimeError('Not all eight selected-card worker contexts remain alive')
    return result


class ATrialSupervisor(base.ProcessSupervisor):
    def __init__(self, group, sample_id=gates.DEFAULT_SAMPLE):
        self.host = profiles.require_host()
        self.sample = gates.sample_profile(sample_id)
        super().__init__(gates.selected_profile(group, sample_id), uuid.uuid4().hex)
        self.frozen = None
        self._workflow_started = False
        self._request_order = []
        self.status = {'case': 'A', 'sample_id': self.sample.sample_id, 'group': group,
                       'host': self.host, 'run_id': self.run_id, 'started_at': base.utc_now(),
                       'supervisor_pid': os.getpid(), 'supervisor_proc_identity': base.proc_identity(os.getpid()),
                       'allocated_physical_npu_ids': list(self.selected.cards), 'port': self.selected.master_port,
                       'parallelism': gates.parallelism(self.selected), 'requested_generation': gates.GENERATION.copy(),
                       'formal_request_attempts': 0, 'formal_50step_started': False, 'formal_50step_completed': False,
                       'quality_evidence': False, 'actual_model_forward_calls': None, 'new_acceleration_results': None,
                       'startup_timeouts': {'init': 2400, 'stage_init': 2400, 'health': HEALTH_TIMEOUT},
                       'request_timeout_seconds': REQUEST_TIMEOUT}

    def update(self, phase, **values):
        self.status.update(values, phase=phase, updated_at=base.utc_now())
        if self.run_dir is not None:
            base.atomic_json(self.run_dir / 'a_status.json', self.status)
            base.atomic_json(self.selected.output_root / 'a_status.json', self.status)
        print(json.dumps({'phase': phase, 'group': self.selected.group, 'run_id': self.run_id, **values}), flush=True)

    def acquire(self):
        for path in (self.selected.output_root, self.selected.lease_root):
            profiles.canonical_private(path)
            path.mkdir(parents=True, exist_ok=True)
            profiles.canonical_private(path)
        self.take_lock(self.selected.output_root / 'run.lock')
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        new_run = self.selected.output_root / 'runs' / f'a_{stamp}_{self.run_id}'
        profiles.canonical_private(new_run)
        new_run.mkdir(parents=True, exist_ok=False)
        self.run_dir = new_run
        for path in (self.run_dir / 'output', gates.ROOT / 'tmp' / f'a_{self.run_id}'):
            profiles.canonical_private(path)
            path.mkdir(parents=True, exist_ok=False)
        self.status['run_directory'] = str(self.run_dir)
        base.atomic_json(self.run_dir / 'host_identity.json', self.host)
        self.update('acquiring_personal_device_leases')
        for card in self.selected.cards:
            self.take_lock(self.selected.lease_root / f'device{card}.lock')
        self.update('personal_device_leases_acquired')

    def evidence_snapshot(self):
        profiles.require_host(self.host)
        sources = gates.source_code_manifest()
        transfer = gates.transfer_gate(self.host)
        runtime = gates.runtime_gate(self.host, sources)
        tiny = gates.tiny_gate(self.selected.group, self.host)
        sample = gates.sample_gate(self.sample.sample_id, transfer=transfer)
        return {'host': self.host, 'run_id': self.run_id, 'cards': list(self.selected.cards),
                'parallelism': gates.parallelism(self.selected), 'generation': gates.GENERATION.copy(),
                'sources': sources, 'transfer': transfer, 'runtime': runtime, 'tiny': tiny, 'sample': sample,
                'weights': gates.weight_manifest(transfer['verified']),
                'precision_policy': {'cli_override': None, 'dit': 'Original source BF16 parameters/activations',
                                     'video_vae_decode': 'Original source FP16 autocast',
                                     'remaining_components': 'Unchanged original pipeline; configs/headers preserved, not a claim of uniform BF16'}}

    def assert_unchanged(self):
        if self.frozen is None or self.evidence_snapshot() != self.frozen:
            raise RuntimeError('Host, source, model assets, migration/runtime/tiny proof or code changed since preflight')

    def preflight(self):
        if (Path(__file__).resolve().parent != self.selected.code_root
                or Path(profiles.__file__).resolve().parent != gates.VALIDATION_CODE
                or Path(base.__file__).resolve().parent != gates.VALIDATION_CODE):
            raise RuntimeError('Trial and validation safety helpers must be at their fixed private deployment roots')
        if len(self.locks) != profiles.WORLD + 1 or any(handle.closed for handle in self.locks):
            raise RuntimeError('Fresh inspection requires all eight card locks plus the trial group lock')
        for executable in ('npu-smi', 'bash', 'curl'):
            if shutil.which(executable) is None:
                raise RuntimeError(f'Missing required executable: {executable}')
        self.frozen = self.evidence_snapshot()
        base.atomic_json(self.run_dir / 'frozen_evidence.json', self.frozen)
        # Preserve the preparation record verbatim as data, not instructions.
        base.atomic_json(self.run_dir / 'sample_manifest.json', self.frozen['sample']['manifest'])
        check = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=False)
        (self.run_dir / 'npu_before.txt').write_text(check.stdout + '\n' + check.stderr)
        if check.returncode:
            raise RuntimeError('npu-smi failed')
        idle = base.selected_idle(check.stdout, self.selected.cards)
        health = base.selected_health_memory(check.stdout, self.selected.cards, gates.MIN_HBM_MIB)
        memory = base.host_memory_snapshot(gates.MIN_RAM)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', self.selected.master_port))
        self.update('preflight_passed', resource_snapshot=idle, card_health_memory=health, host_memory=memory,
                    frozen_evidence_sha256=profiles.digest(self.run_dir / 'frozen_evidence.json'),
                    source_video=str(self.sample.source), edit_prompt=self.sample.prompt,
                    note='A original source only; one fresh two-step smoke must pass before one formal request.')

    def env(self):
        env = os.environ.copy()
        # Never inherit a B checkpoint/candidate activation or foreign Python
        # override. The launcher establishes the approved private environment.
        for key in list(env):
            if key.startswith('ZHONGHAO_H3_OPENVDN') or key in ('PYTHONPATH', 'VLLM_LOGGING_CONFIG_PATH'):
                env.pop(key)
        env.update(H3_ROOT=str(self.run_dir), H3_OUTPUT=str(self.run_dir / 'output'), H3_MINIMAL_RUN_ID=self.run_id,
                   H3_TRIAL_GROUP=self.selected.group, H3_TRIAL_SAMPLE=self.sample.sample_id,
                   H3_TRIAL_SUPERVISOR_PID=str(os.getpid()),
                   H3_PORT=str(self.selected.master_port), H3_GROUP_ID=f'zhonghao_31731_A_{self.selected.group}_{self.run_id}',
                   ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, self.selected.cards)),
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', ZHONGHAO_H3_OPENVDN='0')
        return env

    def launch(self):
        self.assert_unchanged()
        log = (self.run_dir / 'server.log').open('ab', buffering=0)
        self.handles.append(log)
        self.server = self.spawn(['bash', str(self.selected.code_root / 'launch_a_trial.sh')],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.update('server_starting', server_pid=self.server.pid, server_started_at=base.utc_now(),
                    server_proc_identity=base.proc_identity(self.server.pid))
        deadline = time.monotonic() + HEALTH_TIMEOUT
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f'A server exited during startup: {self.server.returncode}')
            try:
                with opener.open(f'http://127.0.0.1:{self.selected.master_port}/health', timeout=5) as response:
                    if response.status == 200:
                        self.update('server_healthy', server_ready_at=base.utc_now())
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError('A server did not become healthy within 2700 seconds')

    def request(self, name, steps):
        expected = [('smoke_2step', 2), ('a_50step', 50)]
        index = len(self._request_order)
        if index >= 2 or (name, steps) != expected[index]:
            raise RuntimeError('Only one two-step smoke followed by one 50-step formal request is allowed')
        if steps == 50 and not self.status.get('formal_smoke_gate_passed'):
            raise RuntimeError('A formal request cannot bypass the fresh same-service smoke gate')
        self._request_order.append((name, steps))  # Count attempts before any I/O; never retry.
        directory = self.run_dir / name
        directory.mkdir(exist_ok=False)
        generation = gates.GENERATION
        extra = {'task': 'ref2va', 'duration': generation['duration_seconds'], 'audio_flow_shift': generation['audio_flow_shift']}
        fields = {key: generation[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
        fields.update(prompt=self.sample.prompt, num_inference_steps=steps, extra_params=json.dumps(extra))
        source_record = gates.file_record(self.sample.source)
        if source_record != self.frozen['sample']['source']:
            raise RuntimeError('Source changed immediately before upload')
        record = {'started_at': base.utc_now(), 'host': self.host, 'group': self.selected.group, 'run_id': self.run_id,
                  'sample_id': self.sample.sample_id, 'fields': fields,
                  'source_video': str(self.sample.source), 'source_sha256': source_record['sha256'],
                  'requested_steps': steps, 'actual_model_forward_calls': None, 'quality_evidence': False,
                  'server_proc_identity': base.proc_identity(self.server.pid)}
        base.atomic_json(directory / 'request.json', record)
        headers, body = directory / 'response.headers', directory / 'response.body'
        command = ['curl', '--silent', '--show-error', '--noproxy', '*', '--connect-timeout', '10',
                   '--max-time', str(REQUEST_TIMEOUT), '--dump-header', str(headers), '--output', str(body),
                   '--write-out', '%{http_code}', '--request', 'POST', f'http://127.0.0.1:{self.selected.master_port}/v1/videos/sync']
        for key, value in fields.items():
            command.extend(['--form-string', f'{key}={value}'])
        command.extend(['--form', f'input_references=@{self.sample.source};type=video/mp4'])
        stdout, stderr = (directory / 'curl.stdout').open('wb'), (directory / 'curl.stderr').open('wb')
        self.handles.extend([stdout, stderr])
        self.update(name + '_request_running')
        start = time.monotonic()
        child = self.spawn(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        while child.poll() is None:
            if self.server.poll() is not None:
                raise RuntimeError('A server exited during request')
            if time.monotonic() - start > REQUEST_TIMEOUT + 20:
                raise TimeoutError('Request exceeded the 1800-second guard plus cleanup allowance')
            time.sleep(1)
        stdout.flush(); stderr.flush()
        elapsed = time.monotonic() - start
        code = (directory / 'curl.stdout').read_text(errors='replace').strip()
        header_text = headers.read_text(errors='replace') if headers.is_file() else ''
        types = re.findall(r'^content-type:\s*([^\r\n]+)', header_text, re.IGNORECASE | re.MULTILINE)
        content_type = types[-1].strip().lower() if types else ''
        success = child.returncode == 0 and code == '200' and content_type.split(';')[0] == 'video/mp4' and body.is_file() and body.stat().st_size > 0
        result = {'completed_at': base.utc_now(), 'requested_steps': steps, 'success': success,
                  'http_code': code, 'content_type': content_type, 'curl_returncode': child.returncode,
                  'request_end_to_end_seconds': elapsed, 'actual_model_forward_calls': None,
                  'dit_latency_seconds': None, 'quality_evidence': False}
        base.atomic_json(directory / 'result.json', result)
        if not success:
            raise RuntimeError(f'{name} failed HTTP/content validation: {result}')
        output = self.run_dir / 'output' / f'{name}.mp4'
        if output.exists() or output.is_symlink():
            raise RuntimeError('Refusing to overwrite an output video')
        body.rename(output)
        probe = gates.video_probe(output, target=True)
        base.atomic_json(directory / 'ffprobe.json', probe)
        result.update(output_video=str(output), output_sha256=profiles.digest(output), ffprobe=probe)
        base.atomic_json(directory / 'result.json', result)
        self.update(name + '_completed', **{name: result})
        return result

    def smoke_gate(self):
        if len(self.locks) != profiles.WORLD + 1 or any(handle.closed for handle in self.locks) or self._request_order != [('smoke_2step', 2)]:
            raise RuntimeError('Formal gate requires unchanged leases and exactly one fresh smoke attempt')
        self.assert_unchanged()
        expected, current = self.status.get('server_proc_identity', {}), base.proc_identity(self.server.pid)
        if (self.server.poll() is not None or current is None
                or any(current.get(key) != expected.get(key) for key in ('pid', 'start_ticks', 'pgrp', 'session'))):
            raise RuntimeError('The exact smoke service is no longer alive')
        smoke = gates.json_file(self.run_dir / 'smoke_2step/result.json')
        request = gates.json_file(self.run_dir / 'smoke_2step/request.json')
        expected_fields = {key: gates.GENERATION[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
        expected_fields.update(prompt=self.sample.prompt, num_inference_steps=2, extra_params=json.dumps({
            'task': 'ref2va', 'duration': gates.GENERATION['duration_seconds'],
            'audio_flow_shift': gates.GENERATION['audio_flow_shift']}))
        if (smoke != self.status.get('smoke_2step') or smoke.get('success') is not True
                or smoke.get('requested_steps') != 2 or smoke.get('http_code') != '200'
                or smoke.get('curl_returncode') != 0 or smoke.get('ffprobe', {}).get('verified') is not True
                or request.get('host') != self.host or request.get('group') != self.selected.group
                or request.get('run_id') != self.run_id or request.get('source_sha256') != self.frozen['sample']['source']['sha256']
                or request.get('source_video') != str(self.sample.source) or request.get('fields') != expected_fields
                or request.get('sample_id') != self.sample.sample_id
                or request.get('requested_steps') != 2
                or any(request.get('server_proc_identity', {}).get(key) != current.get(key)
                       for key in ('pid', 'start_ticks', 'pgrp', 'session'))):
            raise RuntimeError('Smoke result/request does not belong to this source/prompt/service/run')
        output = self.run_dir / 'output/smoke_2step.mp4'
        if (smoke.get('output_video') != str(output) or smoke.get('output_sha256') != profiles.digest(gates.regular(output))
                or gates.video_probe(output, target=True) != smoke['ffprobe']):
            raise RuntimeError('Smoke output changed or failed decoded-frame metadata validation')
        check = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=False)
        (self.run_dir / 'npu_before_formal.txt').write_text(check.stdout + '\n' + check.stderr)
        if check.returncode:
            raise RuntimeError('npu-smi failed before formal')
        health = base.selected_health_memory(check.stdout, self.selected.cards, 0)
        workers = selected_workers(check.stdout, self.selected.cards)
        owned = set(self.owned_group_members(self.server.pid))
        identities = {str(card): base.proc_identity(pid) for card, pid in workers.items()}
        if (not set(workers.values()).issubset(owned)
                or any(identity is None or identity.get('pid') != workers[int(card)]
                       or identity.get('pgrp') != self.server.pid or identity.get('session') != self.server.pid
                       for card, identity in identities.items())):
            raise RuntimeError('Selected-card workers are not all owned by this same A service')
        proof = {'status': 'passed', 'host': self.host, 'run_id': self.run_id, 'group': self.selected.group,
                 'smoke_result': gates.file_record(self.run_dir / 'smoke_2step/result.json'),
                 'smoke_request': gates.file_record(self.run_dir / 'smoke_2step/request.json'),
                 'server_identity': current, 'worker_identities_by_card': identities, 'health': health,
                 'sample_id': self.sample.sample_id,
                 'note': 'Connectivity/metadata only, not edit-quality evidence.'}
        base.atomic_json(self.run_dir / 'formal_smoke_gate.json', proof)
        self.update('formal_smoke_gate_passed', formal_smoke_gate_passed=True, formal_smoke_gate=proof)

    def run_requests(self):
        if self._workflow_started:
            raise RuntimeError('Single-use workflow: no repeated smoke or formal requests')
        self._workflow_started = True
        self.request('smoke_2step', 2)
        self.smoke_gate()
        self.update('formal_50step_authorized', formal_request_attempts=1, formal_50step_started=True,
                    formal_request_started_at=base.utc_now())
        result = self.request('a_50step', 50)
        if result.get('success') is not True or result.get('ffprobe', {}).get('verified') is not True:
            raise RuntimeError('Formal output was not verified')
        self.assert_unchanged()
        self.update('formal_50step_completed_before_cleanup', formal_50step_completed=True,
                    review_required=True, quality_evidence=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=tuple(gates.HTTP_PORTS), required=True)
    parser.add_argument('--sample', choices=gates.SAMPLE_IDS, default=gates.DEFAULT_SAMPLE)
    parser.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args(argv)
    if not args.allow_npu:
        parser.error('Nothing starts without explicit --allow-npu')
    owner = ATrialSupervisor(args.group, args.sample)
    signal.signal(signal.SIGTERM, base.interrupted)
    signal.signal(signal.SIGINT, base.interrupted)
    error = None
    try:
        owner.acquire(); owner.preflight(); owner.launch(); owner.run_requests()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        owner.update('failed_before_cleanup', error=error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean = owner.cleanup()
            if not clean and error is None:
                error = 'Owned A processes remain'
            if owner.status.get('needs_attention') and error is None:
                error = 'Selected-card release could not be verified'
            if error is None and owner.status.get('formal_50step_completed') is not True:
                error = 'No verified formal request completed'
            owner.update('needs_attention' if owner.status.get('needs_attention') else ('failed' if error else 'formal_completed_review_required'),
                         error=error, finished_at=base.utc_now(), note='Original A only. Edit quality, NFE and acceleration remain unreviewed.')
        finally:
            owner.release_locks()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
