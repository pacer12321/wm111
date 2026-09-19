"""One authorized color task measuring only T/S text-state calculations. Existing shared card leases, owned cleanup.

Full unchanged 50-point schedule, one request per case. No retries, training,
old experiment updates, cache optimizations, or external process termination.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path('/cache/zhonghao/h3/text_state_timing_v1')
sys.path.insert(0, '/cache/zhonghao/h3/validation_code')
import profiles
import supervision_base as base


class Diagnostic(base.ProcessSupervisor):
    def __init__(self):
        selected = replace(profiles.profile('01234567'), code_root=ROOT, master_port=19113,
                           cards=(0, 1, 2, 3), group='0123',
                           output_root=ROOT / 'results')
        super().__init__(selected, uuid.uuid4().hex)
        self.root_run = selected.output_root / self.run_id
        self.manifest = json.loads((ROOT / 'manifest.json').read_text())
        self.case = 'D'
        self.status = {'run_id': self.run_id, 'kind': 'text_state_timing_only', 'cases': {},
                       'started_at': base.utc_now(), 'supervisor': base.proc_identity(os.getpid()),
                       'training': False, 'new_speedup_claim': False,
                       'physical_cards': [0, 1, 2, 3], 'world_size': 4,
                       'comparison_scope': 'four_card_diagnostic_not_old_eight_card_latency'}

    def update(self, phase, **values):
        self.status.update(phase=phase, case=self.case, updated_at=base.utc_now(), **values)
        if self.root_run.is_dir():
            base.atomic_json(self.root_run / 'status.json', self.status)
            base.atomic_json(ROOT / 'status.json', self.status)
        print(json.dumps({'phase': phase, 'case': self.case, 'run_id': self.run_id, **values}), flush=True)

    def verify_code(self):
        profiles.require_host(self.manifest['host'])
        if self.selected.cards != (0, 1, 2, 3) or self.manifest['physical_cards'] != [0, 1, 2, 3]:
            raise RuntimeError('Only physical cards 0–3 are authorized; never use 4–7')
        for name, expected in self.manifest['diagnostic_code'].items():
            if profiles.digest(ROOT / name) != expected:
                raise RuntimeError('Diagnostic code changed: ' + name)
        for case, copy in self.manifest['copies'].items():
            for relative, expected in copy['files'].items():
                if profiles.digest(Path(copy['vendor']) / relative) != expected:
                    raise RuntimeError('Diagnostic vendor changed: ' + relative)
        if profiles.digest(Path(self.manifest['source_video'])) != self.manifest['source_sha256']:
            raise RuntimeError('Source video changed')
        # Existing reviewed original candidates remain untouched and validated.
        sys.path.insert(0, '/cache/zhonghao/h3/color_trial_v1/b')
        import b_trial_gates as b
        sys.path.insert(0, '/cache/zhonghao/h3/dualstream_v1/deploy_d')
        import d_trial_gates as d
        if b.source_code_manifest() != self.manifest['original_proofs']['B']:
            raise RuntimeError('Original B evidence changed')
        policy = d.policy_gate(self.manifest['policy_sha256'])
        if d.source_code_manifest(policy['value']) != self.manifest['original_proofs']['D']:
            raise RuntimeError('Original D evidence changed')

    def fresh_idle(self):
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=True)
        (self.run_dir / 'npu_before.txt').write_text(result.stdout)
        idle = base.selected_idle(result.stdout, self.selected.cards)
        health = base.selected_health_memory(result.stdout, self.selected.cards, 40000)
        memory = base.host_memory_snapshot(100 * 1024**3)
        self.update('resources_verified_idle', idle=idle, health=health, memory=memory)

    def env(self):
        result = os.environ.copy()
        for key in list(result):
            if key.startswith(('ZHONGHAO_H3_', 'D_REVIEWED_POLICY', 'S0_')) or key in ('PYTHONPATH', 'VLLM_LOGGING_CONFIG_PATH'):
                result.pop(key)
        result.update(H3_MINIMAL_RUN_ID=self.run_id, H3_PROFILE_CASE=self.case,
                      H3_PROFILE_OUTPUT=str(self.run_dir / 'profile'),
                      H3_PROFILE_VENDOR=self.manifest['copies'][self.case]['vendor'],
                      H3_PROFILE_SUPERVISOR=str(os.getpid()),
                      ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        return result

    def launch(self):
        self.verify_code()
        self.fresh_idle()
        if self.case == 'D':
            smoke_log = (self.run_dir / 'profiler_smoke.log').open('ab', buffering=0)
            self.handles.append(smoke_log)
            command = 'source /cache/zhonghao/h3/env_h3_31731.sh && /cache/zhonghao/h3/env/bin/python /cache/zhonghao/h3/text_state_timing_v1/event_smoke.py'
            smoke = self.spawn(['bash', '-c', command], stdin=subprocess.DEVNULL,
                               stdout=smoke_log, stderr=subprocess.STDOUT)
            self.update('event_timer_smoke_running', smoke=base.proc_identity(smoke.pid))
            if smoke.wait(timeout=300) != 0:
                raise RuntimeError('Profiler export smoke failed; model runs not started')
            self.fresh_idle()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', self.selected.master_port))
        stream = (self.run_dir / 'server.log').open('ab', buffering=0)
        self.handles.append(stream)
        self.server = self.spawn(['bash', str(ROOT / 'launch_server.sh')], stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        self.update('server_starting', server=base.proc_identity(self.server.pid))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 2700
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError('Server exited while loading')
            try:
                with opener.open('http://127.0.0.1:19113/health', timeout=5) as response:
                    if response.status == 200:
                        self.update('server_healthy')
                        return
            except OSError:
                pass
            time.sleep(3)
        raise TimeoutError('Server loading exceeded 2700s')

    def request(self):
        generation = self.manifest['generation']
        fields = {key: generation[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
        fields.update(prompt=self.manifest['prompt'], num_inference_steps=50,
                      extra_params=json.dumps({'task': 'ref2va', 'duration': generation['duration_seconds'],
                                               'audio_flow_shift': generation['audio_flow_shift']}))
        base.atomic_json(self.run_dir / 'request.json', {'fields': fields, 'source_video': self.manifest['source_video'],
                         'source_sha256': self.manifest['source_sha256'], 'text_state_timing_all_49_forwards': True,
                         'not_a_new_speed_benchmark': True})
        command = ['curl', '--silent', '--show-error', '--noproxy', '*', '--connect-timeout', '10',
                   '--max-time', '5400', '--dump-header', str(self.run_dir / 'response.headers'),
                   '--output', str(self.run_dir / 'response.body'), '--write-out', '%{http_code}',
                   '--request', 'POST', 'http://127.0.0.1:19113/v1/videos/sync']
        for key, value in fields.items():
            command += ['--form-string', f'{key}={value}']
        command += ['--form', f'input_references=@{self.manifest["source_video"]};type=video/mp4']
        stdout = (self.run_dir / 'curl.stdout').open('wb')
        stderr = (self.run_dir / 'curl.stderr').open('wb')
        self.handles.extend([stdout, stderr])
        self.update('profiling_request_running')
        start = time.monotonic()
        child = self.spawn(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        while child.poll() is None:
            if self.server.poll() is not None:
                raise RuntimeError('Server exited during profiling request')
            if time.monotonic() - start > 5450:
                raise TimeoutError('Profiling request exceeded guard')
            time.sleep(3)
        stdout.flush()
        stderr.flush()
        code = (self.run_dir / 'curl.stdout').read_text().strip()
        elapsed = time.monotonic() - start
        if child.returncode or code != '200':
            raise RuntimeError(f'Profiling request failed: curl={child.returncode}, HTTP={code}')
        from summarize_text import summarize_run
        summary = summarize_run(self.run_dir / 'profile', elapsed)
        base.atomic_json(self.run_dir / 'text_state_summary.json', summary)
        self.status['cases'][self.case] = {'http_completed': True, 'forward_count_per_rank': 49,
                    'instrumented_request_seconds_NOT_benchmark': elapsed,
                    'summary': str(self.run_dir / 'text_state_summary.json')}
        self.update('profiling_request_completed')

    def run(self):
        profiles.require_host(self.manifest['host'])
        self.selected.output_root.mkdir(parents=True, exist_ok=True)
        self.take_lock(ROOT / 'run.lock')
        self.root_run.mkdir(parents=True, exist_ok=False)
        for card in self.selected.cards:
            self.take_lock(self.selected.lease_root / f'device{card}.lock')
        base.atomic_json(self.root_run / 'manifest.json', self.manifest)
        for case in ('D',):
            self.case = case
            self.run_dir = self.root_run / case
            (self.run_dir / 'profile').mkdir(parents=True, exist_ok=False)
            try:
                self.launch()
                self.request()
            finally:
                clean = self.cleanup()
                self.update('case_cleaned_up')
            if not clean or not self.status.get('selected_cards_verified_idle_after_cleanup'):
                raise RuntimeError('Cannot start next case before owned cleanup and idle verification')
            self.children.clear()
            self.handles.clear()
            self.server = None
        self.verify_code()
        self.update('completed')


if __name__ == '__main__':
    if sys.argv[1:] != ['--run-text-state-once'] or Path(__file__).resolve().parent != ROOT:
        raise SystemExit('Explicit --run-text-state-once at the private deployment root required')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, base.interrupted)
    owner = Diagnostic()
    try:
        owner.run()
    except BaseException as exc:
        owner.update('failed', error=repr(exc))
        raise
    finally:
        owner.release_locks()
