#!/usr/bin/env python3
"""One durable 31731 A-lake -> B-lake continuation. No C, retries or restart.

Waiting reads the fixed A per-run status and takes only a private queue mutex,
never device leases. B takes its own eight device leases and checks idle again.
An fsynced exclusive launch-intent tombstone precedes Popen: a crash can omit B,
but restarting this queue cannot duplicate B. Never remove its ledger to retry.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

ROOT = Path('/cache/zhonghao/h3')
CODE = ROOT / 'trial_queue_code'
QUEUE_ROOT = ROOT / 'trial_queue'
B_CODE = ROOT / 'b_model_trial_code'
A_RUNS = ROOT / 'model_trial/01234567/lake_snow/A/runs'
B_OUTPUT = ROOT / 'b_model_trial/01234567/lake_snow/B'
PYTHON = ROOT / 'env/bin/python'
GROUP, SAMPLE, CARDS = '01234567', 'lake_snow', tuple(range(8))
PREDECESSOR_RUN_ID = '294205bef2e043dc84b41315e544b36d'
PREDECESSOR_STATUS = A_RUNS / ('a_20260913T133502Z_' + PREDECESSOR_RUN_ID) / 'a_status.json'
PREDECESSOR_PID, PREDECESSOR_START_TICKS = 4089275, 124318608
POLL_SECONDS, A_WAIT_TIMEOUT, B_WAIT_TIMEOUT = 30, 8 * 3600, 4 * 3600
OWN_B_TERM_GRACE = 120
PINNED_FILES = {
    'model_trial_code/trial_gates.py': 'b9bfed7e602c9aa030d6838fc8188cfcea2707114a78acc783791978138d029f',
    'model_trial_code/run_a_trial.py': '1c0f8c38637c7a543dc1d462c1d9772422a26d1e0196ff70f9917bcd1c728223',
    'model_trial_code/launch_a_trial.sh': 'd9de5950ee55ea02abb688d226c5b4258a83abda5e8d97aea60c194f0d119144',
    'b_model_trial_code/b_trial_gates.py': 'b0e5b1aafe83b15a9b0fcaa3775f690ec2076c014a9f08842a12f8bb32296507',
    'b_model_trial_code/run_b_trial.py': 'b9b4ca20d0d1fc76bd7868f215c827fcaaad494b029cff919278d9946586d26d',
    'b_model_trial_code/launch_b_trial.sh': '91813c626d515ebbd99548d6c994b750dccefa5934b2a62843430bd67ae71431',
    'validation_code/profiles.py': 'f7ff6a1bfe7d4a91f628e861c435619f731338bf6b36ff81ae780ee0d5511269',
    'validation_code/supervision_base.py': '5889609771d9f80565e52f3d6c0bbb239d4a555c28cc8ffec0b65c8abb4cb344',
    'validation_code/run_validation.py': '74535cafaaa49a77ccea45834e4dec596af6c601b9b985bda6ef1a3d31cfec0a',
    'env_h3_31731.sh': 'c02ecc60ad3ff7a8f6730dab97a282c1cce26a0d1890905a6e392c4d04109e9d',
}
A_ACTIVE_PHASES = {
    'acquiring_personal_device_leases', 'personal_device_leases_acquired', 'preflight_passed',
    'server_starting', 'server_healthy', 'smoke_2step_request_running', 'smoke_2step_completed',
    'formal_smoke_gate_passed', 'formal_50step_authorized', 'a_50step_request_running',
    'a_50step_completed', 'formal_50step_completed_before_cleanup', 'cleaning_up_own_processes',
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def private(path):
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts or not path.is_relative_to(ROOT)
            or path.resolve(strict=False) != path):
        raise RuntimeError(f'Outside/noncanonical private H3 path: {path}')
    return path


def regular(path):
    path = private(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f'Missing/empty/nonregular private file: {path}')
    return path


def file_record(path):
    path = regular(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def read_json(path):
    value = json.loads(regular(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise RuntimeError('Status/proof must be a JSON object')
    return value


def sync_directory(path):
    fd = os.open(private(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_json(path, value):
    """Durable no-overwrite record. A leftover/partial record still blocks retry."""
    path = private(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    sync_directory(path.parent)


def atomic_json(path, value):
    path = private(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    exclusive_json(temporary, value)
    os.replace(temporary, path)
    sync_directory(path.parent)


def production_manifest():
    if Path(__file__).resolve().parent != CODE:
        raise RuntimeError('Deploy the queue only at its fixed private code root')
    files = {name: file_record(ROOT / name) for name in PINNED_FILES}
    for name, expected in PINNED_FILES.items():
        if files[name]['sha256'] != expected:
            raise RuntimeError(f'Frozen production dependency changed: {name}')
    files['trial_queue_code/serial_ab_queue.py'] = file_record(CODE / 'serial_ab_queue.py')
    return files


def load_gates():
    # Verify dependencies before importing even their CPU-only helper modules.
    production_manifest()
    if Path(sys.prefix).resolve() != ROOT / 'env' or not Path(sys.executable).resolve().is_relative_to(ROOT / 'env/bin'):
        raise RuntimeError('Use only the private H3 Python interpreter')
    sys.path.insert(0, str(B_CODE))
    module = importlib.import_module('b_trial_gates')
    if Path(module.__file__).resolve() != B_CODE / 'b_trial_gates.py':
        raise RuntimeError('Wrong B gate module import')
    return module


def validate_predecessor_path(path, run_id):
    path = private(path)
    if (re.fullmatch(r'[0-9a-f]{32}', run_id or '') is None
            or run_id != PREDECESSOR_RUN_ID or path != PREDECESSOR_STATUS
            or path.name != 'a_status.json' or path.parent.parent != A_RUNS
            or re.fullmatch(r'a_\d{8}T\d{6}Z_' + re.escape(run_id), path.parent.name) is None):
        raise RuntimeError('Only the explicitly approved corrected A8 lake per-run status is allowed; never latest')
    return path


def stable_identity(identity):
    if (not isinstance(identity, dict)
            or any(type(identity.get(key)) is not int or identity[key] <= 0
                   for key in ('pid', 'start_ticks', 'pgrp', 'session'))):
        raise RuntimeError('Missing actual supervisor PID/start-ticks/process-group/session identity')
    return {key: identity[key] for key in ('pid', 'start_ticks', 'pgrp', 'session')}


def identity_alive(expected, reader):
    current = reader(expected['pid'])
    return current is not None and current.get('state') != 'Z' and all(current.get(key) == value for key, value in expected.items())


class SerialQueue:
    def __init__(self, predecessor_status, predecessor_run_id, gates):
        self.gates = gates
        self.predecessor_status = validate_predecessor_path(predecessor_status, predecessor_run_id)
        self.predecessor_run_id = predecessor_run_id
        self.host = gates.profiles.require_host()
        self.queue_id = 'a_lake_to_b_lake_' + predecessor_run_id
        self.directory = QUEUE_ROOT / self.queue_id
        self.lock = None
        self.created = False
        self.child = None
        self.child_identity = None
        self.b_status_path = None
        self.b_binding = None
        self.log = None
        self.manifest = None
        self.predecessor_binding = None
        self.status = {'schema_version': 1, 'queue_id': self.queue_id, 'host': self.host,
                       'allowed_jobs': ['A lake predecessor', 'one B lake continuation'],
                       'predecessor_status': str(self.predecessor_status), 'predecessor_run_id': predecessor_run_id,
                       'created_at': utc_now(), 'queue_supervisor_pid': os.getpid(),
                       'queue_supervisor_identity': gates.base.proc_identity(os.getpid()),
                       'b_launch_attempts': 0, 'b_launch_started': False, 'b_formal_completed': False,
                       'quality_evidence': False, 'acceleration_claim': False,
                       'timeouts_seconds': {'waiting_for_A': A_WAIT_TIMEOUT, 'B_child': B_WAIT_TIMEOUT}}

    def update(self, phase, **values):
        changed = self.status.get('phase') != phase
        self.status.update(values, phase=phase, updated_at=utc_now())
        if self.created:
            atomic_json(self.directory / 'queue_status.json', self.status)
        if changed:
            print(json.dumps({'queue_id': self.queue_id, 'phase': phase, **values}, ensure_ascii=False), flush=True)

    def acquire(self):
        private(QUEUE_ROOT).mkdir(parents=True, exist_ok=True)
        private(QUEUE_ROOT)
        fd = os.open(QUEUE_ROOT / 'queue.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            os.close(fd)
            raise RuntimeError('Queue mutex must be a private single-link regular file')
        self.lock = os.fdopen(fd, 'r+b')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The directory itself is a durable, deterministic no-restart ledger.
        # Never reopen, truncate or replace an older queue state, even if failed.
        private(self.directory).mkdir(exist_ok=False)
        sync_directory(QUEUE_ROOT)
        self.created = True
        self.manifest = production_manifest()
        self.predecessor_binding = self.basic_status(read_json(self.predecessor_status), 'A', self.predecessor_status)
        exclusive_json(self.directory / 'queue_binding.json', {
            'queue_id': self.queue_id, 'host': self.host, 'predecessor': self.predecessor_binding,
            'production_manifest': self.manifest, 'created_at': self.status['created_at']})
        self.update('waiting_for_A', note='Only the queue mutex is held; no device lock or model process is started.')

    def assert_bindings(self):
        self.gates.profiles.require_host(self.host)
        if self.manifest != production_manifest():
            raise RuntimeError('Host/boot or fixed queue/A/B/validation/environment code changed')

    def basic_status(self, status, case, path):
        run_id, run = status.get('run_id'), Path(status.get('run_directory', ''))
        private(run)
        expected_runs = A_RUNS if case == 'A' else B_OUTPUT / 'runs'
        if (case not in ('A', 'B') or status.get('case') != case or status.get('sample_id') != SAMPLE
                or status.get('group') != GROUP or status.get('host') != self.host
                or status.get('allocated_physical_npu_ids') != list(CARDS)
                or re.fullmatch(r'[0-9a-f]{32}', run_id or '') is None
                or run.parent != expected_runs or path != run / f'{case.lower()}_status.json'
                or re.fullmatch(case.lower() + r'_\d{8}T\d{6}Z_' + run_id, run.name) is None
                or status.get('requested_generation') != self.gates.GENERATION):
            raise RuntimeError('Wrong case/host/boot/sample/cards/sampler/per-run identity')
        expected_profile = (self.gates.a_gates.selected_profile(GROUP, SAMPLE) if case == 'A'
                            else self.gates.selected_profile(GROUP, SAMPLE))
        if status.get('parallelism') != self.gates.parallelism(expected_profile):
            raise RuntimeError('A and B must use the frozen corrected all-eight compute configuration')
        identity = stable_identity(status.get('supervisor_proc_identity'))
        if status.get('supervisor_pid') != identity['pid']:
            raise RuntimeError('Supervisor PID and actual identity disagree')
        if case == 'A' and (run_id != self.predecessor_run_id or identity['pid'] != PREDECESSOR_PID
                            or identity['start_ticks'] != PREDECESSOR_START_TICKS):
            raise RuntimeError('Status is not the approved corrected A8 predecessor process')
        return {'case': case, 'run_id': run_id, 'run_directory': str(run),
                'status_path': str(path), 'host': self.host, 'supervisor_identity': identity}

    def completed_evidence(self, path, case, expected_binding):
        """Validate terminal artifacts against current frozen source/input/assets."""
        status = read_json(path)
        if self.basic_status(status, case, path) != expected_binding:
            raise RuntimeError('Completed per-run binding changed')
        if (status.get('phase') != 'formal_completed_review_required'
                or status.get('formal_50step_started') is not True or status.get('formal_50step_completed') is not True
                or type(status.get('formal_request_attempts')) is not int or status['formal_request_attempts'] != 1
                or status.get('formal_smoke_gate_passed') is not True
                or status.get('cleanup_completed') is not True
                or status.get('selected_cards_verified_idle_after_cleanup') is not True
                or status.get('needs_attention') is not False or status.get('error') is not None
                or status.get('remaining_owned_process_groups') != {}):
            raise RuntimeError('Formal completion, smoke gate, one attempt and complete cleanup are all mandatory')
        if identity_alive(expected_binding['supervisor_identity'], self.gates.base.proc_identity):
            raise RuntimeError('Completed trial supervisor has not exited yet')
        run = path.parent
        frozen = read_json(run / 'frozen_evidence.json')
        if file_record(run / 'frozen_evidence.json')['sha256'] != status.get('frozen_evidence_sha256'):
            raise RuntimeError('Frozen runtime evidence was changed')
        helper = self.gates.a_gates if case == 'A' else self.gates
        sources = helper.source_code_manifest()
        transfer = helper.transfer_gate(self.host)
        current = {'host': self.host, 'run_id': expected_binding['run_id'], 'cards': list(CARDS),
                   'parallelism': status['parallelism'], 'generation': self.gates.GENERATION,
                   'sources': sources, 'transfer': transfer, 'runtime': helper.runtime_gate(self.host, sources),
                   'tiny': helper.tiny_gate(GROUP, self.host), 'sample': helper.sample_gate(SAMPLE, transfer=transfer),
                   'weights': helper.weight_manifest(transfer['verified'])}
        if any(frozen.get(key) != value for key, value in current.items()):
            raise RuntimeError('Completed evidence is stale for the actual host/source/sample/weights/runtime/tiny proofs')
        gate = read_json(run / 'formal_smoke_gate.json')
        if (gate != status.get('formal_smoke_gate') or gate.get('status') != 'passed'
                or gate.get('host') != self.host or gate.get('run_id') != expected_binding['run_id']
                or gate.get('group') != GROUP or gate.get('sample_id') != SAMPLE):
            raise RuntimeError('Completed trial lacks its same-service smoke gate')
        name = case.lower() + '_50step'
        result = read_json(run / name / 'result.json')
        request = read_json(run / name / 'request.json')
        output = run / 'output' / (name + '.mp4')
        expected_fields = {key: self.gates.GENERATION[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
        expected_fields.update(prompt=self.gates.sample_profile(SAMPLE).prompt, num_inference_steps=50,
                               extra_params=json.dumps({'task': 'ref2va', 'duration': self.gates.GENERATION['duration_seconds'],
                                                        'audio_flow_shift': self.gates.GENERATION['audio_flow_shift']}))
        if (result != status.get(name) or result.get('success') is not True or result.get('requested_steps') != 50
                or result.get('http_code') != '200' or result.get('curl_returncode') != 0
                or result.get('content_type', '').split(';')[0] != 'video/mp4'
                or result.get('output_video') != str(output) or result.get('output_sha256') != file_record(output)['sha256']
                or result.get('ffprobe') != self.gates.video_probe(output, target=True)
                or request.get('run_id') != expected_binding['run_id'] or request.get('host') != self.host
                or request.get('group') != GROUP or request.get('sample_id') != SAMPLE
                or request.get('fields') != expected_fields or request.get('requested_steps') != 50
                or request.get('source_video') != str(self.gates.sample_profile(SAMPLE).source)
                or request.get('source_sha256') != self.gates.a_gates.LAKE_SHA256):
            raise RuntimeError('Formal request/video differs from this exact source, prompt, sampler or successful output')
        return {'binding': expected_binding, 'status_record': file_record(path),
                'frozen_record': file_record(run / 'frozen_evidence.json'),
                'smoke_gate_record': file_record(run / 'formal_smoke_gate.json'),
                'request_record': file_record(run / name / 'request.json'),
                'result_record': file_record(run / name / 'result.json'), 'output_record': file_record(output),
                'request_end_to_end_seconds': result.get('request_end_to_end_seconds'), 'quality_evidence': False}

    def fresh_idle(self, label):
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError('Read-only npu-smi failed')
        idle = self.gates.base.selected_idle(result.stdout, CARDS)
        health = self.gates.base.selected_health_memory(result.stdout, CARDS, 0)
        evidence = {'checked_at': utc_now(), 'idle': idle, 'health': health, 'under_device_leases': False,
                    'note': 'Read-only scheduling check. B must still acquire every lease and recheck.',
                    'npu_smi_stdout': result.stdout, 'npu_smi_stderr': result.stderr}
        exclusive_json(self.directory / f'{label}_idle.json', evidence)
        return evidence

    def wait_for_a(self):
        deadline = time.monotonic() + A_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            self.assert_bindings()
            status = read_json(self.predecessor_status)
            if self.basic_status(status, 'A', self.predecessor_status) != self.predecessor_binding:
                raise RuntimeError('A predecessor identity changed')
            alive = identity_alive(self.predecessor_binding['supervisor_identity'], self.gates.base.proc_identity)
            phase = status.get('phase')
            if phase == 'formal_completed_review_required':
                if alive:
                    self.update('waiting_for_A_exit')
                else:
                    evidence = self.completed_evidence(self.predecessor_status, 'A', self.predecessor_binding)
                    exclusive_json(self.directory / 'completed_a_evidence.json', evidence)
                    self.fresh_idle('after_a')
                    self.update('A_completed_cleaned_and_exited', a_completed_evidence=evidence)
                    return evidence
            elif phase not in A_ACTIVE_PHASES or not alive:
                raise RuntimeError(f'A did not finish successfully: phase={phase!r}, original_supervisor_alive={alive}')
            else:
                self.update('waiting_for_A', predecessor_phase=phase)
            time.sleep(POLL_SECONDS)
        raise TimeoutError('A predecessor exceeded the bounded eight-hour wait; no B launch')

    def child_environment(self):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith('H3_') or key.startswith('ZHONGHAO_H3_OPENVDN') or key in ('PYTHONPATH', 'PYTHONHOME', 'VLLM_LOGGING_CONFIG_PATH'):
                env.pop(key)
        env.update(H3_SERIAL_QUEUE_ID=self.queue_id, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   PATH=f'{ROOT}/env/bin:{ROOT}/bin:/usr/local/Ascend/driver/tools:/usr/local/bin:/usr/bin:/bin')
        return env

    def launch_b_once(self):
        self.assert_bindings()
        if self.child is not None or self.status['b_launch_attempts'] != 0:
            raise RuntimeError('Exactly one B launch attempt is allowed')
        # Recheck the immutable predecessor after the read-only idle snapshot.
        if self.completed_evidence(self.predecessor_status, 'A', self.predecessor_binding) != self.status['a_completed_evidence']:
            raise RuntimeError('A completion evidence changed before B launch')
        command = [str(PYTHON), '-B', str(B_CODE / 'run_b_trial.py'), '--group', GROUP, '--sample', SAMPLE, '--allow-npu']
        intent = {'queue_id': self.queue_id, 'host': self.host, 'created_at': utc_now(), 'command': command,
                  'predecessor': self.predecessor_binding, 'retry_allowed': False}
        exclusive_json(self.directory / 'b_launch_intent.json', intent)
        self.update('starting_B', b_launch_attempts=1, b_launch_started=True, b_command=command)
        self.log = (self.directory / 'b_supervisor.log').open('xb', buffering=0)
        self.child = subprocess.Popen(command, cwd=B_CODE, env=self.child_environment(), start_new_session=True,
                                      pass_fds=(self.lock.fileno(),), stdin=subprocess.DEVNULL,
                                      stdout=self.log, stderr=subprocess.STDOUT)
        self.child_identity = stable_identity(self.gates.base.proc_identity(self.child.pid))
        self.update('B_running', b_supervisor_pid=self.child.pid, b_supervisor_identity=self.child_identity)

    def discover_b_status(self):
        if self.b_status_path is not None:
            value = read_json(self.b_status_path)
            if self.basic_status(value, 'B', self.b_status_path) != self.b_binding:
                raise RuntimeError('Bound B per-run identity changed')
            return value
        latest = B_OUTPUT / 'b_status.json'  # Untrusted hint only; bind once to our child and a per-run file.
        if not latest.exists():
            return None
        value = read_json(latest)
        identity = value.get('supervisor_proc_identity', {})
        if (value.get('supervisor_pid') != self.child.pid
                or any(identity.get(key) != expected for key, expected in self.child_identity.items())):
            return None  # An older B's latest record cannot satisfy this queue.
        path = Path(value.get('run_directory', '')) / 'b_status.json'
        binding = self.basic_status(value, 'B', path)
        # latest can lag a subsequent atomic per-run update; identity, not full
        # snapshot equality, is what must remain the same across that race.
        if self.basic_status(read_json(path), 'B', path) != binding:
            raise RuntimeError('B latest hint does not match its owned per-run identity')
        self.b_status_path, self.b_binding = path, binding
        self.update('B_running', b_status_path=str(path), b_run_id=binding['run_id'])
        return read_json(path)

    def wait_for_b(self):
        deadline = time.monotonic() + B_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            self.assert_bindings()
            status = self.discover_b_status()
            code = self.child.poll()
            if code is not None:
                if code != 0 or status is None:
                    raise RuntimeError(f'Own B supervisor failed or produced no bound per-run status: returncode={code}')
                evidence = self.completed_evidence(self.b_status_path, 'B', self.b_binding)
                exclusive_json(self.directory / 'completed_b_evidence.json', evidence)
                self.fresh_idle('after_b')
                self.update('completed_review_required', b_formal_completed=True, b_exit_code=code,
                            b_completed_evidence=evidence, finished_at=utc_now(),
                            note='One A-to-B continuation completed and cleaned. No C, retry or quality/acceleration claim.')
                return
            if not identity_alive(self.child_identity, self.gates.base.proc_identity):
                raise RuntimeError('Own B child identity changed while its handle still reports running')
            self.update('B_running', b_phase=status.get('phase') if status else 'waiting_for_initial_status')
            time.sleep(POLL_SECONDS)
        raise TimeoutError('Own B exceeded the bounded four-hour queue wait')

    def stop_own_b_if_running(self):
        """SIGTERM only our Popen child after identity + environment proof.

        Its own supervisor cleans its model process groups. Never signal A,
        arbitrary group/PID, or SIGKILL a supervisor and orphan its workers.
        """
        if self.child is None or self.child.poll() is not None:
            return True
        if self.child_identity is None or not identity_alive(self.child_identity, self.gates.base.proc_identity):
            return False
        try:
            marker = f'H3_SERIAL_QUEUE_ID={self.queue_id}'.encode()
            if marker not in Path(f'/proc/{self.child.pid}/environ').read_bytes().split(b'\0'):
                return False
            os.kill(self.child.pid, signal.SIGTERM)
            self.child.wait(timeout=OWN_B_TERM_GRACE)
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False

    def release(self):
        if self.log is not None:
            self.log.close()
        if self.lock is not None:
            self.lock.close()


def interrupted(signum, frame):
    raise InterruptedError(f'Queue received signal {signum}; never stop the predecessor or foreign processes')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-status', type=Path, required=True)
    parser.add_argument('--predecessor-run-id', required=True)
    parser.add_argument('--allow-b-launch', action='store_true')
    args = parser.parse_args(argv)
    if not args.allow_b_launch:
        parser.error('Queue launch requires explicit --allow-b-launch; no monitoring or child starts by default')
    path = validate_predecessor_path(args.predecessor_status, args.predecessor_run_id)
    owner = SerialQueue(path, args.predecessor_run_id, load_gates())
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    error = None
    try:
        owner.acquire(); owner.wait_for_a(); owner.launch_b_once(); owner.wait_for_b()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        # Do not replace an existing ledger when a restart/duplicate was refused.
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        stopped = owner.stop_own_b_if_running()
        if owner.created:
            owner.update('failed' if stopped else 'needs_attention', error=error, finished_at=utc_now(),
                         own_b_supervisor_exited=stopped, retry_allowed=False,
                         note='No retry or next job. Any B cleanup/resource release must be read from its own status.')
        else:
            print(error, file=sys.stderr, flush=True)
    finally:
        owner.release()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
