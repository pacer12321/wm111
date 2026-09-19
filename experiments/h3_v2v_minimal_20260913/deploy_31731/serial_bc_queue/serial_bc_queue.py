#!/usr/bin/env python3
"""One fixed B6362 lake -> C tiny -> C lake chain; never deploy or retry.

Start only after the reviewed C source/entries have been deployed by the owner.
This coordinator reuses frozen AB file/identity/formal-artifact readers without
changing their globals. Child supervisors alone own the nine model/device locks.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/cache/zhonghao/h3')
CODE = ROOT / 'serial_bc_queue_code'
OUTPUT = ROOT / 'serial_bc_queue'
GROUP, SAMPLE, CARDS = '01234567', 'lake_snow', tuple(range(8))
B_RUN_ID = '6362ce1811da4db69d335b3b3ca5d1a4'
B_STATUS = ROOT / ('b_model_trial/01234567/lake_snow/B/runs/b_20260913T140300Z_' + B_RUN_ID + '/b_status.json')
B_PID, B_START = 4692, 124486422
JOBS = ('C_tiny', 'C_full')
CHILDREN = {
    'C_tiny': dict(code=ROOT / 'c_validation_code', script='run_c_validation.py',
                   runs=ROOT / 'c_validation/01234567/runs', status='c_validation_status.json', timeout=1500),
    'C_full': dict(code=ROOT / 'c_model_trial_code', script='run_c_trial.py',
                   runs=ROOT / 'c_model_trial/01234567/lake_snow/C/runs', status='c_status.json', timeout=4*3600),
}
POLL_SECONDS, B_TIMEOUT, STOP_GRACE = 30, 4*3600, 180
AB_SHA = 'aca213d2f11c694ac8a6d0b105d38262dd9be1abd572967c6adb96de56727762'
PINS = {
    'trial_queue_code/serial_ab_queue.py': AB_SHA,
    'c_model_trial_code/c_trial_gates.py': '45e925f83453e225ac42608e9fad1249d75a5d3dc044c54aaf742c66e77fccce',
    'c_model_trial_code/run_c_trial.py': 'ac5755058f7546de0e36821dae6bcad639f27a771a9cd2855c5b39cdb3aa54b2',
    'c_model_trial_code/launch_c_trial.sh': '430e522b84ac70d33560c11163af5c8197078a0f9573af88595d9b5bdfb309d6',
}
FULL_ACTIVE = {'acquiring_personal_device_leases', 'personal_device_leases_acquired', 'preflight_passed',
    'server_starting', 'server_healthy', 'smoke_2step_request_running', 'smoke_2step_completed',
    'formal_smoke_gate_passed', 'formal_50step_authorized', 'formal_50step_completed_before_cleanup',
    'b_50step_request_running', 'b_50step_completed', 'c_50step_request_running', 'c_50step_completed',
    'cleaning_up_own_processes'}
TINY_ACTIVE = {'personal_device_leases_acquired', 'preflight_passed', 'validation_running',
               'validation_passed_before_cleanup', 'cleaning_up_own_processes'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_environment(env=None):
    """Inspect actual resolution before npu-smi/child launch, not a promised PATH."""
    env = os.environ if env is None else env
    if '/usr/local/sbin' not in env.get('PATH', '').split(':'):
        raise RuntimeError('Source the private bootstrap: PATH lacks /usr/local/sbin')
    if env.get('ASCEND_HOME_PATH') != str(ROOT / 'env/Ascend/cann-9.0.1'):
        raise RuntimeError('Source the fixed private CANN9 environment before this queue')
    records = {}
    for name, expected in {'npu-smi': Path('/usr/local/sbin/npu-smi'), 'python': ROOT / 'env/bin/python',
                           'ffmpeg': ROOT / 'bin/ffmpeg', 'bash': Path('/bin/bash')}.items():
        found = shutil.which(name, path=env.get('PATH', ''))
        if (found is None or Path(found).resolve() != expected.resolve()
                or not expected.is_file() or not os.access(expected, os.X_OK)):
            raise RuntimeError(f'Wrong/missing executable resolution for {name}: {found!r}')
        info = expected.stat()
        records[name] = dict(lookup=found, resolved=str(expected.resolve()), size=info.st_size,
                             device=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns)
    return records


def load_dependencies():
    if Path(__file__).resolve().parent != CODE or Path(sys.prefix).resolve() != ROOT / 'env':
        raise RuntimeError('Use the fixed queue deployment and private H3 Python')
    check_environment()
    # Do not import unknown mutable helpers. No arbitrary CLI paths are accepted.
    for relative, expected in PINS.items():
        path = ROOT / relative
        if path.resolve() != path or not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise RuntimeError(f'C preparation/frozen helper is missing or changed: {relative}; no job starts')
    sys.path.insert(0, str(ROOT / 'trial_queue_code'))
    fs = importlib.import_module('serial_ab_queue')
    sys.path.insert(0, str(ROOT / 'c_model_trial_code'))
    gates = importlib.import_module('c_trial_gates')
    if Path(fs.__file__).resolve() != ROOT / 'trial_queue_code/serial_ab_queue.py' or Path(gates.__file__).resolve() != ROOT / 'c_model_trial_code/c_trial_gates.py':
        raise RuntimeError('Imported a helper outside the reviewed private roots')
    return fs, gates


class BCQueue:
    def __init__(self, fs, gates):
        self.fs, self.gates = fs, gates
        self.host = gates.profiles.require_host()
        self.queue_id = 'b_' + B_RUN_ID + '_c_tiny_c_lake'
        self.directory = OUTPUT / self.queue_id
        self.lock, self.log, self.child, self.child_identity = None, None, None, None
        self.created, self.current = False, None
        self.bindings, self.completed, self.intents = {}, {}, set()
        self.manifest, self.executables = None, None
        self.before_runs = set()
        self.status = dict(schema=1, queue_id=self.queue_id, host=self.host, created_at=fs.utc_now(),
                           predecessor_status=str(B_STATUS), predecessor_run_id=B_RUN_ID,
                           allowed_jobs=list(JOBS), launches={'C_tiny': 0, 'C_full': 0},
                           queue_identity=gates.base.proc_identity(os.getpid()), retry_allowed=False,
                           quality_evidence=False, acceleration_claim=False)

    def production_manifest(self):
        pins = {**self.fs.PINNED_FILES, **PINS,
                **{'c_validation_code/' + name: sha for name, sha in self.gates.C_VALIDATION_SHA256.items()}}
        result = {}
        for name, sha in pins.items():
            record = self.fs.file_record(ROOT / name)
            if record['sha256'] != sha: raise RuntimeError(f'Frozen code changed: {name}')
            result[name] = record
        for name in ('serial_bc_queue.py', 'bootstrap_bc.sh'):
            result['queue/' + name] = self.fs.file_record(CODE / name)
        deploy = self.fs.read_json(ROOT / 'candidates/c_v1/deploy_manifest.json')
        if (deploy.get('state') != 'completed' or deploy.get('host') != self.host
                or deploy.get('destination') != str(self.gates.VENDOR)
                or deploy.get('published') is not True or deploy.get('verified_all_destination_size_sha256') is not True
                or deploy.get('source_copy_before_after_unchanged') is not True):
            raise RuntimeError('C vendor has not been completely deployed/verified by the owner; no automatic deployment')
        result['c_deployment'] = self.fs.file_record(ROOT / 'candidates/c_v1/deploy_manifest.json')
        result['c_sources'] = self.gates.source_code_manifest()
        return result

    def update(self, phase, **values):
        changed = self.status.get('phase') != phase
        self.status.update(values, phase=phase, updated_at=self.fs.utc_now())
        if self.created: self.fs.atomic_json(self.directory / 'queue_status.json', self.status)
        if changed: print(json.dumps(dict(phase=phase, queue_id=self.queue_id, **values)), flush=True)

    def acquire(self):
        import fcntl
        # Preparation failure does not consume a ledger or reserve any card.
        self.manifest = self.production_manifest()
        self.executables = check_environment()
        self.bindings['B'] = self.binding(self.fs.read_json(B_STATUS), 'B', B_STATUS)
        self.fs.private(OUTPUT).mkdir(parents=True, exist_ok=True)
        self.fs.private(OUTPUT)
        fd = os.open(OUTPUT / 'queue.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            os.close(fd); raise RuntimeError('Unsafe private queue mutex')
        self.lock = os.fdopen(fd, 'r+b')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.fs.private(self.directory).mkdir(exist_ok=False)  # Deterministic durable no-restart ledger.
        self.fs.sync_directory(OUTPUT)
        self.created = True
        self.fs.exclusive_json(self.directory / 'binding.json', dict(host=self.host, predecessor=self.bindings['B'],
                               production_manifest=self.manifest, executables=self.executables))
        self.update('waiting_for_B', note='Only queue mutex held, no card leases.')

    def assert_frozen(self):
        self.gates.profiles.require_host(self.host)
        if self.manifest != self.production_manifest() or self.executables != check_environment():
            raise RuntimeError('Host/code/deployment/executable binding changed')

    def binding(self, value, kind, path):
        run_id, run = value.get('run_id'), Path(value.get('run_directory', ''))
        self.fs.private(run)
        if (kind not in ('B', *JOBS) or re.fullmatch(r'[0-9a-f]{32}', run_id or '') is None
                or value.get('host') != self.host or value.get('group') != GROUP
                or value.get('allocated_physical_npu_ids') != list(CARDS)):
            raise RuntimeError('Wrong case/host/boot/group/cards/run identity')
        identity = self.fs.stable_identity(value.get('supervisor_proc_identity'))
        if value.get('supervisor_pid') != identity['pid']: raise RuntimeError('Supervisor PID mismatch')
        if kind == 'B':
            if path != B_STATUS or run != B_STATUS.parent or run_id != B_RUN_ID or identity['pid'] != B_PID or identity['start_ticks'] != B_START:
                raise RuntimeError('Only the fixed B6362 predecessor process/path is allowed')
        else:
            job = CHILDREN[kind]
            prefix = '' if kind == 'C_tiny' else 'c_'
            if (run.parent != job['runs'] or path != run / job['status']
                    or re.fullmatch(prefix + r'\d{8}T\d{6}Z_' + run_id, run.name) is None):
                raise RuntimeError('Child status is outside its exact personal per-run namespace')
        if kind == 'C_tiny':
            cp, _ = self.gates.c_validation_modules()
            if value.get('case') != cp.CASE: raise RuntimeError('A/B tiny cannot substitute for C tiny')
        else:
            helper = self.gates.b_gates if kind == 'B' else self.gates
            case = 'B' if kind == 'B' else 'C'
            if (value.get('case') != case or value.get('sample_id') != SAMPLE
                    or value.get('requested_generation') != helper.GENERATION
                    or value.get('parallelism') != helper.parallelism(helper.selected_profile(GROUP, SAMPLE))):
                raise RuntimeError('Full trial must use the frozen lake source/sampler/full-eight parallelism')
        return dict(case='B' if kind == 'B' else ('C' if kind == 'C_full' else 'C_tiny'), run_id=run_id,
                    run_directory=str(run), status_path=str(path), host=self.host, supervisor_identity=identity)

    def completed_evidence(self, kind):
        binding = self.bindings[kind]
        if self.fs.identity_alive(binding['supervisor_identity'], self.gates.base.proc_identity):
            raise RuntimeError('Completed supervisor has not exited')
        path = Path(binding['status_path'])
        if self.binding(self.fs.read_json(path), kind, path) != binding: raise RuntimeError('Bound status identity changed')
        if kind == 'C_tiny':
            proof = self.gates.tiny_gate(GROUP, self.host)  # Actual complete C rank/source/cleanup validator.
            if (proof['status'].get('run_id') != binding['run_id'] or proof['status'].get('run_directory') != binding['run_directory']
                    or self.binding(proof['status'], kind, path) != binding):
                raise RuntimeError('C tiny latest proof is not this queue-owned bound run')
            return proof
        helper = self.gates.b_gates if kind == 'B' else self.gates
        adapter = SimpleNamespace(host=self.host, gates=helper,
                                  basic_status=lambda value, _case, p: self.binding(value, kind, p))
        # Exact previously reviewed full output/source/weights/runtime/cleanup checker;
        # no calls to the AB scheduler or mutations of its module/class globals.
        proof = self.fs.SerialQueue.completed_evidence(adapter, path, binding['case'], binding)
        if kind == 'C_full':
            proof.update(self.strict_completion_evidence(path.parent))
        return proof

    def strict_completion_evidence(self, run):
        """Recheck actual C logs, not merely the existence/hash of a JSON file."""
        gate = self.fs.read_json(run / 'formal_smoke_gate.json')
        raw = self.fs.regular(run / 'server.log').read_bytes()
        prefix_size, prefix_sha = gate.get('server_log_prefix_bytes'), gate.get('server_log_prefix_sha256')
        if (type(prefix_size) is not int or not 0 < prefix_size <= len(raw)
                or re.fullmatch(r'[0-9a-f]{64}', prefix_sha or '') is None
                or hashlib.sha256(raw[:prefix_size]).hexdigest() != prefix_sha):
            raise RuntimeError('C smoke server-log prefix length/SHA differs from its saved proof')
        loads = gate.get('loaded_records_by_pid')
        if (not isinstance(loads, dict) or len(loads) != 8
                or any(re.fullmatch(r'[1-9][0-9]*', key) is None for key in loads)):
            raise RuntimeError('C strict evidence lacks exactly eight canonical recorded load PIDs')
        pids = {int(key) for key in loads}
        workers = gate.get('worker_identities_by_card')
        server = self.fs.stable_identity(gate.get('server_identity'))
        if not isinstance(workers, dict) or set(workers) != {str(card) for card in CARDS}:
            raise RuntimeError('C strict evidence lacks the actual eight selected-card worker identities')
        identities = [self.fs.stable_identity(workers[str(card)]) for card in CARDS]
        if ({identity['pid'] for identity in identities} != pids
                or any(identity['pgrp'] != server['pid'] or identity['session'] != server['pid'] for identity in identities)):
            raise RuntimeError('C strict/load worker PIDs do not match its same-service eight-card smoke proof')
        # The pinned actual parser expects Python tuple/int-key structures;
        # persisted JSON necessarily contains lists/string keys. Normalize only
        # that serialization difference, never repair missing/incorrect fields.
        normalize = lambda value: json.loads(json.dumps(value))
        smoke = self.gates.parse_strict_metadata(raw[:prefix_size].decode('utf-8', errors='replace'), pids, SAMPLE, requests=1)
        if normalize(smoke) != gate.get('strict_metadata'):
            raise RuntimeError('Saved C smoke strict metadata differs from its actual log prefix')
        formal = self.gates.parse_strict_metadata(raw.decode('utf-8', errors='replace'), pids, SAMPLE, requests=2)
        path = run / 'formal_strict_metadata.json'
        if normalize(formal) != self.fs.read_json(path):
            raise RuntimeError('Saved C formal strict metadata differs from its actual two-request log evidence')
        return {'strict_metadata_record': self.fs.file_record(path),
                'strict_server_log_record': self.fs.file_record(run / 'server.log'),
                'strict_log_revalidated': True, 'strict_worker_pids': sorted(pids)}

    def fresh_idle(self, label):
        self.assert_frozen()
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=False)
        if result.returncode: raise RuntimeError('Read-only npu-smi failed')
        record = dict(checked_at=self.fs.utc_now(), under_device_leases=False,
                      idle=self.gates.base.selected_idle(result.stdout, CARDS),
                      health=self.gates.base.selected_health_memory(result.stdout, CARDS, 0),
                      stdout=result.stdout, stderr=result.stderr)
        self.fs.exclusive_json(self.directory / (label + '_idle.json'), record)
        return record

    def wait_for_b(self):
        deadline = time.monotonic() + B_TIMEOUT
        while time.monotonic() < deadline:
            self.assert_frozen()
            value = self.fs.read_json(B_STATUS)
            if self.binding(value, 'B', B_STATUS) != self.bindings['B']: raise RuntimeError('B predecessor changed')
            alive = self.fs.identity_alive(self.bindings['B']['supervisor_identity'], self.gates.base.proc_identity)
            phase = value.get('phase')
            if phase == 'formal_completed_review_required' and not alive:
                self.completed['B'] = self.completed_evidence('B')
                self.fs.exclusive_json(self.directory / 'completed_b.json', self.completed['B'])
                self.fresh_idle('after_b'); self.update('B_completed_cleaned_and_exited'); return
            if phase != 'formal_completed_review_required' and (phase not in FULL_ACTIVE or not alive):
                raise RuntimeError(f'Fixed B did not succeed: phase={phase!r}, alive={alive}')
            self.update('waiting_for_B_exit' if phase == 'formal_completed_review_required' else 'waiting_for_B', b_phase=phase)
            time.sleep(POLL_SECONDS)
        raise TimeoutError('B exceeded the fixed four-hour wait; no C launch')

    def child_environment(self, kind):
        check_environment()
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(('H3_', 'ZHONGHAO_H3_OPENVDN')) or key in ('PYTHONPATH', 'PYTHONHOME', 'VLLM_LOGGING_CONFIG_PATH', 'ASCEND_RT_VISIBLE_DEVICES'):
                env.pop(key)
        env.update(H3_SERIAL_BC_QUEUE_ID=self.queue_id, H3_ROOT=str(self.directory / ('bootstrap_' + kind)),
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        return env  # Preserve verified PATH; fixed child bootstrap sources env again.

    def launch_once(self, kind):
        prior = 'B' if kind == 'C_tiny' else 'C_tiny'
        if kind not in JOBS or (kind == 'C_full' and self.completed.get('C_tiny') is None):
            raise RuntimeError('Only B -> C tiny -> C full is allowed')
        if kind in self.intents or self.status['launches'][kind] != 0 or (self.child is not None and self.child.poll() is None):
            raise RuntimeError('No duplicate/retry or overlapping child launch')
        self.assert_frozen()
        if self.completed_evidence(prior) != self.completed.get(prior): raise RuntimeError('Prerequisite completion evidence changed')
        self.fresh_idle('before_' + kind)
        job = CHILDREN[kind]
        self.fs.private(job['runs'])
        self.before_runs = {str(path) for path in job['runs'].iterdir()} if job['runs'].is_dir() else set()
        command = ['/bin/bash', str(CODE / 'bootstrap_bc.sh'), kind]
        # Tombstone is durable before Popen; crashes may omit work, never duplicate.
        self.fs.exclusive_json(self.directory / (kind + '_launch_intent.json'), dict(kind=kind, host=self.host,
            queue_id=self.queue_id, command=command, predecessor=self.bindings[prior], before_runs=sorted(self.before_runs), retry_allowed=False))
        self.intents.add(kind); self.status['launches'][kind] = 1; self.current = kind
        self.update('starting_' + kind)
        if self.log is not None: self.log.close()
        self.log = (self.directory / (kind + '_supervisor.log')).open('xb', buffering=0)
        self.child = subprocess.Popen(command, cwd=CODE, env=self.child_environment(kind), start_new_session=True,
                                      pass_fds=(self.lock.fileno(),), stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT)
        self.child_identity = self.fs.stable_identity(self.gates.base.proc_identity(self.child.pid))
        self.fs.exclusive_json(self.directory / (kind + '_spawn.json'), dict(pid=self.child.pid, identity=self.child_identity))
        self.update(kind + '_running', child_identity=self.child_identity)

    def discover_child(self, kind):
        if kind in self.bindings:
            binding = self.bindings[kind]; path = Path(binding['status_path'])
            value = self.fs.read_json(path)
            if self.binding(value, kind, path) != binding: raise RuntimeError('Bound child per-run identity changed')
            return value
        # No latest is followed. Discover only new per-run directories and bind
        # exactly once by our Popen PID/start ticks/group/session, not timestamps.
        job = CHILDREN[kind]
        if not job['runs'].is_dir(): return None
        found = []
        for run in job['runs'].iterdir():
            if str(run) in self.before_runs: continue
            self.fs.private(run)
            path = run / job['status']
            if not path.exists(): continue
            value = self.fs.read_json(path)
            if value.get('supervisor_pid') != self.child.pid: continue
            if self.fs.stable_identity(value.get('supervisor_proc_identity')) != self.child_identity: continue
            found.append((self.binding(value, kind, path), value))
        if len(found) > 1: raise RuntimeError('Multiple status runs claim the same queue child identity')
        if not found: return None
        binding, value = found[0]
        self.bindings[kind] = binding
        self.fs.exclusive_json(self.directory / (kind + '_bound_run.json'), binding)
        self.update(kind + '_running', bound_run=binding)
        return value

    def wait_child(self, kind):
        deadline = time.monotonic() + CHILDREN[kind]['timeout']
        while time.monotonic() < deadline:
            self.assert_frozen()
            value = self.discover_child(kind)
            code = self.child.poll()
            if code is not None:
                if code != 0 or value is None: raise RuntimeError(f'{kind} exited {code} or produced no bound status')
                self.completed[kind] = self.completed_evidence(kind)
                self.fs.exclusive_json(self.directory / ('completed_' + kind + '.json'), self.completed[kind])
                self.fresh_idle('after_' + kind); self.update(kind + '_completed_cleaned_and_exited'); return
            if not self.fs.identity_alive(self.child_identity, self.gates.base.proc_identity):
                raise RuntimeError('Owned child PID/start identity changed')
            active = TINY_ACTIVE if kind == 'C_tiny' else FULL_ACTIVE
            terminal = 'completed' if kind == 'C_tiny' else 'formal_completed_review_required'
            if value is not None and value.get('phase') not in active | {terminal}:
                raise RuntimeError(f'{kind} entered failure/unexpected phase: {value.get("phase")}')
            self.update(kind + '_running', child_phase=value.get('phase') if value else 'waiting_for_owned_run')
            time.sleep(POLL_SECONDS)
        raise TimeoutError(f'{kind} exceeded its fixed bounded wait')

    def stop_owned_child(self):
        if self.child is None or self.child.poll() is not None: return True
        if self.child_identity is None or not self.fs.identity_alive(self.child_identity, self.gates.base.proc_identity): return False
        try:
            marker = f'H3_SERIAL_BC_QUEUE_ID={self.queue_id}'.encode()
            if marker not in Path(f'/proc/{self.child.pid}/environ').read_bytes().split(b'\0'): return False
            os.kill(self.child.pid, signal.SIGTERM)  # Supervisor cleans its own workers; never signal B or a foreign group.
            self.child.wait(timeout=STOP_GRACE)
            return True
        except (OSError, subprocess.TimeoutExpired): return False

    def run(self):
        self.acquire(); self.wait_for_b()
        for kind in JOBS: self.launch_once(kind); self.wait_child(kind)
        self.update('completed_review_required', finished_at=self.fs.utc_now(), note='One B-to-C-tiny-to-C chain completed; edit quality/speed remain unreviewed.')

    def release(self):
        if self.log is not None: self.log.close()
        if self.lock is not None: self.lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-c-chain', action='store_true')
    args = parser.parse_args(argv)
    if not args.allow_c_chain: parser.error('Explicit --allow-c-chain required; nothing runs by default')
    fs, gates = load_dependencies()
    owner = BCQueue(fs, gates)
    signal.signal(signal.SIGTERM, fs.interrupted); signal.signal(signal.SIGINT, fs.interrupted)
    error = None
    try: owner.run()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        stopped = owner.stop_owned_child()
        if owner.created: owner.update('failed' if stopped else 'needs_attention', error=error,
            own_child_exited=stopped, retry_allowed=False, finished_at=fs.utc_now())
        else: print(error, file=sys.stderr, flush=True)
    finally: owner.release()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
