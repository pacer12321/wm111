#!/usr/bin/env python3
"""Execute the real offline CPU checker and atomically publish its bound report.

Fixed 31731 private installation only. No NPU call, service, model, download or
installation is requested. Latest is set to running before execution, then to
passed or failed; a failed attempt must never leave an older passing latest.
"""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import traceback
import uuid

ROOT = Path('/cache/zhonghao/h3')
ENV = ROOT / 'env'
ENV_SCRIPT = ROOT / 'env_h3_31731.sh'
CODE = Path('/home/ma-user/workspace/zhonghao/h3_deploy_31731')
CHECKER = CODE / 'validate_h3_runtime.py'
LATEST = ROOT / 'runtime_validation.json'
HISTORY = ROOT / 'runtime_validation_runs'
HOSTNAME = 'ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0'
SOURCE_ROOT = ROOT / 'src/vllm-omni/vllm_omni/diffusion/models/minimax_h3'
SOURCE_FILES = ('minimax_h3_transformer.py', 'pipeline_minimax_h3.py')
PASSED = 'passed_cpu_runtime_checks_only'
TIMEOUT = 300


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def host_identity():
    actual = {'hostname': socket.gethostname(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'machine': os.uname().machine}
    if (actual['hostname'] != HOSTNAME or actual['machine'] != 'aarch64'
            or not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', actual['boot_id'])):
        raise RuntimeError('Runtime reporting is restricted to the actual 31731 host and boot')
    return actual


def canonical(path, allowed=None):
    allowed = ROOT if allowed is None else allowed
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(allowed) or path.resolve() != path:
        raise RuntimeError(f'Noncanonical/out-of-scope runtime evidence path: {path}')
    return path


def digest(path):
    canonical(path, CODE if path.is_relative_to(CODE) else ROOT)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'Missing/nonregular runtime input: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_binding():
    return {'host': host_identity(), 'env_script_sha256': digest(ENV_SCRIPT),
            'source_sha256': {name: digest(SOURCE_ROOT / name) for name in SOURCE_FILES},
            'validator_path': str(CHECKER), 'validator_sha256': digest(CHECKER),
            'reporter_path': str(CODE / 'run_runtime_validation.py'),
            'reporter_sha256': digest(CODE / 'run_runtime_validation.py')}


def write_new(path, data):
    canonical(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def publish_latest(report):
    canonical(LATEST)
    temp = ROOT / ('runtime_validation.' + uuid.uuid4().hex + '.publishing')
    write_new(temp, report)
    os.replace(temp, LATEST)


def verify_checker_completion(process):
    if len(process.stdout) > 16 * 1024**2 or len(process.stderr) > 16 * 1024**2:
        raise RuntimeError('CPU checker output exceeded its evidence size guard')
    report = json.loads(process.stdout)
    guard = report.get('device_guard', {})
    if (process.returncode != 0 or report.get('status') != PASSED or report.get('npu_inference_verified') is not False
            or guard.get('python_npu_initialized') is not False or guard.get('device_operations_requested') != 0
            or guard.get('npu_lazy_init_and_c_init_guarded') is not True):
        raise RuntimeError('The fresh CPU checker did not complete its guarded passing path')
    for required in ('environment', 'metadata', 'runtime_prefix_files', 'modules', 'abi'):
        if not report.get(required):
            raise RuntimeError(f'CPU checker lacks completed {required} evidence')
    return report


def interrupted(signum, _frame):
    raise InterruptedError(f'CPU validation interrupted by signal {signum}')


def run_once():
    # No fixed-root writes on an unrelated host/interpreter. CLI offers no path
    # override, previous-report input or forced-pass mode.
    host = host_identity()
    if Path(sys.prefix).resolve() != ENV or Path(__file__).resolve().parent != CODE:
        raise RuntimeError('Use private env/bin/python and deploy the reporter at its fixed code directory')
    canonical(ROOT)
    lock_path = canonical(ROOT / 'runtime_validation.lock')
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    lock = os.fdopen(fd, 'r+b')
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise RuntimeError('Runtime-validation mutex must be a private single-link regular file')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = uuid.uuid4().hex
        run_dir = HISTORY / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + run_id)
        report = {'status': 'running_cpu_runtime_checks', 'host': host, 'run_id': run_id,
                  'started_at': now(), 'run_directory': str(run_dir), 'reporter_pid': os.getpid(),
                  'npu_inference_verified': False}
        # Invalidate an older pass before inspecting mutable runtime inputs.
        publish_latest(report)
        error = None
        history_created = False
        try:
            canonical(HISTORY)
            HISTORY.mkdir(exist_ok=True)
            canonical(run_dir); run_dir.mkdir(exist_ok=False)
            history_created = True
            write_new(run_dir / 'started.json', report)
            binding = input_binding()
            if binding['host'] != host:
                raise RuntimeError('Host/boot changed before CPU checker execution')
            report.update(binding)
            child_env = os.environ.copy()
            child_env.pop('LD_PRELOAD', None); child_env.pop('LD_AUDIT', None)
            # Actual vLLM 0.26 logger.py uses envs.VLLM_LOGGING_STREAM for its
            # StreamHandler; vllm_omni.logger propagates to that same logger.
            # Its default is stdout, which would corrupt the checker's JSON.
            # A custom inherited config could override this supported setting.
            child_env.pop('VLLM_LOGGING_CONFIG_PATH', None)
            child_env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                             TORCH_DEVICE_BACKEND_AUTOLOAD='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                             H3_ROOT=str(run_dir), H3_OUTPUT=str(run_dir / 'output'),
                             H3_GROUP_ID='runtime_cpu_' + run_id,
                             VLLM_CONFIGURE_LOGGING='1', VLLM_LOGGING_STREAM='ext://sys.stderr',
                             VLLM_LOGGING_COLOR='0', VLLM_LOGGING_LEVEL='INFO')
            report['checker_logging'] = {'stream': child_env['VLLM_LOGGING_STREAM'],
                                         'configure_logging': True, 'custom_config_removed': True,
                                         'stdout_policy': 'Strict single JSON document; never extract a pass from mixed logs.'}
            command = ['/bin/bash', '--noprofile', '--norc', '-c',
                       'source /cache/zhonghao/h3/env_h3_31731.sh\n'
                       'exec /cache/zhonghao/h3/env/bin/python -B '
                       '/home/ma-user/workspace/zhonghao/h3_deploy_31731/validate_h3_runtime.py']
            report['checker_command'] = command
            process = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT,
                                     env=child_env, start_new_session=True, pass_fds=(fd,), check=False)
            write_new(run_dir / 'checker_process.json', {'returncode': process.returncode,
                       'stdout': process.stdout, 'stderr': process.stderr, 'completed_at': now()})
            # Keep every original checker field, including failed-path details.
            try:
                original = json.loads(process.stdout)
                if not isinstance(original, dict):
                    raise ValueError('Expected a JSON object')
                report.update(original)
                report['checker_report'] = original
            except (ValueError, TypeError):
                pass
            verify_checker_completion(process)
            if input_binding() != binding:
                raise RuntimeError('Host/boot/environment/source/checker files changed during validation')
            report.update(binding, status=PASSED)
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            report.update(status='failed', error=error, traceback=traceback.format_exc())
        finally:
            # Preserve original checker status in checker_report, but the latest
            # outer status MUST be failed if execution/binding/finalization failed.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            report.update(finished_at=now(), npu_inference_verified=False)
            if history_created:
                write_new(run_dir / 'report.json', report)
            publish_latest(report)
        print(json.dumps({'status': report['status'], 'latest': str(LATEST), 'run_directory': str(run_dir), 'error': error}), flush=True)
        return 0 if error is None and report['status'] == PASSED else 1
    finally:
        # No LOCK_UN/delete: an inherited checker FD preserves the mutex if the
        # reporting parent is interrupted while its child still exists.
        lock.close()


def main():
    if len(sys.argv) != 1:
        raise SystemExit('No arguments: fixed installation, actual fresh CPU check only')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run_once()


if __name__ == '__main__':
    raise SystemExit(main())
