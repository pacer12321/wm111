"""User-authorized repair: preserve old logs, keep active download, resume once."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import bootstrap_a100 as bootstrap
import run_abcd_cuda as queue

WORK = bootstrap.WORK
HISTORY = WORK / 'history/weights_scope_fix_20260914'


def record(status, **details):
    row = {'status': status, 'updated_at': time.time(), **details}
    bootstrap.save('repair_status.json', row)
    print(json.dumps(row), flush=True)


def script_process(pid, expected):
    identity = queue.pid_identity(pid)
    if identity is None:
        return None
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if str(WORK / expected) not in command:
        raise RuntimeError(f'PID {pid} is not the expected owned script')
    return identity[0]


def main():
    lock = (WORK / 'repair.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if HISTORY.exists() or (WORK / 'results').exists():
        raise RuntimeError('Refusing duplicate repair or any repair after model work started')
    state = queue.read(WORK / 'queue_status.json')
    if state.get('formal_completed') or state.get('status') not in ('waiting_for_environment', 'waiting_for_weights', 'failed'):
        raise RuntimeError('Queue is not safely waiting before generation')
    queue_pid = state['pid']
    queue_start = script_process(queue_pid, 'run_abcd_cuda.py')
    HISTORY.mkdir(parents=True)
    if queue_start is not None:
        record('stopping_only_waiting_queue', queue_pid=queue_pid)
        os.kill(queue_pid, signal.SIGTERM)
        for _ in range(30):
            current = queue.pid_identity(queue_pid)
            if current is None or current[0] != queue_start:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Waiting queue did not exit; no escalation to force kill')
    # Retain the pending VAE download and environment installer. Wait for the
    # original downloader to finish naturally so no two writers share a partial.
    old_pid = 164511
    old_start = script_process(old_pid, 'bootstrap_a100.py')
    record('waiting_for_original_download_to_finish', original_downloader_pid=old_pid)
    deadline = time.monotonic() + 3600
    while old_start is not None and time.monotonic() < deadline:
        current = queue.pid_identity(old_pid)
        if current is None or current[0] != old_start:
            break
        time.sleep(5)
    else:
        if old_start is not None:
            raise TimeoutError('Original downloader still active after1h; no files overwritten')
    for name in ('queue_status.json', 'queue.log', 'weights_status.json'):
        path = WORK / name
        if path.exists():
            path.rename(HISTORY / name)
    record('verifying_corrected_83_file_scope', excluded=sorted(bootstrap.EXCLUDED_TEST_ARTIFACTS))
    bootstrap.weights()  # Reuses and SHA-checks existing files; no deletion.
    if queue.read(WORK / 'weights_status.json').get('status') != 'verified':
        raise RuntimeError('Corrected scope verification did not pass')
    record('resuming_existing_bcd_definition')
    result = subprocess.check_output([str(bootstrap.ENV / 'bin/python'),
                                     str(WORK / 'run_abcd_cuda.py'), '--launch'], text=True)
    record('completed', queue_launch=json.loads(result), weights_verified=True,
           history=str(HISTORY), experiment_definition_unchanged=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (WORK / 'repair_weight_scope.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            print(json.dumps({'repair_pid': child.pid}))
    else:
        try:
            main()
        except BaseException as exc:
            record('failed', error=repr(exc))
            raise
