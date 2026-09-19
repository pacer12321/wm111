"""One authorized pre-inference install transfer change, with pinned hashes."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import bootstrap_a100 as bootstrap
import run_abcd_cuda as queue

WORK = bootstrap.WORK
HISTORY = WORK / 'history/environment_proxy_20260914'
INDEX = 'https://pypi.org/simple'
PROXY = 'http://127.0.0.1:18088'


def record(status, **extra):
    row = dict(status=status, updated_at=time.time(), **extra)
    bootstrap.save('environment_acceleration_status.json', row)
    print(json.dumps(row), flush=True)


def identity(pid, fragment):
    found = queue.pid_identity(pid)
    if found is None:
        raise RuntimeError(f'Expected live owned PID {pid} absent; reassess before mutation')
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if fragment not in cmd:
        raise RuntimeError(f'PID {pid} identity mismatch')
    return found[0]


def terminate(owned):
    for pid, birth in owned.items():
        found = queue.pid_identity(pid)
        if found and found[0] == birth:
            os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        live = []
        for pid, birth in owned.items():
            found = queue.pid_identity(pid)
            if found and found[0] == birth:
                stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                if stat[0] != 'Z':
                    live.append(pid)
        if not live:
            return
        time.sleep(1)
    raise RuntimeError(f'Owned processes did not stop: {live}; no force kill')


def main():
    lock = (WORK / 'environment_acceleration.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if HISTORY.exists() or (WORK / 'results').exists():
        raise RuntimeError('Refuse duplicate or change after inference started')
    state = queue.read(WORK / 'queue_status.json')
    if state['status'] != 'waiting_for_environment' or state.get('formal_completed'):
        raise RuntimeError('Queue not safely waiting')
    if queue.read(WORK / 'environment_status.json')['status'] != 'installing':
        raise RuntimeError('Original install state changed; reassess')
    pins = dict(re.findall(r'^([\w.-]+)==([^\s\\]+)', (WORK / 'cuda_environment.lock.txt').read_text(), re.M))
    assert len(pins) == 224 and pins['torch'] == '2.11.0' and pins['vllm'] == '0.26.0'
    waiting = {state['pid']: identity(state['pid'], str(WORK / 'run_abcd_cuda.py'))}
    installer = {156311: identity(156311, str(WORK / 'bootstrap_a100.py') + ' environment')}
    identity(173820, str(bootstrap.ENV / 'bin/uv') + ' pip install')
    queue.capture_tree(156311, installer)
    if 173820 not in installer:
        raise RuntimeError('uv not a child of the owned installer')
    HISTORY.mkdir(parents=True)
    record('stopping_waiting_queue_and_slow_owned_installer', queue=waiting, installer=installer)
    terminate(waiting)
    terminate(installer)
    for name in ('queue_status.json', 'queue.log', 'environment_status.json', 'environment.log'):
        path = WORK / name
        if path.exists():
            path.rename(HISTORY / name)
    record('installing_identical_locked_dependencies_through_user_proxy', packages=len(pins), index=INDEX,
           lock_sha256=bootstrap.sha(WORK / 'cuda_environment.lock.txt'))
    # No cache deletion, no shared environment edits, no CUDA process launch.
    bootstrap.save('environment_status.json', {'status': 'installing', 'index': INDEX,
                   'proxy': PROXY, 'lock_sha256': bootstrap.sha(WORK / 'cuda_environment.lock.txt')})
    with (WORK / 'environment.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', __file__, '--install'], stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    result = subprocess.check_output([str(bootstrap.ENV / 'bin/python'), str(WORK / 'run_abcd_cuda.py'), '--launch'], text=True)
    record('queue_resumed_installing', installer_pid=child.pid, queue_launch=json.loads(result),
           history=str(HISTORY), index=INDEX, proxy=PROXY, packages=len(pins))


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (WORK / 'environment_acceleration.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            print(json.dumps({'acceleration_pid': child.pid}))
    elif sys.argv[1:] == ['--install']:
        bootstrap.environment(INDEX, locked=True, proxy=PROXY)
    else:
        try:
            main()
        except BaseException as exc:
            record('failed', error=repr(exc))
            raise
