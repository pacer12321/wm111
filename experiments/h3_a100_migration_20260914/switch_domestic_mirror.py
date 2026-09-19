"""Explicitly authorized switch from user proxy to domestic package mirror."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import bootstrap_a100 as bootstrap
import run_abcd_cuda as queue
from accelerate_environment import identity, terminate

WORK = bootstrap.WORK
INDEX = 'https://mirrors.huaweicloud.com/repository/pypi/simple/'
HISTORY = WORK / 'history/environment_domestic_20260914'
LOCK_SHA = 'af81f81237ffcd7cd8b309866bf43a58c94db3f2838a8e14077dd75d183b88e4'


def record(status, **extra):
    row = dict(status=status, updated_at=time.time(), **extra)
    bootstrap.save('mirror_switch_status.json', row)
    print(json.dumps(row), flush=True)


def main():
    lock = (WORK / 'environment_acceleration.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if HISTORY.exists() or (WORK / 'results').exists():
        raise RuntimeError('Duplicate switch or inference already started; no mutations')
    assert bootstrap.sha(WORK / 'cuda_environment.lock.txt') == LOCK_SHA
    state = queue.read(WORK / 'queue_status.json')
    assert state['status'] == 'waiting_for_environment' and not state.get('formal_completed')
    assert queue.read(WORK / 'environment_status.json')['status'] == 'installing'
    installer_pid = queue.read(WORK / 'environment_acceleration_status.json')['installer_pid']
    waiting = {state['pid']: identity(state['pid'], str(WORK / 'run_abcd_cuda.py'))}
    installer = {installer_pid: identity(installer_pid, str(WORK / 'accelerate_environment.py') + ' --install')}
    queue.capture_tree(installer_pid, installer)
    identity(282758, str(bootstrap.ENV / 'bin/uv') + ' pip install')
    assert 282758 in installer
    HISTORY.mkdir(parents=True)
    record('stopping_only_owned_waiting_queue_and_installer', queue=waiting, installer=installer)
    terminate(waiting)
    terminate(installer)
    for name in ('queue_status.json', 'queue.log', 'environment_status.json', 'environment.log',
                 'environment_acceleration_status.json'):
        path = WORK / name
        if path.exists():
            path.rename(HISTORY / name)
    bootstrap.save('environment_status.json', {'status': 'installing', 'index': INDEX,
                   'proxy': None, 'lock_sha256': LOCK_SHA})
    with (WORK / 'environment.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', __file__, '--install'], stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    result = subprocess.check_output([str(bootstrap.ENV / 'bin/python'), str(WORK / 'run_abcd_cuda.py'), '--launch'], text=True)
    record('queue_resumed_installing', installer_pid=child.pid, queue_launch=json.loads(result),
           index=INDEX, proxy=None, lock_sha256=LOCK_SHA, cache_preserved=True, history=str(HISTORY))


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (WORK / 'mirror_switch.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            print(json.dumps({'switch_pid': child.pid}))
    elif sys.argv[1:] == ['--install']:
        bootstrap.environment(INDEX, locked=True, proxy=None)
    else:
        try:
            main()
        except BaseException as exc:
            record('failed', error=repr(exc))
            raise
