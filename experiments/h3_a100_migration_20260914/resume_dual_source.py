"""One-shot authorized mirror-first install with hash-pinned official fallback."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import bootstrap_a100 as bootstrap
import run_abcd_cuda as queue

WORK = bootstrap.WORK
INDEX = 'https://mirrors.huaweicloud.com/repository/pypi/simple/'
FALLBACK = 'https://pypi.org/simple'
LOCK_SHA = 'af81f81237ffcd7cd8b309866bf43a58c94db3f2838a8e14077dd75d183b88e4'
HISTORY = WORK / 'history/environment_dual_source_20260914'
PREFLIGHT_LOG = WORK / 'dual_source_preflight.log'


def record(status, **extra):
    row = dict(status=status, updated_at=time.time(), **extra)
    bootstrap.save('dual_source_status.json', row)
    print(json.dumps(row), flush=True)


def main():
    lock = (WORK / 'environment_acceleration.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not HISTORY.exists() and not (WORK / 'results').exists()
    assert bootstrap.sha(WORK / 'cuda_environment.lock.txt') == LOCK_SHA
    state = queue.read(WORK / 'queue_status.json')
    assert state['status'] == 'failed' and not state.get('formal_completed')
    assert queue.read(WORK / 'environment_status.json')['status'] == 'failed'
    for pid in (state['pid'], 296429):
        if queue.pid_identity(pid):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            assert fields[0] == 'Z', f'Previous process {pid} still alive; refusing duplicate'
    record('verifying_full_locked_dependency_plan')
    with PREFLIGHT_LOG.open('x') as log:
        subprocess.run([str(bootstrap.ENV / 'bin/uv'), 'pip', 'install', '--dry-run',
                        '--python', str(bootstrap.ENV / 'bin/python'), '--cache-dir', str(bootstrap.ROOT / 'uv_cache'),
                        '--index', INDEX, '--default-index', FALLBACK, '--index-strategy', 'unsafe-first-match',
                        '--require-hashes', '-r', str(WORK / 'cuda_environment.lock.txt')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    HISTORY.mkdir(parents=True)
    for name in ('queue_status.json', 'queue.log', 'environment_status.json', 'environment.log', 'dual_source_status.json'):
        path = WORK / name
        if path.exists():
            path.rename(HISTORY / name)
    bootstrap.save('environment_status.json', dict(status='installing', index=INDEX,
                   fallback_index=FALLBACK, lock_sha256=LOCK_SHA))
    with (WORK / 'environment.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', __file__, '--install'], stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    result = subprocess.check_output([str(bootstrap.ENV / 'bin/python'), str(WORK / 'run_abcd_cuda.py'), '--launch'], text=True)
    record('queue_resumed_installing', installer_pid=child.pid, queue_launch=json.loads(result),
           index=INDEX, fallback_index=FALLBACK, lock_sha256=LOCK_SHA, preflight_passed=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (WORK / 'dual_source.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            print(json.dumps({'resume_pid': child.pid}))
    elif sys.argv[1:] == ['--install']:
        bootstrap.environment(INDEX, locked=True, fallback_index=FALLBACK)
    else:
        try:
            main()
        except BaseException as exc:
            record('failed', error=repr(exc))
            raise
