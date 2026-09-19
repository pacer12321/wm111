"""Explicit user-authorized continuation: preserve completed B, run only C then D."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import run_abcd_cuda as q

HISTORY = q.WORK / 'history/b_complete_continue_cd_20260914'
B_SHA = 'b80fe4dedf5a10b4edb75ba4d21bc80d19e2718af7693570cf74c2be1170d808'


def preflight(require_idle=True):
    assert socket.gethostname() == 'os-node-created-dvv7x'
    assert not HISTORY.exists()
    assert not (q.WORK / 'STOP').exists()
    assert all(not (q.WORK / 'results' / case).exists() for case in ('C', 'D'))
    old = q.read(q.WORK / 'queue_status.json')
    assert old['status'] == 'failed' and q.pid_identity(old['pid']) is None
    assert [row['case'] for row in old['formal_completed']] == ['B']
    result = q.read(q.WORK / 'results/B/formal_50step/result.json')
    assert result['http_success'] and result['shape_verified'] and result['requested_steps'] == 50
    assert result == old['formal_completed'][0]
    assert q.sha(q.WORK / 'results/B/formal_50step/output.mp4') == B_SHA
    assert q.read(q.WORK / 'cuda_kernel_test.json')['passed']
    assert q.read(q.WORK / 'finalize_status.json')['installed_omni']
    manifest = q.read(q.WORK / 'cuda_manifest.json')
    q.admission(manifest)
    for case in ('C', 'D'):
        for name, expected in manifest['cases'][case]['model_files'].items():
            assert q.sha(q.WORK / f'candidates/{case}' / q.REL / name) == expected
    q.gpu_rows()  # Always validate exact physical GPU2/3 binding.
    if require_idle:
        q.assert_idle()
    return old, manifest


def wait_idle(timeout_seconds=24 * 3600):
    deadline = time.monotonic() + timeout_seconds
    consecutive = 0
    while time.monotonic() < deadline:
        if (q.WORK / 'STOP').exists():
            raise RuntimeError('Queue STOP requested')
        q.gpu_rows()  # A changed UUID binding is a hard failure, not a busy wait.
        try:
            q.assert_idle()
            consecutive += 1
            if consecutive == 2:
                return
        except RuntimeError:
            consecutive = 0
        time.sleep(15)
    raise RuntimeError('GPU2/3 remained busy until resource-wait deadline; no unknown process touched')


def main():
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old, manifest = preflight(require_idle=False)
    HISTORY.mkdir(parents=True)
    for name in ('queue_status.json', 'queue.log'):
        (q.WORK / name).rename(HISTORY / name)
    with (q.WORK / 'queue.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    q.STATE.update(pid=os.getpid(), formal_completed=old['formal_completed'],
                   case_order=['C', 'D'], previous_completed=['B'],
                   vlm=q.admission(manifest), continued_from=str(HISTORY))
    try:
        q.update('continuing_cd', model_and_settings_unchanged=True)
        for case in ('C', 'D'):
            q.update('waiting_for_idle', next_case=case, resource_wait_timeout_hours=24)
            wait_idle()
            q.run_case(case, manifest)
        wait_idle(timeout_seconds=90)
        assert q.sha(q.WORK / 'results/B/formal_50step/output.mp4') == B_SHA
        q.update('completed_quality_review_required')
    except BaseException as exc:
        q.update('failed', error=repr(exc), no_automatic_retry=True)
        raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (q.WORK / 'cd_dispatch.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            print(json.dumps({'cd_queue_pid': child.pid}))
    elif sys.argv[1:] == ['--check']:
        preflight()
        print('CD-only preflight passed; B preserved; no model launched')
    elif sys.argv[1:] == ['--check-wait']:
        preflight(require_idle=False)
        print('CD resource-wait preflight passed; no GPU allocated and no model launched')
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        main()
