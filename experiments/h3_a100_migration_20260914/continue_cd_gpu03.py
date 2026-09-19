"""User-authorized GPU0/3 retarget; preserve B and untouched C/D computation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import continue_cd as c
import run_abcd_cuda as q

OLD_PID = 385486
TARGETS = ['GPU-7d2dd94f-1ec4-aa0b-b31e-b7b4a8c9c890',
           'GPU-5d12ae6b-1304-87d8-6b26-835e6a754c92']
q.UUIDS = TARGETS
q.STATE.update(gpu_uuids=TARGETS, physical_gpu_indices=[0, 3],
               hardware_changed_from_B=True,
               timing_caveat='B used GPU2/3; C/D use GPU0/3. Check topology before interpreting timing.')
c.HISTORY = q.WORK / 'history/retarget_cd_gpu03_20260914'


def gpu_rows():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(', ') for line in output.strip().splitlines()]
    selected = [row for row in rows if row[1] in TARGETS]
    if [(row[0], row[1]) for row in selected] != list(zip(['0', '3'], TARGETS)):
        raise RuntimeError('Physical GPU0/3 UUID binding changed')
    return selected


def assert_idle():
    rows = gpu_rows()
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(int(r[2]) > 20 or int(r[3]) != 0 for r in rows) or any(u in processes for u in TARGETS):
        raise RuntimeError('GPU0/3 busy; no other process will be touched')


q.gpu_rows = gpu_rows
q.assert_idle = assert_idle


def stop_old_waiter():
    old = q.read(q.WORK / 'queue_status.json')
    assert old['pid'] == OLD_PID and old['status'] == 'waiting_for_idle'
    assert all(not (q.WORK / 'results' / case).exists() for case in ('C', 'D'))
    assert not c.HISTORY.exists()
    cmd = Path(f'/proc/{OLD_PID}/cmdline').read_bytes().split(b'\0')
    assert cmd[:3] == [str(q.ENV / 'bin/python').encode(), b'-u',
                      str(q.WORK / 'continue_cd.py').encode()]
    identity = q.pid_identity(OLD_PID)
    assert identity is not None
    # This process is our CPU-only idle waiter, not a model server or anyone else's job.
    children = subprocess.run(['pgrep', '-P', str(OLD_PID)], capture_output=True, text=True)
    assert children.returncode == 1, 'Waiter has a child; refuse to interrupt until rechecked'
    assert q.pid_identity(OLD_PID) == identity
    os.kill(OLD_PID, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while q.pid_identity(OLD_PID) == identity and time.monotonic() < deadline:
        time.sleep(0.2)
    assert q.pid_identity(OLD_PID) != identity, 'Old waiter did not exit; do not duplicate'
    assert q.read(q.WORK / 'queue_status.json')['status'] == 'failed'
    print('Old CPU-only GPU2/3 waiter exited; no GPU process stopped', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--test']:
        from unittest.mock import patch
        sample = '\n'.join(f'{i}, {u}, 0, 0' for i, u in zip([0, 3], TARGETS))
        with patch.object(subprocess, 'check_output', return_value=sample):
            assert [r[0] for r in gpu_rows()] == ['0', '3']
        with patch.object(subprocess, 'check_output', side_effect=[sample, 'GPU-other, 123']):
            assert_idle()
        with patch.object(subprocess, 'check_output', side_effect=[sample, TARGETS[0] + ', 123']):
            try:
                assert_idle()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Busy target accepted')
        with patch.object(subprocess, 'check_output', return_value=sample.replace('0, GPU-', '2, GPU-', 1)):
            try:
                gpu_rows()
            except RuntimeError:
                pass
            else:
                raise AssertionError('Wrong physical mapping accepted')
        assert q.env_for('C')['CUDA_VISIBLE_DEVICES'] == ','.join(TARGETS)
        print('Five GPU0/3 binding, idle and environment checks passed; no GPU allocation')
    elif sys.argv[1:] == ['--retarget']:
        gpu_rows()
        stop_old_waiter()
        c.preflight(require_idle=False)
        with (q.WORK / 'cd_gpu03_dispatch.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'new_queue_pid': proc.pid, 'physical_gpu_indices': [0, 3]}))
    elif sys.argv[1:] == ['--check']:
        c.preflight(require_idle=False)
        print('GPU0/3 preflight passed; no GPU allocation')
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        c.main()
