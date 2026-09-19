"""D-only destination queue. Original model settings and q.run_case unchanged."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/cache/zhonghao/h3')
sys.path.insert(0, str(ROOT / 'a100_v1'))
import run_abcd_cuda as q

UUIDS = ['GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670',
         'GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3']
HOLDERS = [(0, 1391369), (1, 1385771)]
CONTROL = ROOT / 'd_only_30674_20260914'
ORIGINAL_ENV = q.env_for
q.UUIDS = UUIDS


def env_for(case='D'):
    env = ORIGINAL_ENV(case)
    env['LD_LIBRARY_PATH'] += ':' + str(ROOT / 'python312_runtime/lib')
    return env


def gpu_rows():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(', ') for line in output.strip().splitlines()]
    assert [(r[0], r[1]) for r in rows] == list(zip(['0', '1'], UUIDS)), 'GPU mapping changed'
    return rows


def assert_ready():
    rows = gpu_rows()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
    for line in processes.strip().splitlines():
        if line.strip() and line.split(', ') != [UUIDS[0], '1550469']:
            raise RuntimeError('Unexpected GPU task; no process will be stopped')
    if any(int(r[2]) > (1024 if r[0] == '0' else 20) or int(r[3]) != 0 for r in rows):
        raise RuntimeError('Selected GPUs busy beyond the authorized850MiB residual')


q.env_for = env_for
q.gpu_rows = gpu_rows
q.assert_idle = assert_ready


def preflight():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    assert q.read(ROOT / 'runtime_setup_30674_20260914/status.json')['phase'] == 'runtime_verified'
    assert q.read(ROOT / 'migration_30674_tar_20260914/status.json')['phase'] == 'assets_imported_runtime_setup_required'
    assert not (q.WORK / 'results/D').exists()
    assert not (q.WORK / 'STOP').exists()
    manifest = q.read(q.WORK / 'cuda_manifest.json')
    q.admission(manifest)
    assert q.sha(Path(manifest['source_path'])) == manifest['sample']['source_sha256']
    for name, expected in manifest['cases']['D']['model_files'].items():
        assert q.sha(q.WORK / 'candidates/D' / q.REL / name) == expected, name
    assert q.sha(q.WORK / 'results/B/formal_50step/output.mp4') == 'b80fe4dedf5a10b4edb75ba4d21bc80d19e2718af7693570cf74c2be1170d808'
    gpu_rows()
    return manifest


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ['--launch']:
        preflight()
        with (CONTROL / 'dispatch.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'d_queue_pid': proc.pid}))
        return
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = preflight()
    history = q.WORK / 'history/30674_before_d_only_20260914'
    history.mkdir()
    for name in ('queue_status.json', 'queue.log', 'cuda_kernel_test.json'):
        if (q.WORK / name).exists():
            shutil.copy2(q.WORK / name, history / name)
    q.STATE.update(pid=os.getpid(), case_order=['D'], gpu_uuids=UUIDS,
                   physical_gpu_indices=[0, 1], formal_completed=[],
                   source_B_request_seconds=1613.776763, host=socket.gethostname(),
                   hardware_changed_from_B=True, authorized_residual_host_pid=1550469,
                   timing_caveat='Different host from B; GPU0 may retain another850MiB context; not an isolated benchmark.')
    with (q.WORK / 'queue.log').open('w') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    try:
        q.update('cuda_validation_with_own_holders', case='D')
        subprocess.run([str(q.ENV / 'bin/python'), str(q.WORK / 'test_cuda_kernel.py')],
                       env=env_for('D'), cwd=q.WORK, check=True, timeout=180)
        assert q.read(q.WORK / 'cuda_kernel_test.json')['passed']
        # Verify identity of ONLY our two reservation processes, then release cooperatively.
        for index, pid in HOLDERS:
            directory = ROOT / f'reservation_30674_gpu{index}_20260914'
            held = q.read(directory / 'reservation.json')
            assert held['pid'] == pid and held['owner'] == 'zhonghao' and held['status'] == 'holding'
            assert held['targets'] == {str(index): UUIDS[index]}
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            assert str(directory / f'reserve_30674_gpu{index}.py').encode() in cmd
            assert q.pid_identity(pid) is not None
        q.update('releasing_own_holders', case='D')
        for index, _ in HOLDERS:
            (ROOT / f'reservation_30674_gpu{index}_20260914/STOP').touch(exist_ok=False)
        deadline = time.monotonic() + 90
        while True:
            try:
                assert_ready()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(3)
        (CONTROL / 'topology.txt').write_text(subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))
        q.run_case('D', manifest)
        q.update('completed_quality_review_required', case='D')
    except BaseException as exc:
        q.update('failed', case='D', error=repr(exc), no_automatic_model_retry=True)
        raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    main()
