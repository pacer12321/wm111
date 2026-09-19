"""Authorized one-shot D retry with cgroup evidence and proactive memory guard."""
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys

import run_d_30674 as d
from d_memory_probe import snapshot
from d_memory_guard import GIB, MemoryGuard
from retry_d_after_ffprobe import validate_repair

q = d.q
CONTROL = d.ROOT / 'd_retry_memory_guard_20260914'
HISTORY = q.WORK / 'history/30674_d_worker_failure_20260914'


def launch():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = q.read(q.WORK / 'queue_status.json')
    assert old['status'] == 'failed' and old['pid'] == 1453031
    assert q.pid_identity(1453031) is None and q.pid_identity(1453050) is None
    result = q.read(q.WORK / 'results/D/smoke_2step/result.json')
    assert result['requested_steps'] == 2 and not result['http_success']
    assert not (q.WORK / 'results/D/formal_50step').exists()
    assert not HISTORY.exists() and not CONTROL.exists()
    d.assert_ready()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', q.PORT))
    baseline = snapshot()
    assert baseline['limit_in_bytes'] == 250999996416
    assert baseline['usage_in_bytes'] < 48 * GIB, 'CPU memory baseline too high; no launch'
    evidence = validate_repair()
    old_results = q.WORK / 'results/D'
    assert old_results.resolve().is_relative_to(q.WORK.resolve())
    assert HISTORY.resolve().is_relative_to(q.WORK.resolve())
    HISTORY.mkdir()
    for name in ('queue_status.json', 'queue.log'):
        shutil.copy2(q.WORK / name, HISTORY / name)
    old_results.rename(HISTORY / 'results_D')
    d.preflight()
    assert q.read(q.WORK / 'cuda_kernel_test.json')['passed']
    CONTROL.mkdir()
    (CONTROL / 'preflight.json').write_text(json.dumps({'memory': baseline, 'video': evidence}, indent=2))
    fcntl.flock(lock, fcntl.LOCK_UN)
    with (CONTROL / 'dispatch.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', __file__], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'d_retry_pid': proc.pid, 'history': str(HISTORY), 'control': str(CONTROL)}))


def run():
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = d.preflight()
    d.assert_ready()
    assert snapshot()['usage_in_bytes'] < 48 * GIB
    q.STATE.update(pid=os.getpid(), case_order=['D'], gpu_uuids=d.UUIDS,
                   physical_gpu_indices=[0,1], formal_completed=[], host=socket.gethostname(),
                   source_B_request_seconds=1613.776763, previous_attempt=str(HISTORY),
                   model_and_settings_unchanged=True, authorized_residual_host_pid=1550469,
                   hardware_changed_from_B=True, memory_guard=True,
                   timing_caveat='Different host and CPU quota from B; memory monitor overhead; GPU0 residual850MiB.')
    original_capture, original_cleanup = q.capture_tree, q.cleanup
    guard = MemoryGuard(CONTROL / 'memory_trace.jsonl', lambda: q.STATE.get('status'))
    checking = True

    def capture(server_pid, owned):
        original_capture(server_pid, owned)
        if checking:
            guard.check()

    def cleanup(server, owned):
        nonlocal checking
        checking = False  # cleanup must never be interrupted by the memory guard
        guard.close()
        return original_cleanup(server, owned)

    q.capture_tree, q.cleanup = capture, cleanup
    with (q.WORK / 'queue.log').open('w') as log:
        os.dup2(log.fileno(),1)
        os.dup2(log.fileno(),2)
    try:
        guard.start()
        q.update('starting_memory_guarded_d', case='D')
        q.run_case('D', manifest)  # unchanged: smoke2 -> formal50, only once
        q.update('completed_quality_review_required', case='D')
    except BaseException as exc:
        q.update('failed', case='D', error=repr(exc), memory_guard_reason=guard.reason,
                 no_automatic_model_retry=True)
        raise
    finally:
        guard.close()
        q.capture_tree, q.cleanup = original_capture, original_cleanup
        (CONTROL / 'final_memory.json').write_text(json.dumps(snapshot(), indent=2))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if sys.argv[1:] == ['--launch']:
        launch()
    else:
        assert not sys.argv[1:]
        run()
