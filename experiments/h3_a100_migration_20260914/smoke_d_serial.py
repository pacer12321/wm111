"""One D smoke with native serial checkpoint loading; model settings unchanged."""
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
CONTROL = d.ROOT / 'd_serial_smoke_20260915'
HISTORY = q.WORK / 'history/30674_d_parallel_load_guard_20260915'
q.SERVER_EXTRA_ARGS = ('--disable-multithread-weight-load',)


class SmokeFinished(Exception):
    pass


def launch():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = q.read(q.WORK / 'queue_status.json')
    assert old['status'] == 'failed' and old['pid'] == 1475082
    assert old['memory_guard_reason'].startswith('proactive memory stop:')
    assert q.pid_identity(1475082) is None
    assert not (q.WORK / 'results/D/smoke_2step').exists()
    assert not (q.WORK / 'results/D/formal_50step').exists()
    assert not HISTORY.exists() and not CONTROL.exists()
    d.assert_ready()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', q.PORT))
    baseline = snapshot()
    assert baseline['limit_in_bytes'] == 250999996416
    assert baseline['usage_in_bytes'] < 48 * GIB
    evidence = validate_repair()
    old_results = q.WORK / 'results/D'
    assert old_results.resolve().is_relative_to(q.WORK.resolve())
    assert HISTORY.resolve().is_relative_to(q.WORK.resolve())
    HISTORY.mkdir()
    for name in ('queue_status.json', 'queue.log'):
        shutil.copy2(q.WORK / name, HISTORY / name)
    old_results.rename(HISTORY / 'results_D')
    d.preflight()
    CONTROL.mkdir()
    (CONTROL / 'preflight.json').write_text(json.dumps({'memory': baseline, 'video': evidence}, indent=2))
    fcntl.flock(lock, fcntl.LOCK_UN)
    with (CONTROL / 'dispatch.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', __file__], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'d_smoke_pid': proc.pid, 'history': str(HISTORY), 'control': str(CONTROL)}))


def run():
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = d.preflight()
    d.assert_ready()
    assert snapshot()['usage_in_bytes'] < 48 * GIB
    q.STATE.update(pid=os.getpid(), case_order=['D'], gpu_uuids=d.UUIDS,
                   physical_gpu_indices=[0,1], formal_completed=[], smoke_only=True,
                   host=socket.gethostname(), previous_attempt=str(HISTORY),
                   model_and_settings_unchanged=True, authorized_residual_host_pid=1550469,
                   memory_guard=True, weight_loading='serial',
                   startup_only_change='--disable-multithread-weight-load')
    original_capture, original_cleanup, original_request = q.capture_tree, q.cleanup, q.request
    guard = MemoryGuard(CONTROL / 'memory_trace.jsonl', lambda: q.STATE.get('status'))
    checking = True

    def capture(server_pid, owned):
        original_capture(server_pid, owned)
        if checking:
            guard.check()

    def cleanup(server, owned):
        nonlocal checking
        checking = False
        guard.close()
        return original_cleanup(server, owned)

    def request(server, owned, case, steps, directory, manifest):
        assert case == 'D' and steps == 2, 'Only the smoke request is authorized here'
        result = original_request(server, owned, case, steps, directory, manifest)
        assert result['http_success'] and result['shape_verified']
        q.STATE['smoke_result'] = result
        # q.run_case finally still cleans up its own server, but cannot submit50steps.
        raise SmokeFinished()

    q.capture_tree, q.cleanup, q.request = capture, cleanup, request
    with (q.WORK / 'queue.log').open('w') as log:
        os.dup2(log.fileno(),1)
        os.dup2(log.fileno(),2)
    try:
        guard.start()
        q.update('starting_guarded_smoke_only', case='D')
        q.run_case('D', manifest)
        raise RuntimeError('Smoke unexpectedly returned without completion signal')
    except SmokeFinished:
        q.update('smoke_completed_quality_review_required', case='D', formal_started=False)
    except BaseException as exc:
        q.update('failed', case='D', error=repr(exc), memory_guard_reason=guard.reason,
                 no_automatic_model_retry=True)
        raise
    finally:
        guard.close()
        q.capture_tree, q.cleanup, q.request = original_capture, original_cleanup, original_request
        (CONTROL / 'final_memory.json').write_text(json.dumps(snapshot(), indent=2))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if sys.argv[1:] == ['--launch']:
        launch()
    else:
        assert not sys.argv[1:]
        run()
