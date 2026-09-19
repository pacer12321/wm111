"""Track A: non-file guard; two-step health check then one qualitative 50-step D."""
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import run_d_30674 as d
from d_memory_probe import snapshot
from d_memory_guard_track_a import GIB, MAX_USAGE_BYTES, MemoryGuard
from retry_d_after_ffprobe import validate_repair

q = d.q
CONTROL = d.ROOT / 'd_track_a_nonfile_20260915'
HISTORY = q.WORK / 'history/30674_d_total250_guard_20260915'
q.SERVER_EXTRA_ARGS = ('--disable-multithread-weight-load',)
assert MAX_USAGE_BYTES == 250_000_000_000


class SmokeFinished(Exception):
    pass


def launch():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = q.read(q.WORK / 'queue_status.json')
    assert old['status'] == 'failed' and old['pid'] == 1494846
    assert old['memory_guard_reason'].startswith('proactive memory stop:')
    assert q.pid_identity(1494846) is None
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
    print(json.dumps({'track_a_pid': proc.pid, 'history': str(HISTORY), 'control': str(CONTROL)}))


def run():
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = d.preflight()
    d.assert_ready()
    assert snapshot()['usage_in_bytes'] < 48 * GIB
    q.STATE.update(pid=os.getpid(), case_order=['D'], gpu_uuids=d.UUIDS,
                   physical_gpu_indices=[0,1], formal_completed=[], smoke_only=False, exploratory_only=True, formal_comparison_allowed=False,
                   host=socket.gethostname(), previous_attempt=str(HISTORY),
                   model_and_settings_unchanged=True, authorized_residual_host_pid=1550469,
                   memory_guard=True, threshold_metric='non_file', non_file_threshold_bytes=MAX_USAGE_BYTES, weight_loading='serial',
                   startup_only_change='--disable-multithread-weight-load')
    original_capture, original_cleanup, original_request = q.capture_tree, q.cleanup, q.request
    original_update = q.update
    def exploratory_update(status, **kwargs):
        if status == 'formal_running':
            status = 'exploratory_running'
        return original_update(status, **kwargs)
    q.update = exploratory_update
    guard = MemoryGuard(CONTROL / 'memory_trace.jsonl', lambda: q.STATE.get('status'))
    checking = True
    last_process_sample = 0.0
    process_log = (CONTROL / 'process_memory.jsonl').open('x')

    def capture(server_pid, owned):
        nonlocal last_process_sample
        original_capture(server_pid, owned)
        if checking:
            if time.monotonic() - last_process_sample >= 2:
                rows = []
                for pid, start in owned.items():
                    ident = q.pid_identity(pid)
                    if ident is None or ident[0] != start:
                        continue
                    try:
                        fields = {}
                        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                            key, _, value = line.partition(':')
                            if key in ('VmRSS','VmHWM','RssAnon','RssFile','RssShmem','VmLck','State'):
                                fields[key] = value.strip()
                        rows.append({'pid': pid, 'starttime': start, **fields})
                    except OSError:
                        pass
                process_log.write(json.dumps({'unix_time': time.time(),
                    'stage': q.STATE.get('status'), 'owned_processes': rows}) + '\n')
                process_log.flush()
                last_process_sample = time.monotonic()
            guard.check()

    def cleanup(server, owned):
        nonlocal checking
        checking = False
        guard.close()
        return original_cleanup(server, owned)

    def request(server, owned, case, steps, directory, manifest):
        assert case == 'D' and steps in (2, 50)
        if steps == 50:
            directory = directory.parent / 'exploratory_50step'
        result = original_request(server, owned, case, steps, directory, manifest)
        assert result['http_success'] and result['shape_verified']
        guard.check()
        result.update(scope='Track A exploratory only; not comparable with B/C timing',
                      formal_comparison_allowed=False)
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        q.STATE['smoke_result' if steps == 2 else 'exploratory_result'] = result
        if steps == 2:
            return result
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
        q.update('exploratory_completed_quality_review_required', case='D', formal_started=False)
    except BaseException as exc:
        q.update('failed', case='D', error=repr(exc), memory_guard_reason=guard.reason,
                 no_automatic_model_retry=True)
        raise
    finally:
        guard.close()
        process_log.close()
        q.capture_tree, q.cleanup, q.request = original_capture, original_cleanup, original_request
        (CONTROL / 'final_memory.json').write_text(json.dumps(snapshot(), indent=2))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if sys.argv[1:] == ['--launch']:
        launch()
    else:
        assert not sys.argv[1:]
        run()
