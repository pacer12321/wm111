"""One user-authorized D retry after verified libpulse repair; preserve failed attempt."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import run_d_30674 as d

q = d.q
CONTROL = d.ROOT / 'd_retry_ffprobe_20260914'
HISTORY = q.WORK / 'history/30674_d_ffprobe_failure_20260914'
LIB = d.ROOT / 'python312_runtime/lib/pulseaudio/libpulsecommon-17.0.so'


def validate_repair():
    assert q.sha(LIB) == '49a0412853b48085898f16e24d7a72731f71d55baa9b845bb7f5e1a7120346af'
    manifest = q.read(q.WORK / 'cuda_manifest.json')
    source = Path(manifest['source_path'])
    assert q.sha(source) == manifest['sample']['source_sha256']
    env = d.env_for('D')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames',
        '-of', 'json', str(source)], env=env, capture_output=True, text=True, check=True, timeout=60)
    stream = json.loads(probe.stdout)['streams'][0]
    assert (stream['width'], stream['height'], stream['r_frame_rate'], stream['nb_read_frames']) == (1280,720,'24/1','124')
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-map', '0:v:0',
                              '-f', 'null', '-'], env=env, capture_output=True, text=True, check=True, timeout=60)
    return dict(source_probe=stream, full_source_decode_exit=decoded.returncode,
                library_sha256=q.sha(LIB), probe_stderr=probe.stderr, decode_stderr=decoded.stderr)


def launch():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = q.read(q.WORK / 'queue_status.json')
    assert old['status'] == 'failed' and old['pid'] == 1442555
    assert q.pid_identity(1442555) is None and q.pid_identity(1442788) is None
    result = q.read(q.WORK / 'results/D/smoke_2step/result.json')
    assert result['requested_steps'] == 2 and not result['http_success']
    assert not (q.WORK / 'results/D/formal_50step').exists()
    assert not HISTORY.exists() and not CONTROL.exists()
    d.assert_ready()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', q.PORT))
    evidence = validate_repair()
    # Both exact source and destination lie inside our experiment directory.
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
    (CONTROL / 'video_io_preflight.json').write_text(json.dumps(evidence, indent=2))
    fcntl.flock(lock, fcntl.LOCK_UN)
    with (CONTROL / 'dispatch.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', __file__], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'d_retry_pid': proc.pid, 'failed_attempt_preserved': str(HISTORY)}))


def run():
    lock = (q.WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = d.preflight()
    d.assert_ready()
    q.STATE.update(pid=os.getpid(), case_order=['D'], gpu_uuids=d.UUIDS,
                   physical_gpu_indices=[0,1], formal_completed=[],
                   host=socket.gethostname(), source_B_request_seconds=1613.776763,
                   retry_reason='User authorized retry after matched libpulsecommon supplementation',
                   previous_attempt=str(HISTORY), model_and_settings_unchanged=True,
                   authorized_residual_host_pid=1550469, hardware_changed_from_B=True,
                   timing_caveat='Different host from B; GPU0 may retain850MiB context; not an isolated benchmark.')
    with (q.WORK / 'queue.log').open('w') as log:
        os.dup2(log.fileno(),1)
        os.dup2(log.fileno(),2)
    try:
        q.update('starting_d_after_video_io_preflight', case='D')
        q.run_case('D', manifest)
        q.update('completed_quality_review_required', case='D')
    except BaseException as exc:
        q.update('failed', case='D', error=repr(exc), no_automatic_model_retry=True)
        raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if sys.argv[1:] == ['--launch']:
        launch()
    else:
        assert not sys.argv[1:]
        run()
