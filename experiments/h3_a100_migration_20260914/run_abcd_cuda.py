"""Single-use GPU2/3 BCD queue; user explicitly skipped A. No model retries."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/cache/zhonghao/h3')
WORK = ROOT / 'a100_v1'
ENV = ROOT / 'env_cuda_v1'
UUIDS = ['GPU-856faa17-eeba-f3e1-36b7-720cf8424a84', 'GPU-5d12ae6b-1304-87d8-6b26-835e6a754c92']
HOLDER = ROOT / 'a100_reservation_20260914_gpu23_v1'
REL = 'vllm_omni/diffusion/models/minimax_h3'
PORT = 19123
SERVER_EXTRA_ARGS = ()  # Explicit per-attempt startup options; default remains unchanged.
CASES = ('B', 'C', 'D')
STATE = {'status': 'starting', 'formal_completed': [], 'gpu_uuids': UUIDS, 'case_order': CASES,
         'definition': {'A': 'original dense Ref2VA', 'B': 'VDN hybrid, global source',
                        'C': 'B plus corresponding-frame target-source',
                        'D': 'C plus original VDN local+linear on source-source'},
         'quality_review_required': True}


def update(status, **kwargs):
    STATE.update(status=status, updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **kwargs)
    tmp = WORK / 'queue_status.json.tmp'
    tmp.write_text(json.dumps(STATE, indent=2))
    tmp.replace(WORK / 'queue_status.json')
    print(json.dumps({'status': status, **kwargs}), flush=True)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_rows():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(', ') for line in text.strip().splitlines()]
    selected = [row for row in rows if row[1] in UUIDS]
    if [(row[0], row[1]) for row in selected] != list(zip(['2', '3'], UUIDS)):
        raise RuntimeError('Physical GPU2/3 UUID binding changed')
    return selected


def assert_idle():
    rows = gpu_rows()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
    if any(int(row[2]) > 20 or int(row[3]) != 0 for row in rows) or any(uuid in processes for uuid in UUIDS):
        raise RuntimeError('GPU2/3 are not idle; refusing to touch unknown processes')


def env_for(case='A'):
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT', 'ASCEND_RT_VISIBLE_DEVICES', 'VLLM_LOGGING_CONFIG_PATH'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(UUIDS), PYTHONNOUSERSITE='1',
               PYTHONPATH=str(WORK / f'candidates/{case}'),
               LD_LIBRARY_PATH=str(ROOT / 'cuda_compat13/usr/local/cuda-13.0/compat') + ':' + str(ENV / 'lib'),
               PATH=str(ENV / 'bin') + ':/usr/local/bin:/usr/bin:/bin',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_PLUGINS='vllm_omni_register_models',
               VLLM_OMNI_VIDEO_SYNC_TIMEOUT='7200', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='8', MAX_JOBS='4', NCCL_DEBUG='WARN',
               XDG_CACHE_HOME=str(WORK / 'cache'), TRITON_CACHE_DIR=str(WORK / 'cache/triton'),
               ZHONGHAO_H3_OPENVDN='0' if case == 'A' else '1',
               VLLM_LOGGING_COLOR='0', PYTHONUNBUFFERED='1')
    if case != 'A':
        env['ZHONGHAO_H3_OPENVDN_CHECKPOINT'] = str(ROOT / 'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000')
    else:
        env.pop('ZHONGHAO_H3_OPENVDN_CHECKPOINT', None)
    env['D_REVIEWED_POLICY_PATH'] = str(WORK / 'cuda_d_policy.json')
    env['D_REVIEWED_POLICY_SHA256'] = sha(WORK / 'cuda_d_policy.json')
    return env


def admission(manifest):
    directory = WORK / 'vlm_evidence'
    row = read(directory / 'vlm_admission.json')
    sample = manifest['sample']
    if (row.get('classification') != 'preserve' or row.get('root_reviewed_real_evidence') is not True
            or row.get('edit_prompt') != sample['edit_prompt']
            or row.get('source', {}).get('sha256') != sample['source_sha256']):
        raise RuntimeError('Original real VLM admission does not match source+edit')
    runs = list(directory.glob('vlm_results/*/runs/*'))
    if len(runs) != 1:
        raise RuntimeError('Expected one original VLM evidence run')
    for filename, key in [('vlm_status.json', 'status_sha256'), ('worker_result.json', 'result_sha256')]:
        if sha(runs[0] / filename) != row[key]:
            raise RuntimeError('Original VLM evidence checksum mismatch')
    return {'reused_real_vlm_decision': True, 'run_id': row['run_id'],
            'source_sha256': sample['source_sha256'], 'classification': 'preserve'}


def prerequisites(phase):
    deadline = time.monotonic() + 20 * 3600
    last = None
    while time.monotonic() < deadline:
        e, w = read(WORK / 'environment_status.json'), read(WORK / 'weights_status.json')
        signature = (e.get('status'), w.get('status'), w.get('completed'))
        if signature != last:
            update('waiting_for_' + phase, environment=e.get('status'), weights=w.get('status'),
                   files_completed=w.get('completed'), files_total=w.get('total'))
            last = signature
        if e.get('status') == 'failed' or w.get('status') == 'failed':
            raise RuntimeError(f'Asset preparation failed: environment={e}, weights={w}')
        if phase == 'environment' and e.get('status') == 'installed_not_gpu_validated':
            return
        if phase == 'weights' and w.get('status') == 'verified':
            return
        if (WORK / 'STOP').exists():
            raise RuntimeError('Queue STOP requested')
        time.sleep(15)
    raise TimeoutError('Asset preparation exceeded20h; no experiment launched')


def pid_identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return int(fields[19]), int(fields[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def capture_tree(root_pid, owned):
    rows = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            identity = pid_identity(int(path.name))
            if identity:
                rows[int(path.name)] = identity
    parents = {root_pid} | {pid for pid, start in owned.items() if rows.get(pid, (None,))[0] == start}
    changed = True
    while changed:
        changed = False
        for pid, (start, parent) in rows.items():
            if parent in parents and pid not in parents:
                parents.add(pid)
                changed = True
    for pid in parents:
        if pid in rows:
            owned[pid] = rows[pid][0]


def cleanup(server, owned):
    capture_tree(server.pid, owned)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, start in list(owned.items()):
            identity = pid_identity(pid)
            if identity and identity[0] == start:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    # Do not kill arbitrary GPU processes, even if selected cards remain occupied.


def request(server, owned, case, steps, directory, manifest):
    source = Path(manifest['source_path'])
    sample = manifest['sample']
    if sha(source) != sample['source_sha256']:
        raise RuntimeError('Frozen source changed')
    directory.mkdir()
    g = sample['requested_generation']
    fields = {key: g[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
    fields.update(prompt=sample['edit_prompt'], num_inference_steps=steps,
                  extra_params=json.dumps({'task': 'ref2va', 'duration': g['duration_seconds'],
                                           'audio_flow_shift': g['audio_flow_shift']}))
    (directory / 'request.json').write_text(json.dumps({'case': case, 'fields': fields,
        'source_sha256': sample['source_sha256'], 'source': str(source)}, indent=2))
    command = ['curl', '--silent', '--show-error', '--noproxy', '*', '--connect-timeout', '10',
               '--max-time', '7200', '--dump-header', str(directory / 'headers.txt'),
               '--output', str(directory / 'output.mp4'), '--write-out', '%{http_code} %{time_total}',
               '--request', 'POST', f'http://127.0.0.1:{PORT}/v1/videos/sync']
    for key, value in fields.items():
        command += ['--form-string', f'{key}={value}']
    command += ['--form', f'input_references=@{source};type=video/mp4']
    update('formal_running' if steps == 50 else 'smoke_running', case=case, requested_steps=steps)
    peak = {uuid: 0 for uuid in UUIDS}
    with (directory / 'curl.stdout').open('w') as out, (directory / 'curl.stderr').open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        try:
            while child.poll() is None:
                capture_tree(server.pid, owned)
                if server.poll() is not None:
                    raise RuntimeError('Server exited during generation')
                for _, uuid, memory, _ in gpu_rows():
                    peak[uuid] = max(peak[uuid], int(memory))
                if (WORK / 'STOP').exists():
                    raise RuntimeError('Queue STOP requested')
                time.sleep(2)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    summary = (directory / 'curl.stdout').read_text().split()
    headers = (directory / 'headers.txt').read_text().lower()
    success = child.returncode == 0 and len(summary) == 2 and summary[0] == '200' and 'content-type: video/mp4' in headers
    result = {'http_success': success, 'requested_steps': steps, 'case': case,
              'request_seconds': float(summary[1]) if len(summary) == 2 else None,
              'gpu_memory_sampled_peak_mib': peak, 'quality_verified': False,
              'scope': 'HTTP request excluding model load; same offload/settings across ABCD'}
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    if not success:
        raise RuntimeError(f'{case} {steps}-step HTTP request failed; no retry')
    import av
    with av.open(str(directory / 'output.mp4')) as video:
        stream = video.streams.video[0]
        width, height, rate = stream.width, stream.height, float(stream.average_rate)
        frames = sum(1 for _ in video.decode(stream))
    if (width, height, frames, rate) != (1344, 768, 124, 24.0):
        raise RuntimeError(f'Output shape mismatch: {width} {height} {frames} {rate}')
    result.update(decoded_frames=frames, shape_verified=True)
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def run_case(case, manifest):
    assert_idle()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    directory = WORK / 'results' / case
    directory.mkdir(parents=True, exist_ok=False)
    candidate = WORK / f'candidates/{case}'
    for filename, expected in manifest['cases'][case]['model_files'].items():
        if sha(candidate / REL / filename) != expected:
            raise RuntimeError(f'Candidate changed: {case}/{filename}')
    environment = env_for(case)
    tmp = ROOT / 'tmp' / ('a100_' + case)
    tmp.mkdir(parents=True, exist_ok=True)
    environment['TMPDIR'] = str(tmp)
    command = [str(ENV / 'bin/python'), '-m', 'vllm_omni.entrypoints.cli.main', 'serve',
        str(ROOT / 'models/MiniMax-H3/Ref2VA'), '--omni', '--host', '127.0.0.1', '--port', str(PORT),
        '--trust-remote-code', '--init-timeout', '3600', '--stage-init-timeout', '3600',
        '--num-gpus', '2', '--usp', '2', '--ring', '1', '--text-encoder-tp-size', '2',
        '--enable-layerwise-offload', '--vae-parallel-mode', 'tile', '--vae-use-tiling',
        '--vae-patch-parallel-size', '2', '--diffusion-attention-backend', 'FLASH_ATTN']
    command.extend(SERVER_EXTRA_ARGS)
    (directory / 'launch.json').write_text(json.dumps({'command': command, 'gpu_uuids': UUIDS,
        'sample': manifest['sample'], 'candidate': manifest['cases'][case]}, indent=2))
    owned = {}
    update('loading_model', case=case)
    with (directory / 'server.log').open('w') as log:
        server = subprocess.Popen(command, cwd=candidate, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = pid_identity(server.pid)
        if identity:
            owned[server.pid] = identity[0]
        try:
            deadline = time.monotonic() + 4000
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                capture_tree(server.pid, owned)
                if server.poll() is not None:
                    raise RuntimeError(f'{case} model load failed, see server.log')
                try:
                    with opener.open(f'http://127.0.0.1:{PORT}/health', timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if (WORK / 'STOP').exists():
                    raise RuntimeError('Queue STOP requested')
                time.sleep(2)
            else:
                raise TimeoutError('Model readiness timeout')
            request(server, owned, case, 2, directory / 'smoke_2step', manifest)
            result = request(server, owned, case, 50, directory / 'formal_50step', manifest)
            STATE['formal_completed'].append({'case': case, **result})
            update('case_completed', case=case)
        finally:
            cleanup(server, owned)


def main():
    assert socket.gethostname() == 'os-node-created-dvv7x'
    lock = (WORK / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (WORK / 'results').exists():
        raise RuntimeError('Single-use queue: results already exist')
    STATE['pid'] = os.getpid()
    try:
        manifest = read(WORK / 'cuda_manifest.json')
        STATE['vlm'] = admission(manifest)
        gpu_rows()
        prerequisites('environment')
        update('cuda_validation')
        subprocess.run([str(ENV / 'bin/python'), str(WORK / 'finalize_cuda.py')], check=True, env=env_for())
        subprocess.run([str(ENV / 'bin/python'), str(WORK / 'test_cuda_kernel.py')], check=True, env=env_for())
        prerequisites('weights')
        # Small validation uses spare memory while holder stays. Full model starts
        # only after orderly release and a fresh idle check on preciselyGPU2/3.
        held = read(HOLDER / 'reservation.json')
        if held.get('status') not in ('holding', 'released') or list(held.get('targets', {}).values()) != UUIDS:
            raise RuntimeError('Expected owned reservation missing before handoff')
        if held['status'] == 'holding':
            (HOLDER / 'STOP').touch(exist_ok=False)
        elif held.get('reason') != 'stop_file' or not (HOLDER / 'STOP').exists():
            raise RuntimeError('Released reservation lacks expected owned handoff evidence')
        for _ in range(12):
            if read(HOLDER / 'reservation.json').get('status') == 'released':
                break
            time.sleep(1)
        # The holder writes released before CUDA context teardown is reflected
        # in nvidia-smi. Poll read-only; never terminate an occupying process.
        idle_deadline = time.monotonic() + 60
        while True:
            try:
                assert_idle()
                break
            except RuntimeError:
                if time.monotonic() >= idle_deadline:
                    raise
                time.sleep(2)
        for case in CASES:
            run_case(case, manifest)
        update('completed_quality_review_required')
    except BaseException as exc:
        update('failed', error=repr(exc), no_automatic_retry=True)
        raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        if (WORK / 'queue_status.json').exists():
            raise SystemExit('Queue already dispatched')
        with (WORK / 'queue.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', __file__], stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            print(json.dumps({'queue_pid': process.pid}))
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        main()
