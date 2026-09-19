"""Isolated B/D same-storage smoke and timing queue; no foreign auto-kills."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from gpu_client_admission import scan_clients, check_clients

ROOT = Path('/cache/zhonghao/h3')
CUDA_COMPAT = ROOT / 'cuda_compat13/usr/local/cuda-13.0/compat'
ORIGINAL = ROOT / 'a100_v1'
KIT = ROOT / 'track_b_validation_20260915'
RUN = ROOT / 'bd_prepared_20260915_v2'
UUIDS = ['GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670', 'GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3']
RUNTIME = ['h3_prepared_integration.py', 'prepared_shard_hook.py', 'torch_streaming_shards.py',
           'streaming_shards.py', 'manifest_builder.py', 'meta_model_metadata.py']
sys.path.insert(0, str(ORIGINAL))
import run_abcd_cuda as q
sys.path.insert(0, str(ROOT))
from d_memory_guard_track_a import MemoryGuard
from d_memory_probe import snapshot


def hashes():
    return {name: q.sha(KIT / name) for name in RUNTIME}


def gpu_rows():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(', ') for line in raw.strip().splitlines()]
    assert [(r[0], r[1]) for r in rows] == list(zip(['0', '1'], UUIDS)), 'GPU UUID binding changed'
    return rows


def gpu_pids():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in raw.splitlines() if line.strip()}


def idle():
    rows = gpu_rows()
    if gpu_pids() - {1550469}:
        raise RuntimeError('Foreign GPU jobs present; stop before model launch, no automatic cleanup')
    if any(int(r[2]) > (1024 if r[0] == '0' else 20) or int(r[3]) != 0 for r in rows):
        raise RuntimeError('GPUs not idle beyond known 850MiB residual')


def prepare():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    assert not RUN.exists(), 'Single-use isolated preparation directory already exists'
    manifest = q.read(ORIGINAL / 'cuda_manifest.json')
    for case in ('B', 'D'):
        for name, expected in manifest['cases'][case]['model_files'].items():
            assert q.sha(ORIGINAL / 'candidates' / case / q.REL / name) == expected, (case, name)
    RUN.mkdir()
    for name in ('cuda_d_policy.json',):
        shutil.copy2(ORIGINAL / name, RUN / name)
    shutil.copytree(ORIGINAL / 'vlm_evidence', RUN / 'vlm_evidence')
    from generate_pipeline_patch import BOOTSTRAP
    for case in ('B', 'D'):
        target = RUN / 'candidates' / case
        shutil.copytree(ORIGINAL / 'candidates' / case, target, ignore=shutil.ignore_patterns('__pycache__', '.git'))
        pipeline = target / q.REL / 'pipeline_minimax_h3.py'
        source = pipeline.read_text()
        assert '_h3_prepared_install' not in source
        # Mechanical opt-in bootstrap in a newly created isolated copy only.
        pipeline.write_text(source.rstrip() + '\n' + BOOTSTRAP)
        manifest['cases'][case]['model_files']['pipeline_minimax_h3.py'] = q.sha(pipeline)
    (RUN / 'cuda_manifest.json').write_text(json.dumps(manifest, indent=2))
    (RUN / 'storage_hashes.json').write_text(json.dumps(hashes(), indent=2))
    print(json.dumps({'status': 'prepared_not_launched', 'run': str(RUN)}), flush=True)


def configure(phase):
    original_env = q.env_for
    q.WORK = RUN
    q.PORT = 19125
    q.UUIDS = UUIDS
    q.gpu_rows = gpu_rows
    q.assert_idle = idle
    q.SERVER_EXTRA_ARGS = ('--disable-multithread-weight-load',)
    def env(case):
        value = original_env(case)
        value['PYTHONPATH'] = str(KIT) + ':' + str(RUN / 'candidates' / case)
        # The host has an R535 driver, while this isolated environment uses
        # torch 2.11 + CUDA 13.  Resolve libcuda from NVIDIA's validated
        # forward-compatibility package before the host driver library.
        compat_driver = CUDA_COMPAT / 'libcuda.so.1'
        if not compat_driver.is_file():
            raise RuntimeError(f'Missing CUDA 13 compatibility driver: {compat_driver}')
        library_path = value.get('LD_LIBRARY_PATH', '')
        value['LD_LIBRARY_PATH'] = ':'.join(filter(None, (
            str(CUDA_COMPAT),
            library_path,
            str(ROOT / 'python312_runtime/lib'),
        )))
        value['ZHONGHAO_H3_PREPARED_OFFLOAD'] = '1'
        value['ZHONGHAO_H3_PREPARED_MANIFEST'] = str(KIT / 'validated_manifest_v1.json')
        return value
    q.env_for = env
    q.STATE.clear()
    q.STATE.update(pid=os.getpid(), phase=phase, formal_completed=[], storage_hashes=hashes(),
        case_order=['D'] if phase == 'smoke' else ['B', 'D'], host=socket.gethostname(),
        old_B_time_not_used=True, quality_review_after_timing=True)


class SmokeComplete(Exception):
    pass


def run(phase):
    assert socket.gethostname() == 'os-node-created-mgf6h'
    lock = (ORIGINAL / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert hashes() == q.read(RUN / 'storage_hashes.json'), 'Runtime changed after isolated snapshot'
    for rank in (0, 1):
        record = q.read(KIT / 'gpu_allgather_v2' / f'rank{rank}_status.json')
        assert record.get('status') == 'passed' and record['verified_tensor_slot_pairs'] == 2600
        ring = q.read(KIT / 'gpu_ring_v1' / f'ring_rank{rank}.json')
        assert ring.get('status') == 'passed' and len(ring['cases']) == 3
    if phase == 'compare':
        assert q.read(RUN / 'smoke_D' / 'smoke_2step' / 'result.json').get('shape_verified')
    idle()
    baseline = snapshot()
    assert baseline['limit_in_bytes'] == 250999996416
    file_cache = max(0, baseline['stat']['cache'] - baseline['stat']['shmem'])
    assert baseline['usage_in_bytes'] - file_cache < 48 * 1024**3, 'CPU non-file baseline too high'
    configure(phase)
    manifest = q.read(RUN / 'cuda_manifest.json')
    q.STATE['vlm'] = q.admission(manifest)
    q.STATE['memory_policy'] = 'same approved non_file>250GB + OOM counter guard, full lifetime'
    control = RUN / phase
    control.mkdir()
    guard = MemoryGuard(control / 'memory_trace.jsonl', lambda: q.STATE.get('status'))
    original_capture, original_cleanup, original_request = q.capture_tree, q.cleanup, q.request
    checking = True
    stage_pids = None
    baseline_unreadable = {}
    baseline_clients = scan_clients(unreadable=baseline_unreadable,capture_unreadable=True)

    def scan_clients_strict_with_retry():
        """Absorb short-lived /proc races; keep a persistent unreadable PID fatal."""
        attempts = 40
        for attempt in range(attempts):
            try:
                return scan_clients(unreadable=baseline_unreadable)
            except RuntimeError as exc:
                if (not str(exc).startswith('Cannot inspect GPU client namespace at ')
                        or attempt == attempts - 1):
                    raise
                time.sleep(0.25)
        raise AssertionError('unreachable')

    def capture(pid, owned):
        original_capture(pid, owned)
        if checking:
            guard.check()
            if stage_pids is not None:
                # A request may spawn a CUDA/Triton helper between the first
                # /proc tree snapshot and the GPU-fd scan.  Refresh the tree
                # after scanning so a newly born, genuine server descendant is
                # attributed by (pid, starttime); unrelated jobs still fail.
                clients = scan_clients_strict_with_retry()
                original_capture(pid, owned)
                check_clients(clients, baseline_clients, owned)
            if stage_pids is not None and gpu_pids() - stage_pids:
                raise RuntimeError('New GPU process appeared during generation; timing invalidated')

    def cleanup(server, owned):
        nonlocal checking, stage_pids
        checking = False
        stage_pids = None
        try:
            return original_cleanup(server, owned)
        finally:
            checking = True

    def request(server, owned, case, steps, directory, request_manifest):
        nonlocal stage_pids
        # Compare container identities only. Ngid is NOT a host-process ID.
        original_capture(server.pid, owned)
        clients = scan_clients_strict_with_retry()
        original_capture(server.pid, owned)
        own_clients = check_clients(clients, baseline_clients, owned)
        if len(own_clients) < 2:
            raise RuntimeError('Two verified own GPU workers not yet present')
        stage_pids = gpu_pids()
        if len(stage_pids - {1550469}) > len(own_clients):
            raise RuntimeError('More GPU host contexts than verified own device clients')
        evidence = dict(container_clients=clients, owned_clients=own_clients,
                        unreadable_idle_baseline=baseline_unreadable,
                        baseline_clients=baseline_clients, nvml_host_pids=sorted(stage_pids),
                        method='container PID+starttime+exact GPU device handles; no Ngid mapping')
        (control/f'{case}_{steps}_gpu_admission.json').write_text(json.dumps(evidence,indent=2))
        result = original_request(server, owned, case, steps, directory, request_manifest)
        result.update(storage_hashes=hashes(), scope='same prepared offload B/D; HTTP excludes model initialization')
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        stage_pids = None
        if phase == 'smoke':
            assert case == 'D' and steps == 2
            raise SmokeComplete()
        return result

    q.capture_tree, q.cleanup, q.request = capture, cleanup, request
    guard.start()
    try:
        for case in q.STATE['case_order']:
            idle()
            q.run_case(case, manifest)
        q.update('completed_timing_first_quality_pending')
    except SmokeComplete:
        source, target = RUN / 'results' / 'D', RUN / 'smoke_D'
        assert source.resolve().is_relative_to(RUN.resolve()) and not target.exists()
        source.rename(target)
        q.update('smoke_passed_comparison_not_started')
    except BaseException as exc:
        q.update('failed', error=repr(exc), memory_guard_reason=guard.reason, no_automatic_retry=True)
        raise
    finally:
        guard.close()
        (control / 'final_memory.json').write_text(json.dumps(snapshot(), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--phase', choices=('smoke', 'compare'))
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        assert not args.phase and not args.launch
        return prepare()
    assert args.phase
    if args.launch:
        assert not (RUN / args.phase).exists()
        idle()
        with (RUN / f'{args.phase}_dispatch.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__, '--phase', args.phase],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'phase': args.phase, 'pid': proc.pid, 'dispatched_not_passed': True}), flush=True)
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        run(args.phase)


if __name__ == '__main__':
    main()
