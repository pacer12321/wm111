"""One-shot detached B submission on31731; never retry an existing journal/run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/cache/zhonghao/h3/color_trial_v1')
RELEASE_SHA = 'a70c2cea17e254a7fb2857ea0e5f98816be50fe5e42756eaa60b8df95d659878'
A_RUN = ROOT / 'results/01234567/shirt_red_couple_124/A/runs/a_20260914T035755Z_5bcd246fd46d4965915dbf8d2c990304'


def sha(path):
    assert path.is_file() and not path.is_symlink() and path.resolve() == path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert ROOT.resolve() == ROOT and sha(ROOT / 'release_sha256.json') == RELEASE_SHA
    release = json.loads((ROOT / 'release_sha256.json').read_text())
    for name, expected in release['files'].items():
        assert sha(ROOT / name) == expected, name
    sys.path.insert(0, str(ROOT / 'b'))
    import b_trial_gates as g
    host = g.profiles.require_host()
    lock_path = ROOT / 'color_B_submission.lock'
    g.profiles.canonical_private(lock_path)
    with lock_path.open('a+b') as submission_lock:
        fcntl.flock(submission_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = ROOT / 'color_B_launch.json'
        if journal.exists() or journal.is_symlink():
            raise RuntimeError('B journal already exists: inspect it, never resubmit')
        for case in ('B', 'C'):
            case_root = ROOT / 'results/01234567/shirt_red_couple_124' / case
            if (case_root / (case.lower() + '_status.json')).exists() or ((case_root / 'runs').is_dir() and any((case_root / 'runs').iterdir())):
                raise RuntimeError('B/C already has a run: inspect instead of submitting')
        a = json.loads((A_RUN / 'a_status.json').read_text())
        assert a['run_id'] == '5bcd246fd46d4965915dbf8d2c990304' and a['host'] == host
        assert a['formal_request_attempts'] == 1 and a['formal_50step_completed'] and a['cleanup_completed']
        assert a['phase'] == 'formal_completed_review_required' and a['error'] is None and not a['needs_attention']
        assert not a['remaining_owned_process_groups']
        for identity in [a['supervisor_proc_identity'], a['server_proc_identity'], *a['formal_smoke_gate']['worker_identities_by_card'].values()]:
            current = g.base.proc_identity(identity['pid'])
            assert current is None or current['start_ticks'] != identity['start_ticks'], identity
        sources = g.source_code_manifest()
        transfer = g.transfer_gate(host)
        g.runtime_gate(host, sources)
        g.tiny_gate('01234567', host)
        sample = g.sample_gate('shirt_red_couple_124', transfer=transfer)
        weights = g.weight_manifest(transfer['verified'])
        # Bind the actual completed A request again to the B sample/settings.
        request = json.loads((A_RUN / 'a_50step/request.json').read_text())
        profile = g.sample_profile('shirt_red_couple_124')
        fields = {key: g.GENERATION[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
        fields.update(prompt=profile.prompt, num_inference_steps=50, extra_params=json.dumps({'task': 'ref2va', 'duration': g.GENERATION['duration_seconds'], 'audio_flow_shift': g.GENERATION['audio_flow_shift']}))
        assert request['fields'] == fields and request['source_sha256'] == sample['source']['sha256']
        assert request['source_video'] == str(profile.source) and request['run_id'] == a['run_id'] and request['requested_steps'] == 50
        check = subprocess.run(['/usr/local/sbin/npu-smi', 'info'], capture_output=True, text=True, timeout=30, check=True)
        idle = g.base.selected_idle(check.stdout, tuple(range(8)))
        health = g.base.selected_health_memory(check.stdout, tuple(range(8)), g.MIN_HBM_MIB)
        memory = g.base.host_memory_snapshot(g.MIN_RAM)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        log_path = ROOT / f'color_B_launch_{stamp}.log'
        g.profiles.canonical_private(log_path)
        record = dict(phase='reserved', case='B', sample_id=profile.sample_id, host=host,
                      prepared_at=g.base.utc_now(), release_sha256=RELEASE_SHA,
                      A_run_id=a['run_id'], A_formal_request_sha256=sha(A_RUN / 'a_50step/request.json'),
                      A_request_independent_audit_chunk='9b35dc',
                      fresh_idle=idle, health=health, host_memory=memory,
                      all_B_readonly_prerequisites_verified=True,
                      branch_tensors=weights['branch']['tensor_count'], lora_tensors=weights['lora']['tensor_count'],
                      VLM_run_id=sample['vlm']['evidence']['status']['run_id'],
                      launch_log=str(log_path), formal_attempt_limit=1, smoke_required=True,
                      C_started_by_this_submission=False)
        with journal.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        command = 'source /cache/zhonghao/h3/env_h3_31731.sh\nexec /cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/b/run_b_trial.py --group 01234567 --sample shirt_red_couple_124 --allow-npu'
        with log_path.open('xb', buffering=0) as log:
            child = subprocess.Popen(['bash', '-c', command], stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        identity = g.base.proc_identity(child.pid)
        record.update(phase='submitted', submitted_at=g.base.utc_now(), launcher_pid=child.pid, launcher_identity=identity)
        g.base.atomic_json(journal, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
