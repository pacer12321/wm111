"""Read-only post-hoc D audit. Does not repair the original failed supervisor.

The single observed tqdm prefix is removed ONLY in memory, with location/hash
reported, before the unchanged original strict verifier checks all 800 rows.
No model imports, process control, mutable remote outputs, or new experiments.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/cache/zhonghao/h3/dualstream_v1')
RUN_ID = '7de1704b57104dc5adc17f78211f6395'
RUN = ROOT / ('results/01234567/shirt_red_couple_124/D/runs/d_20260914T072058Z_' + RUN_ID)
POLICY_SHA = 'a412150f34a5f3c564e908e5ba85dcebe32ca641b2d139ce6963cbfc5d3949cd'
sys.path.insert(0, str(ROOT / 'deploy_d'))
import d_trial_gates as g
import d_execution_verifier as verifier


def main():
    host = g.profiles.require_host()
    status = g.json_file(RUN / 'd_status.json')
    frozen = g.json_file(RUN / 'frozen_evidence.json')
    expected_error = 'RuntimeError: Malformed/unlabelled/oversized actual D execution record'
    assert status['host'] == host and status['run_id'] == RUN_ID
    assert status['phase'] == 'failed' and status['error'] == expected_error
    assert status['formal_request_attempts'] == 1 and status['formal_50step_started'] is True
    assert status['formal_50step_completed'] is False
    policy = g.policy_gate(POLICY_SHA)
    assert policy == frozen['policy']
    assert g.source_code_manifest(policy['value']) == frozen['sources']
    transfer = g.transfer_gate(host)
    assert transfer == frozen['transfer']
    assert g.runtime_gate(host, frozen['sources']) == frozen['runtime']
    assert g.tiny_gate('01234567', host, policy['value']) == frozen['tiny']
    assert g.sample_gate(g.DEFAULT_SAMPLE, transfer=transfer) == frozen['sample']
    assert g.weight_manifest(transfer['verified']) == frozen['weights']
    g.contract.verify_submission(frozen['submission'], RUN.parent.parent, RUN_ID, host, POLICY_SHA)
    requests = {}
    for name, steps in [('smoke_2step', 2), ('d_50step', 50)]:
        request = g.json_file(RUN / name / 'request.json')
        result = g.json_file(RUN / name / 'result.json')
        assert result == status[name]
        assert request['run_id'] == RUN_ID and request['host'] == host
        assert request['requested_steps'] == steps and request['policy_sha256'] == POLICY_SHA
        assert request['source_sha256'] == frozen['sample']['source']['sha256']
        fields = {k: g.GENERATION[k] for k in ('width','height','fps','flow_shift','seed')}
        fields.update(prompt=g.sample_profile(g.DEFAULT_SAMPLE).prompt, num_inference_steps=steps,
                      extra_params=json.dumps(dict(task='ref2va',duration=g.GENERATION['duration_seconds'],
                                                   audio_flow_shift=g.GENERATION['audio_flow_shift'])))
        assert request['fields'] == fields
        assert result['success'] is True and result['http_code'] == '200' and result['curl_returncode'] == 0
        output = RUN / 'output' / (name + '.mp4')
        assert result['output_video'] == str(output)
        assert g.profiles.digest(output) == result['output_sha256']
        assert g.video_probe(output, target=True) == result['ffprobe']
        requests[name] = dict(request=g.file_record(RUN/name/'request.json'),
                              result=g.file_record(RUN/name/'result.json'), output=g.file_record(output),
                              elapsed_seconds=result['request_end_to_end_seconds'],
                              media_verified=True, input_and_parameters_verified=True)
    raw = (RUN / 'server.log').read_bytes()
    text = raw.decode('utf-8')
    smoke = g.json_file(RUN / 'formal_smoke_gate.json')
    assert hashlib.sha256(raw[:smoke['server_log_prefix_bytes']]).hexdigest() == smoke['server_log_prefix_sha256']
    loads = g.parse_full_load_evidence(text, frozen['weights'])
    workers = {int(k) for k in smoke['loaded_records_by_pid']}
    assert set(loads) == workers
    try:
        verifier.parse_execution_records(text, workers, run_id=RUN_ID, policy=policy['value'], policy_sha256=POLICY_SHA, requests=2)
    except RuntimeError as exc:
        assert str(exc) == expected_error.removeprefix('RuntimeError: ')
    else:
        raise AssertionError('Original parser failure was not reproduced')
    prefix = '  0%|          | 0/49 [00:00<?, ?it/s]'
    lines = text.splitlines()
    changes = []
    for i, line in enumerate(lines):
        if 'D_EXECUTION_RECORD ' not in line:
            continue
        if line.startswith(prefix + 'H3D pid='):
            changes.append(dict(line=i+1, prefix=prefix,
                                original_line_sha256=hashlib.sha256(line.encode()).hexdigest()))
            lines[i] = line[len(prefix):]
    assert len(changes) == 1 and changes[0]['line'] == 1369
    execution = verifier.parse_execution_records('\n'.join(lines), workers, run_id=RUN_ID,
                           policy=policy['value'], policy_sha256=POLICY_SHA, requests=2)
    assert execution['record_count'] == 800
    assert execution['rank_by_pid'] == {v['pid']:int(k) for k,v in smoke['worker_identities_by_card'].items()}
    originals = [status['supervisor_proc_identity'], status['server_proc_identity'], *smoke['worker_identities_by_card'].values()]
    for original in originals:
        current = g.base.proc_identity(original['pid'])
        assert current is None or current['start_ticks'] != original['start_ticks']
    observer = object.__new__(g.base.ProcessSupervisor)
    observer.run_id = RUN_ID
    assert not observer.owned_group_members(status['server_pid'])
    assert status['cleanup_completed'] is True and status['needs_attention'] is False
    assert status['remaining_owned_process_groups'] == {}
    after = (RUN/'npu_after.txt').read_text()
    assert g.base.selected_idle(after, tuple(range(8))) == status['resource_release_check']
    fresh = subprocess.run(['npu-smi','info'], capture_output=True,text=True,timeout=30,check=True).stdout
    idle = g.base.selected_idle(fresh,tuple(range(8)))
    health = g.base.selected_health_memory(fresh,tuple(range(8)),0)
    assert g.source_code_manifest(policy['value']) == frozen['sources']
    assert hashlib.sha256((RUN/'server.log').read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest()
    print(json.dumps(dict(status='posthoc_checks_passed_original_supervisor_failure_preserved',run_id=RUN_ID,
                         host=host, original_supervisor_phase=status['phase'], original_error=status['error'],
                         original_full_acceptance_flag=False, formal_http_and_media_completed=True,
                         current_dependencies_equal_frozen=True, requests=requests,
                         execution_record_count=execution['record_count'], rank_by_pid=execution['rank_by_pid'],
                         original_parser_failure_reproduced=True, in_memory_tqdm_prefix_normalization=changes,
                         original_log_modified=False, original_log_sha256=hashlib.sha256(raw).hexdigest(),
                         output_sha256=status['d_50step']['output_sha256'], all_ten_original_identities_exited=True,
                         remaining_owned_process_groups={}, fresh_idle=idle,fresh_health=health,
                         no_new_model_run=True,quality_evidence=False),indent=2))


if __name__ == '__main__':
    main()
