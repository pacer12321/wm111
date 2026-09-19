"""CPU-only B8 profile/load/PID/workflow regressions; no remote or device use."""
import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'validation'))
sys.path.insert(0, str(HERE.parent / 'model_trial'))
sys.path.insert(0, str(HERE))
if os.name == 'nt':
    sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import b_trial_gates as gates
import run_b_trial as trial

HOST = {'hostname': gates.profiles.EXPECTED_HOST, 'boot_id': '12345678-abcd-abcd-abcd-123456789abc', 'machine': 'aarch64'}


def weights():
    return {kind: {'path': str(gates.CHECKPOINT / name), 'size_bytes': 1234,
                   'header_sha256': kind + '-header', 'tensor_count': count}
            for kind, name, count in (('branch', 'linear_branch/model.safetensors', 800),
                                      ('lora', 'adapters/default/adapter_model.safetensors', 416))}


def load_record():
    return {'checkpoint': str(gates.CHECKPOINT), 'base_partition': 'ref2va', 'base_tensor_count': 535,
            'branch_tensor_count': 800, 'lora_pairs_merged': 208, 'lora_rank': 64, 'lora_alpha': 64,
            'lora_scale': 1.0, 'official_scale_source_commit': gates.OFFICIAL_COMMIT,
            'merge_dtype': 'FP32 delta, cast to parameter dtype, then add',
            'qkv_merge_layout': 'post-base-loader contiguous Q/K/V thirds', **weights()}


def cpu_record():
    return {'status': 'completed', 'requested_loading_intraop': 4, 'intraop_before': 1, 'intraop_loading': 4,
            'intraop_after': 1, 'interop_before': 32, 'interop_after': 32, 'interop_loading': 32,
            'phase_seconds': {'base': 31.0, 'branch': 2.0, 'lora': 4.0}}


def log_text(*, pids=range(500, 508), mutate=None):
    result = []
    for pid in pids:
        load, cpu = load_record(), cpu_record()
        if mutate:
            mutate(pid, load, cpu)
        for marker, data in (('OPENVDN_B_LOAD_RECORD', load), ('OPENVDN_B_CPU_LOADING_END', cpu)):
            result.append(f'H3B pid={pid} 2026-09-13 INFO vllm_omni.loader {marker} ' + json.dumps(data))
    return '\n'.join(result)


def smi():
    result = []
    for card in range(8):
        result.extend([f'| {card} 910B3 | OK | 0 / 0 |', '| 0 | bus | 0 / 0 3402 / 65536 |'])
    result.append('| NPU Chip | Process id | Process name | Process memory(MB) |')
    result.extend(f'| {card} 0 | {500+card} | VLLM::Worker | 3402 |' for card in range(8))
    return '\n'.join(result)


class ProfileTests(unittest.TestCase):
    def test_independent_outputs_ports_same_locks_no_global_mutation(self):
        tiny = gates.profiles.profile('01234567')
        for sample_id in gates.SAMPLE_IDS:
            a = gates.a_gates.selected_profile('01234567', sample_id)
            b = gates.selected_profile('01234567', sample_id)
            self.assertEqual(b.cards, tuple(range(8)))
            self.assertEqual(b.lease_root, a.lease_root)
            self.assertEqual(b.lease_root, tiny.lease_root)
            self.assertNotEqual(b.output_root, a.output_root)
            self.assertEqual(b.output_root.parts[-1], 'B')
            self.assertNotEqual(b.code_root, a.code_root)
            self.assertNotEqual(b.master_port, a.master_port)
            ap, bp = gates.a_gates.parallelism(a), gates.parallelism(b)
            ap.pop('listen_port'); bp.pop('listen_port')
            self.assertEqual(ap, bp)
        self.assertEqual(tiny, gates.profiles.profile('01234567'))
        self.assertNotEqual(gates.selected_profile('01234567').output_root,
                            gates.selected_profile('01234567', 'explicit_reverse_couple_124').output_root)
        for group in ('0123', '4567', '2367', 'all8'):
            with self.assertRaises(ValueError):
                gates.selected_profile(group)

    def test_same_fixed_sampler_and_source_profiles(self):
        self.assertEqual(gates.GENERATION, gates.a_gates.GENERATION)
        self.assertIsNot(gates.GENERATION, gates.a_gates.GENERATION)
        for name in gates.SAMPLE_IDS:
            self.assertEqual(gates.sample_profile(name), gates.a_gates.sample_profile(name))
        self.assertEqual(gates.GENERATION['num_inference_steps'], 50)
        self.assertEqual(gates.MIN_RAM, 600 * 1024**3)
        self.assertEqual(gates.MIN_HBM_MIB, 55 * 1024)
        self.assertEqual(trial.REQUEST_TIMEOUT, 3600)

    def test_old_text_or_vae_subworld_manifest_cannot_launch_new_eight_group(self):
        selected = gates.selected_profile('01234567')
        correct = gates.a_gates.parallelism(selected)
        correct.update(text_encoder_tp_size=8, vae_patch_parallel_size=8)
        for key in ('text_encoder_tp_size', 'vae_patch_parallel_size'):
            wrong = {**correct, key: 4}
            with mock.patch.object(gates.a_gates, 'parallelism', return_value=wrong):
                with self.assertRaises(RuntimeError):
                    gates.parallelism(selected)

    def test_frozen_a_gate_pin_matches_local_dependency(self):
        self.assertEqual(gates.profiles.digest(HERE.parent / 'model_trial/trial_gates.py'), gates.A_GATES_SHA256)

    def test_candidate_pins_match_current_loader(self):
        for name, sha in gates.profiles.CANDIDATE_SHA256.items():
            self.assertEqual(gates.profiles.digest(HERE.parents[1] / 'b_adaptation/patched' / name), sha)

    def test_logging_uses_real_pid_no_color(self):
        config = gates.logging_configuration()
        self.assertIn('%(process)d', config['formatters']['h3b']['format'])
        self.assertEqual(config['handlers']['h3b']['stream'], 'ext://sys.stderr')
        self.assertFalse(config['loggers']['vllm_omni']['propagate'])

    def test_launcher_has_private_candidate_after_env_and_correct_full_world_groups(self):
        script = (HERE / 'launch_b_trial.sh').read_text()
        self.assertLess(script.index('source /cache/zhonghao/h3/env_h3_31731.sh'),
                        script.index('export PYTHONPATH="/cache/zhonghao/h3/candidates/b_v1'))
        self.assertIn('export ZHONGHAO_H3_OPENVDN=1', script)
        self.assertIn('--num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8', script)
        self.assertIn('--vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8', script)
        self.assertIn('export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600', script)
        self.assertIn('export VLLM_LOGGING_CONFIG_PATH="${H3_ROOT}/formal_logging.json"', script)
        self.assertIn('task_port=19099', script)
        self.assertIn('/b_model_trial/', script)
        self.assertNotIn('/cache/yunfeng', script)
        self.assertNotIn('--dtype', script)


class LoadRecordTests(unittest.TestCase):
    def test_exact_eight_workers_complete_load(self):
        rows = gates.parse_full_load_evidence(log_text(), weights())
        self.assertEqual(set(rows), set(range(500, 508)))
        self.assertEqual(rows[507]['load']['lora_pairs_merged'], 208)

    def test_four_seven_nine_or_duplicate_workers_rejected(self):
        for pids in (range(500, 504), range(500, 507), range(500, 509), [500] * 8):
            with self.subTest(pids=list(pids)), self.assertRaises(RuntimeError):
                gates.parse_full_load_evidence(log_text(pids=pids), weights())

    def test_unlabelled_old_records_and_missing_end_rejected(self):
        for text in (log_text().replace('H3B pid=', 'old pid='),
                     log_text().replace('OPENVDN_B_CPU_LOADING_END', 'OLD_LOADING_END')):
            with self.assertRaises(RuntimeError):
                gates.parse_full_load_evidence(text, weights())

    def test_bad_counts_scale_layout_checkpoint_rejected(self):
        for key, value in (('base_tensor_count', 534), ('branch_tensor_count', 799), ('lora_pairs_merged', 207),
                           ('checkpoint', '/old/shared/checkpoint'), ('lora_scale', .5),
                           ('base_partition', 't2va'), ('lora_alpha', 32), ('qkv_merge_layout', 'head grouped'),
                           ('official_scale_source_commit', 'other')):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                gates.parse_full_load_evidence(log_text(mutate=lambda pid, load, cpu: load.update({key: value}) if pid == 507 else None), weights())

    def test_one_bad_weight_identity_rejected(self):
        for kind in ('branch', 'lora'):
            for key in ('path', 'size_bytes', 'header_sha256', 'tensor_count'):
                with self.subTest(kind=kind, key=key), self.assertRaises(RuntimeError):
                    gates.parse_full_load_evidence(log_text(mutate=lambda pid, load, cpu: load[kind].update({key: 'bad'}) if pid == 500 else None), weights())

    def test_cpu_thread_restoration_and_finite_timing_required(self):
        for key, value in (('status', 'failed'), ('intraop_after', 4), ('interop_after', 1), ('intraop_before', True),
                           ('intraop_loading', 8), ('requested_loading_intraop', 8), ('interop_loading', 1),
                           ('phase_seconds', {'base': float('nan'), 'branch': 1, 'lora': 1}),
                           ('phase_seconds', {'base': 1, 'branch': -1, 'lora': 1}),
                           ('phase_seconds', {})):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                gates.parse_full_load_evidence(log_text(mutate=lambda pid, load, cpu: cpu.update({key: value}) if pid == 507 else None), weights())


class StageBWeightTests(unittest.TestCase):
    def create_weights(self, root, *, branch_count=800, lora_count=416):
        ref = root / 'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000'
        for name, count in (('linear_branch/model.safetensors', branch_count), ('adapters/default/adapter_model.safetensors', lora_count)):
            path = ref / name; path.parent.mkdir(parents=True, exist_ok=True)
            header = {f'tensor{i}': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [i*2, (i+1)*2]} for i in range(count)}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + b'00' * count)
        (ref / 'config.json').write_text('{"fixture":true}')
        return {path.relative_to(root).as_posix(): {'kind': 'file', 'size': path.stat().st_size, 'sha256': gates.profiles.digest(path)}
                for path in ref.rglob('*') if path.is_file()}

    def test_stage_b_current_headers_and_transfer_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); verified = self.create_weights(root)
            with mock.patch.object(gates, 'ROOT', root), mock.patch.object(gates.profiles, 'INSTALL', root):
                record = gates.stage_b_weight_manifest(verified)
                self.assertEqual(record['branch']['tensor_count'], 800)
                self.assertEqual(record['lora']['tensor_count'], 416)
                self.assertEqual(record, gates.stage_b_weight_manifest(verified))
                original = copy.deepcopy(verified)
                for key in original:
                    bad = copy.deepcopy(original); bad[key]['size'] += 1
                    with self.assertRaises(RuntimeError):
                        gates.stage_b_weight_manifest(bad)
                cfg = next(key for key in original if key.endswith('config.json'))
                bad = copy.deepcopy(original); bad[cfg]['sha256'] = 'a' * 64
                with self.assertRaises(RuntimeError):
                    gates.stage_b_weight_manifest(bad)

    def test_partial_checkpoint_counts_fail_closed(self):
        for branch_count, lora_count in ((799, 416), (800, 414)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                verified = self.create_weights(root, branch_count=branch_count, lora_count=lora_count)
                with mock.patch.object(gates, 'ROOT', root), mock.patch.object(gates.profiles, 'INSTALL', root):
                    with self.assertRaises(RuntimeError):
                        gates.stage_b_weight_manifest(verified)


class WorkflowTests(unittest.TestCase):
    def owner(self, sample_id=gates.DEFAULT_SAMPLE):
        with mock.patch.object(gates.profiles, 'require_host', return_value=HOST), mock.patch.object(trial.base, 'proc_identity', return_value=None):
            return trial.BTrialSupervisor('01234567', sample_id)

    def test_cli_requires_allow_and_fixed_group_sample(self):
        for args in (['--group', '01234567'], ['--group', '01234567', '--allow-npu', '--source', 'x'],
                     ['--group', '01234567', '--sample', 'custom', '--allow-npu'], ['--group', '0123', '--allow-npu']):
            with mock.patch.object(trial, 'BTrialSupervisor') as cls, mock.patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    trial.main(args)
                cls.assert_not_called()

    def test_b_state_is_independent_and_timeout_not_timing_claim(self):
        owner = self.owner()
        self.assertEqual(owner.status['case'], 'B')
        self.assertIsNone(owner.status['new_acceleration_results'])
        self.assertEqual(owner.status['request_timeout_seconds'], 3600)
        self.assertFalse(owner.status['quality_evidence'])
        self.assertIn('waiting ceiling', owner.status['timeout_policy'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); owner.run_dir = root
            with mock.patch.object(trial.base, 'atomic_json') as write, mock.patch('sys.stdout', new=io.StringIO()):
                owner.update('test')
                self.assertEqual([call.args[0].name for call in write.call_args_list], ['b_status.json', 'b_status.json'])

    def test_environment_exact_b_candidate_logging_and_run_marker(self):
        owner = self.owner(); owner.run_dir = Path('/private/b_run')
        with mock.patch.dict(os.environ, {'ZHONGHAO_H3_OPENVDN': '0', 'ZHONGHAO_H3_OPENVDN_CHECKPOINT': '/old',
                                         'ZHONGHAO_H3_OPENVDN_C_RULE': '1', 'PYTHONPATH': '/foreign',
                                         'VLLM_LOGGING_CONFIG_PATH': '/other.json'}):
            env = owner.env()
        self.assertEqual(env['ZHONGHAO_H3_OPENVDN'], '1')
        self.assertEqual(env['ZHONGHAO_H3_OPENVDN_CHECKPOINT'], str(gates.CHECKPOINT))
        self.assertNotIn('ZHONGHAO_H3_OPENVDN_C_RULE', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['VLLM_LOGGING_CONFIG_PATH'], str(owner.run_dir / 'formal_logging.json'))
        self.assertEqual(env['VLLM_CONFIGURE_LOGGING'], '1')
        self.assertEqual(env['VLLM_OMNI_VIDEO_SYNC_TIMEOUT'], '3600')
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertIn(owner.run_id, env['H3_GROUP_ID'])

    def test_one_same_service_smoke_then_formal_no_repeat(self):
        owner = self.owner(); calls = []
        owner.request = lambda name, steps: calls.append((name, steps)) or {'success': True, 'ffprobe': {'verified': True}}
        owner.smoke_gate = lambda: calls.append('gate')
        owner.assert_unchanged = lambda: calls.append('unchanged')
        owner.update = lambda phase, **values: owner.status.update(values)
        owner.run_requests()
        self.assertEqual(calls, [('smoke_2step', 2), 'gate', ('b_50step', 50), 'unchanged'])
        self.assertEqual(owner.status['formal_request_attempts'], 1)
        self.assertTrue(owner.status['formal_50step_completed'])
        with self.assertRaises(RuntimeError):
            owner.run_requests()

    def test_gate_failure_never_reaches_formal(self):
        owner = self.owner()
        owner.request = mock.Mock()
        owner.smoke_gate = mock.Mock(side_effect=RuntimeError('missing eight-PID full load'))
        with self.assertRaises(RuntimeError):
            owner.run_requests()
        owner.request.assert_called_once_with('smoke_2step', 2)
        self.assertEqual(owner.status['formal_request_attempts'], 0)

    def test_direct_formal_wrong_order_and_retry_rejected(self):
        owner = self.owner()
        with self.assertRaises(RuntimeError):
            owner.request('b_50step', 50)
        owner._request_order = [('smoke_2step', 2)]
        with self.assertRaises(RuntimeError):
            owner.request('b_50step', 50)
        owner._request_order.append(('b_50step', 50))
        with self.assertRaises(RuntimeError):
            owner.request('smoke_2step', 2)

    def test_real_request_path_keeps_50_steps_and_records_complete_elapsed_not_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); owner = self.owner(); owner.run_dir = root
            (root / 'output').mkdir()
            owner._request_order = [('smoke_2step', 2)]
            owner.status['formal_smoke_gate_passed'] = True
            owner.server = mock.Mock(pid=100)
            owner.frozen = {'sample': {'source': {'sha256': 'same-source'}}}
            owner.update = lambda phase, **values: owner.status.update(values)
            commands = []
            def spawn(command, **kwargs):
                commands.append(command)
                Path(command[command.index('--dump-header') + 1]).write_text('HTTP/1.1 200 OK\r\nContent-Type: video/mp4\r\n')
                Path(command[command.index('--output') + 1]).write_bytes(b'complete CPU toy output')
                kwargs['stdout'].write(b'200')
                return mock.Mock(returncode=0, poll=mock.Mock(return_value=0))
            owner.spawn = spawn
            try:
                with mock.patch.object(gates, 'file_record', return_value={'sha256': 'same-source'}), \
                     mock.patch.object(gates, 'video_probe', return_value={'verified': True}), \
                     mock.patch.object(trial.base, 'proc_identity', return_value={'pid': 100}), \
                     mock.patch.object(trial.base.os, 'O_NOFOLLOW', 0, create=True), \
                     mock.patch.object(trial.time, 'monotonic', side_effect=[10.0, 2010.5]):
                    result = owner.request('b_50step', 50)
                self.assertEqual(result['request_end_to_end_seconds'], 2000.5)
                self.assertEqual(result['requested_steps'], 50)
                self.assertIsNone(result['dit_latency_seconds'])
                self.assertEqual(commands[0][commands[0].index('--max-time') + 1], '3600')
                self.assertIn('num_inference_steps=50', commands[0])
                self.assertIn('http://127.0.0.1:19099/v1/videos/sync', commands[0])
                record = json.loads((root / 'b_50step/request.json').read_text())
                self.assertEqual(record['fields']['seed'], 4101)
                self.assertEqual(record['fields']['prompt'], owner.sample.prompt)
                self.assertEqual(owner._request_order, [('smoke_2step', 2), ('b_50step', 50)])
            finally:
                for stream in owner.handles:
                    stream.close()

    def test_request_timeout_is_fail_closed_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); owner = self.owner(); owner.run_dir = root
            owner.server = mock.Mock(pid=100, poll=mock.Mock(return_value=None))
            owner.frozen = {'sample': {'source': {'sha256': 'same-source'}}}
            owner.update = mock.Mock(); owner.spawn = mock.Mock(return_value=mock.Mock(poll=mock.Mock(return_value=None)))
            try:
                with mock.patch.object(gates, 'file_record', return_value={'sha256': 'same-source'}), \
                     mock.patch.object(trial.base, 'proc_identity', return_value={'pid': 100}), \
                     mock.patch.object(trial.base.os, 'O_NOFOLLOW', 0, create=True), \
                     mock.patch.object(trial.time, 'monotonic', side_effect=[0, 3621]):
                    with self.assertRaises(TimeoutError):
                        owner.request('smoke_2step', 2)
                owner.spawn.assert_called_once()
                self.assertEqual(owner._request_order, [('smoke_2step', 2)])
                self.assertFalse(owner.status['formal_50step_started'])
            finally:
                for stream in owner.handles:
                    stream.close()

    def test_failed_preflight_cleanup_and_release_no_service(self):
        fake = mock.Mock(status={}); fake.preflight.side_effect = RuntimeError('gate failed'); fake.cleanup.return_value = True
        with mock.patch.object(trial, 'BTrialSupervisor', return_value=fake), mock.patch.object(trial.signal, 'signal'):
            self.assertEqual(trial.main(['--group', '01234567', '--allow-npu']), 1)
        fake.launch.assert_not_called(); fake.run_requests.assert_not_called()
        fake.cleanup.assert_called_once(); fake.release_locks.assert_called_once()

    def test_cleanup_exception_still_closes_leases(self):
        fake = mock.Mock(status={}); fake.cleanup.side_effect = RuntimeError('cleanup')
        with mock.patch.object(trial, 'BTrialSupervisor', return_value=fake), mock.patch.object(trial.signal, 'signal'):
            with self.assertRaises(RuntimeError):
                trial.main(['--group', '01234567', '--allow-npu'])
        fake.release_locks.assert_called_once()

    def exercise_smoke_gate(self, corruption=None, sample_id=gates.DEFAULT_SAMPLE):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); owner = self.owner(sample_id); owner.run_dir = root
            owner.server = mock.Mock(pid=100, poll=mock.Mock(return_value=None))
            owner.locks = [SimpleNamespace(closed=False) for _ in range(9)]
            owner._request_order = [('smoke_2step', 2)]
            owner.assert_unchanged = mock.Mock()
            owner.owned_group_members = mock.Mock(return_value=list(range(500, 508)))
            owner.frozen = {'sample': {'source': {'sha256': 'source-sha'}}, 'weights': weights()}
            owner.update = lambda phase, **values: owner.status.update(values)
            server = {'pid': 100, 'start_ticks': 1000, 'pgrp': 100, 'session': 100, 'state': 'S'}
            owner.status['server_proc_identity'] = {**server, 'state': 'R'}
            (root / 'smoke_2step').mkdir(); (root / 'output').mkdir()
            output = root / 'output/smoke_2step.mp4'; output.write_bytes(b'CPU toy video')
            actual_probe = {'verified': True, 'checks': {'dimensions': [1344, 768], 'fps': 24, 'frame_count': 124}}
            result = {'success': True, 'requested_steps': 2, 'http_code': '200', 'curl_returncode': 0,
                      'ffprobe': copy.deepcopy(actual_probe), 'output_video': str(output), 'output_sha256': gates.profiles.digest(output)}
            fields = {key: gates.GENERATION[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
            fields.update(prompt=owner.sample.prompt, num_inference_steps=2, extra_params=json.dumps({
                'task': 'ref2va', 'duration': gates.GENERATION['duration_seconds'], 'audio_flow_shift': gates.GENERATION['audio_flow_shift']}))
            request = {'host': owner.host, 'group': '01234567', 'run_id': owner.run_id, 'sample_id': sample_id,
                       'source_sha256': 'source-sha', 'source_video': str(owner.sample.source), 'fields': fields,
                       'requested_steps': 2, 'server_proc_identity': {**server, 'state': 'R'}}
            logs = log_text()
            if corruption == 'old_server': server['start_ticks'] += 1
            elif corruption == 'foreign_workers': owner.owned_group_members.return_value = [500]
            elif corruption == 'source': request['source_sha256'] = 'other'
            elif corruption == 'fields': fields['seed'] = 1
            elif corruption == 'sample': request['sample_id'] = 'wrong'
            elif corruption == 'output': output.write_bytes(b'changed')
            elif corruption == 'metadata': result['ffprobe']['checks']['frame_count'] = 120
            elif corruption == 'locks': owner.locks.pop()
            elif corruption == 'missing_loads': logs = log_text(pids=range(500, 504))
            elif corruption == 'old_load_pids': logs = log_text(pids=range(600, 608))
            (root / 'server.log').write_text(logs)
            owner.status['smoke_2step'] = copy.deepcopy(result)
            (root / 'smoke_2step/result.json').write_text(json.dumps(result))
            (root / 'smoke_2step/request.json').write_text(json.dumps(request))
            def identity(pid):
                return server.copy() if pid == 100 else {'pid': pid, 'pgrp': 100, 'session': 100, 'start_ticks': 2000+pid, 'state': 'S'}
            with mock.patch.object(gates.profiles, 'INSTALL', root), mock.patch.object(gates, 'video_probe', return_value=actual_probe), \
                 mock.patch.object(trial.base, 'proc_identity', side_effect=identity), \
                 mock.patch.object(trial.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=smi(), stderr='')), \
                 mock.patch.object(trial.base.os, 'O_NOFOLLOW', 0, create=True):
                owner.smoke_gate()
            return owner.status

    def test_eight_pid_same_service_complete_gate_both_samples(self):
        for sample_id in gates.SAMPLE_IDS:
            status = self.exercise_smoke_gate(sample_id=sample_id)
            self.assertTrue(status['formal_smoke_gate_passed'])
            proof = status['formal_smoke_gate']
            self.assertEqual(set(proof['loaded_records_by_pid']), set(range(500, 508)))
            self.assertEqual(proof['sample_id'], sample_id)
            self.assertGreater(proof['server_log_prefix_bytes'], 0)

    def test_gate_blocks_identity_source_metadata_lease_and_load_drift(self):
        for corruption in ('old_server', 'foreign_workers', 'source', 'fields', 'sample', 'output', 'metadata',
                           'locks', 'missing_loads', 'old_load_pids'):
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                self.exercise_smoke_gate(corruption)


if __name__ == '__main__':
    unittest.main(verbosity=2)
