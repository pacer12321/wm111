"""CPU-only C8 profile/load/PID/workflow regressions; no remote or device use."""
import copy
import ast
import importlib.util
import io
import json
import math
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
sys.path.insert(0, str(HERE.parent / 'b_model_trial'))
sys.path.insert(0, '/cache/zhonghao/h3/validation_code')
sys.path.insert(0, '/cache/zhonghao/h3/model_trial_code')
sys.path.insert(0, '/cache/zhonghao/h3/b_model_trial_code')
sys.path.insert(0, str(HERE))
C_ADAPTATION = HERE.parents[1] / 'c_adaptation'
C_TEST_CODE = C_ADAPTATION / 'validation_31731'
if not C_TEST_CODE.is_dir():
    C_TEST_CODE = Path('/cache/zhonghao/h3/c_validation_code')
    C_ADAPTATION = Path('/cache/zhonghao/h3/c_adaptation')
sys.path.insert(0, str(C_TEST_CODE))
if os.name == 'nt':
    sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import c_trial_gates as gates
import run_c_trial as trial

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
            result.append(f'H3C pid={pid} 2026-09-13 INFO vllm_omni.loader {marker} ' + json.dumps(data))
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
            self.assertEqual(b.output_root.parts[-1], 'C')
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
        dep = HERE.parent / 'model_trial/trial_gates.py'
        if not dep.is_file(): dep = gates.b_gates.A_CODE / 'trial_gates.py'
        self.assertEqual(gates.profiles.digest(dep), gates.A_GATES_SHA256)

    def test_c_candidate_and_verifier_pins_match_local_review(self):
        import c_profiles as cp
        model = C_ADAPTATION / 'candidate'
        if not model.is_dir(): model = gates.VENDOR / 'vllm_omni/diffusion/models/minimax_h3'
        for name, sha in cp.PINNED_C.items():
            self.assertEqual(gates.profiles.digest(model / name), sha)
        for name, sha in gates.C_VALIDATION_SHA256.items():
            self.assertEqual(gates.profiles.digest(C_TEST_CODE / name), sha)
        dep = HERE.parent / 'b_model_trial/b_trial_gates.py'
        if not dep.is_file(): dep = gates.B_CODE / 'b_trial_gates.py'
        self.assertEqual(gates.profiles.digest(dep), gates.B_GATES_SHA256)

    def test_logging_uses_real_pid_no_color(self):
        config = gates.logging_configuration()
        self.assertIn('%(process)d', config['formatters']['h3c']['format'])
        self.assertEqual(config['handlers']['h3c']['stream'], 'ext://sys.stderr')
        self.assertFalse(config['loggers']['vllm_omni']['propagate'])

    def test_launcher_has_private_candidate_after_env_and_correct_full_world_groups(self):
        script = (HERE / 'launch_c_trial.sh').read_text()
        self.assertLess(script.index('source /cache/zhonghao/h3/env_h3_31731.sh'),
                        script.index('export PYTHONPATH="/cache/zhonghao/h3/candidates/c_v1'))
        self.assertIn('export ZHONGHAO_H3_OPENVDN=1', script)
        self.assertIn('--num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8', script)
        self.assertIn('--vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8', script)
        self.assertIn('export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600', script)
        self.assertIn('export VLLM_LOGGING_CONFIG_PATH="${H3_ROOT}/formal_logging.json"', script)
        self.assertIn('task_port=19100', script)
        self.assertIn('/c_model_trial/', script)
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
        for text in (log_text().replace('H3C pid=', 'old pid='),
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


class WorkflowTests(unittest.TestCase):
    def owner(self, sample_id=gates.DEFAULT_SAMPLE):
        with mock.patch.object(gates.profiles, 'require_host', return_value=HOST), mock.patch.object(trial.base, 'proc_identity', return_value=None):
            return trial.CTrialSupervisor('01234567', sample_id)

    def test_cli_requires_allow_and_fixed_group_sample(self):
        for args in (['--group', '01234567'], ['--group', '01234567', '--allow-npu', '--source', 'x'],
                     ['--group', '01234567', '--sample', 'custom', '--allow-npu'], ['--group', '0123', '--allow-npu']):
            with mock.patch.object(trial, 'CTrialSupervisor') as cls, mock.patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    trial.main(args)
                cls.assert_not_called()

    def test_b_state_is_independent_and_timeout_not_timing_claim(self):
        owner = self.owner()
        self.assertEqual(owner.status['case'], 'C')
        self.assertIsNone(owner.status['new_acceleration_results'])
        self.assertEqual(owner.status['request_timeout_seconds'], 3600)
        self.assertFalse(owner.status['quality_evidence'])
        self.assertIn('waiting ceiling', owner.status['timeout_policy'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); owner.run_dir = root
            with mock.patch.object(trial.base, 'atomic_json') as write, mock.patch('sys.stdout', new=io.StringIO()):
                owner.update('test')
                self.assertEqual([call.args[0].name for call in write.call_args_list], ['c_status.json', 'c_status.json'])

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
        with tempfile.TemporaryDirectory() as directory:
            owner.run_dir = Path(directory)
            raw = strict_logs(requests=2).encode()
            (owner.run_dir / 'server.log').write_bytes(raw)
            owner.status['formal_smoke_gate'] = {'server_log_prefix_bytes': 0,
                'server_log_prefix_sha256': __import__('hashlib').sha256(b'').hexdigest(),
                'loaded_records_by_pid': {p: {} for p in range(500, 508)}}
            with mock.patch.object(trial.base.os, 'O_NOFOLLOW', 0, create=True):
                owner.run_requests()
            self.assertTrue((owner.run_dir / 'formal_strict_metadata.json').is_file())
        self.assertEqual(calls, [('smoke_2step', 2), 'gate', ('c_50step', 50), 'unchanged'])
        self.assertEqual(owner.status['formal_request_attempts'], 1)
        self.assertTrue(owner.status['formal_50step_completed'])
        with self.assertRaises(RuntimeError): owner.run_requests()

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
            owner.request('c_50step', 50)
        owner._request_order = [('smoke_2step', 2)]
        with self.assertRaises(RuntimeError):
            owner.request('c_50step', 50)
        owner._request_order.append(('c_50step', 50))
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
                    result = owner.request('c_50step', 50)
                self.assertEqual(result['request_end_to_end_seconds'], 2000.5)
                self.assertEqual(result['requested_steps'], 50)
                self.assertIsNone(result['dit_latency_seconds'])
                self.assertEqual(commands[0][commands[0].index('--max-time') + 1], '3600')
                self.assertIn('num_inference_steps=50', commands[0])
                self.assertIn('http://127.0.0.1:19100/v1/videos/sync', commands[0])
                record = json.loads((root / 'c_50step/request.json').read_text())
                self.assertEqual(record['fields']['seed'], 4101)
                self.assertEqual(record['fields']['prompt'], owner.sample.prompt)
                self.assertEqual(owner._request_order, [('smoke_2step', 2), ('c_50step', 50)])
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
        with mock.patch.object(trial, 'CTrialSupervisor', return_value=fake), mock.patch.object(trial.signal, 'signal'):
            self.assertEqual(trial.main(['--group', '01234567', '--allow-npu']), 1)
        fake.launch.assert_not_called(); fake.run_requests.assert_not_called()
        fake.cleanup.assert_called_once(); fake.release_locks.assert_called_once()

    def test_cleanup_exception_still_closes_leases(self):
        fake = mock.Mock(status={}); fake.cleanup.side_effect = RuntimeError('cleanup')
        with mock.patch.object(trial, 'CTrialSupervisor', return_value=fake), mock.patch.object(trial.signal, 'signal'):
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
            if corruption != 'missing_strict':
                logs += '\n' + strict_logs(pids=range(600, 608) if corruption == 'strict_pids' else range(500, 508))
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
                           'locks', 'missing_loads', 'old_load_pids', 'missing_strict', 'strict_pids'):
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                self.exercise_smoke_gate(corruption)


def strict_metadata():
    return dict(mode=gates.STRICT_MODE, reference_count=1, reference_kind='video',
                patch_size=(1, 2, 2), source_shape=(37, 48, 84), target_shape=(37, 48, 84),
                text_len=96, source_audio_t=0, target_audio_t=207)


def strict_logs(*, pids=range(500, 508), requests=1, mutate=None):
    lines = []
    for request in range(requests):
        for pid in pids:
            row = strict_metadata()
            if mutate: mutate(pid, request, row)
            lines.append(f'H3C pid={pid} 2026-09-13 INFO vllm_omni.pipeline {gates.STRICT_MARKER}{row!r}')
    return '\n'.join(lines)


def tiny_report_fixture():
    """CPU-fabricated fixture ONLY; always under a temporary directory.

    Build the real C verifier schema and invoke the actual verifier, not a
    return-value mock pretending a missing production NPU run passed.
    """
    import c_profiles as cp
    import c_cases as cases
    import c_npu_regression as checker
    run_id = 'a' * 32
    manifest = {'validation/c_npu_regression.py': {'sha256': 'f' * 64},
                'strategy/ulysses.py': {'path': '/cpu_fixture', 'sha256': '1' * 64}}
    ranks = []
    for rank in range(8):
        tests, pre = [], {}
        for case in cases.CASES:
            data, layout = cases.fixture(case), cases.geometry(case)
            n = len(data['coords'])
            proof = cases.audit_masks(layout, n)
            pre[case] = dict(rejected=list(cases.REJECTIONS), oracle=proof)
            checks = [dict(check=name, max_abs=0.0, rms_error=0.0, golden_rms=0.0,
                           atol=0.0 if name.endswith('-zero') else 1 / 512,
                           rtol=0.0 if name.endswith('-zero') else 1 / 128) for name in checker.CHECKS]
            tests.append(dict(case=case, packed_rows=n, local_rows=[rank*(n//8), (rank+1)*(n//8)],
                              layout=vars(layout), checks=checks, mask_proof=proof,
                              frozen_qkv_unchanged=True, positive_controls=True, source_is_visual_only=True))
        ranks.append(dict(status='passed', rank=rank, physical_card=rank, host=HOST, run_id=run_id,
                          case=cp.CASE, source_sha256=manifest, tests=tests, precollective_metadata=pre,
                          frozen_weights_unchanged=True, strategy_proof=manifest['strategy/ulysses.py'],
                          parallelism_proof=dict(ulysses_world_size=8, ulysses_rank=rank, ring_world_size=1,
                              tensor_parallel_world_size=1, collective_global_ranks=list(range(8)), heads_per_ulysses_rank=7)))
    report = dict(status='passed', case=cp.CASE, host=HOST, run_id=run_id, world_size=8, physical_cards=list(range(8)),
                  dtype='bfloat16', heads=56, head_dim=128, source_sha256=manifest,
                  test_script_sha256='f'*64, rank_results=ranks)
    return report, manifest, run_id


class TinyGateTests(unittest.TestCase):
    def exercise(self, corruption=None):
        import c_profiles as cp
        import c_npu_regression as checker
        report, manifest, run_id = tiny_report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / 'c_validation/01234567'
            run = output / 'runs' / ('20260913T000000Z_' + run_id)
            run.mkdir(parents=True)
            result_path = run / 'c_tiny.json'
            if corruption == 'rank_count': report['rank_results'].pop()
            if corruption == 'tolerance': report['rank_results'][0]['tests'][0]['checks'][0]['atol'] = 9.0
            if corruption == 'wrong_mask': report['rank_results'][0]['tests'][0]['mask_proof'] = {}
            result_path.write_text(json.dumps(report))
            for item in report['rank_results']:
                (run / f"c_tiny.rank{item['rank']}.json").write_text(json.dumps(item))
            if corruption == 'rank_file':
                (run / 'c_tiny.rank0.json').write_text('{}')
            status = dict(case=cp.CASE, phase='completed', host=HOST.copy(), group='01234567',
                          run_id=run_id, run_directory=str(run), allocated_physical_npu_ids=list(range(8)),
                          validation_passed=True, cleanup_completed=True, selected_cards_verified_idle_after_cleanup=True,
                          needs_attention=False, error=None, remaining_owned_process_groups={}, source_sha256=manifest,
                          result_path=str(result_path), result_sha256=gates.profiles.digest(result_path))
            if corruption == 'b_substitute': status['case'] = 'SP8_31731_v1'
            if corruption == 'boot': status['host'] = {**HOST, 'boot_id': 'other'}
            if corruption == 'cleanup': status['cleanup_completed'] = False
            if corruption == 'not_idle': status['selected_cards_verified_idle_after_cleanup'] = False
            if corruption == 'remaining': status['remaining_owned_process_groups'] = {'999': [999]}
            if corruption == 'bad_hash': status['result_sha256'] = '0' * 64
            if corruption == 'outside_run': status['run_directory'] = str(root / 'other')
            if corruption == 'source_drift': status['source_sha256'] = {}
            for path, data in ((run / 'c_validation_status.json', status),
                               (run / 'source_manifest.json', manifest), (run / 'host_identity.json', HOST),
                               (output / 'c_validation_status.json', status)):
                path.write_text(json.dumps(data))
            if corruption == 'latest_drift': (run / 'c_validation_status.json').write_text('{}')
            if corruption == 'missing': (output / 'c_validation_status.json').unlink()
            adapter = SimpleNamespace(Profile=lambda: SimpleNamespace(output_root=output), CARDS=cp.CARDS, CASE=cp.CASE,
                                      require_host=mock.Mock(), source_manifest=lambda: manifest)
            with mock.patch.object(gates, 'c_validation_modules', return_value=(adapter, checker)), \
                 mock.patch.object(gates.profiles, 'INSTALL', root):
                return gates.tiny_gate('01234567', HOST)

    def test_real_schema_complete_c8_and_cleanup_pass(self):
        proof = self.exercise()
        self.assertEqual(proof['result']['world_size'], 8)
        self.assertEqual(len(proof['result']['rank_results']), 8)
        self.assertIn('Real C SP8 synthetic', proof['scope'])

    def test_no_c_proof_no_fallback_to_b(self):
        for corruption in ('missing', 'b_substitute', 'boot', 'cleanup', 'not_idle', 'remaining', 'bad_hash',
                           'outside_run', 'source_drift', 'latest_drift', 'rank_count', 'tolerance', 'wrong_mask', 'rank_file'):
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                self.exercise(corruption)

    def test_c_verifier_pin_mismatch_refuses_import(self):
        with mock.patch.object(gates, 'file_record', return_value={'sha256': 'wrong'}), \
             mock.patch.object(gates.importlib, 'import_module') as imp:
            with self.assertRaises(RuntimeError): gates.c_validation_modules()
        imp.assert_not_called()

    def test_tiny_reader_has_no_torch_import_or_device_calls(self):
        self.assertNotIn('torch', sys.modules)
        self.exercise()
        self.assertNotIn('torch', sys.modules)


class ActualShapeContractTests(unittest.TestCase):
    """Execute unchanged real pure shape functions, not a duplicate formula."""
    def setUp(self):
        self.support = C_ADAPTATION / 'upstream/vllm_omni/diffusion/models/minimax_h3'
        self.model = C_ADAPTATION / 'candidate'
        if not self.support.is_dir(): self.support = gates.VENDOR / 'vllm_omni/diffusion/models/minimax_h3'
        if not self.model.is_dir(): self.model = gates.VENDOR / 'vllm_omni/diffusion/models/minimax_h3'
        spec = importlib.util.spec_from_file_location('_actual_c_time_request_cpu', self.support / 'time_request.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.planner = module.MINIMAX_H3_SHAPE_PLANNER
        self.namespace = {'MINIMAX_H3_FPS': 24, 'MINIMAX_H3_SHAPE_PLANNER': self.planner,
                          'minimax_h3_align_frame_count': module.minimax_h3_align_frame_count}

    def pure_definitions(self, path, names, namespace):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        selected = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, set(names))
        source = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(node) for node in selected)
        exec(compile(source, str(path), 'exec'), namespace)
        return namespace

    def test_actual_resolve_shape_fps24_duration_rounding_latent_and_audio(self):
        path = self.model / 'pipeline_minimax_h3.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fps = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'MINIMAX_H3_FPS' for t in node.targets)]
        self.assertEqual(fps, [24])
        ns = self.pure_definitions(path, ['_resolve_shape'], self.namespace)
        sampling = SimpleNamespace(fps=gates.GENERATION['fps'], extra_args={'duration': gates.GENERATION['duration_seconds']},
                                   num_frames=0, height=gates.GENERATION['height'], width=gates.GENERATION['width'])
        height, width, frames, latent, audio = ns['_resolve_shape'](None, 'ref2va', sampling, None)
        self.assertEqual((height, width, frames, latent, audio), (768, 1344, 124, 37, 207))
        self.assertEqual((latent, height//16, width//16), strict_metadata()['target_shape'])
        self.assertEqual(audio, strict_metadata()['target_audio_t'])
        self.assertEqual(self.planner.frame_count_from_video_latent_t(latent), frames)
        sampling.fps = 30
        with self.assertRaisesRegex(ValueError, 'fixed at 24'):
            ns['_resolve_shape'](None, 'ref2va', sampling, None)
        self.assertNotIn('torch', sys.modules)

    def test_actual_reference_shape_and_ffmpeg_arguments_no_hidden30fps(self):
        path = self.support / 'reference_video.py'
        run = mock.Mock()
        ns = dict(math=math, Path=Path, subprocess=SimpleNamespace(run=run))
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith('MINIMAX_H3_')):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        ns = self.pure_definitions(path, ['_nearest_multiple', '_reference_video_shape', '_transcode_reference_video'], ns)
        self.assertEqual(ns['MINIMAX_H3_FPS'], 24.0)
        for width, height in ((1344, 768), (1280, 720)):
            prepared = ns['_reference_video_shape'](width, height)
            self.assertEqual(prepared, (1344, 768))
        ns['_transcode_reference_video']('/cpu/source.mp4', target_width=1344, target_height=768,
                                        target_frame_count=self.planner.align_frame_count(120), workdir='/cpu/tmp')
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('-vf') + 1], 'fps=24,scale=1344:768:flags=lanczos,setsar=1')
        self.assertEqual(command[command.index('-frames:v') + 1], '124')
        self.assertTrue(run.call_args.kwargs['check'])

    def test_all_shape_reference_vae_and_packing_sources_are_pinned(self):
        for name, expected in gates.SUPPORT_SHA256.items():
            self.assertEqual(gates.profiles.digest(self.support / name), expected)


class StrictMetadataTests(unittest.TestCase):
    def check(self, text, **kwargs):
        return gates.parse_strict_metadata(text, set(range(500, 508)), 'lake_snow', **kwargs)

    def test_actual_python_repr_and_eight_pids_supported(self):
        result = self.check(strict_logs())
        self.assertEqual(set(result['records_by_pid']), set(range(500, 508)))
        self.assertIn('Not a per-layer', result['limitation'])

    def test_two_requests_requires_same_metadata_and_worker_set(self):
        self.check(strict_logs(requests=2), requests=2)
        with self.assertRaises(RuntimeError): self.check(strict_logs(), requests=2)
        with self.assertRaises(RuntimeError):
            self.check(strict_logs(requests=2, mutate=lambda pid, req, row: row.update(text_len=100) if req else None), requests=2)

    def test_absent_old_b_unlabelled_duplicate_and_unknown_pids_reject(self):
        for text in ('', log_text(), strict_logs().replace('H3C pid=', 'H3B pid='),
                     strict_logs(pids=range(500, 507)), strict_logs(pids=range(500, 509)), strict_logs(requests=2)):
            with self.subTest(text=text[:30]), self.assertRaises(RuntimeError): self.check(text)

    def test_strict_fields_and_shape_temporal_alignment_failclosed(self):
        for key, value in (('mode', 'B'), ('reference_count', 2), ('reference_count', True),
                           ('reference_kind', 'image'), ('patch_size', (1, 1, 1)), ('source_shape', (36, 48, 84)),
                           ('source_shape', (37, 47, 84)), ('source_shape', (True, 48, 84)),
                           ('target_shape', (37, 48, 82)), ('target_audio_t', 200), ('text_len', 0),
                           ('source_audio_t', -1), ('extra', 0)):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.check(strict_logs(mutate=lambda pid, req, row: row.update({key: value})))

    def test_source_spatial_shape_may_differ_and_audio_recorded_not_guessed(self):
        self.check(strict_logs(mutate=lambda pid, req, row: row.update(source_shape=(37, 44, 80), source_audio_t=207)))
        with self.assertRaises(RuntimeError):
            gates.parse_strict_metadata(strict_logs(mutate=lambda pid, req, row: row.update(source_audio_t=207)),
                                        set(range(500, 508)), 'explicit_reverse_couple_124')

    def test_malicious_or_oversized_literals_never_execute(self):
        prefix = f'H3C pid=500 INFO {gates.STRICT_MARKER}'
        for raw in ("{'mode': __import__('os').system('echo NO')}", '{' + ' ' * 5000 + '}', '{bad}'):
            with self.assertRaises(RuntimeError): self.check(prefix + raw)


class StrictProgressPrefixTests(unittest.TestCase):
    """Reconstruct the observed eight-worker smoke metadata, not a new NPU proof.

    Real record ordering, PIDs and metadata are retained; the unimportant log
    header is a fixture. The first observed record had one leading ASCII dot.
    """
    PIDS = set(range(120387, 120395))

    def observed_lines(self, dots=1):
        row = strict_metadata()
        row.update(text_len=6167, source_audio_t=209)
        lines = [f'H3C pid={pid} 2026-09-13 INFO vllm_omni.pipeline {gates.STRICT_MARKER}{row!r}'
                 for pid in (120394, *range(120387, 120394))]
        lines[0] = '.' * dots + lines[0]
        return lines

    def check(self, lines, *, requests=1):
        return gates.parse_strict_metadata('\n'.join(lines), self.PIDS, 'lake_snow', requests=requests)

    def test_observed_eight_worker_smoke_with_single_dot(self):
        result = self.check(self.observed_lines())
        self.assertEqual(set(result['records_by_pid']), self.PIDS)
        self.assertEqual(result['request_count_per_pid'], 1)
        for rows in result['records_by_pid'].values():
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['text_len'], 6167)
            self.assertEqual(rows[0]['source_audio_t'], 209)
            self.assertEqual(rows[0]['target_audio_t'], 207)

    def test_bounded_ascii_progress_dots_preserve_exact_evidence(self):
        expected = self.check(self.observed_lines(0))
        for dots in (0, 1, 2, 64):
            with self.subTest(dots=dots):
                self.assertEqual(self.check(self.observed_lines(dots)), expected)

    def test_bounded_progress_prefix_allows_two_exact_requests(self):
        result = self.check(self.observed_lines(64) + self.observed_lines(2), requests=2)
        self.assertEqual(result['request_count_per_pid'], 2)
        self.assertTrue(all(len(rows) == 2 for rows in result['records_by_pid'].values()))

    def test_excessive_or_unknown_prefix_rejected(self):
        for prefix in ('.' * 65, 'X', 'prefix.', ' ', '. ', '\t', '\u2026', '\uff0e', '\x1b[0m'):
            lines = self.observed_lines(0)
            lines[0] = prefix + lines[0]
            with self.subTest(prefix=repr(prefix)), self.assertRaises(RuntimeError):
                self.check(lines)

    def test_progress_prefix_unknown_pid_rejected(self):
        lines = self.observed_lines()
        lines[0] = lines[0].replace('pid=120394 ', 'pid=999999 ')
        with self.assertRaises(RuntimeError):
            self.check(lines)

    def test_progress_prefix_wrong_logger_label_rejected(self):
        lines = self.observed_lines()
        lines[0] = lines[0].replace('.H3C pid=', '.H3B pid=')
        with self.assertRaises(RuntimeError):
            self.check(lines)

    def test_progress_prefix_duplicate_pid_rejected(self):
        lines = self.observed_lines()
        lines[1] = lines[1].replace('pid=120387 ', 'pid=120394 ')
        with self.assertRaises(RuntimeError):
            self.check(lines)

    def test_progress_prefix_missing_worker_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check(self.observed_lines()[:-1])

    def test_progress_prefix_metadata_mismatch_rejected(self):
        for old, new in (("'text_len': 6167", "'text_len': 6168"),
                         ("'source_audio_t': 209", "'source_audio_t': 210"),
                         ("'target_audio_t': 207", "'target_audio_t': 208")):
            lines = self.observed_lines()
            self.assertIn(old, lines[0])
            lines[0] = lines[0].replace(old, new)
            with self.subTest(field=old), self.assertRaises(RuntimeError):
                self.check(lines)

    def test_progress_prefix_missing_or_excess_requests_rejected(self):
        for copies, requests in ((2, 1), (1, 2), (3, 2)):
            with self.subTest(copies=copies, requests=requests), self.assertRaises(RuntimeError):
                self.check(self.observed_lines() * copies, requests=requests)

    def test_progress_prefix_malformed_oversized_or_suffixed_record_rejected(self):
        for raw in ('{bad}', '{' + ' ' * 4095 + '}',
                    "{'mode': __import__('os').system('echo NO')}"):
            lines = self.observed_lines()
            lines[0] = lines[0].split(gates.STRICT_MARKER)[0] + gates.STRICT_MARKER + raw
            with self.subTest(raw=raw[:40]), self.assertRaises(RuntimeError):
                self.check(lines)
        lines = self.observed_lines()
        lines[0] += ' trailing text'
        with self.assertRaises(RuntimeError):
            self.check(lines)


if __name__ == '__main__':
    unittest.main(verbosity=2)
