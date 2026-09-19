"""CPU-only fixed-sample A gate/workflow tests; no remote or NPU execution."""
import ast
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
sys.path.insert(0, str(HERE))
if os.name == 'nt':
    sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import trial_gates as gates
import run_a_trial as trial

HOST = {'hostname': gates.profiles.EXPECTED_HOST, 'boot_id': '12345678-abcd-abcd-abcd-123456789abc', 'machine': 'aarch64'}
RUN = 'b' * 32


def sample():
    selected = gates.sample_profile('explicit_reverse_couple_124')
    return {'schema_version': 1, 'sample_id': selected.sample_id, 'source_video': str(selected.source),
            'source_sha256': gates.REVERSE_SHA256, 'edit_prompt': selected.prompt, 'requested_generation': gates.GENERATION.copy(),
            'source_metadata': {'width': 1280, 'height': 720, 'frames': 124, 'fps': 24,
                                'duration_seconds': 124/24, 'has_audio': False},
            'provenance': {'start_frame': 104, 'end_frame_exclusive': 228, 'original_fps': 24,
                           'official_source_id': 'Te_Temporal_reordering_04',
                           'original_sha256': 'a18b5f2dcabe5a97d30350de9e402039a7e6c0886442162b164887698d75d638',
                           'instruction_is_self_authored': True, 'official_paired_target_available': False,
                           'gt_target_generated': False},
            'router_control': {'temporal_change_required': True, 'text_only_should_identify_temporal_edit': True,
                               'vlm_necessity_evidence': False}}


def probe_data(width=1344, height=768, fps='24/1', frames='124', duration=str(124/24)):
    return {'streams': [{'codec_type': 'video', 'width': width, 'height': height, 'r_frame_rate': fps,
                          'nb_read_frames': frames, 'nb_frames': frames}], 'format': {'duration': duration}}


def smi(busy=False):
    result = []
    for card in range(8):
        result.extend([f'| {card} 910B3 | OK | 0 / 0 |', '| 0 | bus | 0 / 0 3402 / 65536 |'])
    result.append('| NPU Chip | Process id | Process name | Process memory(MB) |')
    for card in range(8):
        result.append(f'| {card} 0 | {500+card} | Worker | 3402 |' if busy
                      else f'| No running processes found in NPU {card} |')
    return '\n'.join(result)


class FixedProfileTests(unittest.TestCase):
    def test_reuses_card_mutex_without_mutating_tiny_profile(self):
        for group in ('01234567',):
            before = gates.profiles.profile(group)
            selected = gates.selected_profile(group)
            self.assertEqual(before, gates.profiles.profile(group))
            self.assertEqual(before.cards, selected.cards)
            self.assertEqual(before.lease_root, selected.lease_root)
            self.assertNotEqual(before.code_root, selected.code_root)
            self.assertNotEqual(before.output_root, selected.output_root)
            self.assertEqual(selected.master_port, gates.HTTP_PORTS[group])
        self.assertNotEqual(gates.selected_profile('01234567').output_root,
                            gates.selected_profile('01234567', 'explicit_reverse_couple_124').output_root)
        for group in ('all8', '0123', '4567', '2367'):
            with self.assertRaises(ValueError):
                gates.selected_profile(group)

    def test_fixed_sampler_and_parallelism(self):
        params = gates.parallelism(gates.selected_profile('01234567'))
        self.assertEqual((params['num_gpus'], params['usp'], params['ring'], params['dit_tensor_parallel_size'],
                          params['text_encoder_tp_size'], params['vae_patch_parallel_size']), (8, 8, 1, 1, 8, 8))
        self.assertEqual(gates.GENERATION, {'width': 1344, 'height': 768, 'fps': 24, 'duration_seconds': 5.0,
                                          'seed': 4101, 'num_inference_steps': 50, 'flow_shift': 12.0, 'audio_flow_shift': 3.0})
        self.assertEqual(gates.MIN_RAM, 600 * 1024**3)
        self.assertEqual(gates.MIN_HBM_MIB, 55 * 1024)

    def test_original_pipeline_pins_match_reviewed_local_sources(self):
        for name, expected in gates.ORIGINAL_HASHES.items():
            original = HERE.parents[1] / 'b_adaptation/upstream' / name
            self.assertEqual(gates.profiles.digest(original), expected)

    def test_original_selection_after_env_and_no_candidate_execution(self):
        script = (HERE / 'launch_a_trial.sh').read_text()
        self.assertLess(script.index('source /cache/zhonghao/h3/env_h3_31731.sh'), script.index('export PYTHONPATH='))
        self.assertNotIn('candidates/b_v1', script)
        self.assertIn('export ZHONGHAO_H3_OPENVDN=0', script)
        self.assertIn('--init-timeout 2400 --stage-init-timeout 2400', script)
        self.assertNotIn('--dtype', script)
        self.assertNotIn('/cache/yunfeng', script)
        self.assertIn('--num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8', script)
        self.assertIn('--vae-patch-parallel-size 8', script)
        self.assertNotIn('--text-encoder-tp-size 4', script)
        self.assertNotIn('--vae-patch-parallel-size 4', script)
        self.assertIn('${PYTHONPATH:+:${PYTHONPATH}}', script)  # Retain bootstrap's private CANN tail.

    def test_sample_default_and_frozen_choices(self):
        from dataclasses import FrozenInstanceError
        lake = gates.sample_profile()
        reverse = gates.sample_profile('explicit_reverse_couple_124')
        self.assertEqual(lake.sample_id, 'lake_snow')
        self.assertIsNone(lake.manifest_path)
        self.assertNotEqual(lake.source, reverse.source)
        self.assertNotEqual(lake.prompt, reverse.prompt)
        with self.assertRaises(FrozenInstanceError):
            lake.prompt = reverse.prompt
        for name in ('../lake_snow', 'custom', ''):
            with self.assertRaises(ValueError):
                gates.sample_profile(name)

    def test_private_ffmpeg_wrapper_and_binary_are_frozen(self):
        def record(path):
            sha = gates.ORIGINAL_HASHES.get(path.name, 'a' * 64)
            return {'path': str(path), 'sha256': sha, 'size_bytes': 10}
        with mock.patch.object(gates, 'regular', side_effect=lambda path: path), \
             mock.patch.object(gates.os, 'access', return_value=True), \
             mock.patch.object(gates, 'file_record', side_effect=record):
            result = gates.source_code_manifest()
        self.assertEqual(result['media/ffmpeg_wrapper']['path'], str(gates.MEDIA_FFMPEG))
        self.assertEqual(result['media/ffmpeg_binary']['path'], str(gates.MEDIA_FFMPEG_BINARY))
        wrapper = (HERE.parent / 'media_bin/ffmpeg').read_text()
        self.assertIn('exec ' + str(gates.MEDIA_FFMPEG_BINARY).replace('\\', '/') + ' "$@"', wrapper)
        self.assertNotIn('/usr/local/ffmpeg', wrapper)

    def test_missing_or_nonexecutable_encoder_fails_before_model(self):
        with mock.patch.object(gates, 'regular', side_effect=lambda path: path), \
             mock.patch.object(gates.os, 'access', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'not executable'):
                gates.source_code_manifest()


class SampleTests(unittest.TestCase):
    def gate(self, record, *, source_sha=gates.REVERSE_SHA256, width=1280, audio=False):
        probe = {'verified': True, 'checks': {'dimensions': [width, 720], 'container_duration_seconds': 124/24},
                 'metadata': {'streams': [{'codec_type': 'video'}] + ([{'codec_type': 'audio'}] if audio else [])}}
        with mock.patch.object(gates, 'json_file', return_value=record), \
             mock.patch.object(gates, 'file_record', return_value={'sha256': source_sha}), \
             mock.patch.object(gates, 'video_probe', return_value=probe):
            return gates.sample_gate('explicit_reverse_couple_124')

    def test_correct_record_preserved(self):
        value = sample()
        self.assertEqual(self.gate(value)['manifest'], value)

    def test_source_prompt_generation_and_provenance_changes_rejected(self):
        for key, value in (('source_video', '/another/source.mp4'), ('edit_prompt', 'make snow'), ('schema_version', 2),
                           ('requested_generation', {**gates.GENERATION, 'seed': 1})):
            record = sample(); record[key] = value
            with self.assertRaises(RuntimeError):
                self.gate(record)
        for key, value in (('start_frame', 0), ('end_frame_exclusive', 124), ('official_paired_target_available', True)):
            record = sample(); record['provenance'][key] = value
            with self.assertRaises(RuntimeError):
                self.gate(record)

    def test_source_hash_metadata_audio_and_vlm_claim_tamper_rejected(self):
        for kwargs in ({'source_sha': 'changed'}, {'width': 1920}, {'audio': True}):
            with self.assertRaises(RuntimeError):
                self.gate(sample(), **kwargs)
        record = sample(); record['router_control']['vlm_necessity_evidence'] = True
        with self.assertRaises(RuntimeError):
            self.gate(record)


class LakeSampleTests(unittest.TestCase):
    def gate(self, *, entry_change=None, source_change=None, missing=False):
        selected = gates.sample_profile('lake_snow')
        source = {'path': str(selected.source), 'sha256': gates.LAKE_SHA256, 'size_bytes': gates.LAKE_SIZE_BYTES}
        entry = {'kind': 'file', 'size': gates.LAKE_SIZE_BYTES, 'sha256': gates.LAKE_SHA256}
        source.update(source_change or {})
        entry.update(entry_change or {})
        transfer = {'verified': {} if missing else {'data/minimax_h3_t2va_50step.mp4': entry},
                    'verified_record': {'path': str(gates.ROOT / 'transfer_verified.json'), 'sha256': 'transfer-sha'}}
        probe = {'verified': True, 'checks': {'dimensions': [1344, 768], 'fps': 24, 'frame_count': 124,
                                             'container_duration_seconds': 5.207},
                 'metadata': probe_data(duration='5.207')}
        with mock.patch.object(gates, 'file_record', return_value=source) as file_record, \
             mock.patch.object(gates, 'json_file') as json_file, \
             mock.patch.object(gates, 'video_probe', return_value=probe) as video_probe:
            result = gates.sample_gate(transfer=transfer)
            json_file.assert_not_called()  # The lake has no derived preparation manifest.
            file_record.assert_called_once_with(selected.source)
            video_probe.assert_called_once_with(selected.source, target=True)
            return result

    def test_default_lake_freezes_original_source_and_prompt_without_derived_metadata(self):
        result = self.gate()
        manifest = result['manifest']
        self.assertEqual(manifest['sample_id'], 'lake_snow')
        self.assertEqual(manifest['source_sha256'], gates.LAKE_SHA256)
        self.assertEqual(manifest['edit_prompt'], gates.LAKE_PROMPT)
        self.assertIsNone(result['manifest_record'])
        self.assertFalse(manifest['provenance']['source_cropped_or_reencoded'])
        self.assertNotIn('start_frame', manifest['provenance'])
        self.assertFalse(manifest['router_control']['vlm_necessity_evidence'])

    def test_lake_requires_exact_source_size_hash_and_transfer_entry(self):
        for kwargs in ({'missing': True}, {'entry_change': {'kind': 'symlink'}},
                       {'entry_change': {'sha256': 'wrong'}}, {'entry_change': {'size': 3}},
                       {'source_change': {'sha256': gates.REVERSE_SHA256}},
                       {'source_change': {'size_bytes': 3}},
                       {'entry_change': {'sha256': 'f' * 64}, 'source_change': {'sha256': 'f' * 64}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.gate(**kwargs)
        with self.assertRaises(RuntimeError):
            gates.sample_gate()

    def test_original_A_prompt_is_preserved(self):
        original = json.loads((HERE.parents[1] / 'experiment.json').read_text())
        self.assertEqual(gates.LAKE_PROMPT, original['edit_prompt'])
        self.assertIn('minimax_h3_t2va_50step.mp4', str(gates.sample_profile().source))


class TransferRuntimeTests(unittest.TestCase):
    def transfer(self, **changes):
        verified = {name: {} for name in ('env.tar', 'src/vllm/x', 'src/vllm-ascend/x', 'src/vllm-omni/x',
                    'models/MiniMax-H3/Ref2VA/x', 'models/OpenVDN-vdn-minimax-h3/stage-b-step-2000/x')}
        status = {'phase': 'transfer_completed_extraction_and_validation_required', 'deployment': gates.DEPLOYMENT,
                  'role': 'consume', 'hostname': HOST['hostname'], 'verified_entries': len(verified), **changes}
        with mock.patch.object(gates, 'json_file', side_effect=[status, verified]), \
             mock.patch.object(gates, 'file_record', return_value={'sha256': 'proof'}):
            return gates.transfer_gate(HOST)

    def test_only_completed_current_transfer_accepted(self):
        self.assertEqual(self.transfer()['status']['verified_entries'], 6)
        for changes in ({'phase': 'copying_from_shared_staging'}, {'deployment': 'old-v1'},
                        {'hostname': 'source-node'}, {'verified_entries': 3}, {'role': 'produce'}):
            with self.assertRaises(RuntimeError):
                self.transfer(**changes)

    def runtime(self, mutate=None):
        sources = {'runtime/env_h3_31731.sh': {'sha256': 'env-sha'},
                   **{f'original/{name}': {'sha256': sha} for name, sha in gates.ORIGINAL_HASHES.items()}}
        report = {'status': 'passed_cpu_runtime_checks_only', 'host': HOST, 'npu_inference_verified': False,
                  'env_script_sha256': 'env-sha', 'source_sha256': gates.ORIGINAL_HASHES.copy(),
                  'device_guard': {'python_npu_initialized': False, 'device_operations_requested': 0,
                                   'npu_lazy_init_and_c_init_guarded': True}}
        if mutate:
            mutate(report)
        with mock.patch.object(gates, 'json_file', return_value=report), mock.patch.object(gates, 'file_record', return_value={}):
            return gates.runtime_gate(HOST, sources)

    def test_runtime_same_host_env_original_source_required(self):
        self.assertEqual(self.runtime()['report']['host'], HOST)
        for key, value in (('status', 'passed'), ('host', {**HOST, 'boot_id': 'old'}), ('env_script_sha256', 'old'),
                           ('source_sha256', {}), ('npu_inference_verified', True)):
            with self.assertRaises(RuntimeError):
                self.runtime(lambda report: report.update({key: value}))
        with self.assertRaises(RuntimeError):
            self.runtime(lambda report: report['device_guard'].update(device_operations_requested=1))

    def test_tiny_gate_demands_own_group_cleanup_and_current_hashes(self):
        selected = gates.profiles.profile('01234567')
        run = selected.output_root / 'runs' / ('time_' + RUN)
        status = {'phase': 'completed', 'host': HOST, 'group': '01234567', 'allocated_physical_npu_ids': list(range(8)),
                  'validation_passed': True, 'cleanup_completed': True, 'selected_cards_verified_idle_after_cleanup': True,
                  'needs_attention': False, 'run_directory': str(run), 'result_path': str(run / 'npu_wrapper.json'),
                  'result_sha256': 'result-sha', 'source_sha256': {'candidate': 'current'}, 'run_id': RUN}
        def execute(record):
            with mock.patch.object(gates, 'json_file', return_value=record), mock.patch.object(gates, 'regular'), \
                 mock.patch.object(gates.profiles, 'canonical_private'), mock.patch.object(gates.profiles, 'digest', return_value='result-sha'), \
                 mock.patch.object(gates.profiles, 'source_manifest', return_value={'candidate': 'current'}), \
                 mock.patch.object(gates.tiny_validation, 'verify_results', return_value={'status': 'passed'}) as verify, \
                 mock.patch.object(gates, 'file_record', return_value={}):
                result = gates.tiny_gate('01234567', HOST)
                verify.assert_called_once()
                return result
        self.assertEqual(execute(status)['result']['status'], 'passed')
        for changes in ({'group': '4567'}, {'allocated_physical_npu_ids': [0, 1, 2, 3]},
                        {'host': {**HOST, 'boot_id': 'old'}}, {'cleanup_completed': False},
                        {'source_sha256': {'candidate': 'old'}}, {'result_sha256': 'old'}, {'needs_attention': True}):
            with self.assertRaises(RuntimeError):
                execute({**status, **changes})


class VideoAndWeightTests(unittest.TestCase):
    def probe(self, data, target=True):
        with mock.patch.object(gates, 'regular'), mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(gates.os, 'access', return_value=True), \
             mock.patch.object(gates.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(data))) as run:
            result = gates.video_probe(Path('toy.mp4'), target=target)
            self.assertIn('-count_frames', run.call_args.args[0])
            return result

    def test_decoded_124_frame_video_required(self):
        self.assertTrue(self.probe(probe_data())['verified'])
        self.assertTrue(self.probe(probe_data(width=1280, height=720), target=False)['verified'])
        for data in (probe_data(frames='120'), probe_data(fps='25/1'), probe_data(width=1280), probe_data(duration='nan')):
            with self.assertRaises(RuntimeError):
                self.probe(data)

    def test_unavailable_ffprobe_fails_closed(self):
        with mock.patch.object(gates, 'regular'), mock.patch.object(Path, 'is_file', return_value=False):
            with self.assertRaises(RuntimeError):
                gates.video_probe(Path('toy.mp4'))

    def test_weight_identity_headers_and_small_config_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = root / 'models/MiniMax-H3/Ref2VA'
            (ref / 'transformer').mkdir(parents=True)
            mapping = {f'tensor_{i}': f'shard_{i % 13}.safetensors' for i in range(535)}
            (ref / 'transformer/model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
            for index in range(13):
                raw = json.dumps({'toy': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
                (ref / f'transformer/shard_{index}.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + b'00')
            verified = {}
            for path in ref.rglob('*'):
                if path.is_file():
                    verified[path.relative_to(root).as_posix()] = {'kind': 'file', 'size': path.stat().st_size, 'sha256': gates.profiles.digest(path)}
            with mock.patch.object(gates, 'ROOT', root), mock.patch.object(gates, 'REF2VA', ref), \
                 mock.patch.object(gates.profiles, 'INSTALL', root):
                manifest = gates.weight_manifest(verified)
                self.assertEqual(len(manifest['files']), 14)
                self.assertEqual(manifest['files']['transformer/shard_0.safetensors']['tensor_dtypes'], ['BF16'])
                original = copy.deepcopy(verified)
                name = 'models/MiniMax-H3/Ref2VA/transformer/model.safetensors.index.json'
                verified[name]['sha256'] = 'changed'
                with self.assertRaises(RuntimeError):
                    gates.weight_manifest(verified)
                verified = original
                verified['models/MiniMax-H3/Ref2VA/transformer/shard_0.safetensors']['size'] += 1
                with self.assertRaises(RuntimeError):
                    gates.weight_manifest(verified)


class WorkflowTests(unittest.TestCase):
    def owner(self, sample_id=gates.DEFAULT_SAMPLE):
        with mock.patch.object(gates.profiles, 'require_host', return_value=HOST), mock.patch.object(trial.base, 'proc_identity', return_value=None):
            return trial.ATrialSupervisor('01234567', sample_id)

    def test_no_npu_without_allow_and_no_arbitrary_source_cli(self):
        for args in (['--group', '01234567'], ['--group', '01234567', '--allow-npu', '--source', 'elsewhere'],
                     ['--group', '01234567', '--sample', 'custom', '--allow-npu'],
                     ['--group', '0123', '--allow-npu']):
            with mock.patch.object(trial, 'ATrialSupervisor') as cls, mock.patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    trial.main(args)
                cls.assert_not_called()

    def test_single_service_smoke_then_one_formal_and_no_repeat(self):
        owner = self.owner(); calls = []
        owner.request = lambda name, steps: calls.append((name, steps)) or {'success': True, 'ffprobe': {'verified': True}}
        owner.smoke_gate = lambda: calls.append('gate')
        owner.assert_unchanged = lambda: calls.append('unchanged')
        owner.update = lambda _phase, **kw: owner.status.update(kw)
        owner.run_requests()
        self.assertEqual(calls, [('smoke_2step', 2), 'gate', ('a_50step', 50), 'unchanged'])
        self.assertEqual(owner.status['formal_request_attempts'], 1)
        self.assertTrue(owner.status['formal_50step_completed'])
        with self.assertRaises(RuntimeError):
            owner.run_requests()

    def test_failed_smoke_or_gate_never_reaches_formal(self):
        for fail in ('smoke', 'gate'):
            owner = self.owner()
            owner.request = mock.Mock(side_effect=RuntimeError('smoke') if fail == 'smoke' else None)
            owner.smoke_gate = mock.Mock(side_effect=RuntimeError('gate'))
            with self.assertRaises(RuntimeError):
                owner.run_requests()
            owner.request.assert_called_once_with('smoke_2step', 2)
            self.assertEqual(owner.status['formal_request_attempts'], 0)

    def test_direct_formal_wrong_order_and_retry_rejected(self):
        owner = self.owner()
        with self.assertRaises(RuntimeError):
            owner.request('a_50step', 50)
        owner._request_order = [('smoke_2step', 2)]
        with self.assertRaises(RuntimeError):
            owner.request('a_50step', 50)
        owner._request_order.append(('a_50step', 50))
        with self.assertRaises(RuntimeError):
            owner.request('smoke_2step', 2)

    def test_failed_preflight_cleanup_release_no_service(self):
        fake = mock.Mock(status={})
        fake.preflight.side_effect = RuntimeError('migration incomplete')
        fake.cleanup.return_value = True
        with mock.patch.object(trial, 'ATrialSupervisor', return_value=fake), mock.patch.object(trial.signal, 'signal'):
            self.assertEqual(trial.main(['--group', '01234567', '--allow-npu']), 1)
        fake.launch.assert_not_called(); fake.run_requests.assert_not_called()
        fake.cleanup.assert_called_once(); fake.release_locks.assert_called_once()

    def test_cleanup_exception_still_closes_leases(self):
        fake = mock.Mock(status={})
        fake.cleanup.side_effect = RuntimeError('cleanup')
        with mock.patch.object(trial, 'ATrialSupervisor', return_value=fake), mock.patch.object(trial.signal, 'signal'):
            with self.assertRaises(RuntimeError):
                trial.main(['--group', '01234567', '--allow-npu'])
        fake.release_locks.assert_called_once()

    def test_environment_removes_b_activation_and_keeps_identity(self):
        owner = self.owner(); owner.run_dir = Path('/private/run')
        with mock.patch.dict(os.environ, {'ZHONGHAO_H3_OPENVDN': '1', 'ZHONGHAO_H3_OPENVDN_CHECKPOINT': '/shared/b',
                                         'PYTHONPATH': '/candidate', 'VLLM_LOGGING_CONFIG_PATH': '/other.json'}):
            env = owner.env()
        self.assertEqual(env['ZHONGHAO_H3_OPENVDN'], '0')
        self.assertNotIn('ZHONGHAO_H3_OPENVDN_CHECKPOINT', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('VLLM_LOGGING_CONFIG_PATH', env)
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertEqual(env['H3_TRIAL_SAMPLE'], 'lake_snow')
        self.assertIn(owner.run_id, env['H3_GROUP_ID'])

    def test_frozen_evidence_changed_fails(self):
        owner = self.owner(); owner.frozen = {'source': 'old'}
        owner.evidence_snapshot = lambda: {'source': 'new'}
        with self.assertRaises(RuntimeError):
            owner.assert_unchanged()

    def test_selected_worker_map_handles_real_process_names(self):
        self.assertEqual(trial.selected_workers(smi(busy=True), tuple(range(8))), {card: 500+card for card in range(8)})
        for text in (smi(), smi(busy=True) + '\n| 4 0 | 999 | VLLM::Worker | 10 |'):
            with self.assertRaises(RuntimeError):
                trial.selected_workers(text, tuple(range(8)))

    def exercise_smoke_gate(self, corruption=None, sample_id=gates.DEFAULT_SAMPLE):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            owner = self.owner(sample_id); owner.run_dir = root
            owner.server = mock.Mock(pid=100, poll=mock.Mock(return_value=None))
            owner.locks = [SimpleNamespace(closed=False) for _ in range(9)]
            owner._request_order = [('smoke_2step', 2)]
            owner.assert_unchanged = mock.Mock()
            owner.owned_group_members = mock.Mock(return_value=list(range(500, 508)))
            owner.frozen = {'sample': {'source': {'sha256': 'source-sha'}}}
            owner.update = lambda _phase, **values: owner.status.update(values)
            server = {'pid': 100, 'start_ticks': 1000, 'pgrp': 100, 'session': 100, 'state': 'S'}
            owner.status['server_proc_identity'] = {**server, 'state': 'R'}
            (root / 'smoke_2step').mkdir(); (root / 'output').mkdir()
            output = root / 'output/smoke_2step.mp4'; output.write_bytes(b'CPU toy video')
            actual_probe = {'verified': True, 'checks': {'dimensions': [1344, 768], 'fps': 24, 'frame_count': 124}}
            result = {'success': True, 'requested_steps': 2, 'http_code': '200', 'curl_returncode': 0,
                      'ffprobe': copy.deepcopy(actual_probe), 'output_video': str(output),
                      'output_sha256': gates.profiles.digest(output)}
            fields = {key: gates.GENERATION[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
            fields.update(prompt=owner.sample.prompt, num_inference_steps=2, extra_params=json.dumps({
                'task': 'ref2va', 'duration': gates.GENERATION['duration_seconds'],
                'audio_flow_shift': gates.GENERATION['audio_flow_shift']}))
            request = {'host': owner.host, 'group': '01234567', 'run_id': owner.run_id,
                       'sample_id': owner.sample.sample_id,
                       'source_sha256': 'source-sha', 'source_video': str(owner.sample.source), 'fields': fields,
                       'requested_steps': 2, 'server_proc_identity': {**server, 'state': 'R'}}
            if corruption == 'old_server':
                server['start_ticks'] += 1
            elif corruption == 'foreign_workers':
                owner.owned_group_members.return_value = [504, 505, 506]
            elif corruption == 'source':
                request['source_sha256'] = 'other-source'
            elif corruption == 'sample':
                request['sample_id'] = 'explicit_reverse_couple_124'
            elif corruption == 'fields':
                request['fields']['seed'] = 99
            elif corruption == 'metadata':
                result['ffprobe']['checks']['frame_count'] = 120
            elif corruption == 'output':
                output.write_bytes(b'changed output')
            elif corruption == 'locks':
                owner.locks.pop()
            elif corruption == 'steps':
                request['requested_steps'] = 50
            owner.status['smoke_2step'] = copy.deepcopy(result)
            (root / 'smoke_2step/result.json').write_text(json.dumps(result))
            (root / 'smoke_2step/request.json').write_text(json.dumps(request))
            def identity(pid):
                return server.copy() if pid == 100 else {'pid': pid, 'pgrp': 100, 'session': 100, 'start_ticks': 2000+pid, 'state': 'S'}
            with mock.patch.object(gates.profiles, 'INSTALL', root), \
                 mock.patch.object(gates, 'video_probe', return_value=actual_probe), \
                 mock.patch.object(trial.base, 'proc_identity', side_effect=identity), \
                 mock.patch.object(trial.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=smi(busy=True), stderr='')), \
                 mock.patch.object(trial.base.os, 'O_NOFOLLOW', 0, create=True):
                owner.smoke_gate()
            return owner.status

    def test_complete_same_service_smoke_gate_accepts_own_workers(self):
        status = self.exercise_smoke_gate()
        self.assertTrue(status['formal_smoke_gate_passed'])
        self.assertEqual(set(status['formal_smoke_gate']['worker_identities_by_card']), set(map(str, range(8))))
        self.assertEqual(status['formal_smoke_gate']['sample_id'], 'lake_snow')

    def test_explicit_reverse_has_its_own_same_service_gate(self):
        status = self.exercise_smoke_gate(sample_id='explicit_reverse_couple_124')
        self.assertTrue(status['formal_smoke_gate_passed'])
        self.assertEqual(status['formal_smoke_gate']['sample_id'], 'explicit_reverse_couple_124')

    def test_smoke_gate_blocks_changed_identity_metadata_source_and_workers(self):
        for corruption in ('old_server', 'foreign_workers', 'source', 'sample', 'fields', 'metadata', 'output', 'locks', 'steps'):
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                self.exercise_smoke_gate(corruption)

    def test_source_preparation_schema_prompt_consistency(self):
        tree = ast.parse((HERE.parent / 'prepare_explicit_reverse_couple.py').read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'PROMPT' for target in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value), gates.REVERSE_PROMPT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
