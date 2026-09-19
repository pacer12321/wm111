"""Standard-library CPU fixtures only. No torch import, service, SSH or NPU."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.name == 'nt':
    sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import serial_ab_queue as queue

HOST = {'hostname': 'ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0',
        'boot_id': '12345678-abcd-abcd-abcd-123456789abc', 'machine': 'aarch64'}
GENERATION = {'width': 1344, 'height': 768, 'fps': 24, 'duration_seconds': 5.0,
              'seed': 4101, 'num_inference_steps': 50, 'flow_shift': 12.0, 'audio_flow_shift': 3.0}
PROMPT = ('Transform the reference lake video into a photorealistic winter snow scene. '
          'Preserve the original camera movement, framing, shot timing, lake shoreline, mountains, and overall composition. '
          'Cover the landscape with fresh snow, freeze parts of the lake, add gentle falling snow, '
          'and keep synchronized natural winter ambience.')
LAKE_SHA = 'e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9'
B_ID = 'b' * 32
A_IDENTITY = {'pid': queue.PREDECESSOR_PID, 'start_ticks': queue.PREDECESSOR_START_TICKS,
              'pgrp': 4088644, 'session': 4088644, 'state': 'S'}
B_IDENTITY = {'pid': 9911, 'start_ticks': 998877, 'pgrp': 9911, 'session': 9911, 'state': 'S'}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parallelism(port):
    return {'num_gpus': 8, 'usp': 8, 'ring': 1, 'dit_tensor_parallel_size': 1,
            'text_encoder_tp_size': 8, 'layerwise_offload': True, 'vae_use_tiling': True,
            'vae_parallel_mode': 'tile', 'vae_patch_parallel_size': 8,
            'attention_backend': 'FLASH_ATTN', 'listen_host': '127.0.0.1', 'listen_port': port,
            'dtype_cli_override': None, 'hccl_internal_ports': 'Unchanged original vLLM automatic allocation'}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.a_runs = root / 'model_trial/01234567/lake_snow/A/runs'
        self.a_path = self.a_runs / queue.PREDECESSOR_STATUS.parent.name / 'a_status.json'
        self.b_output = root / 'b_model_trial/01234567/lake_snow/B'
        self.b_path = self.b_output / 'runs' / ('b_20260913T142602Z_' + B_ID) / 'b_status.json'
        self.source = root / 'data/minimax_h3_t2va_50step.mp4'
        self.probe = {'verified': True, 'checks': {'dimensions': [1344, 768], 'fps': 24,
                                                'frame_count': 124, 'container_duration_seconds': 5.207}}
        self.common = {'transfer': {'verified': {'fixture_weight': {'size': 2, 'sha256': 'a' * 64}}},
                       'runtime': {'report': {'status': 'passed_cpu_runtime_checks_only', 'host': HOST}},
                       'tiny': {'result': {'status': 'passed', 'physical_cards': list(range(8))}},
                       'sample': {'manifest': {'edit_prompt': PROMPT, 'source_video': str(self.source)},
                                  'source': {'path': str(self.source), 'sha256': LAKE_SHA}}}
        self.sources = {'A': {'original/pipeline_minimax_h3.py': {'sha256': 'original'}},
                        'B': {'original/pipeline_minimax_h3.py': {'sha256': 'original'},
                              'b_candidate/openvdn_checkpoint.py': {'sha256': 'candidate'}}}
        self.weights = {'A': {'files': {'base': {'header_sha256': 'a', 'tensor_count': 535}}},
                        'B': {'ref2va': {'files': {'base': {'tensor_count': 535}}},
                              'branch': {'tensor_count': 800}, 'lora': {'tensor_count': 416}}}
        def helper(case):
            return SimpleNamespace(source_code_manifest=lambda: copy.deepcopy(self.sources[case]),
                transfer_gate=lambda host: copy.deepcopy(self.common['transfer']),
                runtime_gate=lambda host, sources: copy.deepcopy(self.common['runtime']),
                tiny_gate=lambda group, host: copy.deepcopy(self.common['tiny']),
                sample_gate=lambda sample, transfer: copy.deepcopy(self.common['sample']),
                weight_manifest=lambda verified: copy.deepcopy(self.weights[case]))
        self.gates = helper('B')
        self.gates.a_gates = helper('A')
        self.gates.a_gates.LAKE_SHA256 = LAKE_SHA
        self.gates.a_gates.selected_profile = lambda group, sample: 19098
        self.gates.selected_profile = lambda group, sample: 19099
        self.gates.parallelism = parallelism
        self.gates.sample_profile = lambda sample: SimpleNamespace(source=self.source, prompt=PROMPT)
        self.gates.GENERATION = GENERATION.copy()
        self.gates.profiles = SimpleNamespace(require_host=mock.Mock(return_value=HOST))
        self.gates.base = SimpleNamespace(proc_identity=mock.Mock(return_value=None),
                                         selected_idle=mock.Mock(return_value={'explicitly_idle_cards': list(range(8))}),
                                         selected_health_memory=mock.Mock(return_value={'health': 'fixture'}))
        self.gates.video_probe = mock.Mock(return_value=self.probe)

    def patch(self, test):
        for name, value in {'ROOT': self.root, 'QUEUE_ROOT': self.root / 'trial_queue',
                            'CODE': self.root / 'trial_queue_code', 'A_RUNS': self.a_runs,
                            'PREDECESSOR_STATUS': self.a_path, 'B_OUTPUT': self.b_output,
                            'B_CODE': self.root / 'b_model_trial_code', 'PYTHON': self.root / 'env/bin/python'}.items():
            test.enterContext(mock.patch.object(queue, name, value))
        test.enterContext(mock.patch.object(queue.os, 'O_NOFOLLOW', getattr(os, 'O_NOFOLLOW', 0), create=True))
        if os.name == 'nt':
            test.enterContext(mock.patch.object(queue.os, 'getuid', return_value=0, create=True))
            test.enterContext(mock.patch.object(queue.os, 'O_NONBLOCK', 0, create=True))
            test.enterContext(mock.patch.object(queue, 'sync_directory'))
        return self

    def status(self, case='A', phase='formal_completed_review_required'):
        path, identity, run_id = (self.a_path, A_IDENTITY, queue.PREDECESSOR_RUN_ID) if case == 'A' else (self.b_path, B_IDENTITY, B_ID)
        run, name = path.parent, case.lower() + '_50step'
        gate = {'status': 'passed', 'host': HOST, 'run_id': run_id, 'group': '01234567', 'sample_id': 'lake_snow',
                'server_identity': {'pid': 8000, 'start_ticks': 1000, 'session': 8000, 'pgrp': 8000},
                'worker_identities_by_card': {str(card): {'pid': 8100+card} for card in range(8)}}
        if case == 'B':
            gate['loaded_records_by_pid'] = {str(8100+card): {'load': {'base_tensor_count': 535, 'branch_tensor_count': 800, 'lora_pairs_merged': 208}} for card in range(8)}
        fields = {key: GENERATION[key] for key in ('width', 'height', 'fps', 'flow_shift', 'seed')}
        fields.update(prompt=PROMPT, num_inference_steps=50, extra_params=json.dumps({
            'task': 'ref2va', 'duration': 5.0, 'audio_flow_shift': 3.0}))
        request = {'host': HOST, 'run_id': run_id, 'group': '01234567', 'sample_id': 'lake_snow', 'fields': fields,
                   'requested_steps': 50, 'source_video': str(self.source), 'source_sha256': LAKE_SHA}
        output = run / 'output' / (name + '.mp4'); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'CPU fixture only; not a video or model output')
        result = {'success': True, 'requested_steps': 50, 'http_code': '200', 'curl_returncode': 0,
                  'content_type': 'video/mp4', 'output_video': str(output), 'output_sha256': sha(output),
                  'ffprobe': copy.deepcopy(self.probe), 'request_end_to_end_seconds': 1900.5,
                  'actual_model_forward_calls': None, 'dit_latency_seconds': None, 'quality_evidence': False}
        params = parallelism(19098 if case == 'A' else 19099)
        frozen = {'host': HOST, 'run_id': run_id, 'cards': list(range(8)), 'parallelism': params,
                  'generation': GENERATION, 'sources': self.sources[case], **copy.deepcopy(self.common),
                  'weights': self.weights[case]}
        dump(run / 'frozen_evidence.json', frozen)
        dump(run / 'formal_smoke_gate.json', gate)
        dump(run / name / 'request.json', request)
        dump(run / name / 'result.json', result)
        status = {'case': case, 'sample_id': 'lake_snow', 'group': '01234567', 'host': HOST, 'run_id': run_id,
                  'run_directory': str(run), 'supervisor_pid': identity['pid'], 'supervisor_proc_identity': identity.copy(),
                  'allocated_physical_npu_ids': list(range(8)), 'parallelism': params, 'requested_generation': GENERATION,
                  'phase': phase, 'formal_50step_started': True, 'formal_50step_completed': True,
                  'formal_request_attempts': 1, 'formal_smoke_gate_passed': True,
                  'formal_smoke_gate': gate, 'cleanup_completed': True,
                  'selected_cards_verified_idle_after_cleanup': True, 'needs_attention': False,
                  'remaining_owned_process_groups': {}, 'error': None, name: result,
                  'frozen_evidence_sha256': sha(run / 'frozen_evidence.json')}
        dump(path, status)
        return status

    def owner(self):
        return queue.SerialQueue(self.a_path, queue.PREDECESSOR_RUN_ID, self.gates)


class QueueTests(unittest.TestCase):
    def fixture(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        return Fixture(Path(directory).resolve()).patch(self)

    def test_source_pins_match_real_reviewed_local_files(self):
        mapping = {'model_trial_code': 'model_trial', 'b_model_trial_code': 'b_model_trial',
                   'validation_code': 'validation'}
        for relative, expected in queue.PINNED_FILES.items():
            components = relative.split('/')
            if components[0] in mapping:
                components[0] = mapping[components[0]]
            path = HERE.parent.joinpath(*components)
            if not path.is_file():
                path = queue.ROOT / relative  # CPU regression on a reviewed Linux deployment.
            self.assertEqual(sha(path), expected, relative)

    def test_standard_library_only_no_model_runtime_import(self):
        for name in ('torch', 'torch_npu', 'vllm', 'vllm_omni'):
            self.assertNotIn(name, sys.modules)

    def test_only_fixed_per_run_path_runid_allowed_not_latest(self):
        fixture = self.fixture()
        self.assertEqual(queue.validate_predecessor_path(fixture.a_path, queue.PREDECESSOR_RUN_ID), fixture.a_path)
        for path, run_id in ((fixture.a_runs.parent / 'a_status.json', queue.PREDECESSOR_RUN_ID),
                              (fixture.a_path, 'a' * 32), (fixture.root / '../external', queue.PREDECESSOR_RUN_ID),
                              (Path('relative.json'), queue.PREDECESSOR_RUN_ID),
                              (fixture.b_path, queue.PREDECESSOR_RUN_ID)):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                queue.validate_predecessor_path(path, run_id)

    def test_cli_no_allow_does_not_load_gates_or_touch_devices(self):
        with mock.patch.object(queue, 'load_gates') as load, mock.patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit):
                queue.main(['--predecessor-status', str(queue.PREDECESSOR_STATUS), '--predecessor-run-id', queue.PREDECESSOR_RUN_ID])
            load.assert_not_called()

    def test_complete_real_status_shape_accepts_A_and_B_artifacts(self):
        fixture = self.fixture(); owner = fixture.owner()
        for case, path in (('A', fixture.a_path), ('B', fixture.b_path)):
            status = fixture.status(case)
            binding = owner.basic_status(status, case, path)
            result = owner.completed_evidence(path, case, binding)
            self.assertEqual(result['binding']['case'], case)
            self.assertEqual(result['request_end_to_end_seconds'], 1900.5)
            self.assertFalse(result['quality_evidence'])
            self.assertEqual(result['output_record']['sha256'], status[case.lower() + '_50step']['output_sha256'])

    def test_completion_requires_formal_smoke_cleanup_and_exit(self):
        fixture = self.fixture(); owner = fixture.owner()
        original = fixture.status()
        binding = owner.basic_status(original, 'A', fixture.a_path)
        for key, value in (('phase', 'smoke_2step_completed'), ('formal_50step_completed', False),
                           ('formal_request_attempts', 2), ('formal_request_attempts', True),
                           ('formal_smoke_gate_passed', False), ('cleanup_completed', False),
                           ('selected_cards_verified_idle_after_cleanup', False), ('needs_attention', True),
                           ('remaining_owned_process_groups', {'8000': [8001]}), ('error', 'error')):
            dump(fixture.a_path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                owner.completed_evidence(fixture.a_path, 'A', binding)
        dump(fixture.a_path, original)
        fixture.gates.base.proc_identity.return_value = A_IDENTITY.copy()
        with self.assertRaisesRegex(RuntimeError, 'not exited'):
            owner.completed_evidence(fixture.a_path, 'A', binding)

    def test_artifact_source_hash_request_metadata_weight_and_gate_drift_blocked(self):
        fixture = self.fixture(); owner = fixture.owner()
        for corruption in ('frozen_hash', 'source', 'weights', 'gate', 'request_seed', 'request_source', 'output', 'probe'):
            original = fixture.status(); binding = owner.basic_status(original, 'A', fixture.a_path)
            fixture.gates.video_probe.return_value = fixture.probe
            if corruption == 'frozen_hash':
                original['frozen_evidence_sha256'] = 'old'; dump(fixture.a_path, original)
            elif corruption == 'source':
                fixture.sources['A']['original/pipeline_minimax_h3.py']['sha256'] = 'changed-after-run'
            elif corruption == 'weights':
                fixture.weights['A']['files']['base']['tensor_count'] = 534
            elif corruption == 'gate':
                dump(fixture.a_path.parent / 'formal_smoke_gate.json', {'status': 'passed', 'run_id': 'other'})
            elif corruption.startswith('request'):
                path = fixture.a_path.parent / 'a_50step/request.json'; request = queue.read_json(path)
                if corruption == 'request_seed': request['fields']['seed'] = 1
                else: request['source_sha256'] = 'different'
                dump(path, request)
            elif corruption == 'output':
                (fixture.a_path.parent / 'output/a_50step.mp4').write_bytes(b'changed')
            else:
                fixture.gates.video_probe.return_value = {'verified': True, 'wrong': True}
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                owner.completed_evidence(fixture.a_path, 'A', binding)

    def test_wrong_host_run_pid_cards_subworld_or_sampler_cannot_be_predecessor(self):
        fixture = self.fixture(); owner = fixture.owner(); status = fixture.status()
        for key, value in (('host', {**HOST, 'boot_id': 'old'}), ('run_id', B_ID), ('supervisor_pid', 123),
                           ('allocated_physical_npu_ids', [0, 1, 2, 3]), ('requested_generation', {**GENERATION, 'seed': 2}),
                           ('parallelism', {**parallelism(19098), 'text_encoder_tp_size': 4})):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                owner.basic_status({**status, key: value}, 'A', fixture.a_path)

    def test_real_failed_two_step_A_stops_before_any_B_launch(self):
        fixture = self.fixture(); owner = fixture.owner()
        status = fixture.status(phase='failed_before_cleanup')
        status.update(formal_50step_started=False, formal_50step_completed=False, formal_request_attempts=0,
                      error='smoke_2step HTTP 500', cleanup_completed=False)
        dump(fixture.a_path, status)
        owner.predecessor_binding = owner.basic_status(status, 'A', fixture.a_path)
        owner.assert_bindings = mock.Mock(); owner.launch_b_once = mock.Mock()
        fixture.gates.base.proc_identity.return_value = A_IDENTITY.copy()
        with self.assertRaises(RuntimeError):
            owner.wait_for_a()
        owner.launch_b_once.assert_not_called()
        self.assertEqual(owner.status['b_launch_attempts'], 0)

    def test_waiting_A_is_read_only_and_bounded(self):
        fixture = self.fixture(); owner = fixture.owner(); status = fixture.status(phase='server_starting')
        owner.predecessor_binding = owner.basic_status(status, 'A', fixture.a_path)
        owner.assert_bindings = mock.Mock(); owner.update = mock.Mock()
        fixture.gates.base.proc_identity.return_value = A_IDENTITY.copy()
        with mock.patch.object(queue.time, 'monotonic', side_effect=[0, 0, queue.A_WAIT_TIMEOUT + 1]), \
             mock.patch.object(queue.time, 'sleep') as sleep, mock.patch.object(queue.subprocess, 'run') as run, \
             mock.patch.object(queue.subprocess, 'Popen') as popen:
            with self.assertRaises(TimeoutError): owner.wait_for_a()
        sleep.assert_called_once_with(30); run.assert_not_called(); popen.assert_not_called()

    def test_terminal_A_waits_for_its_exact_supervisor_to_exit(self):
        fixture = self.fixture(); owner = fixture.owner(); status = fixture.status()
        owner.predecessor_binding = owner.basic_status(status, 'A', fixture.a_path)
        owner.assert_bindings = mock.Mock(); owner.update = mock.Mock(); owner.fresh_idle = mock.Mock()
        fixture.gates.base.proc_identity.side_effect = [A_IDENTITY.copy(), None, None]
        with mock.patch.object(queue.time, 'sleep') as sleep, mock.patch.object(queue, 'exclusive_json') as write:
            owner.wait_for_a()
        sleep.assert_called_once_with(30); owner.fresh_idle.assert_called_once_with('after_a')
        write.assert_called_once()

    def test_durable_queue_directory_blocks_restart_without_overwriting_prior_state(self):
        fixture = self.fixture(); fixture.status(); owner = fixture.owner()
        with mock.patch.object(queue, 'production_manifest', return_value={'fixed': 'sha'}):
            with mock.patch('sys.stdout', new=io.StringIO()): owner.acquire()
            original = (owner.directory / 'queue_status.json').read_bytes()
            owner.release()
            second = fixture.owner()
            try:
                with self.assertRaises(FileExistsError): second.acquire()
                self.assertFalse(second.created)
                self.assertEqual((owner.directory / 'queue_status.json').read_bytes(), original)
            finally: second.release()

    def prepared_launcher(self):
        fixture = self.fixture(); owner = fixture.owner(); fixture.status()
        owner.directory.mkdir(parents=True); owner.lock = (owner.directory / 'fixture_lock').open('w+b')
        owner.assert_bindings = mock.Mock(); owner.completed_evidence = mock.Mock(return_value={'complete': 'a'})
        owner.status['a_completed_evidence'] = {'complete': 'a'}
        owner.predecessor_binding = {'run_id': queue.PREDECESSOR_RUN_ID}
        owner.update = lambda phase, **values: owner.status.update(values, phase=phase)
        self.addCleanup(owner.release)
        return fixture, owner

    def test_fsynced_launch_intent_precedes_exactly_one_Popen(self):
        fixture, owner = self.prepared_launcher()
        fixture.gates.base.proc_identity.return_value = B_IDENTITY.copy()
        def spawn(command, **kwargs):
            self.assertTrue((owner.directory / 'b_launch_intent.json').is_file())
            self.assertEqual(queue.read_json(owner.directory / 'b_launch_intent.json')['retry_allowed'], False)
            self.assertEqual(owner.status['b_launch_attempts'], 1)
            self.assertEqual(command, [str(queue.PYTHON), '-B', str(queue.B_CODE / 'run_b_trial.py'),
                                       '--group', '01234567', '--sample', 'lake_snow', '--allow-npu'])
            self.assertEqual(kwargs['pass_fds'], (owner.lock.fileno(),))
            self.assertTrue(kwargs['start_new_session'])
            return mock.Mock(pid=B_IDENTITY['pid'])
        with mock.patch.object(queue.subprocess, 'Popen', side_effect=spawn) as popen:
            owner.launch_b_once()
            with self.assertRaises(RuntimeError): owner.launch_b_once()
        popen.assert_called_once()

    def test_failed_Popen_leaves_tombstone_and_cannot_retry(self):
        fixture, owner = self.prepared_launcher()
        with mock.patch.object(queue.subprocess, 'Popen', side_effect=OSError('spawn failed')) as popen:
            with self.assertRaises(OSError): owner.launch_b_once()
            self.assertTrue((owner.directory / 'b_launch_intent.json').is_file())
            with self.assertRaises(RuntimeError): owner.launch_b_once()
        popen.assert_called_once()

    def test_crash_after_intent_before_Popen_still_blocks_new_attempt(self):
        fixture, owner = self.prepared_launcher()
        queue.exclusive_json(owner.directory / 'b_launch_intent.json', {'crash_before_spawn': True})
        with mock.patch.object(queue.subprocess, 'Popen') as popen:
            with self.assertRaises(FileExistsError): owner.launch_b_once()
        popen.assert_not_called()

    def test_B_latest_only_discovers_child_then_binds_immutable_run(self):
        fixture = self.fixture(); owner = fixture.owner(); fixture.status('B')
        owner.child = mock.Mock(pid=B_IDENTITY['pid']); owner.child_identity = queue.stable_identity(B_IDENTITY)
        owner.update = mock.Mock()
        stale = queue.read_json(fixture.b_path); stale['supervisor_pid'] = 42
        dump(fixture.b_output / 'b_status.json', stale)
        self.assertIsNone(owner.discover_b_status())
        dump(fixture.b_output / 'b_status.json', queue.read_json(fixture.b_path))
        self.assertIsNotNone(owner.discover_b_status())
        self.assertEqual(owner.b_status_path, fixture.b_path)
        dump(fixture.b_output / 'b_status.json', stale)
        self.assertEqual(owner.discover_b_status()['run_id'], B_ID)

    def test_B_full_success_requires_own_exit_terminal_artifacts_and_idle(self):
        fixture = self.fixture(); owner = fixture.owner(); status = fixture.status('B')
        owner.child = mock.Mock(pid=B_IDENTITY['pid'], poll=mock.Mock(return_value=0))
        owner.child_identity = queue.stable_identity(B_IDENTITY)
        owner.b_status_path = fixture.b_path; owner.b_binding = owner.basic_status(status, 'B', fixture.b_path)
        owner.assert_bindings = mock.Mock(); owner.fresh_idle = mock.Mock()
        owner.update = lambda phase, **values: owner.status.update(values, phase=phase)
        with mock.patch.object(queue, 'exclusive_json'):
            owner.wait_for_b()
        self.assertEqual(owner.status['phase'], 'completed_review_required')
        self.assertTrue(owner.status['b_formal_completed'])
        owner.fresh_idle.assert_called_once_with('after_b')

    def test_B_failure_and_timeout_do_not_schedule_any_other_job(self):
        fixture = self.fixture(); owner = fixture.owner(); owner.assert_bindings = mock.Mock()
        owner.discover_b_status = mock.Mock(return_value=None)
        owner.child = mock.Mock(poll=mock.Mock(return_value=1))
        with self.assertRaises(RuntimeError): owner.wait_for_b()
        owner.child.poll.return_value = None
        with mock.patch.object(queue.time, 'monotonic', side_effect=[0, queue.B_WAIT_TIMEOUT + 1]):
            with self.assertRaises(TimeoutError): owner.wait_for_b()

    def test_stop_sends_TERM_only_to_exact_own_marked_child_never_A_or_group(self):
        fixture = self.fixture(); owner = fixture.owner()
        owner.child = mock.Mock(pid=B_IDENTITY['pid'], poll=mock.Mock(return_value=None))
        owner.child_identity = queue.stable_identity(B_IDENTITY)
        fixture.gates.base.proc_identity.return_value = B_IDENTITY.copy()
        for marker_ok in (False, True):
            marker = f'H3_SERIAL_QUEUE_ID={owner.queue_id}'.encode() if marker_ok else b'OTHER=1'
            with mock.patch.object(Path, 'read_bytes', return_value=marker + b'\0'), mock.patch.object(queue.os, 'kill') as kill:
                self.assertEqual(owner.stop_own_b_if_running(), marker_ok)
                if marker_ok: kill.assert_called_once_with(B_IDENTITY['pid'], signal.SIGTERM)
                else: kill.assert_not_called()
        fixture.gates.base.proc_identity.return_value = {**B_IDENTITY, 'start_ticks': 123}
        with mock.patch.object(queue.os, 'kill') as kill:
            self.assertFalse(owner.stop_own_b_if_running())
            kill.assert_not_called()

    def test_exclusive_record_never_overwrites_and_symlink_is_rejected(self):
        fixture = self.fixture(); path = fixture.root / 'intent.json'
        queue.exclusive_json(path, {'first': True})
        with self.assertRaises(FileExistsError): queue.exclusive_json(path, {'first': False})
        self.assertEqual(queue.read_json(path), {'first': True})
        if os.name != 'nt':
            link = fixture.root / 'link.json'; link.symlink_to(path)
            with self.assertRaises(RuntimeError): queue.read_json(link)

    def test_global_mutex_conflict_does_not_create_queue_directory(self):
        fixture = self.fixture(); owner = fixture.owner()
        try:
            with mock.patch.object(queue.fcntl, 'flock', side_effect=BlockingIOError('busy')):
                with self.assertRaises(BlockingIOError): owner.acquire()
            self.assertFalse(owner.directory.exists())
        finally: owner.release()

    def test_main_failure_always_releases_queue_mutex_no_automatic_restart(self):
        fake = mock.Mock(created=True, status={}, stop_own_b_if_running=mock.Mock(return_value=True))
        fake.wait_for_a.side_effect = RuntimeError('A HTTP500')
        with mock.patch.object(queue, 'validate_predecessor_path'), mock.patch.object(queue, 'load_gates'), \
             mock.patch.object(queue, 'SerialQueue', return_value=fake), mock.patch.object(queue.signal, 'signal'):
            code = queue.main(['--predecessor-status', str(queue.PREDECESSOR_STATUS),
                               '--predecessor-run-id', queue.PREDECESSOR_RUN_ID, '--allow-b-launch'])
        self.assertEqual(code, 1)
        fake.launch_b_once.assert_not_called(); fake.wait_for_b.assert_not_called()
        fake.release.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
