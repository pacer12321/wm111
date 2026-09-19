"""Synthetic CPU-only VLM protocol/admission tests; never create real evidence."""
import ast
from contextlib import ExitStack
import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
local = HERE.parents[1]/'validation'
sys.path.insert(0, str(local if local.is_dir() else Path('/cache/zhonghao/h3/validation_code')))
sys.path.insert(0, str(HERE))
if os.name == 'nt': sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import vlm_contract as c
import vlm_worker as worker
import run_vlm as run

HOST = dict(hostname=c.profiles.EXPECTED_HOST, boot_id='12345678-abcd-abcd-abcd-123456789abc', machine='aarch64')


def prediction(kind='preserve'):
    effects = dict(order='unchanged', speed='unchanged', duration='unchanged', events='unchanged')
    if kind != 'preserve': effects['order'] = kind
    return dict(classification=kind, confidence=0.8,
        source_evidence=[dict(frame_index=0, observation='Two people face each other.'),
                         dict(frame_index=123, observation='They are embracing.')],
        edit_effects=effects, reason='Relative source and edit analysis in a CPU toy fixture only.')


class ParserTests(unittest.TestCase):
    def test_all_three_classes_preserved_not_defaulted(self):
        for kind in ('preserve', 'change', 'uncertain'):
            row = prediction(kind)
            self.assertEqual(c.parse_classification(json.dumps(row)), row)

    def test_non_json_fences_duplicate_nan_missing_and_extras_fail(self):
        good = json.dumps(prediction())
        for raw in ('', 'preserve', '```json\n'+good+'\n```', good+' trailing',
                    good.replace('0.8', 'NaN'), good.replace('"confidence": 0.8', '"confidence": 0.8, "confidence": 1'),
                    json.dumps({**prediction(), 'label': False}), '{}', '[]'):
            with self.subTest(raw=raw[:25]), self.assertRaises(RuntimeError): c.parse_classification(raw)

    def test_inconsistent_effects_invalid_frames_and_boolean_confidence_fail(self):
        values = []
        for key, value in (('classification', 'other'), ('confidence', True), ('confidence', -0.1), ('reason', ''),
                           ('source_evidence', [dict(frame_index=0, observation='one only')])):
            row = prediction(); row[key] = value; values.append(row)
        row = prediction(); row['edit_effects']['events'] = 'change'; values.append(row)
        row = prediction(); row['source_evidence'][1]['frame_index'] = 1; values.append(row)
        row = prediction(); row['source_evidence'][1]['frame_index'] = 0; values.append(row)
        row = prediction('change'); row['edit_effects']['order'] = 'unchanged'; values.append(row)
        for value in values:
            with self.assertRaises(RuntimeError): c.parse_classification(json.dumps(value))

    def test_prompt_is_actual_source_relative_and_no_router_label(self):
        messages = c.messages(); content = messages[0]['content']
        self.assertEqual(content[0], {'type': 'video'})
        self.assertIn(c.PROMPT, content[1]['text'])
        self.assertIn('already holds', content[1]['text'])
        self.assertNotIn('router_control', json.dumps(messages))
        self.assertNotIn('Te_Temporal_reordering_04', json.dumps(messages))
        self.assertNotIn(c.SAMPLE, json.dumps(messages))

    def test_uniform_frames_and_full_two_device_layer_coverage(self):
        self.assertEqual(c.FRAME_INDICES, (0,8,16,25,33,41,49,57,66,74,82,90,98,107,115,123))
        for i in range(64): self.assertEqual(c.DEVICE_MAP[f'model.language_model.layers.{i}'], 'npu:0' if i < 32 else 'npu:1')
        self.assertEqual(c.DEVICE_MAP['model.visual'], 'npu:0')
        self.assertEqual(c.DEVICE_MAP['lm_head'], 'npu:1')

    def test_v2_prompt_defines_both_appearance_and_action_change_without_expected_label(self):
        # This checks protocol text only, NOT actual VLM semantic accuracy.
        text = c.messages()[0]['content'][1]['text']
        for phrase in (c.CLASSIFIER_PROMPT_VERSION, 'ACTION TIMELINE', 'ACTION OCCURRENCES',
                       'Appearance changes alone', 'is still change', 'Use uncertain',
                       'specific required action-timeline change'):
            self.assertIn(phrase, text)
        self.assertNotIn('expected classification', text.lower())
        self.assertNotIn('confidence 1', text.lower())

    def test_parser_never_relabels_a_semantically_wrong_raw_change_as_preserve(self):
        # Deliberately wrong semantic answer: retained for review, not repaired
        # by a text router. Schema validation cannot establish semantic truth.
        row = prediction()
        row['classification'] = 'change'
        row['edit_effects']['events'] = 'change'
        row['reason'] = 'Changing an object color counts as modifying an event.'
        self.assertEqual(c.parse_classification(json.dumps(row)), row)


class CPUPreparationTests(unittest.TestCase):
    def test_imports_do_not_import_device_or_torch(self):
        self.assertNotIn('torch', sys.modules); self.assertNotIn('torch_npu', sys.modules)
        for path in (HERE/'vlm_contract.py', HERE/'run_vlm.py'):
            tree = ast.parse(path.read_text())
            imported = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            self.assertFalse(any('torch' in ast.unparse(node) for node in imported))

    def test_prepare_only_has_no_inference_path(self):
        fake = dict(source='synthetic', frame_indices=list(c.FRAME_INDICES))
        with mock.patch.object(c, 'extract_video', return_value=(None, fake)), mock.patch.object(worker, 'infer') as infer, \
             mock.patch.object(worker, 'inherited_run') as inherited, mock.patch('sys.stdout', new=io.StringIO()) as stdout:
            self.assertEqual(worker.main(['--prepare-only']), 0)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'cpu_preparation_only')
            infer.assert_not_called(); inherited.assert_not_called()

    def test_no_implicit_npu_or_arbitrary_source_arguments(self):
        for module, args in ((run, []), (run, ['--source', '/other']), (worker, []),
                             (worker, ['--allow-npu', '--prepare-only']), (worker, ['--allow-npu', '--model', '/other'])):
            with mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit): module.main(args)

    def test_raw_reply_is_persisted_before_schema_parse(self):
        text = (HERE/'vlm_worker.py').read_text()
        self.assertLess(text.index("atomic_json(directory/'model_output.json'"), text.index('classification = contract.parse_classification(raw)'))
        self.assertIn('local_files_only=True, trust_remote_code=False', text)
        self.assertIn('do_sample_frames=False', text)
        self.assertIn('videos=[video]', text)
        self.assertIn("attention='eager'", text)


class LoadingInfoSerializationTests(unittest.TestCase):
    def test_empty_hf_sets_round_trip_as_empty_json_arrays(self):
        report = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=[], error_msgs=())
        normalized = worker.loading_info_json(report)
        self.assertEqual(normalized, dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[]))
        self.assertEqual(json.loads(json.dumps(normalized)), normalized)
        self.assertIsInstance(report['missing_keys'], set)
        self.assertIsInstance(report['error_msgs'], tuple)

    def test_nested_tuple_set_and_frozenset_preserve_content_without_mutation(self):
        report = {'nested': ({'pairs': frozenset({('b', 2), ('a', 1)})}, {'names': {'z', 'a'}}),
                  'primitives': [None, True, False, 1, 0.5, 'text']}
        before = copy.deepcopy(report)
        normalized = worker.loading_info_json(report)
        self.assertEqual(normalized, {'nested': [{'pairs': [['a', 1], ['b', 2]]}, {'names': ['a', 'z']}],
                                      'primitives': [None, True, False, 1, 0.5, 'text']})
        self.assertEqual(json.loads(json.dumps(normalized)), normalized)
        self.assertEqual(report, before)

    def test_unsupported_objects_and_nonstring_dictionary_keys_raise(self):
        for value in (object(), Path('/not/a/string'), complex(1, 2), b'bytes', {1: 'nonstring key'},
                      {'nested': [object()]}, {'nested': {('tuple',): 'nonstring key'}}):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(TypeError):
                worker.loading_info_json(value)

    def test_nonempty_loading_errors_are_retained_by_normalization(self):
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            report = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=[], error_msgs=[])
            report[key] = {'actual_nonempty_problem'}
            normalized = worker.loading_info_json(report)
            self.assertEqual(normalized[key], ['actual_nonempty_problem'])
            self.assertTrue(normalized[key])

    def test_original_nonempty_error_rejection_precedes_normalization_and_generation(self):
        tree = ast.parse((HERE/'vlm_worker.py').read_text())
        infer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'infer')
        checks = [(i, node) for i, node in enumerate(infer.body)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                  and isinstance(node.test.func, ast.Name) and node.test.func.id == 'any'
                  and any(isinstance(item, ast.Attribute) and item.attr == 'get'
                          and isinstance(item.value, ast.Name) and item.value.id == 'loading'
                          for item in ast.walk(node.test))]
        self.assertEqual(len(checks), 1)
        check_index, check = checks[0]
        conversions = [i for i, node in enumerate(infer.body) if any(
            isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == 'loading_info_json'
            for item in ast.walk(node))]
        self.assertEqual(len(conversions), 1)
        self.assertLess(check_index, conversions[0])
        generations = [i for i, node in enumerate(infer.body) if any(
            isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == 'generate'
            for item in ast.walk(node))]
        self.assertEqual(len(generations), 1)
        self.assertLess(conversions[0], generations[0])
        self.assertTrue(any(isinstance(node, ast.Raise) for node in check.body))
        code = compile(ast.fix_missing_locations(ast.Module(body=[check], type_ignores=[])), '<CPU loading gate>', 'exec')
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            report = {name: set() for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
            report[key] = {'actual_nonempty_problem'}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'did not load completely'):
                exec(code, {'loading': report})
        exec(code, {'loading': {name: set() for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}})


class CompletionTests(unittest.TestCase):
    def fixture(self, corruption=None):
        stack = ExitStack(); self.addCleanup(stack.close)
        temporary = stack.enter_context(tempfile.TemporaryDirectory())
        root = Path(temporary).resolve(); output = root/'output'
        run_id = 'a'*32; directory = output/'runs'/('20260914T010000Z_'+run_id); directory.mkdir(parents=True)
        def write(name, value): (directory/name).write_text(json.dumps(value), encoding='utf-8')
        source = dict(path=str(c.SOURCE), sha256=c.SOURCE_SHA, size_bytes=c.SOURCE_BYTES)
        wi = dict(pid=501, start_ticks=11, pgrp=501, session=501, state='R')
        si = dict(pid=500, start_ticks=10, pgrp=499, session=499, state='S')
        sources, model = {'actual_api': 'toy'}, {'complete_model': 'toy'}
        runtime, transfer = {'path': '/cpu/runtime', 'sha256': 'r'}, {'path': '/cpu/transfer', 'sha256': 't'}
        frozen = dict(host=HOST, run_id=run_id, source=source, edit_prompt=c.PROMPT, sources=sources, model=model,
                      runtime=runtime, transfer=transfer, device_map=c.DEVICE_MAP, frame_indices=list(c.FRAME_INDICES))
        write('host_identity.json', HOST); write('frozen_evidence.json', frozen)
        prep = dict(host=HOST, run_id=run_id, source=source, edit_prompt=c.PROMPT, frame_indices=list(c.FRAME_INDICES),
                    messages=c.messages(), video_tensor_present=True, video_token_count=448, sampled_frames=[{}]*16,
                    rgb_video_shape=[16,720,1280,3], tensors=dict(pixel_values_videos=dict(numel=1024)), input_token_count=600)
        raw = dict(host=HOST, run_id=run_id, source=source, edit_prompt=c.PROMPT, input_token_count=600,
                   generation=dict(do_sample=False, num_beams=1, max_new_tokens=512, use_cache=True),
                   generated_ids=[1,2], generated_token_count=2, raw_text=json.dumps(prediction()))
        load = dict(host=HOST, run_id=run_id, device_map=c.DEVICE_MAP, parameter_bytes=66714780128,
                    model_class='Qwen3VLForConditionalGeneration', dtype='bfloat16', attention='eager', loading_info={})
        for name, data in (('preprocessing.json',prep), ('model_output.json',raw), ('model_load.json',load)): write(name,data)
        text = '| NPU Chip | Process id | Process name | Process memory(MB) |\n'+'\n'.join(f'| No running processes found in NPU {i} |' for i in range(8))
        (directory/'npu_after.txt').write_text(text)
        stack.enter_context(mock.patch.object(c.profiles, 'INSTALL', root))
        stack.enter_context(mock.patch.object(c, 'OUTPUT', output))
        stack.enter_context(mock.patch.object(c.profiles, 'require_host', return_value=HOST))
        stack.enter_context(mock.patch.object(c, 'source_record', return_value=source))
        stack.enter_context(mock.patch.object(c, 'source_manifest', return_value=sources))
        stack.enter_context(mock.patch.object(c, 'model_identity', return_value=model))
        original_record = c.record
        stack.enter_context(mock.patch.object(c, 'record', side_effect=lambda path: runtime if path == c.ROOT/'runtime_validation.json' else transfer if path == c.ROOT/'consume_status.json' else original_record(path)))
        stack.enter_context(mock.patch.object(run.base, 'proc_identity', return_value=None))
        stack.enter_context(mock.patch.object(run.base.ProcessSupervisor, 'owned_group_members', return_value=[]))
        result = dict(status='passed', case=c.CASE, host=HOST, run_id=run_id, sample_id=c.SAMPLE, source=source,
            edit_prompt=c.PROMPT, worker_proc_identity=wi, leased_physical_cards=list(range(8)), used_physical_cards=[0,1],
            inference_executed=True, video_tensor_present=True, human_router_labels_sent_to_model=False,
            classification=prediction(), records={name:c.record(directory/name) for name in ('preprocessing.json','model_load.json','model_output.json')})
        write('worker_result.json',result)
        status = dict(case=c.CASE, sample_id=c.SAMPLE, host=HOST, run_id=run_id, run_directory=str(directory), phase='completed',
            inference_attempts=1, inference_completed=True, leased_physical_cards=list(range(8)), used_physical_cards=[0,1],
            cleanup_completed=True, selected_cards_verified_idle_after_cleanup=True, needs_attention=False,
            remaining_owned_process_groups={}, error=None, worker_pid=501, worker_proc_identity=wi,
            supervisor_pid=500, supervisor_proc_identity=si, frozen_evidence_sha256=c.profiles.digest(directory/'frozen_evidence.json'),
            result_path=str(directory/'worker_result.json'), result_sha256=c.profiles.digest(directory/'worker_result.json'),
            classification=prediction(), resource_release_check=run.base.selected_idle(text, tuple(range(8))))
        if corruption == 'not_completed': status['phase']='vlm_running'
        elif corruption == 'two_attempts': status['inference_attempts']=2
        elif corruption == 'no_cleanup': status['cleanup_completed']=False
        elif corruption == 'wrong_host': status['host']={**HOST,'boot_id':'other'}
        elif corruption == 'raw_tamper': (directory/'model_output.json').write_text('not JSON')
        elif corruption == 'result_tamper': (directory/'worker_result.json').write_text('{}')
        elif corruption == 'missing_frames': prep['frame_indices']=[0]; write('preprocessing.json',prep)
        elif corruption == 'frozen_tamper': write('frozen_evidence.json',{})
        elif corruption == 'status_class': status['classification']=prediction('change')
        elif corruption == 'wrong_pid': status['worker_pid']=999
        write('vlm_status.json',status)
        return directory,run_id

    def test_valid_completed_schema_uses_actual_read_only_reader(self):
        directory, run_id = self.fixture()
        with mock.patch.object(run.base, 'atomic_json') as write, mock.patch.object(run.subprocess, 'run') as subprocess_run:
            result = run.verify_completed(directory,run_id)
            self.assertEqual(set(result), {'status','result','frozen','records'})
            self.assertEqual(result['result']['classification']['classification'],'preserve')
            write.assert_not_called(); subprocess_run.assert_not_called()

    def test_every_corruption_failclosed(self):
        for corruption in ('not_completed','two_attempts','no_cleanup','wrong_host','raw_tamper','result_tamper',
                           'missing_frames','frozen_tamper','status_class','wrong_pid'):
            with self.subTest(corruption=corruption):
                directory,run_id=self.fixture(corruption)
                with self.assertRaises((RuntimeError,ValueError)): run.verify_completed(directory,run_id)

    def test_original_pid_alive_blocks_but_reused_pid_does_not(self):
        directory, run_id = self.fixture()
        with mock.patch.object(run.base,'proc_identity',side_effect=lambda pid:dict(pid=pid,start_ticks=11 if pid==501 else 10)):
            with self.assertRaises(RuntimeError): run.verify_completed(directory,run_id)
        with mock.patch.object(run.base,'proc_identity',return_value=dict(pid=501,start_ticks=999)):
            run.verify_completed(directory,run_id)

    def test_current_api_drift_blocks(self):
        directory,run_id=self.fixture()
        with mock.patch.object(c,'source_manifest',return_value={'changed':1}), self.assertRaises(RuntimeError):
            run.verify_completed(directory,run_id)

    def test_live_owned_group_blocks(self):
        directory,run_id=self.fixture()
        with mock.patch.object(run.base.ProcessSupervisor,'owned_group_members',return_value=[901]), self.assertRaises(RuntimeError):
            run.verify_completed(directory,run_id)


if __name__ == '__main__': unittest.main(verbosity=2)
