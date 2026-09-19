"""CPU-only color-entry regression. Synthetic files live only in temp roots.

No torch import, SSH, child processes, real HTTP, or NPU operations. Run as an
independent process so fixed production module names cannot reuse old imports.
"""
import ast
from contextlib import ExitStack
import copy
import hashlib
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
VALIDATION = HERE.parent / 'validation'
if not VALIDATION.is_dir():
    VALIDATION = Path('/cache/zhonghao/h3/validation_code')
sys.path.insert(0, str(VALIDATION))
for case in ('a', 'b', 'c'):
    sys.path.insert(0, str(HERE / case))
if os.name == 'nt':
    sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import trial_gates as ag
import b_trial_gates as bg
import c_trial_gates as cg
import run_a_trial as ar
import run_b_trial as br
import run_c_trial as cr

CASES = ((ag, ar, ar.ATrialSupervisor), (bg, br, br.BTrialSupervisor), (cg, cr, cr.CTrialSupervisor))
SAMPLE = 'shirt_red_couple_124'
HOST = dict(hostname=ag.profiles.EXPECTED_HOST, boot_id='12345678-abcd-abcd-abcd-123456789abc', machine='aarch64')
PROMPT = ("Change only the man's pale shirt to a solid red shirt. Preserve the shirt's fabric, shape, and details, "
          "and keep the woman's clothing unchanged. Preserve the original actions, their order and timing, "
          "the people, camera viewpoint and motion, framing, lighting, and background.")


def manifest():
    return dict(schema_version=1, sample_id=SAMPLE, source_video=str(ag.sample_profile().source),
                source_sha256=ag.COLOR_SHA256, edit_prompt=PROMPT, requested_generation=ag.GENERATION.copy(),
                source_metadata=dict(width=1280, height=720, frames=124, fps=24, duration_seconds=124/24, has_audio=False),
                provenance=dict(start_frame=104, end_frame_exclusive=228, original_fps=24,
                    official_source_id='Te_Temporal_reordering_04',
                    original_sha256='a18b5f2dcabe5a97d30350de9e402039a7e6c0886442162b164887698d75d638',
                    instruction_is_self_authored=True, official_paired_target_available=False, gt_target_generated=False),
                router_control=dict(temporal_change_required=False, text_only_should_identify_temporal_edit=False,
                                    vlm_necessity_evidence=False))


def probe():
    return dict(verified=True, checks=dict(dimensions=[1280, 720], fps=24, frame_count=124,
                                          container_duration_seconds=124/24),
                metadata=dict(streams=[dict(codec_type='video')]))


def owner(module, cls):
    with mock.patch.object(ag.profiles, 'require_host', return_value=HOST), mock.patch.object(module.base, 'proc_identity', return_value=None):
        return cls('01234567')


def weights():
    return {kind: dict(path=str(bg.CHECKPOINT / name), size_bytes=1234, header_sha256=kind+'-header', tensor_count=count)
            for kind, name, count in (('branch', 'linear_branch/model.safetensors', 800),
                                      ('lora', 'adapters/default/adapter_model.safetensors', 416))}


def load_logs(label='H3B', pids=range(500, 508), mutate=None):
    lines = []
    for pid in pids:
        load = dict(checkpoint=str(bg.CHECKPOINT), base_partition='ref2va', base_tensor_count=535,
                    branch_tensor_count=800, lora_pairs_merged=208, lora_rank=64, lora_alpha=64, lora_scale=1.0,
                    official_scale_source_commit=bg.OFFICIAL_COMMIT, merge_dtype='FP32 delta, cast to parameter dtype, then add',
                    qkv_merge_layout='post-base-loader contiguous Q/K/V thirds', **weights())
        cpu = dict(status='completed', requested_loading_intraop=4, intraop_before=1, intraop_loading=4,
                   intraop_after=1, interop_before=32, interop_after=32, interop_loading=32,
                   phase_seconds=dict(base=31.0, branch=2.0, lora=4.0))
        if mutate: mutate(load, cpu)
        for marker, data in (('OPENVDN_B_LOAD_RECORD', load), ('OPENVDN_B_CPU_LOADING_END', cpu)):
            lines.append(f'{label} pid={pid} INFO loader {marker} ' + json.dumps(data))
    return '\n'.join(lines)


def strict_logs(pids=range(500, 508), requests=1, prefix='', mutate=None):
    lines = []
    for request in range(requests):
        for pid in pids:
            row = dict(mode=cg.STRICT_MODE, reference_count=1, reference_kind='video', patch_size=(1, 2, 2),
                       source_shape=(37, 48, 84), target_shape=(37, 48, 84), text_len=123,
                       source_audio_t=0, target_audio_t=207)
            if mutate: mutate(pid, request, row)
            lines.append(f'{prefix}H3C pid={pid} INFO pipeline {cg.STRICT_MARKER}{row!r}')
    return '\n'.join(lines)


def smi(busy=True):
    lines = []
    for card in range(8):
        lines += [f'| {card} 910B3 | OK | 0 / 0 |', '| 0 | bus | 0 / 0 3402 / 65536 |']
    lines += ['| NPU Chip | Process id | Process name | Process memory(MB) |']
    lines += [f'| {card} 0 | {500+card} | VLLM::Worker | 3402 |' if busy else
              f'| No running processes found in NPU {card} |' for card in range(8)]
    return '\n'.join(lines)


class ProfileTests(unittest.TestCase):
    def test_new_code_roots_and_only_one_sample(self):
        for index, (g, run, cls) in enumerate(CASES):
            case = 'abc'[index]
            self.assertEqual(g.CODE, ag.ROOT / ('color_trial_v1/' + case))
            self.assertEqual(g.SAMPLE_IDS, (SAMPLE,))
            self.assertEqual(g.DEFAULT_SAMPLE, SAMPLE)
            self.assertEqual(g.sample_profile().prompt, PROMPT)
            self.assertEqual(g.sample_profile().source, ag.ROOT / 'data/shirt_red_couple_124/source.mp4')
            self.assertEqual(Path(g.__file__).resolve().parent, HERE / case)
            for forbidden in ('lake_snow', 'explicit_reverse_couple_124', '', '../shirt_red_couple_124', 'custom'):
                with self.assertRaises(ValueError): g.sample_profile(forbidden)

    def test_outputs_isolated_but_same_eight_locks_and_compute(self):
        original = ag.profiles.profile('01234567')
        selected = [g.selected_profile('01234567') for g, _, _ in CASES]
        self.assertEqual(len({p.output_root for p in selected}), 3)
        self.assertEqual({p.master_port for p in selected}, {19098, 19099, 19100})
        for index, (g, _, _) in enumerate(CASES):
            p = selected[index]
            self.assertEqual(p.output_root, ag.ROOT / 'color_trial_v1/results/01234567' / SAMPLE / 'ABC'[index])
            self.assertEqual(p.lease_root, original.lease_root)
            self.assertEqual(p.cards, tuple(range(8)))
            self.assertEqual(g.GENERATION, dict(width=1344, height=768, fps=24, duration_seconds=5.0, seed=4101,
                num_inference_steps=50, flow_shift=12.0, audio_flow_shift=3.0))
            compute = g.parallelism(p); compute.pop('listen_port')
            if index == 0: baseline = compute
            self.assertEqual(compute, baseline)
            self.assertEqual((compute['usp'], compute['ring'], compute['text_encoder_tp_size'], compute['vae_patch_parallel_size']), (8, 1, 8, 8))
            self.assertEqual(g.MIN_RAM, 600*1024**3)
            self.assertEqual(g.MIN_HBM_MIB, 55*1024)
            for group in ('0123', '4567', '2367'):
                with self.assertRaises(ValueError): g.selected_profile(group)
        self.assertEqual(original, ag.profiles.profile('01234567'))

    def test_dependency_pins_and_originals_preserved(self):
        self.assertEqual(ag.profiles.digest(HERE / 'a/trial_gates.py'), bg.A_GATES_SHA256)
        self.assertEqual(bg.A_GATES_SHA256, cg.A_GATES_SHA256)
        self.assertEqual(ag.profiles.digest(HERE / 'b/b_trial_gates.py'), cg.B_GATES_SHA256)
        self.assertEqual(bg.A_CODE, ag.CODE); self.assertEqual(cg.B_CODE, bg.CODE)
        self.assertEqual(ag.ORIGINAL_VENDOR, ag.ROOT / 'src/vllm-omni')
        self.assertEqual(bg.VENDOR, ag.ROOT / 'candidates/b_v1/vllm-omni')
        self.assertEqual(cg.VENDOR, ag.ROOT / 'candidates/c_v1/vllm-omni')

    def test_launchers_accept_color_only_and_same_math(self):
        for case in 'abc':
            text = (HERE / case / f'launch_{case}_trial.sh').read_text()
            self.assertIn('shirt_red_couple_124)', text)
            self.assertNotIn('lake_snow', text); self.assertNotIn('explicit_reverse', text)
            self.assertIn('/color_trial_v1/results/${H3_TRIAL_GROUP}/${H3_TRIAL_SAMPLE}/' + case.upper(), text)
            self.assertIn('--num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8', text)
            self.assertIn('--vae-patch-parallel-size 8', text)
            self.assertNotIn('--dtype', text)
            self.assertLess(text.index('source /cache/zhonghao/h3/env_h3_31731.sh'), text.index('export PYTHONPATH='))
            self.assertIn('VLLM_OMNI_VIDEO_SYNC_TIMEOUT=' + ('1800' if case == 'a' else '3600'), text)

    def test_cli_rejects_old_samples_arbitrary_paths_no_optin(self):
        for _, module, cls in CASES:
            for args in (['--group', '01234567'], ['--group', '0123', '--allow-npu'],
                         ['--group', '01234567', '--allow-npu', '--source', 'x'],
                         *(['--group', '01234567', '--allow-npu', '--sample', s] for s in ('lake_snow', 'explicit_reverse_couple_124'))):
                with mock.patch.object(module, cls.__name__) as constructor, mock.patch('sys.stderr', new=io.StringIO()):
                    with self.assertRaises(SystemExit): module.main(args)
                    constructor.assert_not_called()

    def test_original_supervisor_function_ast_unchanged(self):
        for index, case in enumerate('abc'):
            old_dir = ('model_trial', 'b_model_trial', 'c_model_trial')[index]
            old = HERE.parent / old_dir / f'run_{case}_trial.py'
            if not old.is_file(): old = ag.ROOT / (old_dir + '_code') / f'run_{case}_trial.py'
            old_tree, new_tree = (ast.parse(p.read_text()) for p in (old, HERE / case / f'run_{case}_trial.py'))
            old_tree.body.pop(0); new_tree.body.pop(0)  # Only module documentation changed.
            self.assertEqual(ast.dump(old_tree), ast.dump(new_tree))

    def test_all_unmodified_gate_functions_match_original_ast(self):
        changes = {'a': {'sample_profile', 'selected_profile', 'sample_gate', 'lake_sample_gate', 'source_code_manifest', 'vlm_admission_gate'},
                   'b': {'selected_profile'}, 'c': {'selected_profile', 'parse_strict_metadata'}}
        for index, case in enumerate('abc'):
            dirname = ('model_trial', 'b_model_trial', 'c_model_trial')[index]
            name = ('trial_gates.py', 'b_trial_gates.py', 'c_trial_gates.py')[index]
            old = HERE.parent / dirname / name
            if not old.is_file(): old = ag.ROOT / (dirname + '_code') / name
            functions = lambda p: {n.name: ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            before, after = functions(old), functions(HERE / case / name)
            for name in set(before) - changes[case]: self.assertEqual(before[name], after[name], (case, name))


class SampleTests(unittest.TestCase):
    def check(self, m=None, record=None, decoded=None):
        with mock.patch.object(ag, 'json_file', return_value=manifest() if m is None else m), \
             mock.patch.object(ag, 'file_record', return_value=record or dict(sha256=ag.COLOR_SHA256, size_bytes=1527004)), \
             mock.patch.object(ag, 'video_probe', return_value=decoded or probe()), \
             mock.patch.object(ag, 'vlm_admission_gate', return_value={'fixture_only_not_real_VLM': True}) as vlm:
            return ag.sample_gate()

    def test_exact_color_manifest_passes(self):
        self.assertEqual(self.check()['manifest'], manifest())
        self.assertIs(bg.sample_gate, ag.sample_gate); self.assertIs(cg.sample_gate, ag.sample_gate)

    def test_real_vlm_is_required_after_source_checks(self):
        source = dict(sha256=ag.COLOR_SHA256, size_bytes=1527004)
        with mock.patch.object(ag, 'json_file', return_value=manifest()), \
             mock.patch.object(ag, 'file_record', return_value=source), \
             mock.patch.object(ag, 'video_probe', return_value=probe()), \
             mock.patch.object(ag, 'vlm_admission_gate', side_effect=RuntimeError('VLM not run')) as vlm:
            with self.assertRaisesRegex(RuntimeError, 'VLM not run'):
                ag.sample_gate()
            vlm.assert_called_once_with(source, PROMPT)

    def test_wrong_prompt_sample_source_generation_rejected(self):
        for key, value in (('edit_prompt', 'Reverse the clip'), ('sample_id', 'explicit_reverse_couple_124'),
                           ('source_video', '/old/source.mp4'), ('source_sha256', '0'*64), ('schema_version', 2),
                           ('requested_generation', {**ag.GENERATION, 'seed': 0})):
            data = manifest(); data[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.check(data)

    def test_source_exact_bytes_hash_and_dimensions(self):
        for data in (dict(sha256='0'*64, size_bytes=1527004), dict(sha256=ag.COLOR_SHA256, size_bytes=1527003)):
            with self.assertRaises(RuntimeError): self.check(record=data)
        for key, value in (('width', 1344), ('height', 768), ('frames', 120), ('fps', 30), ('has_audio', True), ('duration_seconds', 6)):
            data = manifest(); data['source_metadata'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.check(data)

    def test_temporal_router_values_and_extras_fail(self):
        for key in manifest()['router_control']:
            data = manifest(); data['router_control'][key] = True
            with self.assertRaises(RuntimeError): self.check(data)
        data = manifest(); data['router_control']['new_field'] = False
        with self.assertRaises(RuntimeError): self.check(data)

    def test_all_provenance_fields_are_checked(self):
        for key in manifest()['provenance']:
            data = manifest(); data['provenance'].pop(key)
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.check(data)

    def test_decoded_audio_shape_or_duration_mismatch_rejects(self):
        for kind in ('audio', 'shape', 'duration'):
            p = probe()
            if kind == 'audio': p['metadata']['streams'].append(dict(codec_type='audio'))
            elif kind == 'shape': p['checks']['dimensions'] = [1344, 768]
            else: p['checks']['container_duration_seconds'] = 6
            with self.assertRaises(RuntimeError): self.check(decoded=p)


class StrictAndLoadTests(unittest.TestCase):
    def test_strict_actual_silent_color_one_or_two_requests(self):
        for count in (1, 2):
            result = cg.parse_strict_metadata(strict_logs(requests=count), set(range(500, 508)), SAMPLE, requests=count)
            self.assertEqual(result['request_count_per_pid'], count)
        for prefix in ('', '.', '..', '.'*64):
            cg.parse_strict_metadata(strict_logs(prefix=prefix), set(range(500, 508)), SAMPLE)

    def test_color_strict_rejects_source_audio_and_shape_change(self):
        for key, value in (('source_audio_t', 209), ('target_audio_t', 208), ('target_shape', (37, 48, 80)),
                           ('source_shape', (36, 48, 84)), ('mode', 'B'), ('reference_count', 2), ('patch_size', (2, 2, 2))):
            logs = strict_logs(mutate=lambda p, n, row: row.update({key: value}))
            with self.subTest(key=key), self.assertRaises(RuntimeError): cg.parse_strict_metadata(logs, set(range(500, 508)), SAMPLE)

    def test_strict_rejects_unknown_prefix_missing_duplicate_pid_and_malformed(self):
        good = strict_logs()
        for bad in (strict_logs(prefix='x'), strict_logs(prefix=' '), strict_logs(prefix='.'*65),
                    strict_logs(pids=range(500, 507)), strict_logs(pids=range(501, 509)), good+'\n'+good.splitlines()[0],
                    good.replace("'text_len': 123", "'text_len': evil()"), good.replace('H3C', 'H3B')):
            with self.assertRaises(RuntimeError): cg.parse_strict_metadata(bad, set(range(500, 508)), SAMPLE)

    def test_strict_rejects_cross_request_drift_and_old_sample(self):
        bad = strict_logs(requests=2, mutate=lambda p, n, row: row.update(text_len=123+n))
        with self.assertRaises(RuntimeError): cg.parse_strict_metadata(bad, set(range(500, 508)), SAMPLE, requests=2)
        for sample in ('lake_snow', 'explicit_reverse_couple_124'):
            with self.assertRaises(ValueError): cg.parse_strict_metadata(strict_logs(), set(range(500, 508)), sample)

    def test_eight_full_load_and_restore_records_required(self):
        for g, label in ((bg, 'H3B'), (cg, 'H3C')):
            self.assertEqual(set(g.parse_full_load_evidence(load_logs(label), weights())), set(range(500, 508)))
            for bad in (load_logs(label, range(500, 507)), load_logs(label, range(500, 509)),
                        load_logs(label, mutate=lambda load, cpu: load.update(branch_tensor_count=799)),
                        load_logs(label, mutate=lambda load, cpu: load.update(lora_pairs_merged=207)),
                        load_logs(label, mutate=lambda load, cpu: cpu.update(intraop_after=4))):
                with self.assertRaises(RuntimeError): g.parse_full_load_evidence(bad, weights())

    def test_c_missing_own_tiny_never_uses_b(self):
        with mock.patch.object(cg, 'c_validation_modules', side_effect=RuntimeError('missing exact C verifier')), \
             mock.patch.object(bg, 'tiny_gate') as b_tiny:
            with self.assertRaises(RuntimeError): cg.tiny_gate('01234567', HOST)
            b_tiny.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def test_each_request_order_no_retry_or_formal_without_smoke(self):
        for _, module, cls in CASES:
            own = owner(module, cls)
            with self.assertRaises(RuntimeError): own.request(cls.__name__[0].lower()+'_50step', 50)
            own._request_order = [('smoke_2step', 2)]
            with self.assertRaises(RuntimeError): own.request(cls.__name__[0].lower()+'_50step', 50)
            own._request_order = [('smoke_2step', 2), (cls.__name__[0].lower()+'_50step', 50)]
            with self.assertRaises(RuntimeError): own.request('smoke_2step', 2)

    def test_failed_smoke_gate_never_starts_formal(self):
        for _, module, cls in CASES:
            own = owner(module, cls); own.request = mock.Mock()
            own.smoke_gate = mock.Mock(side_effect=RuntimeError('smoke not proven'))
            with self.assertRaises(RuntimeError): own.run_requests()
            own.request.assert_called_once_with('smoke_2step', 2)
            self.assertEqual(own.status['formal_request_attempts'], 0)

    def test_same_service_smoke_then_exactly_one_formal(self):
        for g, module, cls in CASES:
            own = owner(module, cls); calls = []
            own.request = lambda name, steps: calls.append((name, steps)) or dict(success=True, ffprobe=dict(verified=True))
            own.smoke_gate = lambda: calls.append('gate')
            own.assert_unchanged = lambda: calls.append('unchanged')
            own.update = lambda phase, **values: own.status.update(values)
            with tempfile.TemporaryDirectory() as directory:
                own.run_dir = Path(directory)
                raw = strict_logs(requests=2).encode(); (own.run_dir/'server.log').write_bytes(raw)
                own.status['formal_smoke_gate'] = dict(server_log_prefix_bytes=0,
                    server_log_prefix_sha256=hashlib.sha256(b'').hexdigest(), loaded_records_by_pid={p: {} for p in range(500, 508)})
                with mock.patch.object(module.base.os, 'O_NOFOLLOW', 0, create=True): own.run_requests()
                if g is cg: self.assertTrue((own.run_dir/'formal_strict_metadata.json').is_file())
            self.assertEqual(calls, [('smoke_2step', 2), 'gate', (cls.__name__[0].lower()+'_50step', 50), 'unchanged'])
            self.assertEqual(own.status['formal_request_attempts'], 1)
            self.assertTrue(own.status['formal_50step_completed'])
            with self.assertRaises(RuntimeError): own.run_requests()

    def exercise_gate(self, case, corruption=None):
        g, module, cls = CASES[case]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); own = owner(module, cls); own.run_dir = root
            own.server = mock.Mock(pid=100, poll=mock.Mock(return_value=None))
            own.locks = [SimpleNamespace(closed=False) for _ in range(9)]
            own._request_order = [('smoke_2step', 2)]; own.assert_unchanged = mock.Mock()
            own.owned_group_members = mock.Mock(return_value=list(range(500, 508)))
            own.frozen = dict(sample=dict(source=dict(sha256='source-sha')), weights=weights())
            own.update = lambda phase, **values: own.status.update(values)
            server = dict(pid=100, start_ticks=1000, pgrp=100, session=100, state='S')
            own.status['server_proc_identity'] = server.copy()
            (root/'smoke_2step').mkdir(); (root/'output').mkdir()
            output = root/'output/smoke_2step.mp4'; output.write_bytes(b'CPU toy only')
            actual_probe = dict(verified=True, checks=dict(dimensions=[1344, 768], fps=24, frame_count=124))
            result = dict(success=True, requested_steps=2, http_code='200', curl_returncode=0,
                ffprobe=copy.deepcopy(actual_probe), output_video=str(output), output_sha256=ag.profiles.digest(output))
            fields = {k: g.GENERATION[k] for k in ('width', 'height', 'fps', 'flow_shift', 'seed')}
            fields.update(prompt=PROMPT, num_inference_steps=2, extra_params=json.dumps(dict(task='ref2va', duration=5.0, audio_flow_shift=3.0)))
            request = dict(host=HOST, group='01234567', run_id=own.run_id, sample_id=SAMPLE, source_sha256='source-sha',
                           source_video=str(own.sample.source), fields=fields, requested_steps=2, server_proc_identity=server.copy())
            if corruption == 'pid_reuse': server['start_ticks'] += 1
            elif corruption == 'foreign_workers': own.owned_group_members.return_value = [500]
            elif corruption == 'source': request['source_sha256'] = 'old'
            elif corruption == 'old_sample': request['sample_id'] = 'explicit_reverse_couple_124'
            elif corruption == 'prompt': fields['prompt'] = 'Reverse the clip'
            elif corruption == 'output': output.write_bytes(b'tampered')
            elif corruption == 'http': result['http_code'] = '500'
            elif corruption == 'frames': result['ffprobe']['checks']['frame_count'] = 120
            elif corruption == 'locks': own.locks.pop()
            logs = load_logs('H3C' if case == 2 else 'H3B', range(600, 608) if corruption == 'load_pids' else range(500, 508))
            if corruption != 'missing_strict': logs += '\n'+strict_logs()
            (root/'server.log').write_text(logs)
            own.status['smoke_2step'] = copy.deepcopy(result)
            (root/'smoke_2step/result.json').write_text(json.dumps(result))
            (root/'smoke_2step/request.json').write_text(json.dumps(request))
            def identity(pid):
                return server.copy() if pid == 100 else dict(pid=pid, pgrp=100, session=100, start_ticks=2000+pid, state='S')
            with mock.patch.object(ag.profiles, 'INSTALL', root), mock.patch.object(g, 'video_probe', return_value=actual_probe), \
                 mock.patch.object(module.base, 'proc_identity', side_effect=identity), \
                 mock.patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=smi(), stderr='')), \
                 mock.patch.object(module.base.os, 'O_NOFOLLOW', 0, create=True):
                own.smoke_gate()
            return own.status

    def test_actual_eight_worker_smoke_gates_pass_color(self):
        for case in range(3):
            status = self.exercise_gate(case)
            self.assertTrue(status['formal_smoke_gate_passed'])
            self.assertEqual(status['formal_smoke_gate']['sample_id'], SAMPLE)
            if case == 2: self.assertEqual(status['formal_smoke_gate']['strict_metadata']['request_count_per_pid'], 1)

    def test_actual_smoke_gates_fail_identity_source_http_metadata_locks(self):
        for case in range(3):
            for corruption in ('pid_reuse', 'foreign_workers', 'source', 'old_sample', 'prompt', 'output', 'http', 'frames', 'locks'):
                with self.subTest(case=case, corruption=corruption), self.assertRaises(RuntimeError): self.exercise_gate(case, corruption)
        for case in (1, 2):
            with self.assertRaises(RuntimeError): self.exercise_gate(case, 'load_pids')
        with self.assertRaises(RuntimeError): self.exercise_gate(2, 'missing_strict')

    def test_current_evidence_drift_fails(self):
        for _, module, cls in CASES:
            own = owner(module, cls); own.frozen = {'old': 1}
            own.evidence_snapshot = mock.Mock(return_value={'new': 1})
            with self.assertRaises(RuntimeError): own.assert_unchanged()

    def test_worker_table_and_resource_guards_keep_failing_closed(self):
        for _, module, _ in CASES:
            self.assertEqual(set(module.selected_workers(smi(), tuple(range(8))).values()), set(range(500, 508)))
            with self.assertRaises(RuntimeError): module.selected_workers(smi().replace('| 7 0 | 507 | VLLM::Worker | 3402 |', ''), tuple(range(8)))
        ag.base.selected_idle(smi(False), tuple(range(8)))
        with self.assertRaises(RuntimeError): ag.base.selected_idle(smi(), tuple(range(8)))
        ag.base.selected_health_memory(smi(), tuple(range(8)), 55*1024)
        with self.assertRaises(RuntimeError): ag.base.selected_health_memory(smi().replace('3402 / 65536', '64000 / 65536'), tuple(range(8)), 55*1024)

    def test_environment_drops_foreign_candidate_overrides(self):
        for index, (_, module, cls) in enumerate(CASES):
            own = owner(module, cls); own.run_dir = Path('/cpu/toy')
            with mock.patch.dict(os.environ, {'PYTHONPATH': '/foreign', 'ZHONGHAO_H3_OPENVDN_C_RULE': '1'}): env = own.env()
            self.assertNotIn('PYTHONPATH', env); self.assertNotIn('ZHONGHAO_H3_OPENVDN_C_RULE', env)
            self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
            self.assertEqual(env['H3_TRIAL_SAMPLE'], SAMPLE)
            self.assertEqual(env['ZHONGHAO_H3_OPENVDN'], '0' if index == 0 else '1')
            self.assertEqual(module.REQUEST_TIMEOUT, 1800 if index == 0 else 3600)


if __name__ == '__main__':
    unittest.main(verbosity=2)
