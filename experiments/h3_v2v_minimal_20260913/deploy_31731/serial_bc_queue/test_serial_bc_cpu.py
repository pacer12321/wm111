"""Pure CPU filesystem/process fixtures; no real child, environment or NPU starts."""
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
sys.path.insert(0, str(HERE.parent / 'trial_queue'))
sys.path.insert(0, '/cache/zhonghao/h3/trial_queue_code')
if os.name == 'nt': sys.modules.setdefault('fcntl', SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import serial_ab_queue as fs
import serial_bc_queue as q
for folder in ('validation', 'model_trial', 'b_model_trial', 'c_model_trial'):
    sys.path.insert(0, str(HERE.parent / folder))
for folder in ('validation_code', 'model_trial_code', 'b_model_trial_code', 'c_model_trial_code'):
    sys.path.insert(0, str(q.ROOT / folder))
import c_trial_gates as real_c_gates

HOST = dict(hostname='ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0',
            boot_id='12345678-abcd-abcd-abcd-123456789abc', machine='aarch64')
GEN = dict(width=1344, height=768, fps=24, duration_seconds=5.0, seed=4101,
           num_inference_steps=50, flow_shift=12.0, audio_flow_shift=3.0)
LAKE_SHA = 'e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9'
B_IDENT = dict(pid=q.B_PID, start_ticks=q.B_START, pgrp=q.B_PID, session=q.B_PID, state='S')
C_IDENT = dict(pid=8100, start_ticks=5000, pgrp=8100, session=8100, state='S')
T_IDENT = dict(pid=8000, start_ticks=4000, pgrp=8000, session=8000, state='S')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_gates(root):
    probe = dict(verified=True, checks=dict(dimensions=[1344, 768], fps=24, frame_count=124))
    common = dict(transfer={'verified': {'toy': 1}}, runtime={'passed': 'cpu_fixture'},
                  tiny={'synthetic_fixture': True}, sample={'source': {'sha256': LAKE_SHA}})
    def helper(case, port):
        return SimpleNamespace(GENERATION=GEN.copy(), selected_profile=lambda group, sample: port,
            parallelism=lambda port: dict(usp=8, num_gpus=8, ring=1, text_encoder_tp_size=8, vae_patch_parallel_size=8, port=port),
            source_code_manifest=lambda: {'case': case, 'fixture': 'hash'}, transfer_gate=lambda host: copy.deepcopy(common['transfer']),
            runtime_gate=lambda host, sources: copy.deepcopy(common['runtime']), tiny_gate=lambda group, host: copy.deepcopy(common['tiny']),
            sample_gate=lambda sample, transfer: copy.deepcopy(common['sample']), weight_manifest=lambda verified: {'535/800/208': True},
            sample_profile=lambda sample: SimpleNamespace(source=root / 'source.mp4', prompt='frozen lake fixture'),
            video_probe=mock.Mock(return_value=probe))
    b, c = helper('B', 19099), helper('C', 19100)
    c.b_gates = b
    a = SimpleNamespace(LAKE_SHA256=LAKE_SHA)
    b.a_gates = c.a_gates = a
    c.profiles = SimpleNamespace(require_host=mock.Mock(return_value=HOST))
    b.base = c.base = SimpleNamespace(proc_identity=mock.Mock(return_value=None),
        selected_idle=mock.Mock(return_value={'idle': list(range(8))}), selected_health_memory=mock.Mock(return_value={'OK': True}))
    c.c_validation_modules = lambda: (SimpleNamespace(CASE='C_strict_tiny_SP8_31731_v1'), None)
    c.parse_strict_metadata = real_c_gates.parse_strict_metadata
    return c


class Fixtures(unittest.TestCase):
    def fixture(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        bpath = root / q.B_STATUS.relative_to(q.ROOT)
        children = {kind: {**data, 'code': root / data['code'].relative_to(q.ROOT),
                          'runs': root / data['runs'].relative_to(q.ROOT)} for kind, data in q.CHILDREN.items()}
        for module, name, value in ((fs, 'ROOT', root), (q, 'ROOT', root), (q, 'OUTPUT', root/'serial_bc_queue'),
                                    (q, 'CODE', root/'serial_bc_queue_code'), (q, 'B_STATUS', bpath), (q, 'CHILDREN', children)):
            self.enterContext(mock.patch.object(module, name, value))
        self.enterContext(mock.patch.object(os, 'O_NOFOLLOW', getattr(os, 'O_NOFOLLOW', 0), create=True))
        if os.name == 'nt':
            self.enterContext(mock.patch.object(os, 'getuid', return_value=0, create=True))
            self.enterContext(mock.patch.object(os, 'O_NONBLOCK', 0, create=True))
            self.enterContext(mock.patch.object(fs, 'sync_directory'))
        gates = fixture_gates(root)
        owner = q.BCQueue(fs, gates)
        self.addCleanup(owner.release)
        owner.update = lambda phase, **values: owner.status.update(values, phase=phase)
        return root, owner

    def status(self, owner, kind, phase=None):
        if kind == 'B': path, run_id, identity = q.B_STATUS, q.B_RUN_ID, B_IDENT
        else:
            run_id = ('c' if kind == 'C_full' else 'd') * 32
            name = ('c_' if kind == 'C_full' else '') + '20260913T150000Z_' + run_id
            path = q.CHILDREN[kind]['runs'] / name / q.CHILDREN[kind]['status']
            identity = C_IDENT if kind == 'C_full' else T_IDENT
        case = 'B' if kind == 'B' else ('C' if kind == 'C_full' else 'C_strict_tiny_SP8_31731_v1')
        value = dict(case=case, host=HOST, group=q.GROUP, run_id=run_id, run_directory=str(path.parent),
                     supervisor_pid=identity['pid'], supervisor_proc_identity=identity.copy(), allocated_physical_npu_ids=list(q.CARDS),
                     cleanup_completed=True, selected_cards_verified_idle_after_cleanup=True, needs_attention=False,
                     remaining_owned_process_groups={}, error=None)
        if kind == 'C_tiny':
            value.update(phase=phase or 'completed', validation_passed=True)
            owner.gates.tiny_gate = mock.Mock(return_value={'status': value, 'result': {'actual_gate_fixture': True}})
        else:
            helper = owner.gates.b_gates if kind == 'B' else owner.gates
            params = helper.parallelism(helper.selected_profile(q.GROUP, q.SAMPLE))
            value.update(sample_id=q.SAMPLE, parallelism=params, requested_generation=GEN,
                         phase=phase or 'formal_completed_review_required', formal_50step_started=True, formal_50step_completed=True,
                         formal_request_attempts=1, formal_smoke_gate_passed=True)
            run, name = path.parent, case.lower() + '_50step'
            output = run/'output'/(name+'.mp4'); output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b'CPU fixture not a real video')
            result = dict(success=True, requested_steps=50, http_code='200', curl_returncode=0, content_type='video/mp4',
                          output_video=str(output), output_sha256=q.digest(output), ffprobe=helper.video_probe(output, target=True),
                          request_end_to_end_seconds=1234.5)
            fields = {k: GEN[k] for k in ('width','height','fps','flow_shift','seed')}
            fields.update(prompt=helper.sample_profile(q.SAMPLE).prompt, num_inference_steps=50,
                          extra_params=json.dumps(dict(task='ref2va',duration=5.0,audio_flow_shift=3.0)))
            request = dict(run_id=run_id, host=HOST, group=q.GROUP, sample_id=q.SAMPLE, fields=fields, requested_steps=50,
                           source_video=str(helper.sample_profile(q.SAMPLE).source), source_sha256=LAKE_SHA)
            gate = dict(status='passed', host=HOST, run_id=run_id, group=q.GROUP, sample_id=q.SAMPLE)
            sources, transfer = helper.source_code_manifest(), helper.transfer_gate(HOST)
            frozen = dict(host=HOST, run_id=run_id, cards=list(q.CARDS), parallelism=params, generation=GEN, sources=sources,
                transfer=transfer, runtime=helper.runtime_gate(HOST,sources), tiny=helper.tiny_gate(q.GROUP,HOST),
                sample=helper.sample_gate(q.SAMPLE,transfer=transfer),weights=helper.weight_manifest(transfer['verified']))
            if kind == 'C_full':
                pids = set(range(8200,8208))
                metadata = dict(mode=real_c_gates.STRICT_MODE, reference_count=1, reference_kind='video',
                                patch_size=(1,2,2), source_shape=(37,48,84), target_shape=(37,48,84),
                                text_len=96, source_audio_t=207, target_audio_t=207)
                block = ''.join(f'H3C pid={pid} INFO vllm_omni.pipeline {real_c_gates.STRICT_MARKER}{metadata!r}\n' for pid in sorted(pids)).encode()
                raw = block + block
                (run/'server.log').write_bytes(raw)
                gate.update(server_identity=dict(pid=8190,start_ticks=1,pgrp=8190,session=8190),
                            worker_identities_by_card={str(card):dict(pid=8200+card,start_ticks=100+card,pgrp=8190,session=8190) for card in range(8)},
                            loaded_records_by_pid={str(pid):{'fixture_load':True} for pid in sorted(pids)},
                            server_log_prefix_bytes=len(block),server_log_prefix_sha256=hashlib.sha256(block).hexdigest(),
                            strict_metadata=real_c_gates.parse_strict_metadata(block.decode(),pids,q.SAMPLE,requests=1))
                dump(run/'formal_strict_metadata.json',real_c_gates.parse_strict_metadata(raw.decode(),pids,q.SAMPLE,requests=2))
            for file, data in ((run/'frozen_evidence.json',frozen),(run/'formal_smoke_gate.json',gate),
                               (run/name/'request.json',request),(run/name/'result.json',result)):
                dump(file,data)
            value.update(formal_smoke_gate=gate, frozen_evidence_sha256=q.digest(run/'frozen_evidence.json'), **{name:result})
        dump(path,value)
        owner.bindings[kind] = owner.binding(value, kind, path)
        return path,value

    def ready_launch(self):
        root, owner = self.fixture()
        owner.directory.mkdir(parents=True)
        owner.lock = (owner.directory/'fixture.lock').open('w+b')
        owner.assert_frozen = mock.Mock(); owner.fresh_idle = mock.Mock()
        owner.completed_evidence = mock.Mock(return_value={'completed':'fixture'})
        owner.completed['B'] = {'completed':'fixture'}; owner.bindings['B'] = {'run_id': q.B_RUN_ID}
        owner.child_environment = mock.Mock(return_value={'fixture':'no-real-environment'})
        owner.gates.base.proc_identity.return_value = T_IDENT.copy()
        return root,owner


class ContractTests(Fixtures):
    def test_fixed_tuple_paths_and_optin_before_dependency_import(self):
        self.assertEqual(q.JOBS, ('C_tiny','C_full'))
        self.assertEqual((q.B_RUN_ID,q.B_PID,q.B_START), ('6362ce1811da4db69d335b3b3ca5d1a4',4692,124486422))
        with mock.patch.object(q,'load_dependencies') as load, mock.patch('sys.stderr',new=io.StringIO()):
            for args in ([],['--allow-c-chain','--sample','reverse'],['--allow-c-chain','--port','30213']):
                with self.assertRaises(SystemExit): q.main(args)
            load.assert_not_called()

    def test_real_pinned_local_dependencies_unchanged(self):
        maps = {'trial_queue_code':'trial_queue','c_model_trial_code':'c_model_trial'}
        for name,sha in q.PINS.items():
            head,tail = name.split('/',1)
            path = HERE.parent/maps[head]/tail
            if not path.is_file(): path = q.ROOT/name
            self.assertEqual(q.digest(path),sha)

    def test_bootstrap_sources_env_and_checks_real_executables_for_each_child(self):
        script=(HERE/'bootstrap_bc.sh').read_text()
        self.assertLess(script.index('source /cache/zhonghao/h3/env_h3_31731.sh'),script.index('command -v npu-smi'))
        self.assertIn('/usr/local/sbin/npu-smi',script)
        self.assertIn('command -v ffmpeg',script)
        self.assertIn('run_c_validation.py --allow-npu',script)
        self.assertIn('run_c_trial.py --group 01234567 --sample lake_snow --allow-npu',script)
        self.assertNotIn('run_b_trial.py',script)
        self.assertNotIn('deploy_candidate',script)

    def test_environment_resolution_success_and_path_omission_wrong_binary_fail(self):
        env={'PATH':'/cache/zhonghao/h3/bin:/cache/zhonghao/h3/env/bin:/usr/local/sbin:/bin',
             'ASCEND_HOME_PATH':str(q.ROOT/'env/Ascend/cann-9.0.1')}
        mapping={'npu-smi':'/usr/local/sbin/npu-smi','python':str(q.ROOT/'env/bin/python'),
                 'ffmpeg':str(q.ROOT/'bin/ffmpeg'),'bash':'/bin/bash'}
        info=SimpleNamespace(st_size=10,st_dev=1,st_ino=2,st_mtime_ns=3)
        with mock.patch.object(q.shutil,'which',side_effect=lambda name,path: mapping[name]), \
             mock.patch.object(Path,'resolve',lambda p:p),mock.patch.object(Path,'is_file',return_value=True), \
             mock.patch.object(Path,'stat',return_value=info),mock.patch.object(os,'access',return_value=True):
            self.assertEqual(set(q.check_environment(env)),set(mapping))
            with self.assertRaisesRegex(RuntimeError,'sbin'): q.check_environment({**env,'PATH':'/bin'})
            with self.assertRaisesRegex(RuntimeError,'CANN9'): q.check_environment({**env,'ASCEND_HOME_PATH':'/old'})
            for key in mapping:
                old=mapping[key]; mapping[key]='/unapproved/tool'
                with self.subTest(key=key),self.assertRaises(RuntimeError): q.check_environment(env)
                mapping[key]=old

    def test_stale_or_foreign_b_status_not_accepted(self):
        _,owner=self.fixture(); path,value=self.status(owner,'B')
        for key,bad in (('run_id','x'*32),('host',{**HOST,'boot_id':'old'}),('allocated_physical_npu_ids',[0,1,2,3]),
                         ('sample_id','reverse'),('supervisor_pid',q.B_PID+1),('requested_generation',{**GEN,'seed':2})):
            with self.subTest(key=key),self.assertRaises(RuntimeError): owner.binding({**value,key:bad},'B',path)
        with self.assertRaises(RuntimeError): owner.binding(value,'B',path.parent.parent/'b_status.json')

    def test_actual_reused_formal_artifact_reader_accepts_b_and_c(self):
        _,owner=self.fixture()
        for kind in ('B','C_full'):
            self.status(owner,kind)
            proof=owner.completed_evidence(kind)
            self.assertEqual(proof['request_end_to_end_seconds'],1234.5)
            self.assertFalse(proof['quality_evidence'])
        self.assertIn('strict_metadata_record',proof)

    def test_b_requires_real_completion_cleanup_artifacts_and_exit(self):
        _,owner=self.fixture(); path,value=self.status(owner,'B')
        for key,bad in (('formal_50step_completed',False),('formal_request_attempts',2),('cleanup_completed',False),
                         ('selected_cards_verified_idle_after_cleanup',False),('error','fail'),('phase','failed')):
            dump(path,{**value,key:bad})
            with self.subTest(key=key),self.assertRaises(RuntimeError): owner.completed_evidence('B')
        dump(path,value)
        owner.gates.base.proc_identity.return_value=B_IDENT.copy()
        with self.assertRaisesRegex(RuntimeError,'not exited'): owner.completed_evidence('B')
        owner.gates.base.proc_identity.return_value={**B_IDENT,'start_ticks':q.B_START+1}
        owner.completed_evidence('B')  # PID reuse means original is gone; never signal reused PID.

    def test_tiny_final_gate_must_return_exact_bound_run(self):
        _,owner=self.fixture(); _,value=self.status(owner,'C_tiny')
        owner.completed_evidence('C_tiny')
        owner.gates.tiny_gate.assert_called_once_with(q.GROUP,HOST)
        owner.gates.tiny_gate.return_value={'status':{**value,'run_id':'e'*32}}
        with self.assertRaises(RuntimeError): owner.completed_evidence('C_tiny')

    def test_no_model_packages_imported(self):
        for name in ('torch','torch_npu','vllm','vllm_omni'): self.assertNotIn(name,sys.modules)

    def test_formal_strict_file_malformed_or_tampered_is_rejected(self):
        _,owner=self.fixture()
        for corruption in ('not_json','wrong_value','missing_worker','wrong_request_count'):
            path,_=self.status(owner,'C_full');strict_path=path.parent/'formal_strict_metadata.json'
            if corruption=='not_json': strict_path.write_text('not JSON and not strict metadata')
            else:
                record=fs.read_json(strict_path)
                if corruption=='wrong_value': record['records_by_pid']['8200'][0]['mode']='B'
                elif corruption=='missing_worker': record['records_by_pid'].pop('8207')
                else: record['request_count_per_pid']=1
                dump(strict_path,record)
            with self.subTest(corruption=corruption),self.assertRaises((RuntimeError,json.JSONDecodeError)):
                owner.completed_evidence('C_full')

    def test_strict_log_prefix_changed_truncated_or_invalid_size_rejected(self):
        _,owner=self.fixture()
        for corruption in ('prefix_changed','truncated','negative_size','bool_size','oversized','bad_sha'):
            path,value=self.status(owner,'C_full');run=path.parent;gate=value['formal_smoke_gate']
            raw=(run/'server.log').read_bytes()
            if corruption=='prefix_changed': (run/'server.log').write_bytes(raw.replace(b'INFO',b'WARN',1))
            elif corruption=='truncated': (run/'server.log').write_bytes(raw[:10])
            else:
                if corruption=='negative_size': gate['server_log_prefix_bytes']=-1
                elif corruption=='bool_size': gate['server_log_prefix_bytes']=True
                elif corruption=='oversized': gate['server_log_prefix_bytes']=len(raw)+1
                else: gate['server_log_prefix_sha256']='0'*64
                dump(run/'formal_smoke_gate.json',gate);dump(path,value)
            with self.subTest(corruption=corruption),self.assertRaises(RuntimeError): owner.completed_evidence('C_full')

    def test_wrong_pid_or_worker_group_and_smoke_metadata_rejected(self):
        _,owner=self.fixture()
        for corruption in ('load_pid','worker_pid','worker_group','missing_card','saved_smoke','formal_log_pid','formal_log_mode'):
            path,value=self.status(owner,'C_full');run=path.parent;gate=value['formal_smoke_gate']
            raw=(run/'server.log').read_bytes();cut=gate['server_log_prefix_bytes']
            if corruption=='load_pid': gate['loaded_records_by_pid']['9999']=gate['loaded_records_by_pid'].pop('8200')
            elif corruption=='worker_pid': gate['worker_identities_by_card']['0']['pid']=9999
            elif corruption=='worker_group': gate['worker_identities_by_card']['0']['pgrp']=9999
            elif corruption=='missing_card': gate['worker_identities_by_card'].pop('7')
            elif corruption=='saved_smoke': gate['strict_metadata']={}
            elif corruption=='formal_log_pid': (run/'server.log').write_bytes(raw[:cut]+raw[cut:].replace(b'pid=8200',b'pid=9999',1))
            else: (run/'server.log').write_bytes(raw[:cut]+raw[cut:].replace(real_c_gates.STRICT_MODE.encode(),b'B',1))
            dump(run/'formal_smoke_gate.json',gate);dump(path,value)
            with self.subTest(corruption=corruption),self.assertRaises(RuntimeError): owner.completed_evidence('C_full')


class QueueTests(Fixtures):
    def test_b_failure_and_bounded_wait_never_launch(self):
        _,owner=self.fixture(); self.status(owner,'B','failed_before_cleanup')
        owner.assert_frozen=mock.Mock()
        owner.gates.base.proc_identity.return_value=B_IDENT.copy()
        with mock.patch.object(q.subprocess,'Popen') as popen:
            with self.assertRaises(RuntimeError): owner.wait_for_b()
            self.status(owner,'B','b_50step_request_running')
            with mock.patch.object(q.time,'monotonic',side_effect=[0,0,q.B_TIMEOUT+1]),mock.patch.object(q.time,'sleep'):
                with self.assertRaises(TimeoutError): owner.wait_for_b()
            popen.assert_not_called()

    def test_terminal_b_waits_for_own_exit_then_idle(self):
        _,owner=self.fixture(); self.status(owner,'B')
        owner.directory.mkdir(parents=True); owner.assert_frozen=mock.Mock(); owner.fresh_idle=mock.Mock()
        owner.gates.base.proc_identity.side_effect=[B_IDENT.copy(),None,None,None]
        with mock.patch.object(q.time,'sleep') as sleep: owner.wait_for_b()
        sleep.assert_called_once_with(30)
        owner.fresh_idle.assert_called_once_with('after_b')
        self.assertIn('B',owner.completed)

    def test_fsync_intent_precedes_single_popen_and_retry_refused(self):
        _,owner=self.ready_launch()
        def spawn(command,**kwargs):
            self.assertTrue((owner.directory/'C_tiny_launch_intent.json').is_file())
            self.assertEqual(owner.status['launches']['C_tiny'],1)
            self.assertEqual(command,['/bin/bash',str(q.CODE/'bootstrap_bc.sh'),'C_tiny'])
            self.assertEqual(kwargs['pass_fds'],(owner.lock.fileno(),))
            return mock.Mock(pid=T_IDENT['pid'],poll=mock.Mock(return_value=None))
        with mock.patch.object(q.subprocess,'Popen',side_effect=spawn) as popen:
            owner.launch_once('C_tiny')
            with self.assertRaises(RuntimeError): owner.launch_once('C_tiny')
        popen.assert_called_once()

    def test_failed_spawn_and_leftover_intent_do_not_retry(self):
        _,owner=self.ready_launch()
        with mock.patch.object(q.subprocess,'Popen',side_effect=OSError('no spawn')) as popen:
            with self.assertRaises(OSError): owner.launch_once('C_tiny')
            with self.assertRaises(RuntimeError): owner.launch_once('C_tiny')
            popen.assert_called_once()
        self.assertTrue((owner.directory/'C_tiny_launch_intent.json').is_file())

    def test_existing_tombstone_blocks_popen_even_with_fresh_python_object(self):
        _,owner=self.ready_launch()
        fs.exclusive_json(owner.directory/'C_tiny_launch_intent.json',{'crash_before_spawn':True})
        with mock.patch.object(q.subprocess,'Popen') as popen:
            with self.assertRaises(FileExistsError): owner.launch_once('C_tiny')
            popen.assert_not_called()

    def test_full_c_requires_tiny_completion_and_no_other_sample(self):
        _,owner=self.ready_launch()
        with mock.patch.object(q.subprocess,'Popen') as popen:
            for kind in ('C_full','reverse','A','B'):
                with self.assertRaises(RuntimeError): owner.launch_once(kind)
            popen.assert_not_called()

    def test_discovery_binds_only_new_exact_owned_per_run_not_latest(self):
        _,owner=self.fixture(); path,value=self.status(owner,'C_tiny')
        owner.bindings.pop('C_tiny'); owner.directory.mkdir(parents=True)
        owner.child=mock.Mock(pid=T_IDENT['pid']); owner.child_identity=fs.stable_identity(T_IDENT)
        dump(path.parent.parent.parent/'c_validation_status.json',{'malicious_latest':'ignored'})
        owner.before_runs={str(path.parent)}
        self.assertIsNone(owner.discover_child('C_tiny'))
        owner.before_runs=set()
        bad={**value,'supervisor_proc_identity':{**T_IDENT,'start_ticks':1}}
        dump(path,bad); self.assertIsNone(owner.discover_child('C_tiny'))
        dump(path,value); self.assertEqual(owner.discover_child('C_tiny'),value)
        self.assertEqual(owner.bindings['C_tiny']['status_path'],str(path))
        dump(path,bad)
        with self.assertRaises(RuntimeError): owner.discover_child('C_tiny')

    def test_child_failure_never_reaches_next_stage_and_timeout_bounded(self):
        _,owner=self.fixture(); owner.assert_frozen=mock.Mock(); owner.discover_child=mock.Mock(return_value=None)
        owner.child=mock.Mock(poll=mock.Mock(return_value=1))
        with self.assertRaises(RuntimeError): owner.wait_child('C_tiny')
        owner.child.poll.return_value=None
        with mock.patch.object(q.time,'monotonic',side_effect=[0,q.CHILDREN['C_tiny']['timeout']+1]):
            with self.assertRaises(TimeoutError): owner.wait_child('C_tiny')
        self.assertEqual(owner.status['launches']['C_full'],0)

    def test_both_owned_children_finish_exact_gate_exit_and_idle(self):
        _,owner=self.fixture();owner.directory.mkdir(parents=True)
        owner.assert_frozen=mock.Mock();owner.fresh_idle=mock.Mock()
        for kind,identity in (('C_tiny',T_IDENT),('C_full',C_IDENT)):
            self.status(owner,kind)
            owner.child=mock.Mock(pid=identity['pid'],poll=mock.Mock(return_value=0))
            owner.child_identity=fs.stable_identity(identity)
            owner.wait_child(kind)
            self.assertIn(kind,owner.completed)
            self.assertTrue((owner.directory/('completed_'+kind+'.json')).is_file())
        self.assertEqual([c.args[0] for c in owner.fresh_idle.call_args_list],['after_C_tiny','after_C_full'])

    def test_child_environment_preserves_verified_sbin_and_clears_foreign_override(self):
        _,owner=self.fixture()
        path='/cache/zhonghao/h3/bin:/cache/zhonghao/h3/env/bin:/usr/local/sbin:/usr/bin:/bin'
        with mock.patch.object(q,'check_environment'),mock.patch.dict(os.environ,{'PATH':path,'H3_OLD_RUN':'wrong',
            'ZHONGHAO_H3_OPENVDN':'0','PYTHONPATH':'/foreign','ASCEND_RT_VISIBLE_DEVICES':'2,3'}):
            env=owner.child_environment('C_tiny')
        self.assertEqual(env['PATH'],path)
        self.assertEqual(env['H3_SERIAL_BC_QUEUE_ID'],owner.queue_id)
        for key in ('H3_OLD_RUN','ZHONGHAO_H3_OPENVDN','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES'):
            self.assertNotIn(key,env)

    def test_stop_only_own_marked_pid_never_b_or_reused_process(self):
        _,owner=self.fixture(); owner.child=mock.Mock(pid=T_IDENT['pid'],poll=mock.Mock(return_value=None))
        owner.child_identity=fs.stable_identity(T_IDENT); owner.gates.base.proc_identity.return_value=T_IDENT.copy()
        for marked in (False,True):
            marker=(f'H3_SERIAL_BC_QUEUE_ID={owner.queue_id}' if marked else 'OTHER=1').encode()+b'\0'
            with mock.patch.object(Path,'read_bytes',return_value=marker),mock.patch.object(os,'kill') as kill:
                self.assertEqual(owner.stop_owned_child(),marked)
                if marked: kill.assert_called_once_with(T_IDENT['pid'],signal.SIGTERM)
                else: kill.assert_not_called()
        owner.gates.base.proc_identity.return_value={**T_IDENT,'start_ticks':2}
        with mock.patch.object(os,'kill') as kill:
            self.assertFalse(owner.stop_owned_child()); kill.assert_not_called()

    def test_run_fixed_order_and_no_next_after_tiny_failure(self):
        _,owner=self.fixture(); calls=[]
        owner.acquire=lambda:calls.append('acquire');owner.wait_for_b=lambda:calls.append('B')
        owner.launch_once=lambda kind:calls.append('launch '+kind)
        owner.wait_child=lambda kind:calls.append('wait '+kind)
        owner.run()
        self.assertEqual(calls,['acquire','B','launch C_tiny','wait C_tiny','launch C_full','wait C_full'])
        calls.clear();owner.wait_child=mock.Mock(side_effect=RuntimeError('tiny failed'))
        with self.assertRaises(RuntimeError): owner.run()
        self.assertEqual(calls,['acquire','B','launch C_tiny'])

    def test_failed_main_releases_queue_and_has_no_retry(self):
        fake=mock.Mock(created=True);fake.run.side_effect=RuntimeError('stop');fake.stop_owned_child.return_value=True
        with mock.patch.object(q,'load_dependencies',return_value=(fs,None)),mock.patch.object(q,'BCQueue',return_value=fake), \
             mock.patch.object(signal,'signal'):
            self.assertEqual(q.main(['--allow-c-chain']),1)
        fake.stop_owned_child.assert_called_once();fake.release.assert_called_once()

    def test_missing_preparation_does_not_consume_ledger_or_start_child(self):
        _,owner=self.fixture();owner.production_manifest=mock.Mock(side_effect=RuntimeError('C not deployed'))
        with mock.patch.object(q.subprocess,'Popen') as popen:
            with self.assertRaises(RuntimeError): owner.acquire()
            self.assertFalse(owner.directory.exists());self.assertFalse(owner.created);popen.assert_not_called()

    def test_durable_queue_directory_refuses_restart_without_overwrite(self):
        _,owner=self.fixture();self.status(owner,'B')
        with mock.patch.object(owner,'production_manifest',return_value={'frozen':True}), \
             mock.patch.object(q,'check_environment',return_value={'correct':True}):
            owner.acquire();owner.release();owner.lock=None
            sentinel=owner.directory/'preserve.json';dump(sentinel,{'keep':True})
            second=q.BCQueue(fs,owner.gates)
            try:
                with mock.patch.object(second,'production_manifest',return_value={'frozen':True}):
                    with self.assertRaises(FileExistsError): second.acquire()
                self.assertFalse(second.created);self.assertEqual(fs.read_json(sentinel),{'keep':True})
            finally:second.release()


if __name__ == '__main__': unittest.main(verbosity=2)
