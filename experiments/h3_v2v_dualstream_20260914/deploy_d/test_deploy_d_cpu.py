"""Synthetic CPU fixtures only: not real VLM, NPU, teacher or D execution proof."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_contract as c
import d_execution_verifier as e

RUN = "1" * 32
SHA = "a" * 64
WORKERS = set(range(100, 108))
HOST = {"hostname": "synthetic_host_not_actual", "boot_id": "synthetic_boot", "machine": "aarch64"}
IDENTITY = {"pid": 123, "start_ticks": 456, "pgrp": 123, "session": 123}


def policy():
    return {
        "schema_version": 1, "case": "D", "mode": c.MODE,
        "review_status": "approved_for_inference", "policy_revision": "synthetic_test_only",
        "semantic_decisions_complete": True, "training_performed": False,
        "sample_id": c.SAMPLE, "group": c.GROUP, "candidate_vendor": c.VENDOR.as_posix(),
        "candidate_files": {name: SHA for name in c.REQUIRED_CANDIDATE_FILES},
        "execution_verifier_sha256": SHA, "tiny_verifier_sha256": SHA,
        "validation_pins_path": (c.EXPERIMENT/"validation_d/pins.json").as_posix(),
        "validation_pins_sha256": SHA,
        "tiny_proof": {"case": "D_tiny_SP8", "mode": c.MODE, "run_id": RUN,
                       "run_directory": str(c.EXPERIMENT/"validation_results/01234567/runs"/("20260914T000000Z_"+RUN)),
                       "status_sha256": SHA, "result_sha256": SHA, "source_manifest_sha256": SHA},
        "attention": {"endpoint_policy": "vdn_anchors", "auxiliary_policy": "preserve_existing",
                      "source_linear_text": "text", "target_linear_text": "text",
                      "chunk_frames": 5, "chunk_radius": 1,
                      "share_parameters": True, "independent_stream_states": True},
        "execution_contract": copy.deepcopy(e.EXPECTED_EXECUTION_CONTRACT),
    }


def row(rank=0, layer=0, request=1):
    return dict(mode=c.MODE, request_index=request, forward_index=1, layer_index=layer,
                rank=rank, world=8, endpoint_policy="vdn_anchors", auxiliary_policy="preserve_existing",
                policy_sha256=SHA, module_sha256=SHA, module_path=(c.VENDOR/e.MODULE_FILE).as_posix(),
                source_frames=37, target_frames=37,
                source_tokens_per_frame=1008, target_tokens_per_frame=1008, heads=7,
                device="npu:"+str(rank), dtype="torch.bfloat16",
                softmax_source_completed=True, softmax_target_completed=True,
                linear_source_completed=True, linear_target_completed=True)


def line(value, pid=None, prefix=""):
    return prefix + "H3D pid=" + str(100+value["rank"] if pid is None else pid) + " 2026-09-14 INFO fake D_EXECUTION_RECORD " + json.dumps(value)


def logs(requests=1):
    return "\n".join(line(row(rank, layer, request)) for request in range(1, requests+1)
                     for rank in range(8) for layer in range(50))


def parse(log, *, pol=None, requests=1):
    return e.parse_execution_records(log, WORKERS, run_id=RUN, policy=pol or policy(),
                                      policy_sha256=SHA, requests=requests)


class PolicyTests(unittest.TestCase):
    def test_fixture_valid_not_real_admission(self):
        self.assertEqual(c.validate_policy(policy())["mode"], c.MODE)

    def test_missing_approval_training_wrong_case_or_mode_rejected(self):
        for key, value in (("review_status", "draft"), ("training_performed", True),
                           ("semantic_decisions_complete", False), ("case", "C"),
                           ("mode", "S0_dual_stream_vdn_local_linear_v1"), ("schema_version", True)):
            with self.subTest(key=key):
                p = policy(); p[key] = value
                with self.assertRaises(RuntimeError): c.validate_policy(p)

    def test_unapproved_attention_changes_rejected(self):
        for key, value in (("endpoint_policy", "strict_local"), ("auxiliary_policy", "source_closed"),
                           ("source_linear_text", "none"), ("target_linear_text", "none"),
                           ("chunk_frames", 1), ("chunk_radius", True), ("share_parameters", False),
                           ("independent_stream_states", False)):
            with self.subTest(key=key):
                p = policy(); p["attention"][key] = value
                with self.assertRaises(RuntimeError): c.validate_policy(p)

    def test_missing_candidate_and_hash_rejected(self):
        p = policy(); p["candidate_files"].pop(e.MODULE_FILE)
        with self.assertRaises(RuntimeError): c.validate_policy(p)
        for key in ("execution_verifier_sha256", "tiny_verifier_sha256"):
            p=policy(); p[key] = "0"*63
            with self.assertRaises(RuntimeError): c.validate_policy(p)

    def test_candidate_escape_rejected(self):
        for path in ("/root/p.py", "vllm_omni/../p.py", "vllm_omni//p.py", "vllm_omni/./p.py", "vllm_omni\\p.py"):
            p=policy(); p["candidate_files"][path] = SHA
            with self.subTest(path=path), self.assertRaises(RuntimeError): c.validate_policy(p)

    def test_old_tiny_or_latest_rejected(self):
        for key,value in (("case","C_strict_tiny_SP8_31731_v1"),("mode","C_old"),
                          ("run_directory","/cache/zhonghao/h3/c_validation/latest")):
            p=policy(); p["tiny_proof"][key]=value
            with self.subTest(key=key), self.assertRaises(RuntimeError): c.validate_policy(p)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(RuntimeError): c.unique_json(raw)

    def test_policy_gate_requires_existing_exact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/"reviewed_policy.json"
            with patch.object(c,"POLICY",path):
                with self.assertRaises(FileNotFoundError): c.policy_gate(SHA)
                raw=json.dumps(policy()).encode(); path.write_bytes(raw)
                with self.assertRaises(RuntimeError): c.policy_gate("b"*64)
                self.assertEqual(c.policy_gate(c.digest_bytes(raw))["value"]["case"],"D")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.experiment=Path(self.tmp.name).resolve()
        self.patcher=patch.object(c,"EXPERIMENT",self.experiment); self.patcher.start()
        self.output=self.experiment/"results"/c.GROUP/c.SAMPLE/"D"; self.output.mkdir(parents=True)
    def tearDown(self):
        self.patcher.stop(); self.tmp.cleanup()
    def claim(self, run_id=RUN):
        return c.claim_submission(self.output,run_id,HOST,SHA,IDENTITY,"synthetic_timestamp")
    def test_claim_and_verify(self):
        claim=self.claim()
        self.assertEqual(c.verify_submission(claim,self.output,RUN,HOST,SHA),claim)
    def test_duplicate_new_run_and_changed_policy_do_not_overwrite(self):
        original=self.claim(); before=(self.output/c.LEDGER_NAME).read_bytes()
        with self.assertRaises(FileExistsError): self.claim("2"*32)
        with self.assertRaises(FileExistsError):
            c.claim_submission(self.output,RUN,HOST,"b"*64,IDENTITY,"synthetic_timestamp")
        self.assertEqual((self.output/c.LEDGER_NAME).read_bytes(),before)
    def test_incomplete_previous_claim_is_not_resumed(self):
        (self.output/c.LEDGER_NAME).write_bytes(b"{")
        with self.assertRaises(FileExistsError): self.claim()
    def test_ledger_tamper_rejected(self):
        claim=self.claim(); path=self.output/c.LEDGER_NAME
        value=json.loads(path.read_text()); value["case"]="C"; path.write_text(json.dumps(value))
        with self.assertRaises(RuntimeError): c.verify_submission(claim,self.output,RUN,HOST,SHA)
    def test_wrong_output_scope_rejected(self):
        wrong=self.experiment/"results"/c.GROUP/c.SAMPLE/"C"; wrong.mkdir()
        with self.assertRaises(RuntimeError): c.claim_submission(wrong,RUN,HOST,SHA,IDENTITY,"test")
    def test_wrong_identity_rejected(self):
        identity=dict(IDENTITY,start_ticks=False)
        with self.assertRaises(RuntimeError): c.claim_submission(self.output,RUN,HOST,SHA,identity,"test")
    def test_empty_existing_record_rejected(self):
        path=self.output/"empty.json"; path.touch()
        with self.assertRaises(RuntimeError): c.read_record(path)


class ExecutionTests(unittest.TestCase):
    def test_complete_smoke_400_and_formal_800(self):
        for requests in (1,2):
            result=parse(logs(requests),requests=requests)
            self.assertEqual(result["record_count"],400*requests)
            self.assertEqual(set(result["rank_by_pid"].values()),set(range(8)))
    def test_missing_layer_worker_or_request_rejected(self):
        for filtered,requests in (("\n".join(logs().splitlines()[1:]),1),
                                  ("\n".join(logs().splitlines()[50:]),1),
                                  (logs(1),2)):
            with self.assertRaises(RuntimeError): parse(filtered,requests=requests)
    def test_duplicate_unknownpid_or_excessrequest_rejected(self):
        for changed in (logs()+"\n"+line(row()), logs()+"\n"+line(row(),pid=999),
                        logs()+"\n"+line(row(request=2))):
            with self.assertRaises(RuntimeError): parse(changed)
    def test_old_c_or_s0_headers_and_markers_not_accepted(self):
        for changed in (logs().replace("H3D","H3C"),logs().replace("D_EXECUTION_RECORD","S0_EXECUTION_RECORD")):
            with self.assertRaises(RuntimeError): parse(changed)
    def test_only_bounded_ascii_leading_dots(self):
        lines=logs().splitlines()
        self.assertEqual(parse("\n".join(["."*64+lines[0]]+lines[1:]))["record_count"],400)
        for prefix in ("."*65," ", "arbitrary ", "\u2026"):
            with self.subTest(prefix=prefix),self.assertRaises(RuntimeError):
                parse("\n".join([prefix+lines[0]]+lines[1:]))
    def test_branch_false_not_boolean_or_geometry_change_rejected(self):
        fields=[(key,False) for key in e.BOOL_KEYS]+[("linear_source_completed",1),
                ("heads",56),("dtype","torch.float32"),("device","npu:7"),
                ("source_frames",38),("source_tokens_per_frame",999),("target_frames",38),
                ("forward_index",0),("rank",True),("mode","C_old"),
                ("module_sha256","b"*64),("module_path","/old/c_v1/dual_stream_attention.py"),("policy_sha256","b"*64),
                ("endpoint_policy","strict_local"),("auxiliary_policy","source_closed")]
        rest=logs().splitlines()[1:]
        for key,value in fields:
            bad=row(); bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):
                parse("\n".join([line(bad,pid=100)]+rest))
    def test_worker_rank_inconsistent_rejected(self):
        lines=logs().splitlines(); bad=row(rank=1,layer=1)
        lines[1]=line(bad,pid=100)
        with self.assertRaises(RuntimeError): parse("\n".join(lines))
    def test_nonjson_duplicate_key_or_extra_key_rejected(self):
        rest=logs().splitlines()[1:]
        bad=row(); bad["unknown"]=1
        for first in (line(row()).replace('"mode":','"mode":"D_old","mode":',1),
                      line(row()).replace("true","True"),line(bad)):
            with self.assertRaises(RuntimeError): parse("\n".join([first]+rest))
    def test_wrong_execution_contract_rejected(self):
        p=policy(); p["execution_contract"]["first_forward_index"]=0
        with self.assertRaises(RuntimeError): parse(logs(),pol=p)
    def test_invalid_request_count_and_run_identity(self):
        for requests in (0,3,True):
            with self.assertRaises(RuntimeError): parse(logs(),requests=requests)
        with self.assertRaises(RuntimeError):
            e.parse_execution_records(logs(),WORKERS,run_id="old",policy=policy(),policy_sha256=SHA)


class RunnerIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path(__file__).resolve().parent/"run_d_trial.py"
        cls.source=cls.path.read_text(); cls.tree=ast.parse(cls.source)
    def test_preserved_base_subclass_and_new_namespace(self):
        self.assertIn("class DTrialSupervisor(base.ProcessSupervisor):",self.source)
        self.assertIn("import d_trial_gates as gates",self.source)
        self.assertNotIn("import c_trial_gates",self.source)
        for old in ("c_status.json","c_50step","launch_c_trial.sh"):
            self.assertNotIn(old,self.source)
    def test_create_only_submission_before_run_creation(self):
        self.assertLess(self.source.index("claim_submission("),self.source.index("new_run.mkdir("))
        self.assertIn("exist_ok=False",self.source)
        self.assertNotIn("unlink(",self.source)
    def test_no_training_or_h0_cli(self):
        self.assertIn("parser.add_argument('--policy-sha256', required=True",self.source)
        self.assertNotIn("--train",self.source)
        self.assertNotIn("--case",self.source)
    def test_two_requests_guard_and_original_resource_locks_retained(self):
        self.assertIn("expected = [('smoke_2step', 2), ('d_50step', 50)]",self.source)
        self.assertIn("if self._workflow_started:",self.source)
        self.assertIn("self.selected.lease_root / f'device{card}.lock'",self.source)
        self.assertIn("owner.release_locks()",self.source)
    def test_bootstrap_and_launch_are_d_only(self):
        text=(self.path.parent/"launch_d_trial.sh").read_text()
        for token in ('task_port=19101','H3_TRIAL_CASE','D_REVIEWED_POLICY_SHA256',
                      '/D/runs/d_','dualstream_v1/candidate/vllm-omni','--text-encoder-tp-size 8',
                      '--vae-patch-parallel-size 8','--num-gpus 8 --usp 8 --ring 1'):
            self.assertIn(token,text)
        self.assertNotIn("candidates/c_v1",text)


if __name__=="__main__":
    unittest.main(verbosity=2)
