"""Synthetic CPU admission fixtures, NEVER actual NPU/CPU numeric run evidence."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_validation_verifier as v

RUN = "1" * 32
HOST = dict(hostname="synthetic_not_actual", boot_id="synthetic_boot", machine="aarch64")


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root):
        self.code = root / "validation_d"
        self.vendor = root / "candidate/vllm-omni"
        self.runs = root / "validation_results/01234567/runs"
        self.run = self.runs / ("20260914T000000Z_" + RUN)
        model = self.vendor / v.MODEL_RELATIVE
        for path in (self.code, model, self.run): path.mkdir(parents=True)
        self.pins = dict(schema_version=1, case=v.CASE, mode=v.MODE, candidate={}, validation={})
        self.manifest = {}
        for section, names, directory in (("candidate", v.CANDIDATES, model), ("validation", v.VALIDATORS, self.code)):
            for name in sorted(names):
                path = directory / name
                path.write_text("synthetic content not executable evidence: " + name)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                self.pins[section][name] = sha
                self.manifest[section+"/"+name] = dict(path=str(path), sha256=sha)
        pins_sha = save(self.code / "pins.json", self.pins)
        self.pol = dict(case="D", mode=v.MODE, review_status="approved_for_inference", training_performed=False,
                        attention=copy.deepcopy(v.ATTENTION), candidate_vendor=self.vendor.as_posix(),
                        candidate_files={v.MODEL_RELATIVE+n:s for n,s in self.pins["candidate"].items()},
                        validation_pins_path=(self.code/"pins.json").as_posix(), validation_pins_sha256=pins_sha,
                        tiny_verifier_sha256=self.pins["validation"]["d_validation_verifier.py"])
        fixture = dict(schema_version=1, mode=v.MODE, review_status="approved_for_inference",
                       scope="synthetic_validation_only", attention=copy.deepcopy(v.ATTENTION))
        fixture_sha = save(self.run/"validation_policy.json", fixture)
        self.report = dict(fixture_policy_sha256=fixture_sha, rank_results=[dict(rank=i) for i in range(8)],
                           fixture_is_synthetic_not_actual=True)
        result_sha = save(self.run/"d_tiny.json", self.report)
        self.release = dict(selected_physical_cards=list(range(8)), explicitly_idle_cards=list(range(8)), other_busy_cards=[])
        self.status = dict(case=v.CASE, mode=v.MODE, phase="completed", host=HOST, group="01234567", run_id=RUN,
                           run_directory=str(self.run), allocated_physical_npu_ids=list(range(8)), validation_passed=True,
                           cleanup_completed=True, selected_cards_verified_idle_after_cleanup=True, needs_attention=False,
                           remaining_owned_process_groups={}, error=None, result_path=str(self.run/"d_tiny.json"),
                           result_sha256=result_sha, started_at="2026-09-14T00:00:00+00:00", finished_at="2026-09-14T00:01:00+00:00",
                           source_sha256=self.manifest, validation_policy_sha256=fixture_sha,
                           supervisor_pid=100, supervisor_proc_identity=dict(pid=100,start_ticks=200,pgrp=100,session=100),
                           worker_pid=101, worker_proc_identity=dict(pid=101,start_ticks=201,pgrp=101,session=101),
                           resource_release_check=self.release)
        self.owner = dict(case=v.CASE, run_id=RUN, host=HOST, supervisor_pid=100, proc_identity=self.status["supervisor_proc_identity"],
                          cards=list(range(8)), resources_checked=True, lease_fds=list(range(3,12)),
                          lease_paths=[str(self.runs.parent/"run.lock")]+[str(root/"leases"/f"device{i}.lock") for i in range(8)])
        save(self.run/"owner_proof.json",self.owner)
        save(self.run/"host_identity.json",HOST)
        self.pol["tiny_proof"] = dict(case=v.CASE, mode=v.MODE, run_id=RUN, run_directory=str(self.run),
                                      result_sha256=result_sha, status_sha256=save(self.run/"d_validation_status.json",self.status),
                                      source_manifest_sha256=save(self.run/"source_manifest.json",self.manifest))
        (self.run/"npu_after.txt").write_text("synthetic historical npu-smi fixture, never current hardware")
        self.calls, self.live, self.groups = [], {}, []
        selected = types.SimpleNamespace(output_root=self.runs.parent, lease_root=root/"leases")
        fixture_self = self
        class Observer:
            def owned_group_members(self, pgid):
                fixture_self.calls.append(("groups", pgid, self.run_id))
                return fixture_self.groups
        self.base = types.SimpleNamespace(proc_identity=lambda pid:self.live.get(pid), ProcessSupervisor=Observer,
                                          selected_idle=lambda text,cards:copy.deepcopy(self.release))
        def require_host(host):
            if host != HOST: raise RuntimeError("fixture host mismatch")
            return HOST
        self.profiles = types.SimpleNamespace(CODE_ROOT=self.code,VENDOR=self.vendor,CASE=v.CASE,MODE=v.MODE,WORLD=8,CARDS=tuple(range(8)),
                                              Profile=lambda:selected,require_host=require_host,source_manifest=lambda:copy.deepcopy(self.manifest),
                                              load_helper=lambda name:self.base)
        def verify_results(path,host,run_id,manifest):
            self.calls.append(("verify_results",str(path),host,run_id,manifest))
            return copy.deepcopy(self.report)
        self.checker = types.SimpleNamespace(verify_results=verify_results)
    @contextmanager
    def readers(self,pins,pins_sha):
        self.calls.append(("readers",pins,pins_sha))
        yield self.profiles,self.checker
    def refresh_status(self):
        self.pol["tiny_proof"]["status_sha256"] = save(self.run/"d_validation_status.json",self.status)
    def refresh_result(self):
        sha=save(self.run/"d_tiny.json",self.report)
        self.pol["tiny_proof"]["result_sha256"]=self.status["result_sha256"]=sha
        self.refresh_status()


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.f=Fixture(Path(self.tmp.name).resolve())
        self.patches=[patch.object(v,"CODE_ROOT",self.f.code),patch.object(v,"RUNS_ROOT",self.f.runs),
                      patch.object(v,"VENDOR",self.f.vendor),patch.object(v,"__file__",str(self.f.code/"d_validation_verifier.py")),
                      patch.object(v,"verified_readers",self.f.readers)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def verify(self):
        return v.verify_completed(run_directory=str(self.f.run),run_id=RUN,host=HOST,policy=self.f.pol)
    def test_synthetic_valid_full_orchestration_not_actual_numeric_proof(self):
        out=self.verify()
        self.assertIs(out["verified"],True)
        self.assertIs(out["current_card_idle_checked"],False)
        self.assertEqual(out["numeric_rank_count"],8)
        self.assertTrue(any(c[0]=="verify_results" for c in self.f.calls))
        self.assertEqual(len(self.f.calls),3)
    def test_missing_receipt_rejects(self):
        (self.f.run/"d_validation_status.json").unlink()
        with self.assertRaises(FileNotFoundError):self.verify()
    def test_old_case_rejected(self):
        self.f.pol["tiny_proof"]["case"]="C_strict_tiny_SP8"
        with self.assertRaises(RuntimeError):self.verify()
    def test_wrong_run_latest_rejected(self):
        with self.assertRaises(RuntimeError):
            v.verify_completed(run_directory=str(self.f.run),run_id="2"*32,host=HOST,policy=self.f.pol)
    def test_policy_attention_not_preserve_existing_rejected(self):
        self.f.pol["attention"]["auxiliary_policy"]="source_closed"
        with self.assertRaises(RuntimeError):self.verify()
    def test_candidate_changed_rejects(self):
        (self.f.vendor/v.MODEL_RELATIVE/"strict_source_attention.py").write_text("changed")
        with self.assertRaises(RuntimeError):self.verify()
    def test_full_candidate_differs_from_tiny_rejects(self):
        self.f.pol["candidate_files"][v.MODEL_RELATIVE+"dual_stream_attention.py"]="a"*64
        with self.assertRaises(RuntimeError):self.verify()
    def test_missing_pin_or_wrong_sha_rejects(self):
        self.f.pins["candidate"].pop("openvdn_npu.py")
        self.f.pol["validation_pins_sha256"]=save(self.f.code/"pins.json",self.f.pins)
        with self.assertRaises(RuntimeError):self.verify()
    def test_self_verifier_wronghash_rejects(self):
        self.f.pol["tiny_verifier_sha256"]="a"*64
        with self.assertRaises(RuntimeError):self.verify()
    def test_status_changed_without_repin_rejects(self):
        save(self.f.run/"d_validation_status.json",dict(self.f.status,phase="failed"))
        with self.assertRaises(RuntimeError):self.verify()
    def test_current_frozen_manifest_differs_rejects(self):
        self.f.profiles.source_manifest=lambda:{"different":True}
        with self.assertRaises(RuntimeError):self.verify()
    def test_dependency_import_profile_wrongpath_rejects(self):
        self.f.profiles.CODE_ROOT=self.f.code/"other"
        with self.assertRaises(RuntimeError):self.verify()
    def test_readerfailure_propagates_no_status_fallback(self):
        def fail(*args):raise RuntimeError("synthetic numeric verifier rejects")
        self.f.checker.verify_results=fail
        with self.assertRaisesRegex(RuntimeError,"numeric verifier rejects"):self.verify()
    def test_returned_numeric_report_changed_rejects(self):
        self.f.checker.verify_results=lambda *args:dict(self.f.report,status="passed")
        with self.assertRaises(RuntimeError):self.verify()
    def test_fixture_policy_status_binding_rejected(self):
        self.f.status["validation_policy_sha256"]="a"*64;self.f.refresh_status()
        with self.assertRaises(RuntimeError):self.verify()
    def test_fixture_policy_report_binding_rejected(self):
        self.f.report["fixture_policy_sha256"]="a"*64;self.f.refresh_result()
        with self.assertRaises(RuntimeError):self.verify()
    def test_terminal_failed_not_boolean_or_notclean_rejected(self):
        for key,value in (("phase","failed"),("mode","old_C"),("needs_attention",True),("cleanup_completed",1),
                          ("remaining_owned_process_groups",{"101":[102]}),("error","bad")):
            with self.subTest(key=key):
                original=self.f.status[key];self.f.status[key]=value;self.f.refresh_status()
                with self.assertRaises(RuntimeError):self.verify()
                self.f.status[key]=original
    def test_live_original_pid_rejects_and_pid_reuse_is_not_original(self):
        self.f.live[101]=dict(self.f.status["worker_proc_identity"])
        with self.assertRaises(RuntimeError):self.verify()
        self.f.live[101]["start_ticks"]+=1
        self.assertIs(self.verify()["verified"],True)
    def test_original_owned_marked_group_stilllive_rejects(self):
        self.f.groups=[103]
        with self.assertRaises(RuntimeError):self.verify()
    def test_wrong_owner_fd_paths_or_identity_rejects(self):
        for key,value in (("lease_fds",list(range(3,11))),("lease_paths",[]),("supervisor_pid",999)):
            with self.subTest(key=key):
                original=self.f.owner[key];self.f.owner[key]=value;save(self.f.run/"owner_proof.json",self.f.owner)
                with self.assertRaises(RuntimeError):self.verify()
                self.f.owner[key]=original
    def test_historical_idle_differs_rejects_without_fresh_probe(self):
        self.f.base.selected_idle=lambda *_:dict(self.f.release,other_busy_cards=[0])
        with self.assertRaises(RuntimeError):self.verify()
    def test_pinned_record_mutated_during_read_rejects(self):
        old=self.f.checker.verify_results
        def mutate(*args):
            result=old(*args)
            (self.f.run/"npu_after.txt").write_text("changed during verifier")
            save(self.f.run/"source_manifest.json",{"tampered":True})
            return result
        self.f.checker.verify_results=mutate
        with self.assertRaises(RuntimeError):self.verify()


class ReaderContextTests(unittest.TestCase):
    def test_pinsenv_modules_path_restored_success_and_failure(self):
        for fail in (False,True):
            original_path=list(sys.path)
            sentinel=types.ModuleType("old_d_profiles")
            with patch.dict(os.environ,{"D_VALIDATION_PINS_SHA256":"original-value"}),patch.dict(sys.modules,{"d_profiles":sentinel}):
                def loader(name,path,sha):
                    self.assertEqual(os.environ["D_VALIDATION_PINS_SHA256"],"a"*64)
                    if fail:raise RuntimeError("synthetic import error")
                    module=types.ModuleType(name);sys.modules[name]=module;return module
                with patch.object(v,"load_checked",loader):
                    try:
                        with v.verified_readers({"validation":{name+".py":"b"*64 for name in v.IMPORTS}},"a"*64):pass
                    except RuntimeError:
                        self.assertTrue(fail)
                self.assertEqual(os.environ["D_VALIDATION_PINS_SHA256"],"original-value")
                self.assertIs(sys.modules["d_profiles"],sentinel)
                self.assertEqual(sys.path,original_path)
    def test_duplicate_json_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/"x.json";path.write_text('{"a":1,"a":2}')
            with self.assertRaises(RuntimeError):v.read_json(path)


if __name__=="__main__":unittest.main(verbosity=2)
