"""CPU-only structure/oracle/proof gates. Missing torch is explicitly SKIP."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
HERE = Path(__file__).resolve().parent
C_ROOT = HERE.parent
import c_cases as cases
import c_profiles as p
import c_npu_regression as regression
import run_c_validation

layout_module = p.load_file("_c_validation_cpu_layout", C_ROOT / "strict_source_layout.py")


def passed_report():
    host, run_id = {"hostname": "test", "boot_id": "test", "machine": "test"}, "a" * 32
    manifest = {"validation/c_npu_regression.py": {"sha256": "f" * 64},
                "strategy/ulysses.py": {"path": "/test", "sha256": "1" * 64}}
    ranks = []
    for rank in range(cases.WORLD):
        tests, pre = [], {}
        for case in cases.CASES:
            data = cases.fixture(case)
            layout = layout_module.infer_layout(**data)
            n = len(data["coords"])
            proof = cases.audit_masks(layout, n)
            pre[case] = dict(rejected=list(cases.REJECTIONS), layout=vars(layout), oracle=proof)
            checks = [dict(check=name, max_abs=0.0, rms_error=0.0, golden_rms=0.0,
                           atol=0.0 if name.endswith("-zero") else 1 / 512,
                           rtol=0.0 if name.endswith("-zero") else 1 / 128) for name in regression.CHECKS]
            tests.append(dict(case=case, packed_rows=n, local_rows=[rank * (n // 8), (rank + 1) * (n // 8)],
                              layout=vars(layout), checks=checks, mask_proof=proof,
                              frozen_qkv_unchanged=True, positive_controls=True, source_is_visual_only=True))
        ranks.append(dict(status="passed", rank=rank, physical_card=rank, host=host, run_id=run_id,
                          case=p.CASE, source_sha256=manifest, tests=tests, precollective_metadata=pre,
                          frozen_weights_unchanged=True, strategy_proof=manifest["strategy/ulysses.py"],
                          parallelism_proof=dict(ulysses_world_size=8, ulysses_rank=rank, ring_world_size=1,
                              tensor_parallel_world_size=1, collective_global_ranks=list(range(8)), heads_per_ulysses_rank=7)))
    report = dict(status="passed", case=p.CASE, host=host, run_id=run_id, world_size=8, physical_cards=list(range(8)),
                  dtype="bfloat16", heads=56, head_dim=128, source_sha256=manifest,
                  test_script_sha256="f" * 64, rank_results=ranks)
    return report, host, run_id, manifest


class Cases(unittest.TestCase):
    def test_actual_c_inference_accepts_both_legal_layouts(self):
        for case in cases.CASES:
            data = cases.fixture(case)
            layout = layout_module.infer_layout(**data)
            self.assertEqual(len(data["coords"]) % 8, 0)
            self.assertNotEqual(layout.source_times, layout.target_times)
            self.assertNotEqual(layout.source_tokens_per_frame, layout.tokens_per_frame)
            self.assertTrue(cases.audit_masks(layout, len(data["coords"]))["ranks_without_target"])

    def test_all_invalid_metadata_failclosed(self):
        for case in cases.CASES:
            for name, data in cases.invalid_fixtures(case).items():
                with self.subTest(case=case, invalid=name), self.assertRaises(ValueError):
                    layout_module.infer_layout(**data)

    def test_suffix_is_complete_disjoint_grid_not_corruption(self):
        for case in cases.CASES:
            data = cases.invalid_fixtures(case)["source_suffix"]
            rows = data["img_pos"] + data["text_pos"] + data["audio_pos"]
            self.assertEqual(sorted(rows), list(range(data["cu_seqlens"][1])))
            source = [i for i, u in zip(data["img_pos"], data["update_mask"]) if not u]
            target = [i for i, u in zip(data["img_pos"], data["update_mask"]) if u]
            self.assertGreater(source[0], target[-1])
            for name, selected in (("source", source), ("target", target)):
                layout_module.validate_grid(data["coords"], selected, data["metadata"][name + "_shape"], name)

    def test_independent_dense_mask_matches_actual_plan(self):
        for case in cases.CASES:
            data = cases.fixture(case)
            layout = layout_module.infer_layout(**data)
            n = len(data["coords"])
            _, oracle = cases.oracle_masks(layout, n)
            actual = [[False] * n for _ in range(n)]
            dense, groups = layout_module.strict_plan(layout)
            for q in layout_module.expand_spans(dense):
                for k in range(layout.used_len):
                    actual[q][k] = True
            for group in groups:
                for q in range(*group.query_span):
                    for k in layout_module.expand_spans(group.key_spans):
                        actual[q][k] = True
            self.assertEqual(actual, oracle)

    def test_strict_anchors_but_preserved_other_edges(self):
        for case in cases.CASES:
            data = cases.fixture(case)
            layout = layout_module.infer_layout(**data)
            b, c = cases.oracle_masks(layout, len(data["coords"]))
            for frame in (0, layout.num_frames - 1):
                q = layout.video_start + frame * layout.tokens_per_frame
                self.assertEqual(sum(c[q][layout.source_start:layout.source_end]), layout.source_tokens_per_frame)
                self.assertTrue(all(c[q][layout.video_start:layout.video_end]))
            for q in range(layout.used_len):
                for k in range(layout.used_len):
                    if not(layout.video_start <= q < layout.video_end and layout.source_start <= k < layout.source_end):
                        self.assertEqual(c[q][k], b[q][k])
            self.assertFalse(any(any(r) for r in c[layout.used_len:]))

    def test_oracle_never_imports_runtime_plan(self):
        source = (HERE / "c_cases.py").read_text()
        tree = ast.parse(source)
        names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertFalse({"strict_plan", "infer_layout", "_device_plan"} & names)

    def test_explicit_full56heads_one8rank_group(self):
        self.assertEqual((cases.WORLD, cases.HEADS, cases.HEADS // cases.WORLD), (8, 56, 7))
        self.assertEqual(p.CARDS, tuple(range(8)))
        self.assertEqual(p.Profile().lease_root, p.INSTALL / "validation/card_locks")
        self.assertNotEqual(p.CODE_ROOT, p.B_CODE)
        self.assertNotEqual(p.Profile().output_root, p.INSTALL / "validation/01234567")


class ProofGates(unittest.TestCase):
    def setUp(self):
        self.report, self.host, self.run_id, self.manifest = passed_report()

    def verify(self):
        return regression.validate_report(self.report, self.host, self.run_id, self.manifest)

    def test_complete_eight_rank_proof_passes(self):
        self.verify()
        # Production JSON roundtrip must retain exact gate semantics.
        self.report = json.loads(json.dumps(self.report))
        self.verify()

    def test_missing_or_duplicate_rank_rejected(self):
        self.report["rank_results"][7] = self.report["rank_results"][0]
        with self.assertRaises(RuntimeError): self.verify()

    def test_fourcard_or_wrong_heads_rejected(self):
        self.report["heads"] = 48
        with self.assertRaises(RuntimeError): self.verify()

    def test_split_communicator_rejected(self):
        self.report["rank_results"][4]["parallelism_proof"]["collective_global_ranks"] = [4, 5, 6, 7]
        with self.assertRaises(RuntimeError): self.verify()

    def test_missing_suffix_rejection_rejected(self):
        self.report["rank_results"][0]["precollective_metadata"][cases.CASES[0]]["rejected"].remove("source_suffix")
        with self.assertRaises(RuntimeError): self.verify()

    def test_numeric_check_cannot_be_omitted(self):
        self.report["rank_results"][0]["tests"][0]["checks"].pop()
        with self.assertRaises(RuntimeError): self.verify()

    def test_no_tolerance_enlargement(self):
        self.report["rank_results"][0]["tests"][0]["checks"][0]["atol"] = 1
        with self.assertRaises(RuntimeError): self.verify()

    def test_nonfinite_metrics_rejected(self):
        self.report["rank_results"][0]["tests"][0]["checks"][0]["max_abs"] = float("nan")
        with self.assertRaises(RuntimeError): self.verify()

    def test_nonzero_exact_anchor_rejected(self):
        check = next(c for c in self.report["rank_results"][0]["tests"][0]["checks"] if c["check"] == "wrong-source-anchor-zero")
        check["max_abs"] = 1e-10
        with self.assertRaises(RuntimeError): self.verify()

    def test_crossrank_row_proof_rejected(self):
        self.report["rank_results"][0]["tests"][0]["local_rows"] = [0, 1]
        with self.assertRaises(RuntimeError): self.verify()

    def test_wrong_mask_proof_rejected(self):
        self.report["rank_results"][0]["tests"][0]["mask_proof"]["removed_edges"] = 0
        with self.assertRaises(RuntimeError): self.verify()

    def test_weights_and_qkv_preservation_required(self):
        self.report["rank_results"][0]["frozen_weights_unchanged"] = False
        with self.assertRaises(RuntimeError): self.verify()

    def test_different_boot_or_hash_rejected(self):
        self.report["test_script_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError): self.verify()

    def test_no_optin_never_checks_host_or_imports_torch(self):
        with mock.patch.object(p, "require_host") as host, self.assertRaises(SystemExit):
            run_c_validation.main([])
        host.assert_not_called()

    def test_no_optin_regression_never_checks_host(self):
        with mock.patch.object(p, "require_host") as host, self.assertRaises(SystemExit):
            regression.parse_and_validate(["--run-id", self.run_id, "--output", "test.json"])
        host.assert_not_called()


class SourceTests(unittest.TestCase):
    def test_frozen_reused_helpers_match_local_sources(self):
        root = C_ROOT.parent / "deploy_31731/validation"
        # Remote CPU-only review uses the already installed, frozen helper
        # directory read-only; never manufacture or overwrite a sibling tree.
        if not root.is_dir() and C_ROOT == Path("/cache/zhonghao/h3/c_adaptation"):
            root = p.B_CODE
        for name, sha in p.PINNED_HELPERS.items():
            self.assertEqual(p.digest(root / name), sha)
        for name, sha in p.PINNED_C.items():
            self.assertEqual(p.digest(C_ROOT / "candidate" / name), sha)

    def test_precollective_metadata_check_precedes_device_and_collectives(self):
        source = (HERE / "c_npu_regression.py").read_text()
        worker = source[source.index("def worker("):source.index("def validate_report(")]
        self.assertLess(worker.index("metadata_checks(early, torch)"), worker.index("torch.npu.set_device(rank)"))
        self.assertLess(worker.index("metadata_checks(early, torch)"), worker.index("dist.init_process_group("))

    def test_launcher_only_c_paths_and8cards(self):
        script = (HERE / "launch_c_validation.sh").read_text()
        self.assertIn("candidates/c_v1/vllm-omni", script)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", script)
        self.assertIn("exec /cache/zhonghao/h3/env/bin/python", script)
        self.assertNotIn("30213", script)

    def test_cpu_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys, c_cases, c_profiles, c_npu_regression, run_c_validation; assert 'torch' not in sys.modules"],
            cwd=HERE, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class LeaseGateTests(unittest.TestCase):
    """FD/inode/parent proof with fake flock on Windows, real flock on Linux."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output, self.leases = self.root / "out", self.root / "leases"
        self.output.mkdir()
        self.leases.mkdir()
        self.paths = [self.output / "run.lock"] + [self.leases / f"device{c}.lock" for c in range(8)]
        self.handles = [path.open("xb+") for path in self.paths]
        self.addCleanup(lambda: [handle.close() for handle in self.handles])
        self.identity = dict(pid=os.getppid(), start_ticks=100, pgrp=1, session=1)
        self.base = types.SimpleNamespace(proc_identity=lambda pid: self.identity.copy())
        self.host, self.run_id = {"hostname": "fixture"}, "a" * 32
        self.owner = dict(supervisor_pid=os.getppid(), proc_identity=self.identity, host=self.host, run_id=self.run_id,
                          case=p.CASE, cards=list(range(8)), resources_checked=True,
                          lease_paths=[str(x) for x in self.paths], lease_fds=[h.fileno() for h in self.handles])
        self.mock_fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4,
                                               flock=mock.Mock(side_effect=BlockingIOError("held")))
        for patcher in (mock.patch.object(p, "canonical", side_effect=lambda x: Path(x)),
                        mock.patch.object(p, "Profile", return_value=types.SimpleNamespace(output_root=self.output, lease_root=self.leases)),
                        mock.patch.dict(os.environ, H3_C_SUPERVISOR_PID=str(os.getppid())),
                        mock.patch.dict(sys.modules, fcntl=self.mock_fcntl),
                        mock.patch.object(os, "getuid", return_value=os.fstat(self.handles[0].fileno()).st_uid, create=True),
                        mock.patch.object(os, "O_NOFOLLOW", getattr(os, "O_NOFOLLOW", 0), create=True),
                        mock.patch.object(os, "O_NONBLOCK", getattr(os, "O_NONBLOCK", 0), create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def verify(self):
        (self.root / "owner_proof.json").write_text(json.dumps(self.owner))
        regression.verify_parent_leases(self.root, self.run_id, self.host, self.base)

    def test_nine_inherited_locked_inodes_pass(self):
        self.verify()
        self.assertEqual(self.mock_fcntl.flock.call_count, 9)

    def test_missing_or_duplicate_fd_fails(self):
        self.owner["lease_fds"][-1] = self.owner["lease_fds"][0]
        with self.assertRaises(RuntimeError): self.verify()

    def test_wrong_inode_fails(self):
        self.owner["lease_fds"][0], self.owner["lease_fds"][1] = self.owner["lease_fds"][1], self.owner["lease_fds"][0]
        with self.assertRaises(RuntimeError): self.verify()

    def test_unlocked_mutex_fails(self):
        self.mock_fcntl.flock.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "not actually locked"): self.verify()

    def test_parent_start_ticks_mismatch_fails(self):
        self.owner["proc_identity"] = {**self.identity, "start_ticks": 99}
        with self.assertRaises(RuntimeError): self.verify()

    def test_resource_gate_proof_missing_fails(self):
        self.owner["resources_checked"] = False
        with self.assertRaises(RuntimeError): self.verify()

    @unittest.skipUnless(os.name == "posix", "Real inherited flock is Linux-only; mocks are not a kernel proof")
    def test_real_posix_flock_identity_gate(self):
        # Restore the actual module only for this integration check.
        sys.modules.pop("fcntl", None)
        import fcntl
        for handle in self.handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.verify()


@unittest.skipUnless(importlib.util.find_spec("torch"), "No local torch: actual tensor/SP8 emulation unverified")
class TorchCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        package = types.ModuleType("_c_val_real_cpu")
        package.__path__ = [str(C_ROOT / "candidate")]
        sys.modules[package.__name__] = package
        for name in ("openvdn_npu", "strict_source_layout", "strict_source_attention"):
            module = p.load_file(package.__name__ + "." + name, C_ROOT / "candidate" / f"{name}.py")
        cls.adapter = module

    def test_actual_adapter_precollective_metadata(self):
        proof = regression.metadata_checks(self.adapter, self.torch)
        self.assertEqual(set(proof), set(cases.CASES))

    def test_56head_float32_dense_oracle_and8headshard_reassembly(self):
        t = self.torch
        for case in cases.CASES:
            data = cases.fixture(case)
            layout = self.adapter.infer_strict_source_layout(**regression.tensors(data, t))
            rng = t.Generator(device="cpu").manual_seed(810)
            q, k, v = [t.randn((len(data["coords"]), 56, 128), generator=rng, dtype=t.float32) * 0.1 for _ in range(3)]
            before = [x.clone() for x in (q, k, v)]
            golden = regression.dense_cpu(q, k, v, layout, t)
            full = self.adapter.strict_source_softmax_attention(q, k, v, layout, 128 ** -0.5)
            # Emulate exact head-axis ownership; actual HCCL is not claimed.
            parts = [self.adapter.strict_source_softmax_attention(q[:, r*7:(r+1)*7], k[:, r*7:(r+1)*7],
                     v[:, r*7:(r+1)*7], layout, 128 ** -0.5) for r in range(8)]
            combined = t.cat(parts, dim=1)
            t.testing.assert_close(full, golden, atol=1e-6, rtol=1e-5)
            t.testing.assert_close(combined, golden, atol=1e-6, rtol=1e-5)
            for a, b in zip((q, k, v), before): self.assertTrue(t.equal(a, b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
