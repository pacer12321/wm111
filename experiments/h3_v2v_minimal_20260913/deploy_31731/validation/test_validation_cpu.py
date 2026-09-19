"""CPU-only safety/interface tests. Imports no torch and starts no subprocesses."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.name == "nt":
    # Linux-only supervisor imports fcntl, but all locking is mocked in these
    # CPU tests. This stub is not present or used in production files.
    sys.modules.setdefault("fcntl", SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=mock.Mock()))
import profiles
import supervision_base as base
import run_validation as supervisor
import npu_wrapper_regression as wrapper

HOST = {"hostname": profiles.EXPECTED_HOST, "boot_id": "12345678-abcd-abcd-abcd-123456789abc", "machine": "aarch64"}
RUN = "a" * 32


def smi(*, busy=(), health="OK", used=3402, total=65536):
    # Same two-line 910B3 block and process table format read from 31731.
    rows = ["+-------------------+", "| NPU Name | Health | Hugepages-Usage(page) |"]
    for card in range(8):
        rows += [f"| {card} 910B3 | {health} | 99.5 48 0 / 0 |",
                 f"| 0 | 0000:C1:00.0 | 0 0 / 0 {used} / {total} |", "+===================+"]
    rows += ["| NPU Chip | Process id | Process name | Process memory(MB) |", "+===================+"]
    for card in range(8):
        rows += [f"| {card} 0 | 14450 | VLLM::Worker | 64192 |" if card in busy
                 else f"| No running processes found in NPU {card} |", "+===================+"]
    return "\n".join(rows)


def result_fixture(selected, manifest):
    ranks = []
    for rank, card in enumerate(selected.cards):
        tests = []
        for case in sorted(supervisor.EXPECTED_CASES):
            checks = [{"check": check, "max_abs": 0.0, "rms_error": 0.0, "golden_rms": 0.0,
                       "atol": 0.0 if check.endswith("-zero") else 1/512,
                       "rtol": 0.0 if check.endswith("-zero") else 1/128}
                      for check in sorted(supervisor.EXPECTED_CHECKS)]
            tests.append({"case": case, "checks": checks})
        ranks.append({"rank": rank, "physical_card": card, "status": "passed", "host": HOST,
                      "group": selected.group, "run_id": RUN, "source_sha256": manifest,
                      "real_strategy_source": manifest["strategy/ulysses.py"]["path"],
                      "real_strategy_sha256": manifest["strategy/ulysses.py"]["sha256"], "tests": tests,
                      "parallelism_proof": {"ulysses_world_size": 8, "ulysses_rank": rank,
                          "ring_world_size": 1, "tensor_parallel_world_size": 1,
                          "collective_global_ranks": list(range(8)), "heads_per_ulysses_rank": 7}})
    return {"status": "passed", "host": HOST, "group": selected.group, "run_id": RUN,
            "physical_cards": list(selected.cards), "world_size": 8, "dtype": "bfloat16",
            "source_sha256": manifest, "test_script_sha256": "script-sha", "rank_results": ranks}


class ProfileTests(unittest.TestCase):
    def test_only_fixed_single_usp8_group(self):
        selected = profiles.profile("01234567")
        self.assertEqual(selected.cards, tuple(range(8)))
        self.assertEqual(profiles.WORLD, 8)
        self.assertEqual(selected.master_port, 29673)
        self.assertEqual(tuple(profiles.GROUPS), ("01234567",))

    def test_no_arbitrary_group_or_cards(self):
        for group in ("2367", "all", "0123", "4567", "../01234567"):
            with self.assertRaises(ValueError):
                profiles.profile(group)
        with mock.patch.dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"):
            self.assertEqual(profiles.require_cards("01234567", list(range(8))).cards, tuple(range(8)))
            for cards in (list(reversed(range(8))), [0, 1, 2], [4, 5, 6, 7], [0, 0, 2, 3, 4, 5, 6, 7]):
                with self.assertRaises(RuntimeError):
                    profiles.require_cards("01234567", cards)
        with mock.patch.dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"):
            with self.assertRaises(RuntimeError):
                profiles.require_cards("01234567", list(range(8)))

    def test_real_host_and_boot_binding(self):
        with mock.patch.object(profiles, "read_host_identity", return_value=HOST):
            self.assertEqual(profiles.require_host(HOST), HOST)
            with self.assertRaises(RuntimeError):
                profiles.require_host({**HOST, "boot_id": "other-boot"})
        for change in ({"hostname": "30213"}, {"machine": "x86_64"}):
            with mock.patch.object(profiles, "read_host_identity", return_value={**HOST, **change}):
                with self.assertRaises(RuntimeError):
                    profiles.require_host()

    def test_identity_reader_really_reads_os(self):
        with mock.patch.object(Path, "read_text", return_value=HOST["boot_id"]), \
             mock.patch.object(profiles.socket, "gethostname", return_value=HOST["hostname"]), \
             mock.patch.object(profiles.os, "uname", return_value=SimpleNamespace(machine="aarch64"), create=True):
            self.assertEqual(profiles.read_host_identity(), HOST)
        with mock.patch.object(Path, "read_text", return_value="not-a-boot-uuid"):
            with self.assertRaises(RuntimeError):
                profiles.read_host_identity()

    def test_private_path_rejects_escape_and_symlink_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with mock.patch.object(profiles, "INSTALL", root):
                self.assertEqual(profiles.canonical_private(root / "ok"), root / "ok")
                for path in (root.parent / "escape", root / "a/../b", Path("relative")):
                    with self.assertRaises(RuntimeError):
                        profiles.canonical_private(path)
                with mock.patch.object(Path, "resolve", return_value=root.parent / "outside"):
                    with self.assertRaises(RuntimeError):
                        profiles.canonical_private(root / "link")

    def test_golden_matches_exact_reviewed_original(self):
        self.assertEqual(profiles.digest(HERE / "golden/openvdn_npu.py"), profiles.GOLDEN_SHA256)
        self.assertEqual(profiles.digest(HERE.parents[1] / "b_adaptation/upstream/openvdn_npu.py"), profiles.GOLDEN_SHA256)

    def test_source_manifest_pins_candidates_and_tracks_real_strategy(self):
        selected = profiles.profile("01234567")
        def sha(path):
            if "golden" in str(path):
                return profiles.GOLDEN_SHA256
            return profiles.CANDIDATE_SHA256.get(path.name, "other-sha")
        with mock.patch.object(profiles, "canonical_private"), mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "is_symlink", return_value=False), mock.patch.object(profiles, "digest", side_effect=sha):
            manifest = profiles.source_manifest(selected)
            self.assertIn("strategy/ulysses.py", manifest)
            self.assertIn(str(selected.vendor_root), manifest["strategy/ulysses.py"]["path"])
            self.assertEqual(len([key for key in manifest if key.startswith("candidate/")]), 4)
        with mock.patch.object(profiles, "canonical_private"), mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "is_symlink", return_value=False), mock.patch.object(profiles, "digest", return_value="changed"):
            with self.assertRaises(RuntimeError):
                profiles.source_manifest(selected)


class ResourceTests(unittest.TestCase):
    def test_fixed_eight_card_allocation_rejects_any_busy_card(self):
        cards = profiles.profile("01234567").cards
        base.selected_idle(smi(), cards)
        for card in cards:
            with self.assertRaises(RuntimeError):
                base.selected_idle(smi(busy=(card,)), cards)

    def test_idle_other_group_busy_is_allowed(self):
        parsed = base.selected_idle(smi(busy=(4, 5, 6, 7)), (0, 1, 2, 3))
        self.assertEqual(parsed["other_busy_cards"], [4, 5, 6, 7])
        with self.assertRaises(RuntimeError):
            base.selected_idle(smi(busy=(0,)), (0, 1, 2, 3))

    def test_idle_unknown_duplicate_and_missing_rejected(self):
        for text in (smi() + "\nunknown", smi() + "\n| No running processes found in NPU 0 |",
                     smi().replace("| No running processes found in NPU 0 |", ""),
                     smi().replace("Process memory(MB)", "new format")):
            with self.assertRaises(RuntimeError):
                base.selected_idle(text, (0, 1, 2, 3))

    def test_real_smi_health_and_hbm_format(self):
        for cards in ((0, 1, 2, 3), (4, 5, 6, 7)):
            parsed = base.selected_health_memory(smi(), cards, 8192)
            self.assertEqual(parsed["hbm"][cards[0]]["free_mib"], 65536 - 3402)
            self.assertEqual(set(parsed["health"]), set(cards))

    def test_hbm_or_health_ambiguous_low_invalid_rejected(self):
        for text in (smi(health="Warning"), smi(used=60000), smi(used=70000),
                     smi().replace("| 0 910B3 | OK", "| 0 910B3 | Unknown"),
                     smi().replace("| 0 | 0000:C1:00.0 | 0 0 / 0 3402 / 65536 |", ""),
                     smi().replace("+===================+", "| 0 | bus | 3402 / 65536 |", 1)):
            with self.assertRaises(RuntimeError):
                base.selected_health_memory(text, (0, 1, 2, 3), 8192)

    def memory(self, *, host_gib=100, limit="max", used=0, v1=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cg = root / "proc", root / "cgroup"
            proc.mkdir(); cg.mkdir()
            (proc / "meminfo").write_text(f"MemAvailable: {host_gib * 1024**2} kB\n")
            if v1:
                (cg / "memory").mkdir()
                (cg / "memory/memory.limit_in_bytes").write_text(str(limit))
                (cg / "memory/memory.usage_in_bytes").write_text(str(used))
            elif limit is not None:
                (cg / "memory.max").write_text(str(limit))
                (cg / "memory.current").write_text(str(used))
            return base.host_memory_snapshot(64 * 1024**3, proc_root=proc, cgroup_root=cg)

    def test_memory_v2_unlimited_and_v1_finite(self):
        self.assertEqual(self.memory()["effective_available_bytes"], 100 * 1024**3)
        self.assertEqual(self.memory(limit=90 * 1024**3, used=10 * 1024**3, v1=True)["effective_available_bytes"], 80 * 1024**3)
        self.assertEqual(self.memory(limit=2**63-1, v1=True)["effective_available_bytes"], 100 * 1024**3)

    def test_memory_missing_low_cgroup_low_host_rejected(self):
        for kwargs in ({"limit": None}, {"host_gib": 63}, {"limit": 70 * 1024**3, "used": 10 * 1024**3}, {"used": -1}):
            with self.assertRaises(RuntimeError):
                self.memory(**kwargs)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "npu_wrapper.json"
        self.selected = profiles.profile("01234567")
        self.manifest = {"strategy/ulysses.py": {"path": "actual/vendor/ulysses.py", "sha256": "strategy-sha"},
                         "validation/npu_wrapper_regression.py": {"sha256": "script-sha"}}

    def verify(self, result):
        self.path.write_text(json.dumps(result))
        for item in result["rank_results"]:
            self.path.with_name(f"npu_wrapper.rank{item['rank']}.json").write_text(json.dumps(item))
        return supervisor.verify_results(self.path, self.selected, HOST, RUN, self.manifest)

    def test_complete_proof_accepted(self):
        result = result_fixture(self.selected, self.manifest)
        self.assertEqual(self.verify(result), result)

    def test_summary_cross_host_run_group_source_rejected(self):
        for key, value in (("host", {}), ("run_id", "b" * 32), ("group", "0123"), ("world_size", 4),
                           ("test_script_sha256", "old"), ("source_sha256", {}), ("status", "running")):
            result = result_fixture(self.selected, self.manifest)
            result[key] = value
            with self.assertRaises(RuntimeError):
                self.verify(result)

    def test_rank_card_duplicate_strategy_and_case_tamper_rejected(self):
        for key, value in (("rank", 1), ("physical_card", 7), ("real_strategy_source", "old/vendor.py"),
                           ("real_strategy_sha256", "changed"), ("host", {}), ("tests", [])):
            result = result_fixture(self.selected, self.manifest)
            result["rank_results"][0][key] = value
            with self.assertRaises(RuntimeError):
                self.verify(result)

    def test_four_rank_or_split_collective_proof_rejected(self):
        for mutation in ("missing-ranks", "four-world", "four-peers", "wrong-head-split"):
            result = result_fixture(self.selected, self.manifest)
            if mutation == "missing-ranks":
                result["rank_results"] = result["rank_results"][:4]
            else:
                proof = result["rank_results"][0]["parallelism_proof"]
                if mutation == "four-world":
                    proof["ulysses_world_size"] = 4
                elif mutation == "four-peers":
                    proof["collective_global_ranks"] = list(range(4))
                else:
                    proof["heads_per_ulysses_rank"] = 14
            with self.assertRaises(RuntimeError):
                self.verify(result)

    def test_missing_duplicate_nonfinite_and_loosened_checks_rejected(self):
        for mutation in ("case", "check", "nan", "tolerance", "exact"):
            result = result_fixture(self.selected, self.manifest)
            test = result["rank_results"][0]["tests"][0]
            if mutation == "case":
                test["case"] = "not-executed"
            elif mutation == "check":
                test["checks"][0]["check"] = "not-executed"
            elif mutation == "nan":
                test["checks"][0]["max_abs"] = float("nan")
            elif mutation == "tolerance":
                test["checks"][0]["atol"] = 0.5
            else:
                next(check for check in test["checks"] if check["check"].endswith("-zero"))["max_abs"] = 0.001
            with self.assertRaises(RuntimeError):
                self.verify(result)

    def test_standalone_rank_tamper_rejected(self):
        result = result_fixture(self.selected, self.manifest)
        self.verify(result)
        self.path.with_name("npu_wrapper.rank0.json").write_text("{}")
        with self.assertRaises(RuntimeError):
            supervisor.verify_results(self.path, self.selected, HOST, RUN, self.manifest)


class LifecycleTests(unittest.TestCase):
    def test_no_npu_without_explicit_opt_in(self):
        with mock.patch.object(supervisor, "ValidationSupervisor") as cls, mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                supervisor.main(["--group", "01234567"])
            cls.assert_not_called()

    def test_failure_unwinds_cleanup_and_releases(self):
        fake = mock.Mock(status={})
        fake.preflight.side_effect = RuntimeError("busy card")
        fake.cleanup.return_value = True
        with mock.patch.object(supervisor, "ValidationSupervisor", return_value=fake), mock.patch.object(supervisor.signal, "signal"):
            self.assertEqual(supervisor.main(["--group", "01234567", "--allow-npu"]), 1)
        fake.launch.assert_not_called()
        fake.cleanup.assert_called_once()
        fake.release_locks.assert_called_once()

    def test_cleanup_failure_still_releases(self):
        fake = mock.Mock(status={})
        fake.cleanup.side_effect = RuntimeError("inspection failed")
        with mock.patch.object(supervisor, "ValidationSupervisor", return_value=fake), mock.patch.object(supervisor.signal, "signal"):
            with self.assertRaises(RuntimeError):
                supervisor.main(["--group", "01234567", "--allow-npu"])
        fake.release_locks.assert_called_once()

    def test_spawn_preserves_all_leases_and_new_session(self):
        obj = base.ProcessSupervisor(profiles.profile("01234567"), RUN)
        obj.env = lambda: {"H3_MINIMAL_RUN_ID": RUN}
        obj.locks = [mock.Mock(fileno=mock.Mock(return_value=n)) for n in range(10, 19)]
        child = SimpleNamespace(pid=500)
        with mock.patch.object(base.subprocess, "Popen", return_value=child) as popen, \
             mock.patch.object(base, "proc_identity", return_value={"pid": 500}):
            obj.spawn(["bash", "private-launcher"])
        self.assertEqual(popen.call_args.kwargs["pass_fds"], tuple(range(10, 19)))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["H3_MINIMAL_RUN_ID"], RUN)

    def test_reused_foreign_pid_not_signaled(self):
        obj = base.ProcessSupervisor(profiles.profile("01234567"), RUN)
        child = mock.Mock(pid=500)
        obj.children = [(child, {"start_ticks": 100})]
        obj.owned_group_members = mock.Mock(return_value=[])
        with mock.patch.object(base, "proc_identity", return_value={"start_ticks": 101, "pgrp": 500, "session": 500}), \
             mock.patch.object(base.os, "killpg", create=True) as killpg:
            self.assertTrue(obj.cleanup())
            killpg.assert_not_called()

    def test_owned_identity_cleanup_and_close_only_leases(self):
        obj = base.ProcessSupervisor(profiles.profile("01234567"), RUN)
        child = mock.Mock(pid=500)
        obj.children = [(child, {"start_ticks": 100})]
        obj.owned_group_members = mock.Mock(return_value=[])
        obj.locks = [mock.Mock(), mock.Mock()]
        handles = list(obj.locks)
        with mock.patch.object(base, "proc_identity", return_value={"start_ticks": 100, "pgrp": 500, "session": 500}), \
             mock.patch.object(base.os, "killpg", create=True) as killpg:
            self.assertTrue(obj.cleanup())
            killpg.assert_called_once_with(500, base.signal.SIGTERM)
        obj.release_locks()
        for handle in handles:
            handle.close.assert_called_once()
        self.assertFalse(obj.locks)

    def test_private_lock_rejects_nonregular_or_hardlink_without_flock(self):
        obj = base.ProcessSupervisor(profiles.profile("01234567"), RUN)
        for mode, links in ((stat.S_IFIFO, 1), (stat.S_IFREG, 2)):
            with mock.patch.object(base.os, "O_NOFOLLOW", 0, create=True), mock.patch.object(base.os, "O_NONBLOCK", 0, create=True), \
                 mock.patch.object(base.os, "open", return_value=20), mock.patch.object(base.os, "close") as close, \
                 mock.patch.object(base.os, "getuid", return_value=1000, create=True), \
                 mock.patch.object(base.os, "fstat", return_value=SimpleNamespace(st_mode=mode, st_nlink=links, st_uid=1000)), \
                 mock.patch.object(base.fcntl, "flock") as flock:
                with self.assertRaises(RuntimeError):
                    obj.take_lock(Path("device0.lock"))
                flock.assert_not_called(); close.assert_called_once_with(20)

    def test_fixed_group_acquires_all_eight_card_locks_and_state(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(profiles, "require_host", return_value=HOST), \
             mock.patch.object(base, "proc_identity", return_value=None), mock.patch("sys.stdout", new=io.StringIO()), \
             mock.patch.object(base.os, "O_NOFOLLOW", 0, create=True):
            install = Path(directory).resolve()
            with mock.patch.object(profiles, "INSTALL", install):
                a = supervisor.ValidationSupervisor("01234567")
                a.take_lock = mock.Mock()
                a.acquire()
                a_locks = {str(call.args[0]) for call in a.take_lock.call_args_list}
                self.assertEqual(len(a_locks), 9)
                self.assertTrue({str(a.selected.lease_root / f"device{i}.lock") for i in range(8)} <= a_locks)
                for instance in (a,):
                    state = json.loads((instance.selected.output_root / "validation_status.json").read_text())
                    self.assertEqual(state["run_id"], instance.run_id)
                    self.assertEqual(state["group"], instance.selected.group)
                self.assertIn("01234567", a.env()["H3_GROUP_ID"])


class WrapperTests(unittest.TestCase):
    def test_full_cpu_launch_gate_accepts_only_matching_proofs_and_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            install = Path(directory).resolve()
            selected = profiles.Profile("01234567", tuple(range(8)), 29673, install / "validation/01234567",
                                        install / "validation/card_locks", HERE, install / "candidate", install / "env.sh")
            root = selected.output_root / "runs" / ("20260913T123456Z_" + RUN)
            root.mkdir(parents=True)
            host_path, manifest_path = root / "host_identity.json", root / "source_manifest.json"
            host_path.write_text(json.dumps(HOST)); manifest_path.write_text('{"reviewed": "source"}')
            argv = ["--allow-npu", "--group", "01234567", "--physical-cards", "0,1,2,3,4,5,6,7", "--run-id", RUN,
                    "--host-proof", str(host_path), "--manifest-proof", str(manifest_path),
                    "--output", str(root / "npu_wrapper.json"), "--master-port", "29673"]
            environment = {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "H3_MINIMAL_RUN_ID": RUN,
                           "H3_VALIDATION_GROUP": "01234567", "H3_VALIDATION_SUPERVISOR_PID": str(os.getppid()),
                           "H3_ROOT": str(root), "H3_OUTPUT": str(root)}
            with mock.patch.dict(os.environ, environment), mock.patch.object(profiles, "INSTALL", install), \
                 mock.patch.object(profiles, "profile", return_value=selected), \
                 mock.patch.object(profiles, "read_host_identity", return_value=HOST), \
                 mock.patch.object(profiles, "source_manifest", return_value={"reviewed": "source"}), \
                 mock.patch.object(wrapper.socket, "socket"):
                args, got_profile, manifest = wrapper.parse_and_validate(argv)
                self.assertEqual(args.physical_cards, list(range(8)))
                self.assertEqual(args.host, HOST)
                self.assertEqual(got_profile, selected)
                self.assertEqual(manifest, {"reviewed": "source"})
                for key, value in (("H3_VALIDATION_SUPERVISOR_PID", "wrong-pid"), ("H3_ROOT", str(root.parent)),
                                   ("H3_MINIMAL_RUN_ID", "b" * 32)):
                    with mock.patch.dict(os.environ, {key: value}), self.assertRaises(RuntimeError):
                        wrapper.parse_and_validate(argv)
                host_path.write_text(json.dumps({**HOST, "boot_id": "old-boot"}))
                with self.assertRaises(RuntimeError):
                    wrapper.parse_and_validate(argv)
                host_path.write_text(json.dumps(HOST))
                manifest_path.write_text('{"reviewed": "old-source"}')
                with self.assertRaises(RuntimeError):
                    wrapper.parse_and_validate(argv)
                manifest_path.write_text('{"reviewed": "source"}')
                (root / "npu_wrapper.json").write_text("existing user result")
                with self.assertRaises(FileExistsError):
                    wrapper.parse_and_validate(argv)
                self.assertEqual((root / "npu_wrapper.json").read_text(), "existing user result")

    def test_worker_rejects_wrong_explicit_allocation_before_torch(self):
        with mock.patch.dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="4,5,6,7"):
            with self.assertRaises(RuntimeError):
                wrapper.worker(0, {"group": "01234567", "physical_cards": list(range(8))}, {})
        self.assertNotIn("torch", sys.modules)

    def test_spawn_receives_explicit_eight_cards_and_host(self):
        for group in ("01234567",):
            with tempfile.TemporaryDirectory() as directory:
                selected = profiles.profile(group)
                output = Path(directory) / "npu_wrapper.json"
                args = SimpleNamespace(output=output, physical_cards=list(selected.cards), host=HOST, group=group,
                                       run_id=RUN, master_port=selected.master_port, collective_timeout=180)
                manifest = {"fake": "CPU toy only"}
                calls = []
                def fake_spawn(worker, *, args, nprocs, join):
                    data, received_manifest = args
                    calls.append((data, received_manifest, nprocs, join))
                    for rank in range(8):
                        output.with_name(f"npu_wrapper.rank{rank}.json").write_text('{"status": "passed"}')
                torch = ModuleType("torch")
                mp = ModuleType("torch.multiprocessing")
                mp.spawn = fake_spawn; torch.multiprocessing = mp
                with mock.patch.dict(sys.modules, {"torch": torch, "torch.multiprocessing": mp}), \
                     mock.patch.object(wrapper, "parse_and_validate", return_value=(args, selected, manifest)), \
                     mock.patch.object(profiles, "require_host", return_value=HOST), \
                     mock.patch.object(profiles, "source_manifest", return_value=manifest), mock.patch("sys.stdout", new=io.StringIO()):
                    wrapper.main([])
                self.assertEqual(calls[0][0]["physical_cards"], list(selected.cards))
                self.assertEqual(calls[0][0]["host"], HOST)
                self.assertEqual(calls[0][2:], (8, True))

    def test_wrapper_math_helpers_match_existing_validated_test(self):
        original = HERE.parents[1] / "b_adaptation/tests/npu_wrapper_regression.py"
        def funcs(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef)}
        old, new = funcs(original), funcs(HERE / "npu_wrapper_regression.py")
        for name in ("initialize_small", "compare", "build_holder", "make_inputs", "prepare_qkv"):
            self.assertEqual(new[name], old[name])

    def test_launcher_order_isolation_and_no_30213_import(self):
        launcher = (HERE / "launch_validation.sh").read_text()
        self.assertLess(launcher.index("source /cache/zhonghao/h3/env_h3_31731.sh"), launcher.index('export PYTHONPATH='))
        self.assertIn('--physical-cards "${task_cards}"', launcher)
        self.assertIn("01234567) task_cards=0,1,2,3,4,5,6,7; task_port=29673", launcher)
        for file in ("profiles.py", "supervision_base.py", "run_validation.py", "npu_wrapper_regression.py"):
            source = (HERE / file).read_text()
            self.assertNotIn("run_a_supervised", source)
            self.assertNotIn("/cache/yunfeng", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
