"""CPU-only formal-B gates; no process launches, NPU imports, or model loads."""
import copy
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if os.name == "nt":
    # Production supervisor remains Linux-only. No lock operation is exercised
    # by these temporary-fixture/unit tests on Windows.
    sys.modules.setdefault("fcntl", SimpleNamespace(LOCK_EX=2, LOCK_NB=4))
b = importlib.import_module("run_b_supervised")


def config(source):
    return {"source_video": str(source), "edit_prompt": "Make the lake winter; preserve timing.",
            "allocated_physical_npu_ids": [2, 3, 6, 7],
            "requested_generation": {"width": 1344, "height": 768, "fps": 24,
                                     "duration_seconds": 5.0, "seed": 4101,
                                     "num_inference_steps": 50, "flow_shift": 12.0, "audio_flow_shift": 3.0}}


def weight_identity(kind):
    branch = kind == "branch"
    suffix = "linear_branch/model.safetensors" if branch else "adapters/default/adapter_model.safetensors"
    return {"path": str(b.CHECKPOINT / suffix), "size_bytes": 100,
            "header_sha256": "a" * 64 if branch else "b" * 64, "tensor_count": 800 if branch else 416}


def load_record(weights):
    return {"checkpoint": str(b.CHECKPOINT), "base_partition": "ref2va", "base_tensor_count": 535,
            "branch_tensor_count": 800, "lora_pairs_merged": 208, "lora_rank": 64, "lora_alpha": 64,
            "lora_scale": 1.0, "official_scale_source_commit": b.OFFICIAL_COMMIT,
            "merge_dtype": "FP32 delta, cast to parameter dtype, then add",
            "qkv_merge_layout": "post-base-loader contiguous Q/K/V thirds",
            "branch": weights["branch"], "lora": weights["lora"]}


def cpu_record():
    return {"status": "completed", "intraop_loading": 4, "intraop_before": 192,
            "intraop_after": 192, "interop_before": 192, "interop_after": 192,
            "interop_loading": 192, "phase_seconds": {"base": 1.0, "branch": 2.0, "lora": 3.0}}


def log_records(weights, loads=None, cpus=None):
    lines = []
    for pid in (101, 102, 103, 104):
        lines.append(f"H3B pid={pid} INFO test OPENVDN_B_LOAD_RECORD " + json.dumps((loads or {}).get(pid, load_record(weights))))
        lines.append(f"H3B pid={pid} INFO test OPENVDN_B_CPU_LOADING_END " + json.dumps((cpus or {}).get(pid, cpu_record())))
    return "\n".join(lines)


def npu_table():
    lines = []
    for card, pid in zip(b.CARDS, (101, 102, 103, 104)):
        lines.append(f"| {card} 910B3 | OK | 0 |")
    lines.append("| NPU Chip | Process id | Process name | Process memory(MB) |")
    for card, pid in zip(b.CARDS, (101, 102, 103, 104)):
        lines.append(f"| {card} 0 | {pid} | python | 12000 |")
    return "\n".join(lines)


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3-formal-gate-")
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"synthetic-source-not-a-real-video")
        self.config = config(self.source)
        self.weights = {kind: weight_identity(kind) for kind in ("branch", "lora")}

    def tearDown(self):
        self.temp.cleanup()

    def supervisor(self, mode="smoke-then-formal"):
        instance = b.BSupervisor(self.config, self.root / "unused_tiny.json", mode=mode)
        instance.run_dir = self.root / "run"
        instance.run_dir.mkdir(exist_ok=True)
        instance.update = mock.Mock(side_effect=lambda phase, **values: instance.status.update(phase=phase, **values))
        instance.request = mock.Mock()
        instance.require_fresh_smoke_for_formal = mock.Mock()
        return instance

    def artifacts(self, run_dir):
        (run_dir / "smoke_2step").mkdir()
        (run_dir / "output").mkdir()
        output = run_dir / "output/smoke_2step.mp4"
        output.write_bytes(b"synthetic-video-fixture-no-decode-claim")
        smoke = {"success": True, "http_code": "200", "content_type": "video/mp4", "curl_returncode": 0,
                 "requested_steps": 2, "output_video": str(output),
                 "ffprobe": {"verified": True, "checks": {"dimensions": [1344, 768], "fps": 24, "frame_count": 124}}}
        generation = self.config["requested_generation"]
        fields = {key: generation[key] for key in ("width", "height", "fps", "flow_shift", "seed")}
        fields.update(prompt=self.config["edit_prompt"], num_inference_steps=2,
                      extra_params=json.dumps({"task": "ref2va", "duration": 5.0, "audio_flow_shift": 3.0}))
        request = {"requested_steps": 2, "source_video": str(self.source), "fields": fields}
        (run_dir / "smoke_2step/result.json").write_text(json.dumps(smoke))
        (run_dir / "smoke_2step/request.json").write_text(json.dumps(request))
        return smoke

    def test_default_mode_only_runs_smoke(self):
        instance = self.supervisor("smoke-only")
        instance.run_requests()
        instance.request.assert_called_once_with("smoke_2step", 2)
        instance.require_fresh_smoke_for_formal.assert_not_called()
        self.assertFalse(instance.status["formal_50step_started"])
        self.assertEqual(instance.status["formal_request_attempts"], 0)

    def test_explicit_mode_gates_before_exactly_one_formal_request(self):
        instance = self.supervisor()
        events = []
        def request(name, steps):
            events.append((name, steps))
            if name == "b_50step":
                instance.status[name] = {"success": True, "ffprobe": {"verified": True}}
        instance.request.side_effect = request
        instance.require_fresh_smoke_for_formal.side_effect = lambda: events.append("gate")
        instance.run_requests()
        self.assertEqual(events, [("smoke_2step", 2), "gate", ("b_50step", 50)])
        self.assertTrue(instance.status["formal_50step_started"])
        self.assertTrue(instance.status["formal_50step_completed"])
        self.assertEqual(instance.status["formal_request_attempts"], 1)
        with self.assertRaises(RuntimeError):
            instance.run_requests()
        self.assertEqual(instance.request.call_count, 2)

    def test_failed_smoke_never_calls_gate_or_formal(self):
        instance = self.supervisor()
        instance.request.side_effect = RuntimeError("smoke failed")
        with self.assertRaises(RuntimeError):
            instance.run_requests()
        instance.request.assert_called_once_with("smoke_2step", 2)
        instance.require_fresh_smoke_for_formal.assert_not_called()
        self.assertEqual(instance.status["formal_request_attempts"], 0)

    def test_failed_gate_never_requests_formal(self):
        instance = self.supervisor()
        instance.require_fresh_smoke_for_formal.side_effect = RuntimeError("wrong candidate")
        with self.assertRaises(RuntimeError):
            instance.run_requests()
        instance.request.assert_called_once_with("smoke_2step", 2)
        self.assertFalse(instance.status["formal_50step_started"])

    def test_failed_formal_is_not_retried(self):
        instance = self.supervisor()
        instance.request.side_effect = [None, RuntimeError("formal failed")]
        with self.assertRaises(RuntimeError):
            instance.run_requests()
        with self.assertRaises(RuntimeError):
            instance.run_requests()
        self.assertEqual(instance.request.call_args_list, [mock.call("smoke_2step", 2), mock.call("b_50step", 50)])
        self.assertEqual(instance.status["formal_request_attempts"], 1)
        self.assertFalse(instance.status["formal_50step_completed"])

    def test_unverified_formal_output_cannot_be_marked_completed(self):
        instance = self.supervisor()
        instance.status["b_50step"] = {"success": True, "ffprobe": {"verified": False}}
        with self.assertRaises(RuntimeError):
            instance.run_requests()
        self.assertFalse(instance.status["formal_50step_completed"])

    def test_four_distinct_workers_with_restored_loader_pass(self):
        result = b.parse_full_load_evidence(log_records(self.weights), self.weights)
        self.assertEqual(set(result), {101, 102, 103, 104})

    def test_old_unlabelled_or_no_cpu_loader_logs_fail(self):
        for raw in ("\n".join("OPENVDN_B_LOAD_RECORD " + json.dumps(load_record(self.weights)) for _ in range(4)),
                    "\n".join(line for line in log_records(self.weights).splitlines() if "CPU_LOADING" not in line)):
            with self.assertRaises(RuntimeError):
                b.parse_full_load_evidence(raw, self.weights)

    def test_duplicate_pid_record_fails(self):
        raw = log_records(self.weights)
        with self.assertRaises(RuntimeError):
            b.parse_full_load_evidence(raw + "\n" + raw.splitlines()[0], self.weights)

    def test_incomplete_or_wrong_weight_coverage_fails(self):
        for key, value in (("base_tensor_count", 534), ("branch_tensor_count", 799), ("lora_pairs_merged", 207),
                           ("checkpoint", "wrong"), ("official_scale_source_commit", "wrong")):
            row = load_record(self.weights)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                b.parse_full_load_evidence(log_records(self.weights, loads={101: row}), self.weights)
        row = copy.deepcopy(load_record(self.weights))
        row["branch"]["header_sha256"] = "wrong"
        with self.assertRaises(RuntimeError):
            b.parse_full_load_evidence(log_records(self.weights, loads={101: row}), self.weights)

    def test_cpu_thread_restore_and_phase_timing_required(self):
        for key, value in (("status", "failed"), ("intraop_loading", 192), ("intraop_after", 4), ("interop_after", 1)):
            row = cpu_record()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                b.parse_full_load_evidence(log_records(self.weights, cpus={101: row}), self.weights)
        row = cpu_record()
        row["phase_seconds"]["lora"] = float("nan")
        with self.assertRaises(RuntimeError):
            b.parse_full_load_evidence(log_records(self.weights, cpus={101: row}), self.weights)

    def test_smoke_artifacts_bind_request_and_output(self):
        instance = self.supervisor()
        smoke = self.artifacts(instance.run_dir)
        result = b.verify_smoke_artifacts(instance.run_dir, smoke, self.config)
        self.assertEqual(result["output_sha256"], b.digest(Path(smoke["output_video"])))
        changed = copy.deepcopy(self.config)
        changed["edit_prompt"] = "different"
        with self.assertRaises(RuntimeError):
            b.verify_smoke_artifacts(instance.run_dir, smoke, changed)

    def test_old_output_or_unverified_metadata_rejected(self):
        instance = self.supervisor()
        original = self.artifacts(instance.run_dir)
        for change in ({"output_video": str(self.source)}, {"success": False},
                       {"ffprobe": {"verified": False}}, {"requested_steps": 50}):
            smoke = dict(original, **change)
            (instance.run_dir / "smoke_2step/result.json").write_text(json.dumps(smoke))
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                b.verify_smoke_artifacts(instance.run_dir, smoke, self.config)

    def test_worker_map_rejects_missing_or_duplicate_cards(self):
        self.assertEqual(b.selected_worker_map(npu_table()), dict(zip(b.CARDS, (101, 102, 103, 104))))
        for table in (npu_table().replace("| 7 0 | 104", "| 0 0 | 104"), npu_table() + "\n| 2 0 | 999 | other | 100 |"):
            with self.assertRaises(RuntimeError):
                b.selected_worker_map(table)

    def prepare_live_gate(self):
        instance = self.supervisor()
        del instance.require_fresh_smoke_for_formal
        instance.status["smoke_2step"] = self.artifacts(instance.run_dir)
        instance.formal_manifest = {"host": {"hostname": "same-host", "boot_id": "same-boot"}, "weights": self.weights}
        instance.runtime_manifest = mock.Mock(return_value=copy.deepcopy(instance.formal_manifest))
        instance.validated_gate = {"candidate": "new-loader-hash"}
        instance.locks = [SimpleNamespace(closed=False) for _ in range(5)]
        identity = {"pid": 99, "start_ticks": 200, "pgrp": 99, "session": 99}
        instance.status["server_proc_identity"] = identity
        instance.status["resource_snapshot"] = {"selected_physical_cards": list(b.CARDS), "explicitly_idle_cards": list(b.CARDS)}
        instance.server = SimpleNamespace(pid=99, poll=lambda: None)
        instance.owned_group_members = mock.Mock(return_value=[101, 102, 103, 104])
        (instance.run_dir / "server.log").write_text(log_records(self.weights))
        return instance, identity

    def test_full_gate_cross_checks_pid_ownership_and_writes_evidence(self):
        instance, identity = self.prepare_live_gate()
        with mock.patch.object(b.base, "proc_identity", side_effect=lambda pid: dict(identity, pid=pid)), mock.patch.object(
                b.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=npu_table(), stderr="")):
            evidence = instance.require_fresh_smoke_for_formal()
        self.assertEqual(evidence["status"], "passed")
        self.assertTrue((instance.run_dir / "formal_smoke_gate.json").is_file())
        self.assertEqual(set(evidence["loaded_records_by_pid"]), {101, 102, 103, 104})

    def test_changed_manifest_or_released_lease_blocks_formal(self):
        instance, _ = self.prepare_live_gate()
        instance.runtime_manifest.return_value = {"host": "another-host"}
        with self.assertRaises(RuntimeError):
            instance.require_fresh_smoke_for_formal()
        instance.runtime_manifest.return_value = copy.deepcopy(instance.formal_manifest)
        instance.locks[0].closed = True
        with self.assertRaises(RuntimeError):
            instance.require_fresh_smoke_for_formal()

    def test_unowned_worker_or_restarted_service_blocks_formal(self):
        instance, identity = self.prepare_live_gate()
        instance.owned_group_members.return_value = [101, 102, 103]
        with mock.patch.object(b.base, "proc_identity", return_value=identity), mock.patch.object(
                b.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=npu_table(), stderr="")):
            with self.assertRaises(RuntimeError):
                instance.require_fresh_smoke_for_formal()
        with mock.patch.object(b.base, "proc_identity", return_value=dict(identity, start_ticks=999)):
            with self.assertRaises(RuntimeError):
                instance.require_fresh_smoke_for_formal()

    def test_formal_only_cli_does_not_exist(self):
        with self.assertRaises(SystemExit) as caught:
            b.main(["--validation-result", "/tmp/proof.json", "--mode", "formal-only"])
        self.assertEqual(caught.exception.code, 2)

    def test_logging_format_really_includes_pid(self):
        import logging
        value = logging.LogRecord("vllm_omni.test", logging.INFO, __file__, 1, "OPENVDN_B_LOAD_RECORD {}", (), None)
        formatter = logging.Formatter(b.logging_configuration()["formatters"]["h3b"]["format"])
        self.assertIn(f"H3B pid={os.getpid()}", formatter.format(value))

    def test_missing_preflight_idle_evidence_blocks(self):
        instance, _ = self.prepare_live_gate()
        instance.status.pop("resource_snapshot")
        with self.assertRaises(RuntimeError):
            instance.require_fresh_smoke_for_formal()

    def test_main_success_preserves_formal_flags_and_cleans_up(self):
        script = self.root / "scripts"
        script.mkdir()
        (script / "experiment.json").write_text(json.dumps(self.config))
        fake = SimpleNamespace(run_dir=self.root, status={}, acquire=mock.Mock(), preflight=mock.Mock(),
                               launch=mock.Mock(), cleanup=mock.Mock(return_value=True), release_locks=mock.Mock())
        fake.update = lambda phase, **values: fake.status.update(phase=phase, **values)
        fake.run_requests = lambda: fake.status.update(formal_50step_started=True, formal_50step_completed=True,
                                                       formal_request_attempts=1)
        with mock.patch.object(b, "SCRIPT_DIR", script), mock.patch.object(b, "BSupervisor", return_value=fake), \
                mock.patch.object(b.signal, "signal"):
            code = b.main(["--validation-result", str(self.root / "tiny.json"), "--mode", "smoke-then-formal"])
        self.assertEqual(code, 0)
        self.assertEqual(fake.status["phase"], "formal_completed_review_required")
        self.assertTrue(fake.status["formal_50step_started"])
        self.assertTrue(fake.status["formal_50step_completed"])
        self.assertEqual(fake.status["formal_request_attempts"], 1)
        fake.cleanup.assert_called_once()
        fake.release_locks.assert_called_once()

    def test_main_gate_failure_still_cleans_up_and_releases(self):
        script = self.root / "scripts"
        script.mkdir()
        (script / "experiment.json").write_text(json.dumps(self.config))
        fake = SimpleNamespace(run_dir=self.root, status={"formal_50step_started": False},
                               acquire=mock.Mock(), preflight=mock.Mock(), launch=mock.Mock(),
                               run_requests=mock.Mock(side_effect=RuntimeError("gate rejected")),
                               cleanup=mock.Mock(return_value=True), release_locks=mock.Mock())
        fake.update = lambda phase, **values: fake.status.update(phase=phase, **values)
        with mock.patch.object(b, "SCRIPT_DIR", script), mock.patch.object(b, "BSupervisor", return_value=fake), \
                mock.patch.object(b.signal, "signal"):
            code = b.main(["--validation-result", str(self.root / "tiny.json"), "--mode", "smoke-then-formal"])
        self.assertEqual(code, 1)
        self.assertEqual(fake.status["phase"], "failed")
        self.assertFalse(fake.status["formal_50step_started"])
        fake.cleanup.assert_called_once()
        fake.release_locks.assert_called_once()

    def test_weight_manifest_headers_and_changes_are_tracked(self):
        ref, stage = self.root / "weights/ref2va", self.root / "weights/stage"
        (ref / "transformer").mkdir(parents=True)
        (stage / "linear_branch").mkdir(parents=True)
        (stage / "adapters/default").mkdir(parents=True)
        def safetensors(path, count):
            header = {f"t{i}": {"dtype": "F32", "shape": [1], "data_offsets": [4*i, 4*i+4]} for i in range(count)}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * count))
        mapping = {f"t{i}": f"shard-{i % 13}.safetensors" for i in range(535)}
        for name in set(mapping.values()):
            safetensors(ref / "transformer" / name, 1)
        (ref / "transformer/model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        metadata = ref / "config.json"
        metadata.write_text('{"version": 1}')
        safetensors(stage / "linear_branch/model.safetensors", 800)
        safetensors(stage / "adapters/default/adapter_model.safetensors", 416)
        with mock.patch.object(b, "REF2VA_ROOT", ref), mock.patch.object(b, "CHECKPOINT", stage):
            first = b.runtime_weight_manifest()
            self.assertEqual(first["branch"]["tensor_count"], 800)
            self.assertEqual(first["lora"]["tensor_count"], 416)
            metadata.write_text('{"version": 2}')
            self.assertNotEqual(first, b.runtime_weight_manifest())
            (ref / "transformer/shard-0.safetensors").unlink()
            with self.assertRaises(RuntimeError):
                b.runtime_weight_manifest()


if __name__ == "__main__":
    unittest.main(verbosity=2)
