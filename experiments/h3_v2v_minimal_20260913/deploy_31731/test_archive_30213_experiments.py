"""CPU/filesystem toys only. No remote writes, NPU or real experiment mutation."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import archive_30213_experiments as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.path = self.source / "sample.py"
        self.path.write_text("x = 1\n", encoding="utf-8")
        self.selected = [(self.source, "sample.py", "code/sample.py")]
        self.entries = archive.inspect_files(self.selected)
        self.archive_id = "a" * 32
        self.manifest = dict(schema=1, archive_id=self.archive_id, source_host=archive.SOURCE_HOST,
                             files=self.entries, completed_runs=[], excluded_unsealed_runs=[])

    def gate(self):
        return dict(phase="formal_completed_review_required", finished_at="2026-09-13T13:00:00+00:00",
                    cleanup_completed=True, remaining_owned_process_groups={}, needs_attention=False,
                    run_directory=str(self.source), run_id="toy", supervisor_proc_identity=dict(pid=123, start_ticks=42))

    def make_tar(self):
        path = self.root / "evidence.tar"
        archive.write_archive(path, self.selected, self.entries, self.manifest)
        checksum, identity = archive.hash_file(path)
        record = dict(status="complete", archive_id=self.archive_id, archive_name=path.name,
                      archive_sha256=checksum, archive_bytes=identity[2], file_count=len(self.entries),
                      manifest_sha256=hashlib.sha256(archive.canonical_json(self.manifest)).hexdigest())
        return path, record

    def test_terminal_cleanup_gate_accepts_exited_not_active(self):
        self.assertTrue(archive.completed_gate(self.gate(), self.source, lambda pid: None)["cleanup_completed"])
        with self.assertRaisesRegex(ValueError, "has not exited"):
            archive.completed_gate(self.gate(), self.source, lambda pid: dict(start_ticks=42, state="Z"))

    def test_reused_pid_is_not_original_process(self):
        archive.completed_gate(self.gate(), self.source, lambda pid: dict(start_ticks=43))

    def test_incomplete_or_ambiguous_cleanup_fails(self):
        for changes in (dict(phase="b_50step_request_running"), dict(finished_at=None), dict(cleanup_completed=None),
                        dict(remaining_owned_process_groups={1: [2]}), dict(needs_attention=True),
                        dict(run_directory="/another/path"), dict(supervisor_proc_identity=None)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                archive.completed_gate({**self.gate(), **changes}, self.source, lambda pid: None)

    def test_success_and_failed_terminal_evidence_not_conflated(self):
        result = archive.completed_gate({**self.gate(), "phase": "failed", "error": "toy failure"}, self.source, lambda pid: None)
        self.assertEqual(result["phase"], "failed")
        self.assertEqual(result["error"], "toy failure")

    def test_path_traversal_and_credentials_rejected(self):
        for path in ("../escape", "/absolute", "a//b", "a/./b", "a\\b", "C:/file", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                archive.safe_relative(path)
        for path in ("keys/key.pem", ".ssh/config", "envs/model/file", ".env.local", "models/file.safetensors"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                archive.source_policy(path)
        archive.source_policy("b_adaptation/vendor/vllm_omni/diffusion/models/minimax_h3/model.py")

    def test_private_key_and_token_split_across_chunks_rejected(self):
        for payload in (b"x-----BEGIN PRIVATE KEY-----y", b"hf_" + b"a" * 30, b"https://person:password@example.org"):
            with self.subTest(payload_len=len(payload)), mock.patch.object(archive, "CHUNK", 7), self.assertRaises(ValueError):
                archive.scan_stream(io.BytesIO(payload), check_credentials=True)

    def test_hash_roundtrip_and_source_preserved(self):
        original = self.path.read_bytes()
        path, record = self.make_tar()
        parsed = archive.inspect_tar(path, record)
        pending = self.root / ".partial-toy"
        archive.extract_verified(path, pending, parsed)
        final = self.root / "final"
        archive.atomic_rename_new(pending, final)
        self.assertEqual((final / "code/sample.py").read_bytes(), original)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue((final / "ARCHIVE_OWNER.json").is_file())

    def test_no_overwrite_archive_or_destination(self):
        path, record = self.make_tar()
        with self.assertRaises(FileExistsError):
            archive.write_archive(path, self.selected, self.entries, self.manifest)
        pending = self.root / "pending"
        pending.mkdir()
        final = self.root / "existing"
        final.mkdir()
        (final / "keep").write_text("keep")
        with self.assertRaises(OSError):
            archive.atomic_rename_new(pending, final)
        self.assertEqual((final / "keep").read_text(), "keep")
        self.assertTrue(pending.is_dir())
        empty = self.root / "empty-existing"
        empty.mkdir()
        original_inode = empty.stat().st_ino
        with self.assertRaises(OSError):
            archive.atomic_rename_new(pending, empty)
        self.assertTrue(pending.is_dir())
        self.assertEqual(empty.stat().st_ino, original_inode)

    def test_changed_source_before_copy_fails(self):
        self.path.write_text("x = 2222\n")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            archive.write_archive(self.root / "notsealed.tar", self.selected, self.entries, self.manifest)

    def test_changed_source_during_copy_fails(self):
        original_reader = archive.HashingReader
        source_path = self.path

        class MutatingReader(original_reader):
            def read(self, count=-1):
                data = super().read(count)
                source_path.write_text("x = 333333\n")
                return data

        with mock.patch.object(archive, "HashingReader", MutatingReader), self.assertRaisesRegex(ValueError, "mutated"):
            archive.write_archive(self.root / "notsealed.tar", self.selected, self.entries, self.manifest)

    def test_content_hash_mismatch_detected(self):
        path, record = self.make_tar()
        parsed = archive.inspect_tar(path, record)
        parsed["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            archive.extract_verified(path, self.root / "bad-dest", parsed)

    def test_tar_symlink_hardlink_fifo_absolute_duplicate_rejected(self):
        for kind in ("symlink", "hardlink", "fifo", "absolute", "duplicate"):
            with self.subTest(kind=kind):
                path = self.root / (kind + ".tar")
                payload = archive.canonical_json(self.manifest)
                with tarfile.open(path, "w") as handle:
                    info = tarfile.TarInfo("archive_manifest.json")
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))
                    info = tarfile.TarInfo("/escape" if kind == "absolute" else "code/sample.py")
                    if kind == "symlink":
                        info.type, info.linkname = tarfile.SYMTYPE, "/outside"
                    elif kind == "hardlink":
                        info.type, info.linkname = tarfile.LNKTYPE, "archive_manifest.json"
                    elif kind == "fifo":
                        info.type = tarfile.FIFOTYPE
                    handle.addfile(info)
                    if kind == "duplicate":
                        handle.addfile(info)
                record = dict(archive_id=self.archive_id, file_count=1, manifest_sha256=hashlib.sha256(payload).hexdigest())
                with self.assertRaises(ValueError):
                    archive.inspect_tar(path, record)

    def test_tar_unknown_extra_file_rejected(self):
        path, record = self.make_tar()
        with tarfile.open(path, "a") as handle:
            handle.addfile(tarfile.TarInfo("code/extra.py"))
        with self.assertRaisesRegex(ValueError, "Extra/missing"):
            archive.inspect_tar(path, record)

    def test_source_symlink_rejected(self):
        linked = self.source / "link.py"
        try:
            linked.symlink_to(self.path)
        except OSError:
            self.skipTest("native symlinks unavailable")
        with self.assertRaises((ValueError, OSError)):
            archive.inspect_files([(self.source, "link.py", "code/link.py")])

    def test_host_and_archive_id_fail_closed(self):
        with mock.patch.object(archive.socket, "gethostname", return_value="wrong-host"), self.assertRaises(ValueError):
            archive.require_host(archive.SOURCE_HOST)
        for value in ("../bad", "A" * 32, "a" * 31, "a" * 33):
            with self.assertRaises(ValueError):
                archive.validate_archive_id(value)

    def test_exclusive_record_never_overwrites(self):
        path = self.root / "record.json"
        archive.fsync_json_exclusive(path, {"status": "complete"})
        with self.assertRaises(FileExistsError):
            archive.fsync_json_exclusive(path, {"status": "failed"})
        self.assertEqual(json.loads(path.read_text()), {"status": "complete"})

    def setup_selection(self):
        code, results = self.root / "code", self.root / "results"
        code.mkdir()
        results.mkdir()
        for name in archive.CODE_FILES:
            (code / name).write_text("{}" if name.endswith(".json") else "example\n")
        for name in archive.CODE_TREES:
            (code / name).mkdir()
            (code / name / "example.py").write_text("x = 1\n")
        for group, _ in archive.GROUPS:
            (results / group).mkdir(parents=True)
        run = results / "b_model/runs" / archive.REQUIRED_RUN
        run.mkdir()
        state = {**self.gate(), "run_directory": str(run)}
        (run / "b_status.json").write_text(json.dumps(state))
        return code, results, run

    def test_selection_excludes_old_unsealed_but_requires_latest_complete(self):
        code, results, run = self.setup_selection()
        orphan = results / "b_model/runs/old_orphan"
        orphan.mkdir()
        (orphan / "b_status.json").write_text(json.dumps({**self.gate(), "run_directory": str(orphan), "phase": "cleaning_up_own_processes", "finished_at": None}))
        selected, accepted, excluded = archive.select_sources(code, results, process_reader=lambda pid: None)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(excluded), 1)
        self.assertFalse(any("old_orphan" in item[2] for item in selected))
        state = json.loads((run / "b_status.json").read_text())
        state["phase"] = "b_50step_request_running"
        (run / "b_status.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "Required latest B"):
            archive.select_sources(code, results, process_reader=lambda pid: None)

    def test_second_inventory_detects_new_files(self):
        code, results, run = self.setup_selection()
        before, _, _ = archive.select_sources(code, results, process_reader=lambda pid: None)
        (run / "late.json").write_text("{}")
        after, _, _ = archive.select_sources(code, results, process_reader=lambda pid: None)
        self.assertNotEqual(before, after)

    @unittest.skipUnless(os.name == "posix", "Linux flock toy")
    def test_existing_locks_are_read_only_and_block_active_job(self):
        import fcntl
        root = self.root / "locks"
        root.mkdir()
        for name in archive.LOCKS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"owned existing lease")
        with (root / archive.LOCKS[1]).open("rb") as job:
            fcntl.flock(job, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with archive.quiescence_locks(root):
                    self.fail("archive entered while the existing job held its lock")
        with archive.quiescence_locks(root):
            pass
        for name in archive.LOCKS:
            self.assertEqual((root / name).read_bytes(), b"owned existing lease")

    def test_destination_parent_symlink_cannot_escape(self):
        destination, outside = self.root / "destination", self.root / "outside"
        destination.mkdir()
        outside.mkdir()
        try:
            (destination / "escape").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("native symlinks unavailable")
        with self.assertRaises((ValueError, OSError)):
            with archive.create_destination_file(destination, "escape/wrong.py", 0o644) as output:
                output.write(b"should not happen")
        self.assertEqual(list(outside.iterdir()), [])

    def test_only_run_root_runtime_cache_excluded_and_recorded(self):
        code, results, run = self.setup_selection()
        cache = run / "cache/triton"
        cache.mkdir(parents=True)
        (cache / "npu_utils.so").write_bytes(b"fake runtime binary")
        code_cache = code / "b_adaptation/vendor/cache"
        code_cache.mkdir(parents=True)
        (code_cache / "keep.py").write_text("keep = 1\n")
        for name in ("server.log", "request.json", "video.mp4"):
            (run / name).write_bytes(b"keep evidence")
        selected, _, excluded = archive.select_sources(code, results, process_reader=lambda pid: None)
        names = {item[2] for item in selected}
        self.assertFalse(any("npu_utils.so" in name for name in names))
        self.assertIn("code/b_adaptation/vendor/cache/keep.py", names)
        for name in ("server.log", "request.json", "video.mp4"):
            self.assertTrue(any(item.endswith("/" + name) for item in names))
        self.assertEqual(excluded, [{"type": "runtime_cache", "path": str(run / "cache"),
                          "reason": "Explicitly excluded run-root compiled runtime cache; source retained"}])
        # A .so outside the explicitly excluded run-root cache remains forbidden.
        (run / "unexpected.so").write_bytes(b"must fail")
        with self.assertRaisesRegex(ValueError, "binary rejected"):
            archive.select_sources(code, results, process_reader=lambda pid: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
