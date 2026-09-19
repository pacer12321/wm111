import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import transfer_h3 as m


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3-transfer-test-")
        self.root = Path(self.temp.name).resolve()
        self.stage = self.root / "stage"
        self.target = self.root / "target"
        self.stage.mkdir()
        self.target.mkdir()
        self.stage_patch = patch.object(m, "STAGE", self.stage)
        self.target_patch = patch.object(m, "TARGET", self.target)
        self.stage_patch.start()
        self.target_patch.start()

    def tearDown(self):
        self.target_patch.stop()
        self.stage_patch.stop()
        self.temp.cleanup()

    def transfer(self):
        obj = m.Transfer.__new__(m.Transfer)
        obj.mode = "consume"
        obj.root = self.target
        obj.state = {"bytes_copied_this_process": 0, "files_completed": 0}
        obj.last_status = time.monotonic()
        return obj

    def test_copy_hash_and_content(self):
        source = self.stage / "source"
        source.write_bytes(b"h3" * 10000)
        dest = self.target / "weights"
        digest, size = self.transfer().copy_file(source, dest)
        self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(size, source.stat().st_size)
        self.assertEqual(source.read_bytes(), dest.read_bytes())

    def test_same_file_resume(self):
        source = self.stage / "source"
        source.write_bytes(b"same")
        dest = self.target / "weights"
        dest.write_bytes(b"same")
        self.transfer().copy_file(source, dest)

    def test_different_destination_never_overwritten(self):
        source = self.stage / "source"
        source.write_bytes(b"new")
        dest = self.target / "weights"
        dest.write_bytes(b"old")
        with self.assertRaises(RuntimeError):
            self.transfer().copy_file(source, dest)
        self.assertEqual(dest.read_bytes(), b"old")

    def test_bad_transport_hash_not_published(self):
        source = self.stage / "source"
        source.write_bytes(b"corrupted")
        dest = self.target / "weights"
        with self.assertRaises(RuntimeError):
            self.transfer().copy_file(source, dest, "0" * 64)
        self.assertFalse(dest.exists())

    def test_relative_escape_rejected(self):
        for name in ("../escape", "/tmp/escape"):
            with self.assertRaises(RuntimeError):
                m.within(self.target, name)

    def test_symlink_parent_escape_rejected(self):
        (self.target / "link").symlink_to(self.stage, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            m.within(self.target, "link/file")

    def test_existing_owner_required(self):
        with self.assertRaises(RuntimeError):
            m.claim(self.target)
        owned = self.root / "owned"
        m.claim(owned)
        m.claim(owned)

    def test_atomic_json_not_follow_old_temp_link(self):
        external = self.stage / "outside"
        external.write_text("unchanged")
        (self.target / "status.json.writing").symlink_to(external)
        m.atomic_json(self.target / "status.json", {"ok": True})
        self.assertEqual(external.read_text(), "unchanged")

    def test_failed_producer_before_manifest_exits(self):
        m.atomic_json(self.stage / "produce_status.json", {"phase": "failed"})
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            self.transfer().consume()

    def test_nfs_metadata_never_renames_or_overwrites(self):
        path = self.stage / "status.json"
        with patch.object(m.os, "replace", side_effect=AssertionError("rename forbidden")):
            m.atomic_json(path, {"step": 1})
            m.atomic_json(path, {"step": 2})
        self.assertEqual(m.read_json(path), {"step": 2})
        self.assertEqual(len(list(self.stage.glob("status.json.version-*.ready"))), 2)

    def test_nfs_data_never_renames_or_links(self):
        source = self.target / "source"
        source.write_bytes(b"immutable bytes")
        dest = self.stage / "object"
        obj = self.transfer()
        obj.mode = "produce"
        obj.root = self.stage
        with patch.object(m.os, "replace", side_effect=AssertionError("rename forbidden")), \
                patch.object(m.os, "link", side_effect=AssertionError("link forbidden")):
            obj.copy_file(source, dest)
        self.assertEqual(source.read_bytes(), dest.read_bytes())

    def test_created_but_unrecorded_symlink_resume(self):
        (self.target / "src").mkdir()
        (self.target / "src/link").symlink_to("../env/python")
        m.atomic_json(self.stage / "manifest.json", {
            "deployment": m.DEPLOYMENT, "target_host": m.TARGET_HOST,
            "complete": True,
            "entries": [{"path": "src/link", "kind": "symlink", "link": "../env/python"}],
        })
        obj = self.transfer()
        obj.consume()
        self.assertEqual(obj.state["phase"], "transfer_completed_extraction_and_validation_required")
        self.assertIn("src/link", json.loads((self.target / "transfer_verified.json").read_text()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
