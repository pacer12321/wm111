"""CPU-only toy regressions; no actual environment, NPU, or migration needed."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import pack_h3_actual_environment as pack


class File:
    def __init__(self, source, target, **kwargs):
        self.source, self.target = source, target
        self.__dict__.update(kwargs)


class PackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_redirect_only_selected_preserves_old_objects(self):
        (self.root / "a.py").write_text("actual")
        selected, untouched = File("cache/a.py", "a.py"), File("cache/b.py", "b.py")
        env = types.SimpleNamespace(files=[selected, untouched])
        with patch.object(pack.audit, "ALLOWED_MISSING", set()):
            result = pack.redirect_legacy_files(env, {"a.py"}, File, self.root)
        self.assertEqual(result, ["a.py"])
        self.assertEqual(env.files[0].source, str(self.root / "a.py"))
        self.assertEqual(env.files[0].file_mode, "unknown")
        self.assertIsNone(env.files[0].prefix_placeholder)
        self.assertFalse(env.files[0].is_conda)
        self.assertIs(env.files[1], untouched)
        self.assertEqual(selected.source, "cache/a.py")

    def test_manifest_coverage_failure(self):
        with patch.object(pack.audit, "ALLOWED_MISSING", set()), self.assertRaises(RuntimeError):
            pack.redirect_legacy_files(types.SimpleNamespace(files=[]), {"a.py"}, File, self.root)

    def test_duplicate_targets_failure(self):
        with self.assertRaises(RuntimeError):
            pack.redirect_legacy_files(types.SimpleNamespace(files=[File("a", "x"), File("b", "x")]), set(), File, self.root)

    def test_escaping_source_failure(self):
        for path in ("../escape", "/absolute"):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                pack.regular_source(self.root, path)

    def test_missing_source_failure(self):
        with self.assertRaises(RuntimeError):
            pack.regular_source(self.root, "absent.py")

    def make_tar(self, entries):
        path = self.root / "toy.tar"
        with tarfile.open(path, "w") as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return path

    def test_archive_hash_match(self):
        path = self.make_tar([("init.py", b"actual"), ("unrelated", b"ignored")])
        self.assertEqual(pack.verify_archive(path, {"init.py": {hashlib.sha256(b"actual").hexdigest()}}), 1)

    def test_archive_old_cache_rejected(self):
        path = self.make_tar([("init.py", b"old cache")])
        with self.assertRaises(RuntimeError):
            pack.verify_archive(path, {"init.py": {hashlib.sha256(b"actual").hexdigest()}})

    def test_archive_missing_rejected(self):
        path = self.make_tar([])
        with self.assertRaises(RuntimeError):
            pack.verify_archive(path, {"init.py": {"unused"}})

    def test_archive_duplicate_rejected(self):
        path = self.make_tar([("init.py", b"actual"), ("init.py", b"actual")])
        with self.assertRaises(RuntimeError):
            pack.verify_archive(path, {"init.py": {hashlib.sha256(b"actual").hexdigest()}})

    def test_output_boundary_owner_and_no_overwrite(self):
        stage, objects = self.root, self.root / "objects"
        objects.mkdir()
        marker = stage / "deployment_owner.json.version-0001"
        marker.write_text(json.dumps({"deployment": pack.DEPLOYMENT}))
        marker.with_name(marker.name + ".ready").touch()
        output = objects / "env-toy.tar"
        with patch.object(pack, "STAGE", stage), patch.object(pack, "OBJECTS", objects):
            self.assertEqual(pack.checked_output(str(output)), output)
            for invalid in (stage / "env-bad.tar", objects / "wrong.tar", objects / "env-a.tar.exe"):
                with self.assertRaises(RuntimeError):
                    pack.checked_output(str(invalid))
            output.touch()
            with self.assertRaises(RuntimeError):
                pack.checked_output(str(output))

    def test_wrong_stage_owner_rejected(self):
        objects = self.root / "objects"
        objects.mkdir()
        marker = self.root / "deployment_owner.json.version-0001"
        marker.write_text('{"deployment":"other"}')
        marker.with_name(marker.name + ".ready").touch()
        with patch.object(pack, "STAGE", self.root), patch.object(pack, "OBJECTS", objects), self.assertRaises(RuntimeError):
            pack.checked_output(str(objects / "env-toy.tar"))


if __name__ == "__main__":
    unittest.main()
