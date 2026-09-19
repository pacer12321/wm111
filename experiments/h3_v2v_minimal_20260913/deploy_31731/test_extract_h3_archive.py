"""Small local CPU-only archives; never touch production paths."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import extract_h3_archive as extract


def entry(name, data=b"python", kind=tarfile.REGTYPE, link=""):
    member = tarfile.TarInfo(name)
    member.type, member.linkname = kind, link
    member.size = len(data) if kind == tarfile.REGTYPE else 0
    return member, data


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "deployment_owner.json").write_text(json.dumps({"deployment": extract.DEPLOYMENT}))

    def make_archive(self, entries):
        path = self.root / "env.tar"
        with tarfile.open(path, "w") as tar:
            for member, data in entries:
                tar.addfile(member, io.BytesIO(data) if member.isfile() else None)
        record = {"env.tar": {"path": "env.tar", "kind": "file", "size": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
        (self.root / "transfer_verified.json").write_text(json.dumps(record))
        return path

    def preflight(self, entries):
        with tarfile.open(self.make_archive(entries), "r:") as tar:
            return extract.preflight(tar, self.root / "env")

    def test_valid_file_preflight(self):
        plan, counts = self.preflight([entry("bin/python")])
        self.assertEqual(counts["regular_files"], 1)
        self.assertEqual(plan[0].name, "bin/python")

    def test_absolute_internal_link_is_mapped_relative(self):
        plan, counts = self.preflight([entry("lib/a"), entry("bin/a", kind=tarfile.SYMTYPE,
                                                        link=extract.SOURCE_ENV + "/lib/a")])
        self.assertEqual(plan[-1].linkname, "../lib/a")
        self.assertEqual(counts["mapped_absolute_symlinks"], 1)

    def test_external_absolute_and_escaping_relative_links_rejected(self):
        for link in ("/etc/passwd", "../../escape", extract.SOURCE_ENV + "/../other"):
            with self.subTest(link=link), self.assertRaises(RuntimeError):
                self.preflight([entry("data"), entry("bin/link", kind=tarfile.SYMTYPE, link=link)])

    def test_noncanonical_member_names_rejected(self):
        for name in ("../escape", "/absolute", "a/../b", "a//b", "./a", "a\\b", "C:/file", "a\x00b"):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                # NULs are truncated by tar headers, so validate TarInfo directly.
                extract.normalized_name(entry(name)[0])

    def test_duplicate_member_rejected(self):
        with self.assertRaises(RuntimeError):
            self.preflight([entry("file"), entry("file")])

    def test_hardlinks_and_special_files_rejected(self):
        for kind in (tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                self.preflight([entry("normal"), entry("bad", kind=kind, link="normal")])

    def test_member_below_symlink_rejected_regardless_of_order(self):
        for entries in ([entry("dir/child"), entry("dir", kind=tarfile.SYMTYPE, link="real")],
                        [entry("dir", kind=tarfile.SYMTYPE, link="real"), entry("dir/child")]):
            with self.assertRaises(RuntimeError):
                self.preflight(entries)

    def test_symlink_chain_escape_not_just_lexical(self):
        with self.assertRaises(RuntimeError):
            self.preflight([entry("data"), entry("a", kind=tarfile.SYMTYPE, link="."),
                            entry("x", kind=tarfile.SYMTYPE, link="a/../outside")])

    def test_symlink_cycle_rejected(self):
        with self.assertRaises(RuntimeError):
            self.preflight([entry("data"), entry("a", kind=tarfile.SYMTYPE, link="b"),
                            entry("b", kind=tarfile.SYMTYPE, link="a")])

    def test_finite_repeated_symlink_resolution(self):
        _, counts = self.preflight([entry("data"), entry("a", kind=tarfile.SYMTYPE, link="."),
                                    entry("x", kind=tarfile.SYMTYPE, link="a/a/data")])
        self.assertEqual(counts["symlinks"], 2)

    def test_sha_mismatch_creates_no_env(self):
        self.make_archive([entry("bin/python")])
        records = json.loads((self.root / "transfer_verified.json").read_text())
        records["env.tar"]["sha256"] = "0" * 64
        (self.root / "transfer_verified.json").write_text(json.dumps(records))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            extract.extract_once(self.root)
        self.assertFalse((self.root / "env").exists())
        states = list(self.root.glob("env-extraction-*.json"))
        self.assertEqual(json.loads(states[0].read_text())["phase"], "failed")

    def test_inspect_only_has_no_writes(self):
        self.make_archive([entry("bin/python")])
        before = set(self.root.iterdir())
        with contextlib.redirect_stdout(io.StringIO()):
            result = extract.extract_once(self.root, inspect_only=True)
        self.assertEqual(result["phase"], "preflight_passed")
        self.assertEqual(before, set(self.root.iterdir()))

    def test_extract_fresh_and_refuse_resume(self):
        self.make_archive([entry("bin/python", b"toy")])
        with contextlib.redirect_stdout(io.StringIO()):
            result = extract.extract_once(self.root)
        self.assertEqual(result["phase"], "extracted_runtime_validation_required")
        self.assertEqual((self.root / "env/bin/python").read_bytes(), b"toy")
        self.assertTrue((self.root / "env" / extract.OWNER_NAME).is_file())
        with self.assertRaises(RuntimeError):
            extract.extract_once(self.root)

    def test_interruption_retains_owned_partial_and_failed_record(self):
        self.make_archive([entry("bin/python")])
        with patch.object(tarfile.TarFile, "extractall", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            extract.extract_once(self.root)
        self.assertTrue((self.root / "env" / extract.OWNER_NAME).is_file())
        state = json.loads(next(self.root.glob("env-extraction-*.json")).read_text())
        self.assertEqual(state["phase"], "failed")
        self.assertFalse(state["automatic_resume_permitted"])

    def test_marker_collision_rejected(self):
        with self.assertRaises(RuntimeError):
            self.preflight([entry(extract.OWNER_NAME)])


if __name__ == "__main__":
    unittest.main()
