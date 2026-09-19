"""CPU-only filesystem safety regression; fixtures stay in tempfile directories."""
import os
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import relocate_h3_environment as relocation


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h3-relocation-test-")
        self.base = Path(self.temporary.name).resolve()
        self.layout = relocation.Layout(self.base / "env", self.base / "src")
        for root in self.layout.roots:
            root.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, text, root=None):
        path = (root or self.layout.env) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def symlink(self, target, path, directory=False):
        try:
            os.symlink(target, path, target_is_directory=directory)
        except OSError as exc:
            self.skipTest(f"This OS does not permit test symlinks: {exc}")

    def test_dry_run_is_read_only_and_apply_is_idempotent(self):
        finder = self.write("lib/python3.12/site-packages/__editable___vllm_finder.py",
                            f"MAPPING = {{'vllm': '{relocation.OLD_SRC}/vllm/vllm'}}\n")
        cann = self.write("Ascend/cann-9.0.1/set_env.sh",
                          f'version_dirpath="{relocation.OLD_ENV}/Ascend/cann-9.0.1"\n')
        entry = self.write("bin/vllm", f"#!{relocation.OLD_ENV}/bin/python\nprint('test')\n")
        old = {path: path.read_bytes() for path in (finder, cann, entry)}
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["change_count"], 3)
        self.assertFalse(plan["blockers"])
        self.assertEqual(old, {path: path.read_bytes() for path in old})
        result = relocation.apply_plan(plan, self.layout)
        self.assertEqual(result["status"], "relocated")
        self.assertEqual(result["applied_count"], 3)
        self.assertEqual(relocation.make_plan(self.layout)["change_count"], 0)

    def test_generated_source_compile_paths_are_relocated(self):
        root = self.layout.src / "vllm-ascend"
        path = self.write("vllm_ascend/_cann_ops_custom/dynamic/chunk.py",
                          f"options = ['-include{relocation.OLD_SRC}/vllm-ascend/csrc/common/include/cann_compat.h']\n", root)
        result = relocation.apply_plan(relocation.make_plan(self.layout), self.layout)
        self.assertEqual(result["applied_count"], 1)
        self.assertIn((root / "csrc/common/include/cann_compat.h").as_posix(), path.read_text())

    def test_pth_egg_link_and_direct_url_are_repaired(self):
        for name in ("package.pth", "package.egg-link", "vllm.dist-info/direct_url.json"):
            self.write(f"lib/python3.12/site-packages/{name}", f'"file://{relocation.OLD_SRC}/vllm"\n')
        self.assertEqual(relocation.make_plan(self.layout)["change_count"], 3)

    def test_unknown_source_prefix_blocks_entire_apply(self):
        good = self.write("bin/vllm", f"#!{relocation.OLD_ENV}/bin/python\n")
        self.write("Ascend/config.json", f'"{relocation.OLD_SRC}/unapproved-project/kernel"')
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "blocked")
        before = good.read_bytes()
        with self.assertRaises(ValueError):
            relocation.apply_plan(plan, self.layout)
        self.assertEqual(good.read_bytes(), before)

    def test_boundary_does_not_match_a_different_source_name(self):
        original = f"{relocation.OLD_SRC}/vllm-extra/file.py"
        self.assertEqual(relocation.rewrite_text(original, self.layout), original)

    def test_binary_is_not_rewritten(self):
        path = self.layout.env / "native.so"
        data = b"\x7fELF\x00" + relocation.OLD_ENV.encode()
        path.write_bytes(data)
        self.assertEqual(relocation.make_plan(self.layout)["change_count"], 0)
        self.assertEqual(path.read_bytes(), data)
        entry = self.layout.env / "bin/native"
        entry.parent.mkdir()
        entry.write_bytes(data)
        plan = relocation.make_plan(self.layout)
        self.assertFalse(plan["blockers"])
        self.assertIn(str(entry), plan["skipped_native_elf"])
        self.assertEqual(entry.read_bytes(), data)

    def test_large_native_elf_skipped_before_text_size_limit(self):
        entry = self.layout.env / "bin/large_native"
        entry.parent.mkdir()
        with entry.open("wb") as stream:
            stream.write(b"\x7fELF\x00" + relocation.OLD_ENV.encode())
            stream.truncate(relocation.MAX_TEXT_BYTES + 1024)
        before = entry.stat().st_size
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["skipped_native_elf"], [str(entry)])
        self.assertFalse(plan["skipped_oversize"])
        self.assertEqual(plan["change_count"], 0)
        self.assertEqual(entry.stat().st_size, before)

    def test_large_text_still_fails_closed(self):
        entry = self.layout.env / "bin/large_script"
        entry.parent.mkdir()
        with entry.open("wb") as stream:
            stream.write(f"#!{relocation.OLD_ENV}/bin/python\n".encode())
            stream.write(b"#" * relocation.MAX_TEXT_BYTES)
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "blocked")
        self.assertEqual(plan["skipped_oversize"], [str(entry)])

    def test_streaming_prefix_match_across_small_chunk_boundaries(self):
        raw = b"12345" + relocation.OLD_ENV.encode() + b"--" + relocation.OLD_SRC.encode() + b"tail"
        path = self.layout.env / "large.json"
        path.write_bytes(raw)
        audit = relocation.scan_large_json(path, chunk_size=7)
        self.assertEqual(audit["markers"], sorted(relocation.LEGACY_MARKERS))
        self.assertEqual(audit["size_bytes"], len(raw))
        self.assertEqual(audit["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(path.read_bytes(), raw)

    def test_large_clean_json_is_fully_hashed_and_never_written(self):
        raw = json.dumps({"data": "x" * 1000}).encode()
        path = self.layout.env / "table.json"
        path.write_bytes(raw)
        with mock.patch.object(relocation, "MAX_TEXT_BYTES", 64), \
                mock.patch.object(relocation, "STREAM_CHUNK_BYTES", 13):
            plan = relocation.make_plan(self.layout)
            self.assertEqual(plan["status"], "ready")
            self.assertEqual(plan["change_count"], 0)
            audit = plan["streamed_large_json"][0]
            self.assertEqual(audit["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(audit["size_bytes"], len(raw))
            result = relocation.apply_plan(plan, self.layout)
        self.assertEqual(result["status"], "relocated")
        self.assertEqual(path.read_bytes(), raw)

    def test_large_json_with_legacy_prefix_is_blocked_not_rewritten(self):
        path = self.write("table.json", json.dumps({"path": relocation.OLD_SRC + "/vllm", "data": "x" * 1000}))
        before = path.read_bytes()
        with mock.patch.object(relocation, "MAX_TEXT_BYTES", 64):
            plan = relocation.make_plan(self.layout)
            self.assertEqual(plan["status"], "blocked")
            self.assertEqual(plan["streamed_large_json"][0]["markers"], [relocation.OLD_SRC])
            with self.assertRaises(ValueError):
                relocation.apply_plan(plan, self.layout)
        self.assertEqual(path.read_bytes(), before)

    def test_preserved_large_json_changed_before_apply_blocks_other_writes(self):
        path = self.write("table.json", json.dumps({"data": "x" * 1000}))
        entry = self.write("bin/entry", relocation.OLD_ENV)
        with mock.patch.object(relocation, "MAX_TEXT_BYTES", 64):
            plan = relocation.make_plan(self.layout)
            before = entry.read_bytes()
            path.write_text(json.dumps({"data": relocation.OLD_ENV + "x" * 1000}))
            with self.assertRaises(ValueError):
                relocation.apply_plan(plan, self.layout)
        self.assertEqual(entry.read_bytes(), before)

    def test_noneditable_local_wheel_provenance_is_preserved(self):
        data = {"url": "file://" + relocation.OLD_SRC + "/MindIE-SD/dist/mindiesd.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}
        path = self.write("lib/python3.12/site-packages/mindiesd.dist-info/direct_url.json", json.dumps(data))
        before = path.read_bytes()
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["change_count"], 0)
        self.assertEqual(len(plan["preserved_archive_provenance"]), 1)
        result = relocation.apply_plan(plan, self.layout)
        self.assertEqual(result["status"], "relocated")
        self.assertEqual(path.read_bytes(), before)

    def test_editable_direct_url_is_not_treated_as_provenance(self):
        data = {"url": "file://" + relocation.OLD_SRC + "/vllm", "dir_info": {"editable": True}}
        path = self.write("lib/python3.12/site-packages/vllm.dist-info/direct_url.json", json.dumps(data))
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["change_count"], 1)
        self.assertFalse(plan["preserved_archive_provenance"])
        relocation.apply_plan(plan, self.layout)
        self.assertNotIn(relocation.OLD_SRC, path.read_text())

    def test_archive_info_cannot_hide_unmapped_editable_binding(self):
        data = {"url": "file://" + relocation.OLD_SRC + "/unknown/project.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}, "dir_info": {"editable": True}}
        self.write("lib/python3.12/site-packages/pkg.dist-info/direct_url.json", json.dumps(data))
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "blocked")
        self.assertFalse(plan["preserved_archive_provenance"])

    def test_provenance_changed_to_editable_before_apply_blocks_other_writes(self):
        data = {"url": "file://" + relocation.OLD_SRC + "/unknown/pkg.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}
        path = self.write("lib/python3.12/site-packages/pkg.dist-info/direct_url.json", json.dumps(data))
        entry = self.write("bin/entry", relocation.OLD_ENV)
        plan = relocation.make_plan(self.layout)
        before = entry.read_bytes()
        data["dir_info"] = {"editable": True}
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            relocation.apply_plan(plan, self.layout)
        self.assertEqual(entry.read_bytes(), before)

    def test_non_elf_binary_like_text_with_legacy_prefix_blocks(self):
        entry = self.layout.env / "bin/non_elf_binary"
        entry.parent.mkdir()
        data = b"custom\x00" + relocation.OLD_ENV.encode()
        entry.write_bytes(data)
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "blocked")
        self.assertFalse(plan["skipped_native_elf"])
        self.assertIn("binary-like", plan["blockers"][0]["reason"])
        self.assertEqual(entry.read_bytes(), data)

    def test_target_outside_whitelist_is_rejected(self):
        outside = self.base / "outside.py"
        outside.write_text("unchanged", encoding="utf-8")
        with self.assertRaises(ValueError):
            relocation.authorized_path(outside, self.layout)
        with self.assertRaises(ValueError):
            relocation.authorized_path(self.layout.src / "vllm-extra/file.py", self.layout)
        with self.assertRaises(ValueError):
            relocation.authorized_path(self.layout.env / "../outside.py", self.layout)

    def test_changed_file_fails_before_any_write(self):
        first = self.write("bin/first", f"#!{relocation.OLD_ENV}/bin/python\n")
        second = self.write("bin/second", f"#!{relocation.OLD_ENV}/bin/python\n")
        plan = relocation.make_plan(self.layout)
        original = first.read_bytes()
        second.write_text("changed independently", encoding="utf-8")
        with self.assertRaises(ValueError):
            relocation.apply_plan(plan, self.layout)
        self.assertEqual(first.read_bytes(), original)

    def test_symlink_rewrite_changes_link_not_referent(self):
        target = self.layout.env / "Ascend/cann-9.0.1"
        target.mkdir(parents=True)
        link = self.layout.env / "Ascend/cann"
        self.symlink(f"{relocation.OLD_ENV}/Ascend/cann-9.0.1", link, directory=True)
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["change_count"], 1)
        relocation.apply_plan(plan, self.layout)
        self.assertEqual(Path(os.readlink(link)), target)
        self.assertEqual(relocation.make_plan(self.layout)["change_count"], 0)

    def test_external_symlink_is_not_followed(self):
        outside = self.base / "external"
        outside.mkdir()
        file = self.write("secret.py", f"path = '{relocation.OLD_ENV}'", outside)
        link = self.layout.env / "external"
        self.symlink(outside, link, directory=True)
        with self.assertRaises(ValueError):
            relocation.authorized_path(link / "secret.py", self.layout)
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["change_count"], 0)
        self.assertEqual(len(plan["external_links_preserved"]), 1)
        self.assertIn(relocation.OLD_ENV, file.read_text())

    def test_external_leaf_symlink_is_rejected_for_text_write(self):
        target = self.write("external.py", "unchanged", self.base)
        link = self.layout.env / "link.py"
        self.symlink(target, link)
        with self.assertRaises(ValueError):
            relocation.authorized_path(link, self.layout)

    def test_symlink_guard_also_covered_without_os_symlink_privilege(self):
        target = self.write("file.py", "unchanged")
        original = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda path: path == target or original(path)):
            with self.assertRaises(ValueError):
                relocation.authorized_path(target, self.layout)

    def test_symlink_parent_guard_without_os_symlink_privilege(self):
        target = self.write("nested/file.py", "unchanged")
        original = Path.is_symlink
        with mock.patch.object(Path, "is_symlink", lambda path: path == target.parent or original(path)):
            with self.assertRaises(ValueError):
                relocation.authorized_path(target, self.layout)

    def test_unknown_binary_and_provenance_are_not_silently_claimed_relocated(self):
        provenance = self.write("conda-meta/history.json", f'"{relocation.OLD_ENV}"')
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["change_count"], 0)
        self.assertTrue(any("provenance" in item for item in plan["limitations"]))
        self.assertIn(relocation.OLD_ENV, provenance.read_text())

    def test_hardlinked_text_is_blocked_without_changing_other_name(self):
        target = self.write("bin/entry", f"#!{relocation.OLD_ENV}/bin/python\n")
        outside = self.base / "other-hardlink"
        os.link(target, outside)
        plan = relocation.make_plan(self.layout)
        self.assertEqual(plan["status"], "blocked")
        with self.assertRaises(ValueError):
            relocation.apply_plan(plan, self.layout)
        self.assertIn(relocation.OLD_ENV, outside.read_text())

    def test_production_cli_has_no_root_override(self):
        with self.assertRaises(SystemExit) as captured:
            relocation.main(["--root", str(self.base)])
        self.assertEqual(captured.exception.code, 2)
        self.assertEqual(relocation.production_layout().env, Path("/cache/zhonghao/h3/env"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
