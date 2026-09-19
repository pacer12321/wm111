"""CPU-only filesystem toy tests; never invoke remote host, env, model or NPU."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import deploy_candidate_31731 as d


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Toy(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.paths = d.Paths(root, root / "candidates/b_v1/vllm-omni", root / "c_adaptation/candidate",
                             root / "candidates/c_v1")
        (self.paths.source_b / d.MODEL_REL).mkdir(parents=True)
        self.paths.source_c.mkdir(parents=True)
        self.b_pins, self.c_pins = {}, {}
        for name in d.B_PINS:
            path = self.paths.source_b / d.MODEL_REL / name
            path.write_text("B original " + name)
            self.b_pins[name] = sha(path)
        for name in d.C_PINS:
            path = self.paths.source_c / name
            path.write_text(("B original " if name in ("openvdn_npu.py", "openvdn_checkpoint.py") else "C candidate ") + name)
            self.c_pins[name] = sha(path)
        (self.paths.source_b / "README.md").write_text("ordinary source documentation")
        (self.paths.source_b / "empty").mkdir()
        self.host = dict(hostname="toy", machine="test", boot_id="fixture")

    def host_check(self, previous=None):
        if previous is not None and previous != self.host:
            raise RuntimeError("host changed")
        return self.host.copy()

    def inspect(self, **kwargs):
        return d.inspect(self.paths, b_pins=self.b_pins, c_pins=self.c_pins, **kwargs)

    def deploy(self, **kwargs):
        return d.deploy(self.paths, self.host_check, b_pins=self.b_pins, c_pins=self.c_pins, **kwargs)

    def failed_manifest(self, published=False):
        manifest = json.loads((self.paths.destination_root / "deploy_manifest.json").read_text())
        self.assertEqual(manifest["state"], "failed")
        self.assertEqual(manifest["published"], published)
        self.assertFalse(manifest["source_copy_before_after_unchanged"])
        return manifest

    def symlink(self, target, link, isdir=False):
        try:
            link.symlink_to(target, target_is_directory=isdir)
        except OSError as exc:
            self.skipTest(f"Native symlink unavailable: {exc}")

    def test_inspect_is_readonly(self):
        before = d.inventory(self.paths.source_b)
        proof = self.inspect()
        self.assertEqual(proof["b"], before)
        self.assertFalse(self.paths.destination_root.exists())

    def test_success_exact_overlay_copy_and_no_source_change(self):
        b_before, c_before = d.inventory(self.paths.source_b), d.overlay_inventory(self.paths.source_c, self.c_pins)
        manifest = self.deploy()
        self.assertEqual(manifest["state"], "completed")
        self.assertTrue(manifest["verified_all_destination_size_sha256"])
        self.assertTrue(manifest["source_copy_before_after_unchanged"])
        self.assertEqual(d.inventory(self.paths.source_b), b_before)
        self.assertEqual(d.overlay_inventory(self.paths.source_c, self.c_pins), c_before)
        self.assertFalse(self.paths.staging_vendor.exists())
        for name, digest in self.c_pins.items():
            dest = self.paths.destination_vendor / d.MODEL_REL / name
            self.assertEqual(sha(dest), digest)
            source = self.paths.source_c / name
            self.assertNotEqual((source.stat().st_dev, source.stat().st_ino), (dest.stat().st_dev, dest.stat().st_ino))
        self.assertTrue((self.paths.destination_vendor / "empty").is_dir())
        self.assertFalse(manifest["npu_invoked"])

    def test_preexisting_destination_never_overwritten(self):
        self.paths.destination_root.mkdir()
        sentinel = self.paths.destination_root / "user.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError): self.deploy()
        self.assertEqual(sentinel.read_text(), "preserve")
        self.assertEqual(sorted(p.name for p in self.paths.destination_root.iterdir()), ["user.txt"])

    def test_completed_deployment_cannot_resume(self):
        self.deploy()
        before = (self.paths.destination_root / "deploy_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError): self.deploy()
        self.assertEqual((self.paths.destination_root / "deploy_manifest.json").read_bytes(), before)

    def test_b_hash_mismatch_stops_before_mkdir(self):
        (self.paths.source_b / d.MODEL_REL / "pipeline_minimax_h3.py").write_text("unexpected edit")
        with self.assertRaisesRegex(RuntimeError, "Pinned B"): self.deploy()
        self.assertFalse(self.paths.destination_root.exists())

    def test_c_hash_mismatch_stops_before_mkdir(self):
        (self.paths.source_c / "strict_source_layout.py").write_text("unexpected edit")
        with self.assertRaisesRegex(RuntimeError, "Pinned C"): self.deploy()
        self.assertFalse(self.paths.destination_root.exists())

    def test_rejects_missing_b_or_c_file(self):
        (self.paths.source_c / "strict_source_layout.py").unlink()
        with self.assertRaises(FileNotFoundError): self.inspect()
        self.assertFalse(self.paths.destination_root.exists())

    def test_ignores_git_pycache_pyc_without_copy(self):
        for name in (".git", "__pycache__"):
            folder = self.paths.source_b / name
            folder.mkdir()
            (folder / "not_source.bin").write_bytes(b"ignored")
        (self.paths.source_b / "module.pyc").write_bytes(b"ignored")
        manifest = self.deploy()
        self.assertEqual(len(manifest["source_b_before"]["excluded"]), 3)
        for name in (".git", "__pycache__", "module.pyc"):
            self.assertFalse((self.paths.destination_vendor / name).exists())

    def test_symlink_even_ignored_name_is_rejected(self):
        self.symlink(self.paths.source_c, self.paths.source_b / "__pycache__", True)
        with self.assertRaisesRegex(RuntimeError, "Symlink"): self.inspect()

    def test_symlink_regular_file_is_rejected(self):
        self.symlink(self.paths.source_c / "strict_source_layout.py", self.paths.source_b / "link.py")
        with self.assertRaisesRegex(RuntimeError, "Symlink"): self.inspect()

    def test_symlink_overlay_is_rejected(self):
        path = self.paths.source_c / "strict_source_layout.py"
        path.unlink()
        self.symlink(self.paths.source_b / "README.md", path)
        with self.assertRaisesRegex(RuntimeError, "Symlink"): self.inspect()

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO filesystem test requires POSIX")
    def test_fifo_is_rejected(self):
        os.mkfifo(self.paths.source_b / "pipe")
        with self.assertRaisesRegex(RuntimeError, "special"): self.inspect()

    def test_single_file_cap_rejected(self):
        (self.paths.source_b / "large.py").write_bytes(b"x" * 101)
        with self.assertRaisesRegex(RuntimeError, "exceeds"): self.inspect(max_file=100)

    def test_cumulative_source_cap_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "cumulative"): self.inspect(max_tree=100)

    def test_combined_overlay_size_cap_rejected(self):
        before = d.inventory(self.paths.source_b)["total_bytes"]
        (self.paths.source_c / "strict_source_layout.py").write_bytes(b"x" * 1000)
        self.c_pins["strict_source_layout.py"] = sha(self.paths.source_c / "strict_source_layout.py")
        with self.assertRaisesRegex(RuntimeError, "Combined"): self.inspect(max_tree=before + 200)

    def test_outside_private_root_is_rejected(self):
        bad = d.Paths(self.paths.private_root, self.paths.source_b, self.paths.source_c, self.paths.private_root.parent / "outside")
        with self.assertRaises(RuntimeError): d.inspect(bad, b_pins=self.b_pins, c_pins=self.c_pins)

    def test_overlapping_source_destination_is_rejected(self):
        bad = d.Paths(self.paths.private_root, self.paths.source_b, self.paths.source_c, self.paths.source_b / "new")
        with self.assertRaisesRegex(RuntimeError, "disjoint"): d.inspect(bad, b_pins=self.b_pins, c_pins=self.c_pins)

    def test_b_file_changed_during_copy_is_rejected_and_retained(self):
        called = False
        def mutate(src, dst, record, **kwargs):
            nonlocal called
            if not called:
                called = True
                src.write_bytes(b"changed by another actor")
            d.copy_new(src, dst, record, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "changed"): self.deploy(copy_fn=mutate)
        self.failed_manifest()
        self.assertFalse(self.paths.destination_vendor.exists())
        self.assertTrue(self.paths.staging_vendor.is_dir())

    def test_source_overlay_changed_after_copy_is_rejected(self):
        def mutate(src, dst, record, **kwargs):
            d.copy_new(src, dst, record, **kwargs)
            if src == self.paths.source_c / "strict_source_attention.py":
                src.write_bytes(b"changed by another actor after copy")
        with self.assertRaisesRegex(RuntimeError, "Pinned C"): self.deploy(copy_fn=mutate)
        self.failed_manifest()
        self.assertFalse(self.paths.destination_vendor.exists())

    def test_unexpected_staging_file_blocks_publish(self):
        done = False
        def extra(src, dst, record, **kwargs):
            nonlocal done
            d.copy_new(src, dst, record, **kwargs)
            if not done:
                done = True
                (self.paths.staging_vendor / "extra.txt").write_text("unexpected")
        with self.assertRaisesRegex(RuntimeError, "Destination contains"): self.deploy(copy_fn=extra)
        self.failed_manifest()
        self.assertTrue((self.paths.staging_vendor / "extra.txt").exists())

    def test_hardlinked_staging_output_is_rejected(self):
        linked = False
        def external_link(src, dst, record, **kwargs):
            nonlocal linked
            d.copy_new(src, dst, record, **kwargs)
            if not linked:
                linked = True
                os.link(dst, self.paths.private_root / "external_hardlink")
        with self.assertRaisesRegex(RuntimeError, "hardlinked"):
            self.deploy(copy_fn=external_link)
        self.failed_manifest()
        self.assertFalse(self.paths.destination_vendor.exists())

    def test_failed_deployment_blocks_retry_without_deleting(self):
        def fail(*args, **kwargs): raise RuntimeError("synthetic copy failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic"): self.deploy(copy_fn=fail)
        before = (self.paths.destination_root / "deploy_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError): self.deploy()
        self.assertEqual((self.paths.destination_root / "deploy_manifest.json").read_bytes(), before)

    def test_no_hardlink_or_broad_delete_calls(self):
        tree = ast.parse(Path(d.__file__).read_text())
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse({"link", "unlink", "rmtree", "remove", "system", "Popen"} & called)

    def test_empty_destination_race_does_not_overwrite(self):
        source = self.paths.private_root / "stage_dir"
        destination = self.paths.private_root / "raced_dir"
        source.mkdir(); destination.mkdir()
        with self.assertRaises(OSError): d.rename_new_dir(source, destination)
        self.assertTrue(source.is_dir())
        self.assertTrue(destination.is_dir())

    def test_host_change_stops_before_publication(self):
        count = 0
        def check(previous=None):
            nonlocal count
            count += 1
            if count == 3: raise RuntimeError("host changed")
            return self.host.copy()
        with self.assertRaisesRegex(RuntimeError, "host changed"):
            d.deploy(self.paths, check, b_pins=self.b_pins, c_pins=self.c_pins)
        self.failed_manifest()
        self.assertFalse(self.paths.destination_vendor.exists())

    def test_late_source_change_marks_published_vendor_failed(self):
        rename = d.rename_new_dir
        def late(source, destination):
            rename(source, destination)
            (self.paths.source_b / "README.md").write_text("external late edit")
        with mock.patch.object(d, "rename_new_dir", side_effect=late), self.assertRaisesRegex(RuntimeError, "Source changed"):
            self.deploy()
        self.failed_manifest(published=True)
        self.assertTrue(self.paths.destination_vendor.is_dir())


class FixedPolicy(unittest.TestCase):
    def test_exact_paths_caps_and_no_cli_overrides(self):
        paths = d.Paths()
        self.assertEqual(paths.destination_vendor.as_posix(), "/cache/zhonghao/h3/candidates/c_v1/vllm-omni")
        self.assertEqual((d.MAX_FILE, d.MAX_TREE), (20 * 1024**2, 128 * 1024**2))
        with self.assertRaises(SystemExit): d.main(["--source", "/somewhere"])

    def test_pins_match_current_candidate_and_b_shared_files(self):
        root = Path(__file__).resolve().parent
        for name, expected in d.C_PINS.items():
            self.assertEqual(sha(root / "candidate" / name), expected)
        for name in ("openvdn_npu.py", "openvdn_checkpoint.py"):
            self.assertEqual(d.B_PINS[name], d.C_PINS[name])

    def test_imports_no_torch_or_npu_library(self):
        tree = ast.parse(Path(d.__file__).read_text())
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse(any(m and (m.startswith("torch") or m.startswith("vllm")) for m in modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
