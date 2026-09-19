"""Pure CPU helpers; no torch imports or production installation required."""
import pathlib
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

import validate_h3_runtime as validate


class ValidateTests(unittest.TestCase):
    def test_exact_source_versions(self):
        self.assertEqual(validate.EXPECTED_VERSIONS["torch"], "2.10.0")
        self.assertEqual(validate.EXPECTED_VERSIONS["torch-npu"], "2.10.0.post2")
        self.assertEqual(validate.EXPECTED_VERSIONS["vllm-omni"], "0.26.0+npu")

    def test_old_active_prefix_detected_but_comment_ignored(self):
        old = validate.OLD_PREFIXES[0]
        self.assertTrue(validate.runtime_text_has_old_prefix("#!" + old + "/bin/python"))
        self.assertTrue(validate.runtime_text_has_old_prefix(old + "/lib/python3.12"))
        self.assertTrue(validate.has_old_prefix(old + ":/usr/bin"))
        self.assertFalse(validate.runtime_text_has_old_prefix("# migrated from " + old + "/bin/python"))
        self.assertFalse(validate.has_old_prefix(old + "-other/path"))

    def test_missing_library_rejected(self):
        with self.assertRaises(RuntimeError):
            validate.parse_ldd("libtorch_npu.so => not found")

    def test_old_source_library_rejected(self):
        with self.assertRaises(RuntimeError):
            validate.parse_ldd("libx.so => " + validate.OLD_PREFIXES[0] + "/lib/libx.so (0x1)")

    def test_resolved_dependency_parser(self):
        with patch.object(validate, "allowed_library_path", return_value=True):
            self.assertEqual(validate.parse_ldd("libx.so => /private/libx.so (0x1)"),
                             [{"library": "libx.so", "path": "/private/libx.so"}])

    def test_foreign_library_rejected(self):
        with patch.object(validate, "allowed_library_path", return_value=False), self.assertRaises(RuntimeError):
            validate.parse_ldd("libx.so => /foreign/libx.so (0x1)")

    def test_system_cann_library_rejected_even_when_system_path_allowed(self):
        with patch.object(validate, "allowed_library_path", return_value=True), \
                patch.object(validate, "inside", return_value=False), self.assertRaises(RuntimeError):
            validate.parse_ldd("libascendcl.so => /usr/lib/libascendcl.so (0x1)")

    def test_device_and_network_guards(self):
        for function in (validate.forbid_device_init, validate.forbid_network):
            with self.assertRaises(RuntimeError):
                function()

    def test_elf_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tiny.so"
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = struct.pack("<H", 183)
            path.write_bytes(header)
            self.assertEqual(validate.elf_machine(path), 183)
            header[18:20] = struct.pack("<H", 62)
            path.write_bytes(header)
            with self.assertRaises(RuntimeError):
                validate.elf_machine(path)

    def test_module_origin_private_only(self):
        module = types.SimpleNamespace(__name__="example", __file__="/foreign/example.py")
        with patch.object(validate, "inside", return_value=False), self.assertRaises(RuntimeError):
            validate.module_origin(module)

    def test_active_env_source_prefix_rejected(self):
        values = {"H3_INSTALL": str(validate.ROOT), "H3_ENV": str(validate.ENV),
                  "ASCEND_HOME_PATH": str(validate.ENV / "Ascend/cann-9.0.1"),
                  "ASCEND_TOOLKIT_HOME": str(validate.ENV / "Ascend/cann-9.0.1"),
                  "LD_LIBRARY_PATH": validate.OLD_PREFIXES[0] + "/lib"}
        with self.assertRaises(RuntimeError):
            validate.check_environment(values)


if __name__ == "__main__":
    unittest.main()
