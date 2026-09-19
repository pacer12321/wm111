"""Header-only tests against the collected real checkpoint metadata."""
import copy
import json
import unittest
from pathlib import Path

from manifest_builder import audit, attach_model_metadata


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads(Path(__file__).with_name("real_header_snapshot.json").read_text(encoding="utf-8"))

    def test_real_coverage_and_not_runtime_ready(self):
        manifest = audit(self.snapshot)
        self.assertEqual((manifest["base_count"], manifest["branch_count"], manifest["lora_tensor_count"], manifest["lora_pair_count"]), (535, 800, 416, 208))
        self.assertEqual(len(manifest["plans"]), 1335)
        self.assertFalse(manifest["runtime_usable"])
        self.assertTrue(all(p["model_iteration_index"] is None for p in manifest["plans"]))

    def test_real_non_square_projection_shapes(self):
        plans = {p["model_name"]: p for p in audit(self.snapshot)["plans"]}
        qkv, out = plans["blocks.0.attn.qkv_proj.weight"], plans["blocks.0.attn.out_proj.weight"]
        self.assertEqual(qkv["shape"], [21504, 5376])
        self.assertEqual(out["shape"], [5376, 7168])
        self.assertEqual([(x["start_row"], x["end_row"]) for x in qkv["lora_pairs"]], [(0, 7168), (7168, 14336), (14336, 21504)])
        self.assertEqual(out["lora_pairs"][0]["a"]["shape"], [64, 7168])
        self.assertEqual(out["lora_pairs"][0]["b"]["shape"], [5376, 64])

    def test_main_block_dtype_count_and_remaining_items(self):
        manifest = audit(self.snapshot)
        for index in range(50):
            block = [p for p in manifest["plans"] if p["offload_unit"] == f"blocks.{index}"]
            self.assertEqual(len(block), 26)
            self.assertEqual({p["dtype"] for p in block}, {"BF16"})
            self.assertEqual(sum(p["numel"] * 2 for p in block), 1376730080)
        nonblock = [p for p in manifest["plans"] if p["offload_unit"] == "non_main_block_requires_explicit_policy"]
        self.assertEqual(len(nonblock), 35)
        self.assertEqual(sum(p["dtype"] == "F32" for p in nonblock), 13)

    def test_duplicate_file_keys_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["files"].append(snapshot["files"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit(snapshot)

    def test_missing_lora_file_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["files"] = [f for f in snapshot["files"] if f["group"] != "lora"]
        with self.assertRaisesRegex(ValueError, "Missing LoRA"):
            audit(snapshot)

    def test_missing_branch_file_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["files"] = [f for f in snapshot["files"] if f["group"] != "branch"]
        with self.assertRaisesRegex(ValueError, "Stage-B branch"):
            audit(snapshot)

    def test_truncated_file_extent_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["files"][0]["file_size"] -= 1
        with self.assertRaisesRegex(ValueError, "Bad file size"):
            audit(snapshot)

    def test_metadata_order_is_explicit_and_shape_checked(self):
        manifest = audit(self.snapshot)
        # Synthetic enumeration tests the validator only. This is not saved
        # as a real model order and must not be described as meta validation.
        entries = [dict(name=p["model_name"], shape=p["shape"], dtype=p["dtype"]) for p in reversed(manifest["plans"])]
        metadata = dict(device="meta", dit_tp=1, parameters_and_persistent_buffers=entries)
        result = attach_model_metadata(manifest, metadata)
        self.assertEqual(result["plans"][0]["model_iteration_index"], 1334)
        self.assertFalse(result["runtime_usable"])
        metadata["parameters_and_persistent_buffers"][0]["shape"] = [1]
        with self.assertRaisesRegex(ValueError, "Model/checkpoint mismatch"):
            attach_model_metadata(manifest, metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
