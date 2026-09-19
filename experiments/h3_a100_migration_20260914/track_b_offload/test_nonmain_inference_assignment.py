"""Real torch regression for builder inference tensors -> non-main Params.

No vLLM initialization, process group, GPU, or large checkpoint is needed.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch
    from h3_prepared_integration import _assign_loaded_nonmain
    from streaming_shards import RangeReader, TensorPlan, TensorRef, fp32_to_bf16
    from test_streaming_shards import write_fixture
    from torch_streaming_shards import build_torch_block_shard


@unittest.skipUnless(HAS_TORCH, "torch unavailable: actual inference assignment regression NOT executed")
class NonMainAssignmentTests(unittest.TestCase):
    def test_builder_payload_parameter_alias_flags_and_module_to(self):
        for requires_grad in (True, False):
            with self.subTest(requires_grad=requires_grad), tempfile.TemporaryDirectory() as tmp:
                values = np.arange(12, dtype="f4").reshape(3, 4) / 8
                path = Path(tmp) / "tiny.safetensors"
                write_fixture(path, dict(weight=("BF16", fp32_to_bf16(values))))
                plan = TensorPlan("weight", TensorRef(RangeReader(path), "weight"))
                shards, _ = build_torch_block_shard([plan], 1, 0, pin_memory=False)
                value = shards["BF16"].reshape(3, 4)
                self.assertFalse(torch.is_inference(value))
                self.assertFalse(torch.is_inference_mode_enabled())
                module = torch.nn.Module()
                module.register_parameter("weight", torch.nn.Parameter(
                    torch.empty((3, 4), dtype=torch.bfloat16, device="meta"), requires_grad=requires_grad))
                module.weight.marker = "preserve original attributes"
                pointer = value.untyped_storage().data_ptr()
                _assign_loaded_nonmain(module, "weight", value)
                self.assertFalse(torch.is_inference_mode_enabled())
                self.assertEqual(module.weight.requires_grad, requires_grad)
                self.assertEqual(module.weight.marker, "preserve original attributes")
                self.assertEqual(module.weight.untyped_storage().data_ptr(), pointer)
                self.assertTrue(torch.equal(module.weight, torch.from_numpy(values).to(torch.bfloat16)))
                # This is the operation that failed in the real D smoke.
                output = torch.nn.functional.linear(torch.ones(2, 4, dtype=torch.bfloat16), module.weight)
                self.assertEqual(tuple(output.shape), (2, 3))
                # Exercise nn.Module._apply outside inference mode as backend
                # placement does, using a tiny CPU dtype transfer instead of GPU.
                module.to(dtype=torch.float32)
                self.assertEqual(module.weight.requires_grad, requires_grad)
                self.assertTrue(torch.equal(module.weight, torch.from_numpy(values)))

    def test_builder_f32_buffer_alias_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.safetensors"
            values = np.array([.5, 1., 2.], dtype="f4")
            write_fixture(path, dict(scale=("F32", values)))
            plan = TensorPlan("scale", TensorRef(RangeReader(path), "scale"))
            shards, _ = build_torch_block_shard([plan], 1, 0, pin_memory=False)
            value = shards["F32"]
            self.assertFalse(torch.is_inference(value))
            module = torch.nn.Module()
            module.register_buffer("scale", torch.empty(3, device="meta", dtype=torch.float32))
            _assign_loaded_nonmain(module, "scale", value)
            self.assertIs(module.scale, value)
            self.assertTrue(torch.equal(module.scale, torch.from_numpy(values)))
            with self.assertRaises(ValueError):
                _assign_loaded_nonmain(module, "scale", value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
