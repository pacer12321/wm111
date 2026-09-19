"""Torch CPU equivalence gates. Explicitly skipped if torch is unavailable."""
import importlib.util
import ast
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streaming_shards import LoRA, RangeReader, TensorPlan, TensorRef, fp32_to_bf16
from test_streaming_shards import write_fixture

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch_streaming_shards import build_torch_block_shard


def original_production_merge():
    """Compile only the exact audited CPU functions, not the serving module."""
    source_path = Path(os.environ.get("TRACK_B_PRODUCTION_MERGE_SOURCE",
        str(Path(__file__).parent.parent / "dualstream_v1/candidate/vllm-omni/vllm_omni/diffusion/models/minimax_h3/openvdn_checkpoint.py")))
    raw = source_path.read_bytes()
    expected = "fbe011bae524bea16f54a14032e61e82f2c68aa4da6d426b98a13e097fb8f19f"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise AssertionError("Production merge source changed; re-audit before equivalence test")
    tree = ast.parse(raw.decode("utf-8"))
    wanted = {"_check_cpu_tensor", "_assert_finite", "merge_lora_pair_"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if len(nodes) != 3:
        raise AssertionError("Missing production CPU helper functions")
    isolated = ast.Module(body=nodes, type_ignores=[])
    namespace = {"torch": torch, "STAGE_B_SCALE": 1.0}
    exec(compile(ast.fix_missing_locations(isolated), str(source_path), "exec"), namespace)
    return namespace["merge_lora_pair_"]


@unittest.skipUnless(HAS_TORCH, "torch unavailable: production CPU equivalence NOT executed")
class TorchTests(unittest.TestCase):
    def test_original_256_row_merge_grid_and_midrow_shards(self):
        generator = np.random.default_rng(4101)
        heads, dim, cols, rank = 3, 128, 13, 64
        rows = heads * dim
        qkv = fp32_to_bf16(generator.normal(size=(rows * 3, cols)).astype("f4") / 10)
        a = fp32_to_bf16(generator.normal(size=(rank, cols)).astype("f4") / 10)
        b = fp32_to_bf16(generator.normal(size=(rows, rank)).astype("f4") / 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.safetensors"
            write_fixture(path, dict(qkv=("BF16", qkv), a=("BF16", a), b=("BF16", b)))
            reader = RangeReader(path)
            ref = lambda key: TensorRef(reader, key)
            plan = TensorPlan("qkv", ref("qkv"), heads, dim,
                tuple(LoRA(i * rows, (i + 1) * rows, ref("a"), ref("b")) for i in range(3)))
            original = torch.from_numpy(qkv).view(torch.bfloat16).reshape(heads, 3 * dim, cols)
            expected = torch.cat([original[:, i * dim:(i + 1) * dim].reshape(rows, cols) for i in range(3)]).clone()
            a_bf16 = torch.from_numpy(a).view(torch.bfloat16)
            b_bf16 = torch.from_numpy(b).view(torch.bfloat16)
            production_merge = original_production_merge()
            for projection in range(3):
                target = expected[projection * rows:(projection + 1) * rows]
                production_merge(target, a_bf16, b_bf16, name=f"synthetic_qkv[{projection}]")
            for world in (2, 7):
                shards = [build_torch_block_shard([plan], world, r, chunk_elements=19)[0]["BF16"] for r in range(world)]
                actual = torch.cat(shards)[:expected.numel()].reshape_as(expected)
                self.assertTrue(torch.equal(expected, actual), "Bitwise CPU weight mismatch")

    def test_f32_preservation_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f32.safetensors"
            data = np.array([.125, -4, 5.5], dtype="<f4")
            write_fixture(path, dict(x=("F32", data)))
            ref = TensorRef(RangeReader(path), "x")
            shards = [build_torch_block_shard([TensorPlan("x", ref)], 2, r)[0]["F32"] for r in range(2)]
            self.assertTrue(torch.equal(torch.cat(shards)[:3], torch.from_numpy(data)))
            self.assertEqual(shards[1][-1].item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
