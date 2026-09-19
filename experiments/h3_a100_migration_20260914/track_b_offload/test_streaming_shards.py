"""CPU synthetic tests, not a real-weight/GPU equivalence claim."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from streaming_shards import (
    LoRA, RangeReader, TensorPlan, TensorRef, bf16_to_fp32,
    build_block_shard, fp32_to_bf16,
)


def write_fixture(path, tensors):
    header, payload = {}, bytearray()
    for key, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[key] = dict(dtype=dtype, shape=list(array.shape), data_offsets=[len(payload), len(payload) + len(raw)])
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "tiny.safetensors"

    def tearDown(self):
        self.temp.cleanup()

    def ref(self, tensors):
        write_fixture(self.path, tensors)
        reader = RangeReader(self.path)
        return reader, lambda key: TensorRef(reader, key)

    def test_identity_mixed_dtype_padding_and_metadata(self):
        x = fp32_to_bf16(np.arange(15, dtype=np.float32).reshape(3, 5))
        y = fp32_to_bf16(np.arange(4, dtype=np.float32))
        z = np.arange(3, dtype="<f4")
        reader, ref = self.ref(dict(x=("BF16", x), z=("F32", z), y=("BF16", y)))
        plans = [TensorPlan(k, ref(k)) for k in ("x", "z", "y")]
        all_shards = [build_block_shard(plans, 2, rank, chunk_elements=3) for rank in range(2)]
        self.assertEqual(all_shards[0][1], all_shards[1][1])
        np.testing.assert_array_equal(np.concatenate([s[0]["BF16"] for s in all_shards])[:19], np.concatenate((x.ravel(), y)))
        np.testing.assert_array_equal(np.concatenate([s[0]["F32"] for s in all_shards])[:3], z)
        self.assertEqual(all_shards[1][0]["BF16"][-1], 0)
        self.assertLessEqual(max(hi - lo for _, lo, hi in reader.payload_reads), 3)

    def test_rank_only_reads_own_source_intersection(self):
        x = fp32_to_bf16(np.arange(100, dtype=np.float32))
        reader, ref = self.ref(dict(x=("BF16", x)))
        build_block_shard([TensorPlan("x", ref("x"))], 2, 1, chunk_elements=7)
        self.assertEqual(sum(hi - lo for _, lo, hi in reader.payload_reads), 50)
        self.assertTrue(all(lo >= 50 for _, lo, hi in reader.payload_reads))

    def test_qkv_reorder_all_ranks_exact(self):
        heads, dim, width = 3, 2, 5
        x = fp32_to_bf16(np.arange(3 * heads * dim * width, dtype=np.float32).reshape(3 * heads * dim, width))
        reader, ref = self.ref(dict(qkv=("BF16", x)))
        plan = TensorPlan("qkv", ref("qkv"), heads, dim)
        # Independent implementation follows original reshape/split/cat.
        grouped = x.reshape(heads, 3 * dim, width)
        expected = np.concatenate([grouped[:, i * dim:(i + 1) * dim, :].reshape(heads * dim, width) for i in range(3)])
        for world in (1, 2, 3, 7):
            shards = [build_block_shard([plan], world, rank, chunk_elements=4)[0]["BF16"] for rank in range(world)]
            np.testing.assert_array_equal(np.concatenate(shards)[:x.size], expected.ravel())

    def test_qkv_lora_then_shard_matches_full_reference(self):
        heads, dim, width, rank = 2, 2, 5, 2
        # Binary-exact inputs make reduction order irrelevant for this fixture.
        x = fp32_to_bf16(np.arange(60, dtype=np.float32).reshape(12, 5) / 8)
        a = fp32_to_bf16(np.arange(rank * width, dtype=np.float32).reshape(rank, width) / 16)
        b = fp32_to_bf16(np.arange(4 * rank, dtype=np.float32).reshape(4, rank) / 8)
        reader, ref = self.ref(dict(qkv=("BF16", x), a=("BF16", a), b=("BF16", b)))
        loras = tuple(LoRA(i * 4, (i + 1) * 4, ref("a"), ref("b")) for i in range(3))
        plan = TensorPlan("qkv", ref("qkv"), heads, dim, loras)
        grouped = x.reshape(heads, 3 * dim, width)
        base = np.concatenate([grouped[:, i * dim:(i + 1) * dim, :].reshape(4, width) for i in range(3)])
        delta = fp32_to_bf16(bf16_to_fp32(b) @ bf16_to_fp32(a))
        expected = fp32_to_bf16(bf16_to_fp32(base) + np.tile(bf16_to_fp32(delta), (3, 1)))
        for world in (2, 7):
            shards = [build_block_shard([plan], world, r, chunk_elements=3)[0]["BF16"] for r in range(world)]
            np.testing.assert_array_equal(np.concatenate(shards)[:x.size], expected.ravel())

    def test_branch_and_out_lora_identity_mapping(self):
        x = fp32_to_bf16(np.arange(12, dtype=np.float32).reshape(3, 4) / 8)
        a = fp32_to_bf16(np.ones((2, 4), dtype=np.float32) / 4)
        b = fp32_to_bf16(np.ones((3, 2), dtype=np.float32) / 2)
        _, ref = self.ref(dict(out=("BF16", x), branch=("BF16", x), a=("BF16", a), b=("BF16", b)))
        plans = [TensorPlan("out", ref("out"), loras=(LoRA(0, 3, ref("a"), ref("b")),)), TensorPlan("branch", ref("branch"))]
        shards = [build_block_shard(plans, 2, r, chunk_elements=5)[0]["BF16"] for r in range(2)]
        expected = np.concatenate([fp32_to_bf16(bf16_to_fp32(x) + .25).ravel(), x.ravel()])
        np.testing.assert_array_equal(np.concatenate(shards), expected)

    def test_invalid_group_and_duplicate_name_fail_closed(self):
        _, ref = self.ref(dict(x=("F32", np.zeros(2, dtype="<f4"))))
        plan = TensorPlan("x", ref("x"))
        for world, rank in ((0, 0), (2, 2), (2, -1)):
            with self.assertRaises(ValueError):
                build_block_shard([plan], world, rank)
        with self.assertRaises(ValueError):
            build_block_shard([plan, plan], 2, 0)

    def test_invalid_qkv_and_lora_fail_closed(self):
        _, ref = self.ref(dict(x=("BF16", np.zeros((3, 4), dtype="<u2"))))
        with self.assertRaises(ValueError):
            build_block_shard([TensorPlan("x", ref("x"), 2, 2)], 2, 0)
        with self.assertRaises(ValueError):
            build_block_shard([TensorPlan("x", ref("x"), loras=(LoRA(0, 3, ref("x"), ref("x")),))], 2, 0)

    def test_payload_corruption_rejected(self):
        write_fixture(self.path, dict(x=("F32", np.zeros(2, dtype="<f4"))))
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            RangeReader(self.path)

    def test_nonfinite_rejected(self):
        _, ref = self.ref(dict(x=("F32", np.array([np.nan], dtype="<f4"))))
        with self.assertRaises(ValueError):
            build_block_shard([TensorPlan("x", ref("x"))], 1, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
