"""Optional actual torch CPU adapter tests; never select/initialize an NPU.

Missing torch is a reported SKIP, not a successful runtime validation. These
tests do not import vllm/model weights, and do not implement HCCL collectives.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
HERE = Path(__file__).resolve().parent
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed: real CPU adapter/runtime unverified")
class TorchCPUAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        pkg_name = "_isolated_c_cpu_test"
        package = types.ModuleType(pkg_name)
        package.__path__ = [str(HERE)]
        sys.modules[pkg_name] = package
        base = HERE / "candidate" / "openvdn_npu.py"
        if not base.exists():
            base = HERE.parent / "b_adaptation" / "patched" / "openvdn_npu.py"
        cls.b = _load(pkg_name + ".openvdn_npu", base)
        cls.layout_module = _load(pkg_name + ".strict_source_layout", HERE / "strict_source_layout.py")
        cls.adapter = _load(pkg_name + ".strict_source_attention", HERE / "strict_source_attention.py")
        from test_strict_source import fixture, c_oracle, b_oracle
        cls.fixture = staticmethod(fixture)
        cls.c_oracle = staticmethod(c_oracle)
        cls.b_oracle = staticmethod(b_oracle)

    def tensors(self, data):
        t = self.torch
        return dict(img_pos=t.tensor(data["img_pos"], dtype=t.int64, device="cpu"),
                    update_mask=t.tensor(data["update_mask"], dtype=t.bool, device="cpu"),
                    text_pos=t.tensor(data["text_pos"], dtype=t.int64, device="cpu"),
                    audio_pos=t.tensor(data["audio_pos"], dtype=t.int64, device="cpu"),
                    img_position_ids=t.tensor(data["coords"], dtype=t.float64, device="cpu").unsqueeze(0),
                    cu_seqlens=t.tensor(data["cu_seqlens"], dtype=t.int32, device="cpu"), metadata=data["metadata"])

    def dense(self, q, k, v, layout, oracle):
        t = self.torch
        out = t.zeros_like(q)
        # SDPA all-false padding behavior varies with backend: leave padding
        # zero explicitly, matching both production B and C implementations.
        mask = t.tensor(oracle(layout, q.shape[0]), dtype=t.bool, device="cpu")[:layout.used_len]
        out[:layout.used_len] = t.nn.functional.scaled_dot_product_attention(
            q[:layout.used_len].transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=0.5
        ).squeeze(0).transpose(0, 1)
        return out

    def qkv(self, rows, heads=4):
        t = self.torch
        generator = t.Generator(device="cpu").manual_seed(128)
        return tuple(t.randn(rows, heads, 8, dtype=t.float64, device="cpu", generator=generator) for _ in range(3))

    def test_real_tensor_grouped_c_equals_independent_dense_oracle(self):
        t = self.torch
        data = self.fixture(frames=17, source_hw=(4, 4), target_hw=(2, 6))
        layout = self.adapter.infer_strict_source_layout(**self.tensors(data))
        q, k, v = self.qkv(len(data["coords"]))
        frozen = tuple(x.clone() for x in (q, k, v))
        expected = self.dense(q, k, v, layout, self.c_oracle)
        for batch in (1, 4, 37):
            actual = self.adapter.strict_source_softmax_attention(q, k, v, layout, 0.5, batch)
            t.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        for actual, old in zip((q, k, v), frozen):
            self.assertTrue(t.equal(actual, old))

    def test_actual_b_c_b_caches_do_not_cross_contaminate(self):
        t = self.torch
        data = self.fixture(frames=12, ra=1, ta=3)
        inputs = self.tensors(data)
        layout = self.adapter.infer_strict_source_layout(**inputs)
        b_layout = self.b.infer_openvdn_layout(inputs["img_pos"][inputs["update_mask"]],
                                              inputs["text_pos"], inputs["img_position_ids"], inputs["cu_seqlens"])
        q, k, v = self.qkv(len(data["coords"]))
        b1 = self.b.openvdn_softmax_attention(q, k, v, b_layout, 0.5)
        c1 = self.adapter.strict_source_softmax_attention(q, k, v, layout, 0.5)
        b2 = self.b.openvdn_softmax_attention(q, k, v, b_layout, 0.5)
        t.testing.assert_close(b1, self.dense(q, k, v, layout, self.b_oracle), atol=1e-10, rtol=1e-10)
        self.assertTrue(t.equal(b1, b2))
        self.assertFalse(t.equal(c1, b1))
        changed = self.fixture(frames=12, ra=2, ta=2)
        layout2 = self.adapter.infer_strict_source_layout(**self.tensors(changed))
        c2 = self.adapter.strict_source_softmax_attention(q, k, v, layout2, 0.5)
        t.testing.assert_close(c2, self.dense(q, k, v, layout2, self.c_oracle), atol=1e-10, rtol=1e-10)

    def test_tensor_head_shards_and_row_splits_reassemble(self):
        t = self.torch
        data = self.fixture(frames=12, source_hw=(4, 6), target_hw=(2, 6), pad=13)
        layout = self.adapter.infer_strict_source_layout(**self.tensors(data))
        q, k, v = self.qkv(len(data["coords"]), heads=4)
        full = self.adapter.strict_source_softmax_attention(q, k, v, layout, 0.5)
        heads = [self.adapter.strict_source_softmax_attention(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1], layout, 0.5) for i in range(4)]
        combined = t.cat(heads, dim=1)
        t.testing.assert_close(full, combined, atol=1e-10, rtol=1e-10)
        cuts = [0, layout.source_start + 1, layout.source_end - 1, layout.video_start + 1, len(data["coords"])]
        rows = t.cat([combined[lo:hi] for lo, hi in zip(cuts, cuts[1:])])
        self.assertTrue(t.equal(rows, combined))

    def test_actual_vendor_packer_not_synthetic_prefix(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("actual vendor packer also requires numpy")
        packer = _load("_isolated_c_cpu_test.packed_sequence", HERE / "upstream/vllm_omni/diffusion/models/minimax_h3/packed_sequence.py")
        t = self.torch
        refs = [dict(kind="video", latent_t=7, latent_h=4, latent_w=6, ref_audio_t=2)]
        packed = packer.minimax_h3_packed_sequence_ref2va_blocks(text_len=5, latent_t=7, latent_h=2, latent_w=4, audio_t=3, ref_blocks=refs)
        meta = self.layout_module.make_metadata(task="ref2va", ref_blocks=refs,
                    visual_condition_shapes=[(7, 4, 6)], target_shape=(7, 2, 4), text_len=5, audio_t=3)
        layout = self.adapter.infer_strict_source_layout(img_pos=packed["img_pos"], update_mask=packed["update_mask"],
                    text_pos=packed["text_pos"], audio_pos=packed["audio_pos"], img_position_ids=packed["img_position_ids"],
                    cu_seqlens=packed["cu_seqlens"], metadata=meta)
        self.assertNotEqual(layout.source_times, layout.target_times)
        q, k, v = self.qkv(int(packed["seq_len"]))
        out = self.adapter.strict_source_softmax_attention(q, k, v, layout, 0.5)
        t.testing.assert_close(out, self.dense(q, k, v, layout, self.c_oracle), atol=1e-10, rtol=1e-10)

    def test_bad_metadata_rejected_before_attention(self):
        data = self.fixture()
        data["metadata"]["reference_count"] = 2
        with self.assertRaises(ValueError):
            self.adapter.infer_strict_source_layout(**self.tensors(data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
