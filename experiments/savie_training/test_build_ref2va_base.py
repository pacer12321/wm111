"""Tiny-config fixtures for build_ref2va_base.py and the trainer's Ref2VA base gate.

The fixture h3-base is built the way OpenVDN's was presumably built -- diffusers' converter
applied to native FL2VA -- and a separate, distinct native Ref2VA is converted against it.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

import h3_diffusers_convert as conv
from ref2va_base_contract import RECEIPT_NAME, verify_ref2va_base

HERE = Path(__file__).resolve().parent
CONFIG = conv.MINIMAX_H3_TEST_TRANSFORMER_CONFIG


def native_checkpoint(root, seed):
    """Every planned native key with random values, split over two shards like a release."""
    generator = torch.Generator().manual_seed(seed)
    plan = conv.get_transformer_key_plan(CONFIG)
    heads, head_dim, hidden = CONFIG["num_attention_heads"], CONFIG["attention_head_dim"], CONFIG["hidden_size"]
    tensors = {}
    for key, targets in plan.items():
        if not targets:
            freq = CONFIG["rope_freq_dim"]
            tensors[key] = 1.0 / CONFIG["rope_theta"] ** (torch.arange(0, 2 * freq, 2).float() / (2 * freq))
            continue
        shape = [3 * heads * head_dim, hidden] if key.endswith("qkv_proj.weight") else targets[0][1]
        dtype = torch.float32 if key.startswith(conv.MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16
        tensors[key] = torch.randn(shape, generator=generator).to(dtype)
    transformer = Path(root) / "transformer"
    transformer.mkdir(parents=True)
    keys = sorted(tensors)
    for index, part in enumerate((keys[::2], keys[1::2]), start=1):
        save_file({k: tensors[k] for k in part}, str(transformer / f"model-{index:05d}-of-00002.safetensors"))
    return tensors


def vllm_reorder(weight, heads, head_dim):
    """vllm-omni `_reorder_grouped_qkv_to_qkv(num_query_groups=heads, heads_per_group=1)`, copied."""
    per_group = 3 * head_dim
    grouped = weight.reshape(heads, per_group, *weight.shape[1:])
    q, k, v = torch.split(grouped, [head_dim, head_dim, head_dim], dim=1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0)


class BuildRef2VABaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.models = root / "models"
        self.fl2va_dir = self.models / "MiniMax-H3" / "FL2VA"
        self.ref2va_dir = self.models / "MiniMax-H3" / "Ref2VA"
        self.fl2va = native_checkpoint(self.fl2va_dir, seed=1)
        self.ref2va = native_checkpoint(self.ref2va_dir, seed=2)
        self.h3_base = self.models / "OpenVDN" / "h3-base"
        conv.convert_transformer(str(self.fl2va_dir), str(self.h3_base / "transformer"), CONFIG, 1 << 20)
        (self.h3_base / "transformer" / "config.json").write_text(
            json.dumps({"_class_name": "MiniMaxH3Transformer3DModel", **CONFIG}))
        self.output = self.models / "OpenVDN" / "ref2va-base"

    def build(self, ref2va=None, fl2va=True, extra=()):
        command = [sys.executable, str(HERE / "build_ref2va_base.py"), "--version", "test",
                   "--ref2va", str(ref2va or self.ref2va_dir), "--template", str(self.h3_base),
                   "--output", str(self.output), "--max-shard-size", str(1 << 20), *extra]
        if fl2va:
            command += ["--fl2va", str(self.fl2va_dir)]
        return subprocess.run(command, cwd=HERE, capture_output=True, text=True)

    def load_output(self, key):
        from safetensors import safe_open
        transformer = self.output / "transformer"
        index = json.loads((transformer / conv.SAFE_WEIGHTS_INDEX_NAME).read_text())
        with safe_open(str(transformer / index["weight_map"][key]), framework="pt") as handle:
            return handle.get_tensor(key)

    def test_build_passes_and_trainer_gate_accepts(self):
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = verify_ref2va_base(self.output)
        self.assertEqual(receipt["partition"], "ref2va")
        self.assertTrue(receipt["checks"]["template_is_fl2va"]["passed"])
        self.assertGreater(receipt["checks"]["differs_from_template"]["differing"], 0)
        self.assertFalse((self.output.parent / "ref2va-base.partial").exists())

    def test_converted_weights_match_inference_loader_semantics(self):
        self.assertEqual(self.build().returncode, 0)
        heads, head_dim, ffn = CONFIG["num_attention_heads"], CONFIG["attention_head_dim"], CONFIG["ffn_dim"]
        for i in range(CONFIG["num_layers"]):
            fused = torch.cat([self.load_output(f"transformer_blocks.{i}.attn.to_{p}.weight") for p in "qkv"])
            self.assertTrue(torch.equal(fused, vllm_reorder(self.ref2va[f"blocks.{i}.attn.qkv_proj.weight"],
                                                            heads, head_dim)))
            # vllm-omni fc1 is [gate, up] -> silu(gate) * up; diffusers SwiGLU is [up; gate].
            native = self.ref2va[f"blocks.{i}.mlp.fc1.weight"]
            ff = self.load_output(f"transformer_blocks.{i}.ff.net.0.proj.weight")
            self.assertTrue(torch.equal(ff[:ffn], native[ffn:]) and torch.equal(ff[ffn:], native[:ffn]))
        self.assertTrue(torch.equal(self.load_output("proj_in.weight"), self.ref2va["video_patch_proj.weight"]))

    def test_h3_base_is_refused_as_training_base(self):
        with self.assertRaisesRegex(ValueError, "FL2VA h3-base"):
            verify_ref2va_base(self.h3_base)

    def test_tampered_base_is_refused(self):
        self.assertEqual(self.build().returncode, 0)
        shard = sorted((self.output / "transformer").glob("diffusion_pytorch_model-*.safetensors"))[0]
        with shard.open("r+b") as handle:  # rename a tensor inside the header
            raw = handle.read(4096)
            position = raw.index(b"transformer_blocks")
            handle.seek(position)
            handle.write(b"Transformer_blocks")
        with self.assertRaisesRegex(ValueError, "changed after conversion"):
            verify_ref2va_base(self.output)

    def test_fl2va_weights_under_a_ref2va_name_are_rejected(self):
        fake = self.models / "copy" / "Ref2VA"
        fake.mkdir(parents=True)
        os.symlink(self.fl2va_dir / "transformer", fake / "transformer")
        result = self.build(ref2va=fake)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a distinct Ref2VA", result.stderr)
        self.assertFalse(self.output.exists())

    def test_never_overwrites_and_dry_run_writes_nothing(self):
        result = self.build(extra=["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("never overwrites", result.stderr)


if __name__ == "__main__":
    unittest.main()
