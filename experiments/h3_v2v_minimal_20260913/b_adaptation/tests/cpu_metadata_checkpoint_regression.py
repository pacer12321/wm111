"""CPU-only dynamic-layout and small-matrix checkpoint-loader unit tests.

No actual checkpoint, H3 model, vLLM service, or NPU tensor is accessed.
The 50+2 fake attention objects are only tiny CPU matrices for mapping checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from cpu_head_shard_regression import BASE, load_module


def rejects(label, callback):
    try:
        callback()
    except (ValueError, RuntimeError) as error:
        return {"check": label, "rejected": type(error).__name__, "reason": str(error)}
    raise AssertionError(f"{label}: invalid input was accepted")


def check_layout(module):
    frames, height, width, prefix, text_length, padding = 7, 2, 3, 11, 3, 5
    per_frame = height * width
    used = prefix + frames * per_frame + text_length
    packed = used + padding
    coords = torch.full((packed, 3), -99.0, device="cpu")
    # H3 temporal coordinates may have nonuniform initial spacing and spatial
    # coordinates need not be integers. The layout must count, not round them.
    times = torch.tensor([0.0, 0.025, 0.058333, 0.091667, 0.125, 0.158333, 0.191667])
    hh, ww = torch.meshgrid(torch.tensor([-0.75, 0.25]), torch.tensor([-0.8, 0.0, 0.8]), indexing="ij")
    spatial = torch.stack((hh.flatten(), ww.flatten()), -1)
    for index, time in enumerate(times):
        coords[prefix + index * per_frame:prefix + (index + 1) * per_frame, 0] = time
        coords[prefix + index * per_frame:prefix + (index + 1) * per_frame, 1:] = spatial
    target = torch.arange(prefix, prefix + frames * per_frame, device="cpu")
    text = torch.arange(prefix + frames * per_frame, used, device="cpu")
    cu = torch.tensor([0, used, packed], device="cpu")
    result = module.infer_openvdn_layout(target, text, coords, cu)
    assert (result.num_frames, result.frame_height, result.frame_width) == (frames, height, width)
    assert (result.video_start, result.video_end, result.used_len, result.text_len) == (prefix, prefix + frames * per_frame, used, text_length)
    records = [{"check": "dynamic-grid-noninteger-coordinates-padding", "layout": result.__dict__}]
    empty_text = module.infer_openvdn_layout(target, torch.empty(0, dtype=torch.long), coords, cu)
    assert empty_text.text_len == 0
    records.append({"check": "empty-text-layout", "passed": True})
    shuffled = target.clone()
    shuffled[[0, 1]] = shuffled[[1, 0]]
    records.append(rejects("out-of-order-target-rows", lambda: module.infer_openvdn_layout(shuffled, text, coords, cu)))
    wrong_selection = torch.cat((target[:2], target[3:]))
    records.append(rejects("missing-target-row", lambda: module.infer_openvdn_layout(wrong_selection, text, coords, cu)))
    source_selected = torch.cat((torch.tensor([prefix - 1]), target))
    records.append(rejects("source-row-in-target-mask", lambda: module.infer_openvdn_layout(source_selected, text, coords, cu)))
    inconsistent = coords.clone()
    inconsistent[prefix + per_frame, 1] += 0.2
    records.append(rejects("inconsistent-grid-across-frames", lambda: module.infer_openvdn_layout(target, text, inconsistent, cu)))
    unordered_grid = coords.clone()
    unordered_grid[prefix:prefix + 2] = unordered_grid[torch.tensor([prefix + 1, prefix])]
    records.append(rejects("non-row-major-grid", lambda: module.infer_openvdn_layout(target, text, unordered_grid, cu)))
    records.append(rejects("text-overlaps-target", lambda: module.infer_openvdn_layout(target, target[:2], coords, cu)))
    records.append(rejects("multiple-real-documents", lambda: module.infer_openvdn_layout(target, text, coords, torch.tensor([0, 4, used, packed]))))
    not_finite = coords.clone()
    not_finite[prefix, 0] = float("nan")
    records.append(rejects("nonfinite-grid", lambda: module.infer_openvdn_layout(target, text, not_finite, cu)))
    return records


def fake_attention(index):
    heads, dim, hidden = 2, 4, 6
    qkv = torch.arange(3 * heads * dim * hidden, dtype=torch.float32, device="cpu").reshape(3 * heads * dim, hidden) + index * 1000
    out = torch.zeros(hidden, heads * dim, device="cpu")
    return SimpleNamespace(
        total_num_heads=heads, num_heads=heads, head_dim=dim,
        qkv_proj=SimpleNamespace(weight=qkv), out_proj=SimpleNamespace(weight=out),
    )


def check_mapping(module):
    model = SimpleNamespace(
        blocks=[SimpleNamespace(attn=fake_attention(i)) for i in range(50)],
        token_refiner=SimpleNamespace(blocks=[SimpleNamespace(attn=fake_attention(i + 50)) for i in range(2)]),
    )
    pairs = list(module.lora_targets(model))
    assert len(pairs) == 208
    assert len({name for name, _, _ in pairs}) == 208
    all_blocks = model.blocks + model.token_refiner.blocks
    for index, block in enumerate(all_blocks):
        qkv = block.attn.qkv_proj.weight
        rows = 8
        family = f"transformer_blocks.{index}.attn.orig" if index < 50 else f"token_refiner.refiner_blocks.{index - 50}.attn"
        for offset, projection in enumerate(("to_q", "to_k", "to_v")):
            name, target, _ = pairs[4 * index + offset]
            assert name == f"{family}.{projection}"
            expected = qkv[offset * rows:(offset + 1) * rows]
            assert target.data_ptr() == expected.data_ptr()
            torch.testing.assert_close(target, expected, atol=0, rtol=0)
        name, target, _ = pairs[4 * index + 3]
        assert name == f"{family}.to_out.0" and target is block.attn.out_proj.weight
    mapping = module.branch_name_map()
    assert len(mapping) == len(set(mapping.values())) == 800
    for index in range(50):
        for suffix in module.BRANCH_SUFFIXES:
            assert mapping[f"transformer_blocks.{index}.attn.{suffix}"] == f"blocks.{index}.attn.{suffix}"
    return [{"check": "lora-post-reorder-QKV-thirds-and-O", "pairs": 208}, {"check": "all-branch-name-mapping", "tensors": 800}]


def check_merge(module):
    records = []
    for seed in (71, 83):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for dtype in (torch.float32, torch.bfloat16):
            a = (torch.randn(64, 13, generator=generator, device="cpu") * 0.05).to(dtype)
            b = (torch.randn(11, 64, generator=generator, device="cpu") * 0.05).to(dtype)
            base = (torch.randn(11, 13, generator=generator, device="cpu") * 0.1).to(dtype)
            expected = base + (b.float() @ a.float()).to(dtype)
            for chunk in (1, 3, 256):
                actual = base.clone()
                module.merge_lora_pair_(actual, a, b, name="synthetic", chunk_rows=chunk)
                atol, rtol = (0.0, 0.0) if dtype == torch.bfloat16 else (1e-7, 1e-6)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                records.append({"check": "official-scale-one-LoRA", "seed": seed, "dtype": str(dtype), "chunk_rows": chunk, "max_abs": float((actual.float() - expected.float()).abs().max()), "atol": atol, "rtol": rtol})
    records.append(rejects("invalid-LoRA-shape", lambda: module.merge_lora_pair_(torch.zeros(3, 4), torch.zeros(2, 5), torch.zeros(3, 2), name="bad")))
    records.append(rejects("nonfinite-LoRA", lambda: module.merge_lora_pair_(torch.zeros(3, 4), torch.full((2, 4), float("nan")), torch.zeros(3, 2), name="nan")))
    return records


def check_headers(module, temp):
    valid = temp / "valid.safetensors"
    save_file({"tiny": torch.arange(12, device="cpu").to(torch.bfloat16).reshape(3, 4)}, valid)
    header, record = module.inspect_header(valid)
    assert header["tiny"]["shape"] == [3, 4] and record["tensor_count"] == 1
    records = [{"check": "valid-BF16-header", "passed": True}]
    truncated = temp / "truncated.safetensors"
    truncated.write_bytes(valid.read_bytes()[:-1])
    records.append(rejects("truncated-data", lambda: module.inspect_header(truncated)))
    f32 = temp / "unexpected-dtype.safetensors"
    save_file({"tiny": torch.zeros(3, 4, device="cpu")}, f32)
    records.append(rejects("unexpected-FP32-checkpoint", lambda: module.inspect_header(f32)))
    duplicate = temp / "duplicate-key.safetensors"
    fragment = '"tiny":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
    raw = ("{" + fragment + "," + fragment + "}").encode()
    duplicate.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(2))
    records.append(rejects("duplicate-header-key", lambda: module.inspect_header(duplicate)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source_paths = [args.base / "patched/openvdn_npu.py", args.base / "patched/openvdn_checkpoint.py"]
    attention = load_module("openvdn_metadata_candidate", source_paths[0])
    checkpoint = load_module("openvdn_checkpoint_candidate", source_paths[1])
    with torch.inference_mode():
        checks = check_layout(attention) + check_mapping(checkpoint) + check_merge(checkpoint)
        with tempfile.TemporaryDirectory(prefix="synthetic_headers_", dir=args.output.parent) as directory:
            checks.extend(check_headers(checkpoint, Path(directory)))
    report = {"status": "passed", "device": "cpu", "source_sha256": {str(path.relative_to(args.base)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, "checks": checks, "limitations": "Synthetic unit tests only; no full checkpoint loading, vLLM loader integration, actual NPU execution, or actual H3 sample metadata."}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
