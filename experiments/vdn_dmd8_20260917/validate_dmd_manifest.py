import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from manifest_builder import attach_model_metadata, tensor_plans_for_main_block
from streaming_shards import RangeReader, bf16_to_fp32, fp32_to_bf16, read_model_interval


MANIFEST = HERE / "manifest.json"
META = Path("/cache/zhonghao/h3/track_b_validation_20260915/real_meta_v1.json")


manifest = attach_model_metadata(json.loads(MANIFEST.read_text()), json.loads(META.read_text()))
readers = {}


def reader(path):
    readers.setdefault(path, RangeReader(path))
    return readers[path]


plans = tensor_plans_for_main_block(manifest, 0, reader)
by_name = {plan.name: plan for plan in plans}
assert len(plans) == 26
assert len([p for p in manifest["plans"] if p["offload_unit"] == "non_main_block_requires_explicit_policy"]) == 35
assert manifest["lora_merge_entry_count"] == 623
assert len(by_name["blocks.0.attn.qkv_proj.weight"].loras) == 6
assert len(by_name["blocks.0.attn.out_proj.weight"].loras) == 2
assert {x.b_row_offset for x in by_name["blocks.0.mlp.fc1.weight"].loras} == {0, 14336}


def assembled_row(plan, row):
    cols = plan.source.info["shape"][1]
    fragments = list(read_model_interval(plan, row * cols, (row + 1) * cols, chunk_elements=cols))
    assert len(fragments) == 1 and fragments[0][0] == row * cols
    return fragments[0][1]


def reference_row(plan, row):
    rows, cols = plan.source.info["shape"]
    source_row = row
    if plan.qkv_heads is not None:
        rows_per_projection = plan.qkv_heads * plan.head_dim
        projection, within = divmod(row, rows_per_projection)
        head, dim = divmod(within, plan.head_dim)
        source_row = head * 3 * plan.head_dim + projection * plan.head_dim + dim
    result = plan.source.reader.read_flat(plan.source.key, source_row * cols, (source_row + 1) * cols)
    for lora in plan.loras:
        if not lora.start_row <= row < lora.end_row:
            continue
        rank = lora.a.info["shape"][0]
        a_raw = lora.a.reader.read_flat(lora.a.key, 0, rank * cols)
        a = bf16_to_fp32(a_raw).reshape(rank, cols)
        b_row = lora.b_row_offset + row - lora.start_row
        b_raw = lora.b.reader.read_flat(lora.b.key, b_row * rank, (b_row + 1) * rank)
        delta = fp32_to_bf16(bf16_to_fp32(b_raw) @ a)
        result = fp32_to_bf16(bf16_to_fp32(result) + bf16_to_fp32(delta))
    return result


checks = {
    "blocks.0.attn.qkv_proj.weight": (0, 7168, 14336),
    "blocks.0.attn.out_proj.weight": (0,),
    "blocks.0.mlp.fc1.weight": (0, 14336),
    "blocks.0.mlp.fc2.weight": (0,),
    "blocks.0.adaln_proj.linear.weight": (0,),
}
for name, rows in checks.items():
    for row in rows:
        actual = assembled_row(by_name[name], row)
        expected = reference_row(by_name[name], row)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"numeric mismatch {name} row={row}")
        print("PASS", name, "row", row)

print("DMD8_MANIFEST_VALID", json.dumps({
    "plans": len(manifest["plans"]),
    "default_pairs": manifest["default_lora_pair_count"],
    "turbo_pairs": manifest["turbo_lora_pair_count"],
    "merge_entries": manifest["lora_merge_entry_count"],
}, sort_keys=True))
