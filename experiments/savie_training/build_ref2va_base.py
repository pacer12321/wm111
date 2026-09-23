"""Build the Ref2VA training base: native `MiniMax-H3/Ref2VA` -> the diffusers layout of `h3-base`.

OpenVDN's `h3-base/transformer` is the diffusers export of the FL2VA partition, while inference
serves the native Ref2VA partition. This writes `<output>/transformer` from the Ref2VA shards with
diffusers' own converter (vendored in `h3_diffusers_convert.py`) and `h3-base`'s config.json, so
`--base <output>` is a drop-in for `--base h3-base` that differs only in the weights.

    python build_ref2va_base.py \
        --ref2va   /cache/zhonghao/h3/models/MiniMax-H3/Ref2VA \
        --template /cache/zhonghao/h3/models/OpenVDN-vdn-minimax-h3/h3-base \
        --output   /cache/zhonghao/h3/models/OpenVDN-vdn-minimax-h3/ref2va-base \
        [--fl2va   /cache/zhonghao/h3/models/MiniMax-H3/FL2VA] [--dry-run]

Checks, all recorded in the receipt the trainer verifies at startup:
  source_key_plan                 every native Ref2VA key is planned, dtypes/shapes as planned
  template_layout                 the planned diffusers keys/shapes/dtypes are exactly h3-base's
  converted_layout                the written keys/shapes/dtypes are exactly h3-base's
  independent_numeric_spot_check  sampled tensors equal an independent re-derivation from the
                                  inference loader's semantics (per-head [q,k,v] rows, fc1 [gate; up])
  differs_from_template           sampled weights are not FL2VA's (catches pointing at the wrong source)
  template_is_fl2va  (--fl2va)    the same derivation from native FL2VA reproduces h3-base bitwise,
                                  i.e. h3-base was built by this conversion
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open

import h3_diffusers_convert as conv
from ref2va_base_contract import (INDEX_NAME, PARTITION, RECEIPT_NAME, SCHEMA,
                                  fingerprint_transformer, sha256_file)

UPSTREAM_CONVERTER_SHA256 = "86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c"


def native_header(checkpoint):
    """key -> (dtype, shape, file) over every shard of a native `<variant>/transformer`."""
    header = {}
    for shard in sorted(glob.glob(os.path.join(checkpoint, "transformer", "*.safetensors"))):
        for key, info in conv.read_safetensors_header(shard).items():
            if key in header:
                raise ValueError(f"duplicate key {key} in {shard}")
            header[key] = (info["dtype"], list(info["shape"]), shard)
    if not header:
        raise FileNotFoundError(f"no safetensors shards under {checkpoint}/transformer")
    return header


def diffusers_header(transformer_dir):
    """key -> (dtype, shape, file) for a diffusers transformer, resolved through its index."""
    index = json.loads((Path(transformer_dir) / INDEX_NAME).read_text(encoding="utf-8"))
    header, by_file = {}, {}
    for key, name in index["weight_map"].items():
        by_file.setdefault(name, set()).add(key)
    for name, keys in by_file.items():
        path = os.path.join(transformer_dir, name)
        shard = conv.read_safetensors_header(path)
        if set(shard) != keys:
            raise ValueError(f"{path}: header keys disagree with {INDEX_NAME}")
        for key in keys:
            header[key] = (shard[key]["dtype"], list(shard[key]["shape"]), path)
    return header


def planned_layout(config):
    """diffusers key -> (dtype, shape) the converter writes for `config`."""
    layout = {}
    for source_key, targets in conv.get_transformer_key_plan(config).items():
        dtype = "F32" if source_key.startswith(conv.MINIMAX_H3_FP32_SOURCE_PREFIXES) else "BF16"
        for target_key, shape in targets:
            layout[target_key] = (dtype, list(shape))
    return layout


def compare_layout(actual, expected, label):
    """Exact key set, dtype and shape equality; returns a check record or raises."""
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    wrong = [key for key in expected.keys() & actual.keys()
             if tuple(actual[key][:2]) != tuple(expected[key][:2])]
    if missing or extra or wrong:
        raise ValueError(f"{label}: missing={missing[:8]} unexpected={extra[:8]} "
                         f"dtype/shape mismatches={sorted(wrong)[:8]}")
    return {"passed": True, "tensor_count": len(expected)}


def check_source_plan(source, config):
    plan = conv.get_transformer_key_plan(config)
    missing, extra = sorted(plan.keys() - source.keys()), sorted(source.keys() - plan.keys())
    if missing or extra:
        raise ValueError(f"native Ref2VA keys vs plan: missing={missing[:8]} unexpected={extra[:8]}")
    wrong = []
    for key, targets in plan.items():
        dtype, shape, _ = source[key]
        expected = "F32" if key.startswith(conv.MINIMAX_H3_FP32_SOURCE_PREFIXES) else "BF16"
        if targets and dtype != expected:
            wrong.append(f"{key}: dtype {dtype} != {expected}")
        if len(targets) == 1 and shape != targets[0][1]:
            wrong.append(f"{key}: shape {shape} != {targets[0][1]}")
    if wrong:
        raise ValueError(f"native Ref2VA header disagrees with the plan: {wrong[:8]}")
    return {"passed": True, "source_tensor_count": len(source)}


def check_template_config(template, config):
    """h3-base's config.json must describe the architecture the converter plans for."""
    stored = json.loads((Path(template) / "transformer" / "config.json").read_text(encoding="utf-8"))
    wrong = {key: (stored.get(key), value) for key, value in config.items()
             if json.dumps(stored.get(key)) != json.dumps(value)}
    if wrong:
        raise ValueError(f"h3-base config.json disagrees with the converter config: {wrong}")


def sample_source_keys(config):
    """All non-block keys, both refiner blocks, and the first, middle and last DiT blocks."""
    last = config["num_layers"] - 1
    blocks = {f"blocks.{i}." for i in (0, last // 2, last)}
    keep = []
    for key, targets in conv.get_transformer_key_plan(config).items():
        if not targets:
            continue
        if key.startswith("blocks.") and not any(key.startswith(prefix) for prefix in blocks):
            continue
        keep.append(key)
    return keep


def independent_targets(source_key, tensor, config):
    """Re-derive the diffusers tensors from the inference loader's semantics, not the converter's code.

    vllm-omni's `_install_qkv_weight_loader` reads fused QKV rows as per-head [q, k, v]
    (`_reorder_grouped_qkv_to_qkv`, one head per group) and `_install_fc1_weight_loader` reads
    fc1 as [gate, up]; diffusers' attention takes separate to_q/to_k/to_v and its SwiGLU reads
    [up; gate]. Target names are the converter's (names cannot change values).
    """
    names = [name for name, _ in conv.get_transformer_key_plan(config)[source_key]]
    if source_key.endswith(".attn.qkv_proj.weight"):
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        rows = tensor.reshape(heads, 3, head_dim, tensor.shape[-1])
        return dict(zip(names, (rows[:, i].reshape(heads * head_dim, -1) for i in range(3))))
    if source_key.endswith(".mlp.fc1.weight"):
        half = tensor.shape[0] // 2
        return {names[0]: torch.cat([tensor[half:], tensor[:half]], dim=0)}
    (name,) = names
    return {name: tensor}


class Reader:
    """Lazily opened safetensors handles, keyed by file."""

    def __init__(self):
        self.handles = {}

    def get(self, header, key):
        path = header[key][2]
        if path not in self.handles:
            self.handles[path] = safe_open(path, framework="pt", device="cpu")
        return self.handles[path].get_tensor(key)


def spot_check(source_header, target_header, config, reader, label):
    """Bitwise: independent_targets(native tensor) == the diffusers tensor at the same key."""
    checked, unequal = 0, []
    for source_key in sample_source_keys(config):
        expected = independent_targets(source_key, reader.get(source_header, source_key), config)
        for key, value in expected.items():
            actual = reader.get(target_header, key)
            checked += 1
            if actual.dtype != value.dtype or not torch.equal(actual, value):
                unequal.append(key)
    if unequal:
        raise ValueError(f"{label}: {len(unequal)}/{checked} sampled tensors differ, e.g. {unequal[:8]}")
    return {"passed": True, "tensors_compared": checked}


def check_differs(output_header, template_header, config, reader):
    """Sampled block matrices must not all equal FL2VA's; an all-equal result means a wrong source."""
    same, differ = [], []
    for source_key in sample_source_keys(config):
        for key, _ in conv.get_transformer_key_plan(config)[source_key]:
            if "blocks." not in key or not key.endswith(".weight") or len(output_header[key][1]) != 2:
                continue
            equal = torch.equal(reader.get(output_header, key), reader.get(template_header, key))
            (same if equal else differ).append(key)
    if not differ:
        raise ValueError(f"all {len(same)} sampled block matrices equal h3-base: the source is "
                         "not a distinct Ref2VA checkpoint")
    return {"passed": True, "sampled_block_matrices": len(same) + len(differ),
            "differing": len(differ), "identical": same}


def sha256_file_header(path):
    """Digest of one native shard's safetensors header (the payload is covered by the spot check)."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return hashlib.sha256(handle.read(size)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref2va", type=Path, required=True, help="native MiniMax-H3/Ref2VA directory")
    parser.add_argument("--template", type=Path, required=True, help="OpenVDN h3-base directory")
    parser.add_argument("--output", type=Path, required=True, help="new directory; must not exist")
    parser.add_argument("--fl2va", type=Path, default=None,
                        help="optional native MiniMax-H3/FL2VA directory, to prove h3-base == convert(FL2VA)")
    parser.add_argument("--max-shard-size", type=int, default=5 * 1024**3)
    parser.add_argument("--dry-run", action="store_true", help="header-level checks only; write nothing")
    parser.add_argument("--version", choices=["h3", "test"], default="h3", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # abspath, not resolve: the partition is identified by its directory name, which a symlink may hide.
    ref2va, template, output = (Path(os.path.abspath(path)) for path in (args.ref2va, args.template, args.output))
    if ref2va.name != "Ref2VA":
        raise ValueError(f"--ref2va must be the native Ref2VA partition directory, got {ref2va}")
    config = conv.MINIMAX_H3_TEST_TRANSFORMER_CONFIG if args.version == "test" else conv.MINIMAX_H3_TRANSFORMER_CONFIG
    partial = output.with_name(output.name + ".partial")
    if not args.dry_run and (output.exists() or partial.exists()):
        raise FileExistsError(f"{output} or {partial} already exists; this never overwrites")

    checks = {}
    source = native_header(ref2va)
    checks["source_key_plan"] = check_source_plan(source, config)
    check_template_config(template, config)
    template_header = diffusers_header(template / "transformer")
    checks["template_layout"] = compare_layout(template_header, planned_layout(config), "h3-base vs plan")
    reader = Reader()
    if args.fl2va is not None:
        checks["template_is_fl2va"] = spot_check(native_header(os.path.abspath(args.fl2va)), template_header,
                                                 config, reader, "convert(FL2VA) vs h3-base")
    print(json.dumps({"stage": "preflight", "checks": checks}, indent=2), flush=True)
    if args.dry_run:
        return

    transformer = partial / "transformer"
    conv.convert_transformer(str(ref2va), str(transformer), config, args.max_shard_size)
    shutil.copyfile(template / "transformer" / "config.json", transformer / "config.json")
    output_header = diffusers_header(transformer)
    checks["converted_layout"] = compare_layout(output_header, template_header, "ref2va-base vs h3-base")
    checks["independent_numeric_spot_check"] = spot_check(source, output_header, config, reader,
                                                          "convert(Ref2VA) vs re-derivation")
    checks["differs_from_template"] = check_differs(output_header, template_header, config, reader)

    receipt = {
        "schema": SCHEMA,
        "partition": PARTITION,
        "source": str(ref2va),
        "source_shards": {Path(path).name: sha256_file_header(path)
                          for path in sorted({info[2] for info in source.values()})},
        "template": str(template),
        "template_fingerprint": fingerprint_transformer(template / "transformer"),
        "converter": {"module": "h3_diffusers_convert.py",
                      "module_sha256": sha256_file(conv.__file__),
                      "upstream_sha256": UPSTREAM_CONVERTER_SHA256},
        "config_source": str(template / "transformer" / "config.json"),
        "checks": checks,
        "fingerprint": fingerprint_transformer(transformer),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (partial / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    partial.rename(output)
    print(json.dumps({"stage": "done", "output": str(output), "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
