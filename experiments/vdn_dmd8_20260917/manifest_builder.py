"""Header-only checkpoint audit and explicit tensor plan manifest.

Never constructs H3 or reads payload bytes. Optional SSH collection is
read-only on the specified host; output artifacts are created locally.
Model iteration order remains UNVERIFIED until attach_model_metadata() gets
a real meta-model enumeration. Checkpoint key order is NOT a substitute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = "/cache/zhonghao/h3/models"
BRANCH_SUFFIXES = (
    "linear_attention.alpha.A_log", "linear_attention.alpha.down.weight",
    "linear_attention.alpha.dt_bias", "linear_attention.alpha.up.weight",
    "linear_attention.beta_proj.weight", "linear_attention.norm.weight",
    "linear_attention.output_gate.down.weight", "linear_attention.output_gate.up.bias",
    "linear_attention.output_gate.up.weight", "linear_attention.short_conv.k_sp.weight",
    "linear_attention.short_conv.k_tm.weight", "linear_attention.short_conv.v_sp.weight",
    "linear_attention.short_conv.v_tm.weight", "softmax_gate.up.bias",
    "softmax_gate.up.weight", "to_out_linear.weight",
)
ITEMSIZE = {"BF16": 2, "F32": 4}


def collect_headers(ssh_config, port=30674):
    if port != 30674:
        raise ValueError("This audit is scoped to port 30674")
    # Read-only remote Python: only open/read/stat/print. No model imports,
    # tensor payloads, directories/files written, or GPU initialization.
    remote = r'''
import hashlib, json, struct
from pathlib import Path
root = Path('/cache/zhonghao/h3/models')
base = root / 'MiniMax-H3/Ref2VA/transformer'
stage = root / 'OpenVDN-vdn-minimax-h3/stage-dmd-step-250'
groups = {'base': sorted(base.glob('*.safetensors')),
          'branch': [stage / 'linear_branch/model.safetensors'],
          'default_lora': [stage / 'adapters/default/adapter_model.safetensors'],
          'turbo_lora': [stage / 'adapters/turbo/adapter_model.safetensors']}
result = {'config': json.loads((base / 'config.json').read_text()), 'files': []}
for group, files in groups.items():
    for path in files:
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if not 2 <= size <= 16 * 1024 * 1024: raise ValueError('header size')
            raw = stream.read(size)
            if len(raw) != size: raise ValueError('truncated header')
        result['files'].append({'group':group, 'path':str(path), 'file_size':path.stat().st_size,
            'header_size':size, 'header_sha256':hashlib.sha256(raw).hexdigest(), 'header':json.loads(raw)})
print(json.dumps(result, separators=(',', ':')))
'''
    command = ["ssh", "-F", str(ssh_config), "-p", str(port), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=8", "dev-modelarts-cnnorth9.huaweicloud.com",
               "/cache/zhonghao/h3/env_cuda_v1/bin/python -c " + "'" + remote.replace("'", "'\"'\"'") + "'"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(completed.stdout)


def collect_local_headers():
    """Collect the same bounded safetensors headers on the serving host."""
    import struct
    root = Path(ROOT)
    base = root / "MiniMax-H3/Ref2VA/transformer"
    stage = root / "OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
    groups = {
        "base": sorted(base.glob("*.safetensors")),
        "branch": [stage / "linear_branch/model.safetensors"],
        "default_lora": [stage / "adapters/default/adapter_model.safetensors"],
        "turbo_lora": [stage / "adapters/turbo/adapter_model.safetensors"],
    }
    result = {"config": json.loads((base / "config.json").read_text()), "files": []}
    for group, files in groups.items():
        for path in files:
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if not 2 <= size <= 16 * 1024 * 1024:
                    raise ValueError("header size")
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError("truncated header")
            result["files"].append({"group": group, "path": str(path), "file_size": path.stat().st_size,
                "header_size": size, "header_sha256": hashlib.sha256(raw).hexdigest(), "header": json.loads(raw)})
    return result


def flatten_headers(snapshot):
    groups = {name: {} for name in ("base", "branch", "default_lora", "turbo_lora")}
    for file in snapshot["files"]:
        group = groups[file["group"]]
        offset = 0
        entries = [(k, v) for k, v in file["header"].items() if k != "__metadata__"]
        for key, info in sorted(entries, key=lambda pair: pair[1]["data_offsets"][0]):
            if key in group:
                raise ValueError(f"Duplicate {file['group']} key {key}")
            shape, dtype = info["shape"], info["dtype"]
            if dtype not in ITEMSIZE or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError(f"Invalid shape/dtype {key}")
            lo, hi = info["data_offsets"]
            if lo != offset or hi - lo != math.prod(shape) * ITEMSIZE[dtype]:
                raise ValueError(f"Invalid extent {key}")
            offset = hi
            group[key] = dict(source_file=file["path"], source_key=key, shape=shape, dtype=dtype,
                              data_offset=8 + file["header_size"] + lo, numel=math.prod(shape))
        if 8 + file["header_size"] + offset != file["file_size"]:
            raise ValueError(f"Bad file size: {file['path']}")
    return groups


def exact_keys(actual, expected, label):
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"{label}: missing={missing}, unexpected={extra}")


def audit(snapshot):
    groups = flatten_headers(snapshot)
    base, branch, default_lora, turbo_lora = (
        groups[k] for k in ("base", "branch", "default_lora", "turbo_lora")
    )
    if len(base) != 535:
        raise ValueError(f"Expected 535 base tensors, got {len(base)}")
    config = snapshot["config"]
    heads = config.get("num_attention_heads")
    head_dim = config.get("attention_head_dim")
    if not heads or not head_dim or config.get("num_layers") != 50:
        raise ValueError("Unexpected full-model config")
    hidden = config["hidden_size"]
    attention_width = heads * head_dim
    # Header names are native DiT names, not the pipeline's transformer.
    # Prefix. Refuse to infer a prefix if an incompatible checkpoint appears.
    plans = {key: dict(value, model_name=key, transformation="identity", lora_pairs=[]) for key, value in base.items()}
    branch_mapping = {f"transformer_blocks.{i}.attn.{suffix}": f"blocks.{i}.attn.{suffix}"
                      for i in range(50) for suffix in BRANCH_SUFFIXES}
    exact_keys(branch, branch_mapping, "Stage-B branch 800")
    for source_key, name in branch_mapping.items():
        if name in plans or branch[source_key]["dtype"] != "BF16":
            raise ValueError(f"Invalid branch tensor {name}")
        plans[name] = dict(branch[source_key], model_name=name, transformation="identity", lora_pairs=[])
    expected_default = set()
    expected_turbo = set()
    default_pair_count = 0
    turbo_pair_count = 0
    merge_entry_count = 0

    def append_pair(target_name, a, b, *, start_row, end_row, b_row_offset=0, adapter):
        nonlocal merge_entry_count
        target = base[target_name]
        rank = a["shape"][0]
        if (a["shape"] != [rank, target["shape"][1]] or len(b["shape"]) != 2
                or b["shape"][1] != rank or b_row_offset < 0
                or b_row_offset + end_row - start_row > b["shape"][0]):
            raise ValueError(f"Bad {adapter} LoRA shapes for {target_name}")
        if any(x["dtype"] != "BF16" for x in (target, a, b)):
            raise ValueError(f"Bad {adapter} LoRA/target dtype for {target_name}")
        plans[target_name]["lora_pairs"].append(dict(
            start_row=start_row, end_row=end_row, b_row_offset=b_row_offset,
            scale=1, adapter=adapter, a=a, b=b,
            merge="FP32_BA_then_cast_delta_then_parameter_dtype_add"))
        merge_entry_count += 1

    def attention_pairs(adapter_name, source_tensors, expected_keys):
        nonlocal default_pair_count, turbo_pair_count
        pair_count = 0
        for family, count, source in (("blocks", 50, "transformer_blocks.{}.attn.orig"),
                                      ("token_refiner.blocks", 2, "token_refiner.refiner_blocks.{}.attn")):
            for i in range(count):
                prefix = f"{family}.{i}.attn"
                qkv_name, out_name = prefix + ".qkv_proj.weight", prefix + ".out_proj.weight"
                if qkv_name not in base or out_name not in base:
                    raise ValueError(f"Missing attention projections {prefix}")
                if base[qkv_name]["shape"] != [3 * attention_width, hidden] or base[out_name]["shape"] != [hidden, attention_width]:
                    raise ValueError(f"Unexpected projection shape {prefix}")
                plans[qkv_name]["transformation"] = "grouped_head_qkv_to_contiguous_qkv"
                plans[qkv_name]["qkv_heads"] = heads
                plans[qkv_name]["head_dim"] = head_dim
                for projection, name, start, stop in (("to_q", qkv_name, 0, attention_width),
                                                       ("to_k", qkv_name, attention_width, 2 * attention_width),
                                                       ("to_v", qkv_name, 2 * attention_width, 3 * attention_width),
                                                       ("to_out.0", out_name, 0, hidden)):
                    adapter_prefix = source.format(i) + "." + projection
                    a_key = adapter_prefix + f".lora_A.{adapter_name}.weight"
                    b_key = adapter_prefix + f".lora_B.{adapter_name}.weight"
                    expected_keys.update((a_key, b_key))
                    if a_key not in source_tensors or b_key not in source_tensors:
                        raise ValueError(f"Missing {adapter_name} LoRA pair {adapter_prefix}")
                    append_pair(name, source_tensors[a_key], source_tensors[b_key],
                                start_row=start, end_row=stop, adapter=adapter_name)
                    pair_count += 1
        if adapter_name == "default":
            default_pair_count += pair_count
        else:
            turbo_pair_count += pair_count

    attention_pairs("default", default_lora, expected_default)
    attention_pairs("turbo", turbo_lora, expected_turbo)

    # DMD turbo additionally adapts all FFNs, per-block AdaLN and final AdaLN.
    # Turbo is already released in diffusers layout.  Native H3 fc1 stores
    # [gate; value], opposite to diffusers [value; gate], so address the two B
    # halves in reversed order without allocating a transformed full tensor.
    for family, count, source_family in (("blocks", 50, "transformer_blocks"),
                                         ("token_refiner.blocks", 2, "token_refiner.refiner_blocks")):
        for i in range(count):
            fc1 = f"{family}.{i}.mlp.fc1.weight"
            fc2 = f"{family}.{i}.mlp.fc2.weight"
            if base[fc1]["shape"] != [28672, hidden] or base[fc2]["shape"] != [hidden, 14336]:
                raise ValueError(f"Unexpected FFN target shape {family}.{i}")
            for suffix, target in (("ff.net.0.proj", fc1), ("ff.net.2", fc2)):
                prefix = f"{source_family}.{i}.{suffix}"
                a_key = prefix + ".lora_A.turbo.weight"
                b_key = prefix + ".lora_B.turbo.weight"
                expected_turbo.update((a_key, b_key))
                a, b = turbo_lora[a_key], turbo_lora[b_key]
                if target == fc1:
                    half = base[target]["shape"][0] // 2
                    append_pair(target, a, b, start_row=0, end_row=half,
                                b_row_offset=half, adapter="turbo")
                    append_pair(target, a, b, start_row=half, end_row=2 * half,
                                b_row_offset=0, adapter="turbo")
                else:
                    append_pair(target, a, b, start_row=0, end_row=base[target]["shape"][0],
                                adapter="turbo")
                turbo_pair_count += 1

    for i in range(50):
        target = f"blocks.{i}.adaln_proj.linear.weight"
        prefix = f"transformer_blocks.{i}.adaln_proj.linear"
        a_key = prefix + ".lora_A.turbo.weight"
        b_key = prefix + ".lora_B.turbo.weight"
        expected_turbo.update((a_key, b_key))
        append_pair(target, turbo_lora[a_key], turbo_lora[b_key],
                    start_row=0, end_row=base[target]["shape"][0], adapter="turbo")
        turbo_pair_count += 1

    target = "final_layer.adaln_proj.linear.weight"
    a_key = "norm_out.linear.lora_A.turbo.weight"
    b_key = "norm_out.linear.lora_B.turbo.weight"
    expected_turbo.update((a_key, b_key))
    append_pair(target, turbo_lora[a_key], turbo_lora[b_key],
                start_row=0, end_row=base[target]["shape"][0], adapter="turbo")
    turbo_pair_count += 1

    exact_keys(default_lora, expected_default, "DMD default LoRA 416 tensors")
    exact_keys(turbo_lora, expected_turbo, "DMD turbo LoRA 726 tensors")
    if default_pair_count != 208 or turbo_pair_count != 363:
        raise ValueError(f"LoRA pair count mismatch: default={default_pair_count}, turbo={turbo_pair_count}")
    # Complete production order will be attached only from a meta-model
    # enumeration; JSON/safetensors order is deliberately not trusted.
    all_plans = sorted(plans.values(), key=lambda p: p["model_name"])
    for plan in all_plans:
        match = re.match(r"^(blocks\.\d+)\.", plan["model_name"])
        plan["offload_unit"] = match.group(1) if match else "non_main_block_requires_explicit_policy"
        plan["model_iteration_index"] = None
    return dict(version=2, source="header_only_dmd8", runtime_usable=False,
        missing_gate="real_meta_model_parameter_buffer_order_shape_dtype_and_ownership",
        stage="dmd", expected_actual_dit_forwards=8,
        base_count=len(base), branch_count=len(branch),
        default_lora_tensor_count=len(default_lora), default_lora_pair_count=default_pair_count,
        turbo_lora_tensor_count=len(turbo_lora), turbo_lora_pair_count=turbo_pair_count,
        lora_merge_entry_count=merge_entry_count,
        qkv_conversion_count=52, config=config, plans=all_plans,
        storage_bytes={group: sum(v["numel"] * ITEMSIZE[v["dtype"]] for v in tensors.values())
                       for group, tensors in groups.items()},
        file_provenance=[{k: v for k, v in f.items() if k != "header"} for f in snapshot["files"]])


def attach_model_metadata(manifest, metadata):
    """Validate and attach a real meta-only model enumeration.

    Entries must be gathered as params then buffers for each main block,
    exactly as DLO does; main-block dtype order follows first occurrence.
    Never accept sorted checkpoint order as a runtime layout.
    """
    if metadata.get("device") != "meta" or metadata.get("dit_tp") != 1:
        raise ValueError("Require a real meta DiT with TP=1 metadata")
    entries = metadata["parameters_and_persistent_buffers"]
    by_name = {e["name"]: e for e in entries}
    if len(by_name) != len(entries):
        raise ValueError("Duplicate model metadata name")
    exact_keys(by_name, [p["model_name"] for p in manifest["plans"]], "model/checkpoint coverage")
    for plan in manifest["plans"]:
        entry = by_name[plan["model_name"]]
        if entry["shape"] != plan["shape"] or entry["dtype"] != plan["dtype"]:
            raise ValueError(f"Model/checkpoint mismatch {plan['model_name']}")
    # Metadata attachment alone is not full runtime integration: pinned
    # buffers, non-persistent buffers, and non-block lifecycles remain gates.
    result = json.loads(json.dumps(manifest))
    order = {e["name"]: i for i, e in enumerate(entries)}
    for plan in result["plans"]:
        plan["model_iteration_index"] = order[plan["model_name"]]
    result["model_metadata_validated"] = True
    return result


def tensor_plans_for_main_block(manifest, block_index, reader_factory):
    """Resolve one verified main-block manifest to streaming TensorPlans.

    reader_factory(path) must return/cached-open the read-only RangeReader.
    This function itself never determines production parameter order.
    """
    if not manifest.get("model_metadata_validated") or not 0 <= block_index < 50:
        raise ValueError("Require real meta-model validation and a valid main block")
    from streaming_shards import LoRA, TensorPlan, TensorRef

    def ref(entry):
        reader = reader_factory(entry["source_file"])
        actual = reader.header[entry["source_key"]]
        if actual["shape"] != entry["shape"] or actual["dtype"] != entry["dtype"]:
            raise ValueError("Checkpoint header changed since manifest creation")
        return TensorRef(reader, entry["source_key"])

    selected = [p for p in manifest["plans"] if p["offload_unit"] == f"blocks.{block_index}"]
    if len(selected) != 26 or any(p["model_iteration_index"] is None for p in selected):
        raise ValueError("Incomplete block or unverified order")
    selected.sort(key=lambda p: p["model_iteration_index"])
    return [TensorPlan(p["model_name"], ref(p), p.get("qkv_heads"), p.get("head_dim"),
                       tuple(LoRA(x["start_row"], x["end_row"], ref(x["a"]), ref(x["b"]),
                                  x.get("b_row_offset", 0))
                             for x in p["lora_pairs"])) for p in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-readonly", action="store_true")
    parser.add_argument("--collect-local", action="store_true")
    parser.add_argument("--ssh-config", default="C:/Users/DZH/.ssh/config")
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).with_name("real_header_snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("real_manifest.json"))
    args = parser.parse_args()
    if args.collect_readonly and args.collect_local:
        raise ValueError("Choose only one collection mode")
    if args.collect_readonly:
        if args.snapshot.exists():
            raise FileExistsError(args.snapshot)
        snapshot = collect_headers(args.ssh_config)
        args.snapshot.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    elif args.collect_local:
        if args.snapshot.exists():
            raise FileExistsError(args.snapshot)
        snapshot = collect_local_headers()
        args.snapshot.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    else:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    manifest = audit(snapshot)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("plans", "config", "file_provenance")}, indent=2))
    print("snapshot_sha256=" + hashlib.sha256(args.snapshot.read_bytes()).hexdigest())
    print("manifest_sha256=" + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
