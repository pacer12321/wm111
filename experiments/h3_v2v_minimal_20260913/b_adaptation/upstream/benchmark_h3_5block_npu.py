#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
    parse_attention_config,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence


SOURCE_BLOCKS = (0, 12, 24, 37, 49)


def init_dist() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def build_model(
    model_dir: Path,
    *,
    openvdn: bool = False,
    groups_per_call: int = 4,
    interior_group_size: int = 0,
    group_impl: str = "varlen",
) -> tuple[MiniMaxH3DiTModel, OmniDiffusionConfig]:
    with (model_dir / "config.json").open() as f:
        raw_config = json.load(f)
    raw_config["num_layers"] = len(SOURCE_BLOCKS)
    raw_config["openvdn_enabled"] = openvdn
    raw_config["openvdn_num_frames"] = 102
    raw_config["openvdn_frame_height"] = 24
    raw_config["openvdn_frame_width"] = 42
    raw_config["openvdn_groups_per_call"] = groups_per_call
    raw_config["openvdn_interior_group_size"] = interior_group_size
    raw_config["openvdn_group_impl"] = group_impl
    parallel = DiffusionParallelConfig(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    od_config = OmniDiffusionConfig(
        model=str(model_dir),
        model_class_name="MiniMaxH3DiTModel",
        dtype=torch.bfloat16,
        tf_model_config=TransformerConfig.from_dict(raw_config),
        parallel_config=parallel,
        diffusion_attention_config=parse_attention_config(None, attention_backend="FLASH_ATTN"),
    )
    with set_current_diffusion_config(od_config):
        model = MiniMaxH3DiTModel(od_config=od_config)
    return model, od_config


def load_selected_weights(model: MiniMaxH3DiTModel, model_dir: Path) -> set[str]:
    with (model_dir / "model.safetensors.index.json").open() as f:
        weight_map = json.load(f)["weight_map"]
    old_to_new = {old: new for new, old in enumerate(SOURCE_BLOCKS)}
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    block_pattern = re.compile(r"^blocks\.(\d+)\.(.+)$")
    for source_name, shard in weight_map.items():
        match = block_pattern.match(source_name)
        if match:
            old_idx = int(match.group(1))
            if old_idx not in old_to_new:
                continue
            target_name = f"blocks.{old_to_new[old_idx]}.{match.group(2)}"
        else:
            target_name = source_name
        by_shard[shard].append((source_name, target_name))

    loaded: set[str] = set()
    for shard in sorted(by_shard):
        print(f"loading {shard}: {len(by_shard[shard])} tensors", flush=True)
        tensors = []
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            for source_name, target_name in by_shard[shard]:
                tensors.append((target_name, f.get_tensor(source_name)))
        loaded.update(model.load_weights(tensors))
        del tensors
        gc.collect()

    print(f"loaded {len(loaded)} tensors; source blocks={SOURCE_BLOCKS}", flush=True)
    return loaded


def load_openvdn_branch_weights(model: MiniMaxH3DiTModel, path: Path) -> set[str]:
    old_to_new = {old: new for new, old in enumerate(SOURCE_BLOCKS)}
    pattern = re.compile(r"^transformer_blocks\.(\d+)\.attn\.(.+)$")
    weights = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for source_name in f.keys():
            match = pattern.match(source_name)
            if match is None:
                continue
            old_idx = int(match.group(1))
            if old_idx not in old_to_new:
                continue
            target_name = f"blocks.{old_to_new[old_idx]}.attn.{match.group(2)}"
            weights.append((target_name, f.get_tensor(source_name)))
    loaded = model.load_weights(weights)
    print(f"loaded {len(loaded)} OpenVDN branch tensors from {path}", flush=True)
    return loaded


def _lora_delta(f, prefix: str, device: torch.device) -> torch.Tensor:
    a = f.get_tensor(prefix + ".lora_A.default.weight").to(device)
    b = f.get_tensor(prefix + ".lora_B.default.weight").to(device)
    return torch.matmul(b, a)


def fold_openvdn_lora(model: MiniMaxH3DiTModel, path: Path, device: torch.device) -> None:
    old_to_new = {old: new for new, old in enumerate(SOURCE_BLOCKS)}
    with safe_open(path, framework="pt", device="cpu") as f, torch.inference_mode():
        for old_idx, new_idx in old_to_new.items():
            attn = model.blocks[new_idx].attn
            rows = attn.total_num_heads * attn.head_dim
            for offset, name in enumerate(("to_q", "to_k", "to_v")):
                prefix = f"transformer_blocks.{old_idx}.attn.orig.{name}"
                attn.qkv_proj.weight[offset * rows : (offset + 1) * rows].add_(
                    _lora_delta(f, prefix, device).to(attn.qkv_proj.weight.dtype)
                )
            prefix = f"transformer_blocks.{old_idx}.attn.orig.to_out.0"
            attn.out_proj.weight.add_(
                _lora_delta(f, prefix, device).to(attn.out_proj.weight.dtype)
            )

        for block_idx, block in enumerate(model.token_refiner.blocks):
            attn = block.attn
            rows = attn.total_num_heads * attn.head_dim
            for offset, name in enumerate(("to_q", "to_k", "to_v")):
                prefix = f"token_refiner.refiner_blocks.{block_idx}.attn.{name}"
                attn.qkv_proj.weight[offset * rows : (offset + 1) * rows].add_(
                    _lora_delta(f, prefix, device).to(attn.qkv_proj.weight.dtype)
                )
            prefix = f"token_refiner.refiner_blocks.{block_idx}.attn.to_out.0"
            attn.out_proj.weight.add_(
                _lora_delta(f, prefix, device).to(attn.out_proj.weight.dtype)
            )
    synchronize()
    print(f"folded OpenVDN rank-64 LoRA from {path}", flush=True)


def make_inputs(device: torch.device, text_len: int):
    latent_t, latent_h, latent_w, audio_t = 102, 48, 84, 575
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(1234)
    text = torch.randn(text_len, 5120, dtype=torch.bfloat16, generator=generator)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=text,
        token_tags=packed["token_tags"],
        device=device,
    )
    video_rows = torch.randn(
        packed["img_pos"].numel(), 96, dtype=torch.float32, generator=generator
    ).to(device)
    audio_rows = torch.randn(
        packed["audio_pos"].numel(), 32, dtype=torch.float32, generator=generator
    ).to(device)
    kwargs = branch.forward_kwargs(
        video_rows=video_rows,
        audio_rows=audio_rows,
        t_video=0.0,
        t_audio=0.0,
        imgvid_cond_timestep=0.999,
        audio_ref_cond_timestep=1.0,
    )
    return kwargs, int(packed["seq_len"]), int(packed["img_pos"].numel())


def synchronize() -> None:
    torch.npu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/cache/yunfeng/models/MiniMax-H3/FL2VA/transformer")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    # 98 makes text + 2*575 audio + 102*24*42 video rows exactly 64-aligned.
    # The NPU backend otherwise materializes a dense QK padding mask (~40 GiB).
    parser.add_argument("--text-len", type=int, default=98)
    parser.add_argument("--openvdn", action="store_true")
    parser.add_argument("--openvdn-weights")
    parser.add_argument("--openvdn-lora")
    parser.add_argument("--groups-per-call", type=int, default=4)
    parser.add_argument("--interior-group-size", type=int, default=0)
    parser.add_argument("--group-impl", choices=("varlen", "loop", "batch"), default="varlen")
    parser.add_argument("--disable-openvdn-linear", action="store_true")
    parser.add_argument(
        "--openvdn-delta-rule",
        choices=("vdn_solve", "sana_scaled", "relu_linear"),
        default="vdn_solve",
    )
    args = parser.parse_args()

    init_dist()
    device = torch.device("npu:0")
    model, _ = build_model(
        Path(args.model_dir),
        openvdn=args.openvdn,
        groups_per_call=args.groups_per_call,
        interior_group_size=args.interior_group_size,
        group_impl=args.group_impl,
    )
    loaded = load_selected_weights(model, Path(args.model_dir))
    if args.openvdn:
        if not args.openvdn_weights or not args.openvdn_lora:
            raise ValueError("--openvdn requires --openvdn-weights and --openvdn-lora")
        loaded.update(load_openvdn_branch_weights(model, Path(args.openvdn_weights)))
    expected = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"missing {len(missing)} tensors; first entries: {missing[:20]}")
    model.post_load_weights()
    model.eval().to(device)
    if args.openvdn:
        fold_openvdn_lora(model, Path(args.openvdn_lora), device)
        for block in model.blocks:
            block.attn.linear_attention.delta_rule = args.openvdn_delta_rule
        if args.disable_openvdn_linear:
            for block in model.blocks:
                block.attn.openvdn_linear_enabled = False
    kwargs, seq_len, video_rows = make_inputs(device, args.text_len)
    torch.npu.reset_peak_memory_stats(device)
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(0),
                "frames": 345,
                "latent_shape": [102, 48, 84],
                "audio_t": 575,
                "text_len": args.text_len,
                "packed_seq_len": seq_len,
                "video_rows": video_rows,
                "source_blocks": list(SOURCE_BLOCKS),
                "openvdn": args.openvdn,
                "openvdn_linear": args.openvdn and not args.disable_openvdn_linear,
                "openvdn_delta_rule": args.openvdn_delta_rule if args.openvdn else None,
                "groups_per_call": args.groups_per_call if args.openvdn else None,
                "interior_group_size": args.interior_group_size if args.openvdn else None,
                "group_impl": args.group_impl if args.interior_group_size else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with torch.inference_mode():
        for index in range(args.warmup):
            start = time.perf_counter()
            model(**kwargs)
            synchronize()
            print(f"warmup[{index}]={time.perf_counter() - start:.6f}s", flush=True)

        elapsed = []
        for index in range(args.repeat):
            synchronize()
            start = time.perf_counter()
            model(**kwargs)
            synchronize()
            duration = time.perf_counter() - start
            elapsed.append(duration)
            print(f"measure[{index}]={duration:.6f}s", flush=True)

    result = {
        "seconds_per_nfe": elapsed,
        "mean_seconds": sum(elapsed) / len(elapsed),
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "estimated_8step_dit_seconds": 8 * sum(elapsed) / len(elapsed),
        "peak_allocated_gib": torch.npu.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.npu.max_memory_reserved(device) / 2**30,
    }
    print("RESULT=" + json.dumps(result), flush=True)
    destroy_distributed_env()


if __name__ == "__main__":
    main()
