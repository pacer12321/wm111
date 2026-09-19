"""Four-GPU Ref2VA dual-stream training entrypoint.

The smoke mode intentionally uses a tiny synthetic latent geometry while loading the
real Ref2VA base, VDN branch and both official DMD8 adapters.  It validates the exact
production path: attention LoRA plus small VDN trainables, FSDP2 CPU-offloaded
masters, dual-stream Flex
mask, forward, backward and AdamW update.  Dataset mode is added on the same packing
path; the smoke never substitutes a small model or a different attention graph.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file

from src.inference.utils.lora import merge_lora_state
from src.models.factory import build_model, load_model_weights
from src.models.hybrid_transform import (install_token_skip_training, set_layout,
                                         set_softmax_backend, set_token_skip)
from src.training import fsdp_stage as fs
from src.training.ref2va_batch import (latent_selector_mask, pack_ref2va_batch,
                                       unpatchify_video_rows)
from src.training.t2va_batch import x0_from_velocity
from src.training.shared_clip_contract import audio_latent_count, merge_encoded_pair


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--dmd8", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32-masters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument(
        "--train-token-skip", action=argparse.BooleanOptionalAction, default=True,
        help=("Enable the exact latent-selector token-skip path during training. "
              "Disable for connectivity training, then enable the validated selector "
              "at inference or during a short skip-aware calibration stage."),
    )
    parser.add_argument(
        "--shard-size", type=int, default=0,
        help=("FSDP shard-group size. 0 uses the whole world; with world=4 and "
              "shard-size=2, FSDP2 builds a 2-replica x 2-shard HSDP mesh."),
    )
    parser.add_argument("--synthetic-t", type=int, default=2)
    parser.add_argument("--synthetic-h", type=int, default=8)
    parser.add_argument("--synthetic-w", type=int, default=8)
    parser.add_argument(
        "--sample-dir", default=None,
        help="Growing directory containing sample_XXXXXX.pt + .done streaming pairs.",
    )
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--sample-wait-timeout", type=float, default=300.0)
    parser.add_argument("--audio-input-policy", choices=["fixed-silent"], default=None,
        help="Requires explicit approval and matching inference policy; no audio objective.")
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def configure_savie_lora(model, rank, alpha):
    """Freeze the DMD8 trunk and train only SAViE LoRA plus the VDN branch.

    DMD8's released adapters have already been folded into the base weights by
    ``load_dmd8_model``.  This is a new adapter, restricted to the 50 DiT attention
    projections; the token refiner, FFN, VAE and text encoder remain frozen.
    """
    model.requires_grad_(False)
    target = (
        r"transformer_blocks\.\d+\.attn\."
        r"(orig\.(to_q|to_k|to_v|to_out\.0)|to_out_linear)"
    )
    model = inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=target,
        ),
        model,
        adapter_name="savie",
    )
    # Keep only inexpensive, structurally specific VDN parameters fully trainable.
    # The two large readout/projection families stay frozen or receive LoRA above.
    small_vdn = (
        ".attn.linear_attention.short_conv.",
        ".attn.linear_attention.alpha.A_log",
        ".attn.linear_attention.alpha.dt_bias",
        ".attn.linear_attention.beta_proj.",
        ".attn.linear_attention.output_gate.",
        ".attn.linear_attention.norm.",
        ".attn.softmax_gate.",
    )
    for name, parameter in model.named_parameters():
        if any(fragment in name for fragment in small_vdn):
            parameter.requires_grad_(True)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable)
    forbidden = [
        name for name, _ in trainable
        if (".ff." in name or ".adaln_proj." in name
            or (".attn.orig." in name and ".lora_" not in name)
            or name.endswith(".attn.to_out_linear.base_layer.weight"))
    ]
    if forbidden:
        raise RuntimeError(f"LoRA whitelist leaked {len(forbidden)} full trunk weights: "
                           f"{forbidden[:8]}")
    if trainable_count > 500_000_000:
        raise RuntimeError(
            f"LoRA scope unexpectedly exposes {trainable_count/1e9:.3f}B parameters; "
            "hard limit is 0.500B"
        )
    return model


def load_dmd8_model(base, checkpoint, rank):
    # ``base`` is the complete Ref2VA Diffusers pipeline. The checkpoint ModelSpec
    # deliberately owns the ``transformer`` subfolder contract, so do not append it
    # here (doing so would bypass or duplicate that contract).
    with open(os.path.join(checkpoint, "model_spec.json"), encoding="utf-8") as handle:
        spec = json.load(handle)
    model = build_model(spec, device="cpu", base_source=base)
    meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta:
        raise RuntimeError(
            f"base checkpoint left {len(meta)} parameters on meta device; "
            f"first entries: {meta[:8]}"
        )
    branch = load_file(os.path.join(checkpoint, "linear_branch", "model.safetensors"),
                       device="cpu")
    loaded = load_model_weights(model, branch)
    del branch
    merged = []
    for name in ("default", "turbo"):
        path = os.path.join(checkpoint, "adapters", name, "adapter_model.safetensors")
        state = load_file(path, device="cpu")
        merged.append((name, merge_lora_state(model, state, scale=1.0)))
        del state
    meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta:
        raise RuntimeError(
            f"DMD8 assembly left {len(meta)} parameters on meta device; "
            f"first entries: {meta[:8]}"
        )
    if rank == 0:
        print(f"DMD8 initialization: {loaded} VDN tensors; adapters {merged}", flush=True)
    return model, spec


def synthetic_sample(args):
    generator = torch.Generator().manual_seed(args.seed)
    shape = (24, args.synthetic_t, args.synthetic_h, args.synthetic_w)
    source = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    # A small, deterministic non-identity target makes the target-only loss meaningful.
    target = (source.float() * 0.95 + 0.05).to(torch.bfloat16)
    return {
        "source_video_latents": source,
        "target_video_latents": target,
        "target_audio_latents": torch.randn((2, 32, 2), generator=generator,
                                              dtype=torch.bfloat16),
        "prompt_embeds": torch.randn((8, 5120), generator=generator,
                                      dtype=torch.bfloat16),
        "text_token_tags": torch.ones(8, dtype=torch.long),
    }


def stream_sample(args, sample_index, rank):
    """Wait for one atomically committed precomputed Ref2VA training sample."""
    video_stem = os.path.join(args.sample_dir, f"video_{sample_index:06d}")
    prompt_stem = os.path.join(args.sample_dir, f"prompt_{sample_index:06d}")
    video_payload, video_done = video_stem + ".pt", video_stem + ".done"
    prompt_payload, prompt_done = prompt_stem + ".pt", prompt_stem + ".done"
    started = time.time()
    announced = False
    while not (os.path.isfile(video_done) and os.path.isfile(prompt_done)):
        if time.time() - started > args.sample_wait_timeout:
            raise TimeoutError(
                f"timed out waiting for {video_done} and {prompt_done}"
            )
        if rank == 0 and not announced:
            print(
                f"waiting for streaming sample {sample_index}: "
                f"video={os.path.isfile(video_done)} prompt={os.path.isfile(prompt_done)}",
                flush=True,
            )
            announced = True
        time.sleep(2.0)
    video = torch.load(video_payload, map_location="cpu", weights_only=True)
    prompt = torch.load(prompt_payload, map_location="cpu", weights_only=True)
    sample = merge_encoded_pair(video, prompt)
    if args.audio_input_policy != "fixed-silent":
        raise ValueError("real-data audio input policy has not been explicitly selected")
    audio_t = audio_latent_count(sample["frame_count"], sample["video_fps"])
    sample["target_audio_latents"] = torch.zeros((2, 32, audio_t), dtype=torch.bfloat16)
    sample["audio_input_policy"] = args.audio_input_policy
    required = {
        "source_video_latents", "target_video_latents", "target_audio_latents",
        "prompt_embeds", "text_token_tags",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise RuntimeError(f"stream sample {sample_index} missing keys: {missing}")
    return sample


def tensor_fingerprint(tensor):
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def check_input_identity(batch, sample, world, shard_size):
    # Sums/squared sums can agree even if token order/tags differ. Compare exact
    # bytes, shapes, dtype AND semantic metadata on the first two training steps.
    signature = {k: tensor_fingerprint(v) for k, v in batch["inputs"].items()
                 if isinstance(v, torch.Tensor)}
    signature.update(sample_id=sample.get("sample_id", "synthetic"),
        clip_contract_sha256=sample.get("clip_contract_sha256", "synthetic"),
        audio_input_policy=sample.get("audio_input_policy", "synthetic"),
        target_video_rows=tensor_fingerprint(batch["target_video_rows"]))
    signatures = [None] * world
    dist.all_gather_object(signatures, signature)
    for leader in range(0, world, shard_size):
        if any(value != signatures[leader] for value in signatures[leader:leader+shard_size]):
            raise RuntimeError(f"input/label/noise/index mismatch inside shard group {leader//shard_size}")
    replicas = signatures[::shard_size]
    if len({s["hidden_states"] for s in replicas}) != len(replicas):
        raise RuntimeError("different DP replicas unexpectedly have identical inputs/noise")
    return signatures


def require_finite_on_all_ranks(value, name, device):
    ok = torch.isfinite(value.detach()).all().to(device=device, dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not bool(ok.item()):
        raise FloatingPointError(f"{name} is non-finite on at least one rank; optimizer not updated")


def _synchronize_replica_mask(mask, world, shard_size):
    """Use one deterministic selector mask inside each FSDP shard group.

    Shard ranks execute the same sample, but tiny floating-point differences in the
    selector bootstrap can move boundary tokens across the two-means threshold.  All
    ranks must still take identical sparse-control-flow paths.  The first rank of each
    shard group is therefore authoritative.
    """
    gathered = [torch.empty_like(mask, dtype=torch.uint8) for _ in range(world)]
    dist.all_gather(gathered, mask.to(torch.uint8))
    leader = (dist.get_rank() // shard_size) * shard_size
    reference = gathered[leader].to(torch.bool)
    disagreement = int(torch.count_nonzero(mask ^ reference))
    return reference, disagreement


def _fuse_stable_velocity(pred_target_v, batch, active_target_mask):
    """Exact x0-space cosine fusion used by the validated fastest inference path."""
    if active_target_mask is None:
        return pred_target_v, 1.0
    stable = ~active_target_mask
    if not bool(stable.any()):
        return pred_target_v, 1.0
    sigma = 1.0 - float(batch["t_v"])
    if sigma <= 0:
        raise ValueError("latent selector fusion requires positive sigma")
    current = batch["target_noisy_rows"].float()
    target_x0 = current + sigma * pred_target_v
    progress = torch.as_tensor(batch["t_v"], device=current.device,
                               dtype=torch.float32).clamp(0.0, 1.0)
    alpha = torch.cos(progress * (math.pi / 2.0)).square()
    fused_x0 = alpha * target_x0 + (1.0 - alpha) * batch["source_clean_rows"].float()
    fused_velocity = (fused_x0 - current) / sigma
    output = pred_target_v.clone()
    output[stable] = fused_velocity[stable].to(output.dtype)
    return output, float(alpha)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.sample_dir is not None and args.audio_input_policy is None:
        raise ValueError("select and verify the SAME audio input policy for training and inference first")
    # A fresh retrain must never silently pick up bad-tag weights or reset Adam
    # moments while pretending a weights-only artifact is an exact continuation.
    if list(Path(args.output).glob("savie_step*.pt")) or list(Path(args.output).glob("train_state_step*.pt")):
        raise RuntimeError("fresh DMD8 retrain requires a new output directory; no implicit resume")
    torch.manual_seed(args.seed)
    shard_size = args.shard_size or world
    if world % shard_size:
        raise ValueError(f"world size {world} is not divisible by shard size {shard_size}")
    replica_id = rank // shard_size
    shard_rank = rank % shard_size

    t0 = time.time()
    model, spec = load_dmd8_model(args.base, args.dmd8, rank)
    model = configure_savie_lora(model, args.lora_rank, args.lora_alpha)
    install_token_skip_training(model)
    if args.fp32_masters:
        # Only trainables need fp32 masters.  The frozen 33B DMD8 trunk stays bf16.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    resume = fs.Resume()
    gc.collect()
    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        print(f"real H3 model loaded: {params/1e9:.3f}B total, "
              f"{trainable_params/1e9:.3f}B trainable; "
              f"fp32_masters={args.fp32_masters}; load={time.time()-t0:.1f}s", flush=True)
        print(f"trainable whitelist sample: {trainable_names[:16]}", flush=True)

    set_softmax_backend(model, "flex")
    model = fs.shard_model(
        model, world, shard_size, device, rank,
        activation_checkpointing=args.activation_checkpointing,
        cpu_offload=args.cpu_offload,
        shard_root=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    # Every rank in one FSDP shard group must execute the same sample/noise path.
    # Different HSDP replicas receive different streams and FSDP2 reduces their
    # gradients across the replicate mesh on every backward.
    noise_generator, timestep_generator = fs.make_generators(
        args.seed, replica_id, device
    )
    replicas = world // shard_size
    if args.sample_dir is not None and args.max_steps * replicas > args.sample_count:
        raise ValueError(
            f"{args.max_steps} steps x {replicas} replicas exceeds "
            f"sample_count={args.sample_count}"
        )
    sample = None if args.sample_dir is not None else synthetic_sample(args)

    model.train()
    rows = []
    for step in range(resume.start_step, args.max_steps):
        step_t0 = time.time()
        selector_mask_disagreement = 0
        selector_raw_active_ratio = None
        sample_index = step * replicas + replica_id
        if args.sample_dir is not None:
            sample = stream_sample(args, sample_index, rank)
        batch = pack_ref2va_batch(
            sample, device, noise_generator, timestep_generator, num_steps=8,
        )
        if step < 2 and world > 1:
            signatures = check_input_identity(batch, sample, world, shard_size)
            if rank == 0:
                print(json.dumps(dict(test="hsdp_exact_inputs", passed=True,
                    step=step, signatures=signatures)), flush=True)
        set_layout(model, batch["layout"])
        # Exact validated latent selector: the first DMD8 x0 prediction is compared
        # with the clean source, split by exact two-means and spatially dilated r=1.
        # The resulting mask stays fixed. For sampled skip steps, reconstruct the
        # appropriate refresh cache(s) with no graph before the one gradient forward.
        refresh_indices = (0, 4)
        if not args.train_token_skip:
            skip_stats = set_token_skip(model, mode="refresh_all")
            active_target_mask = None
            selector_score = None
            selector_threshold = None
        elif batch["step_index"] in refresh_indices:
            skip_stats = set_token_skip(model, mode="refresh_all")
            active_target_mask = None
            selector_score = None
            selector_threshold = None
        else:
            first_batch = pack_ref2va_batch(
                sample, device, noise_generator, timestep_generator, num_steps=8,
                step_index=0, noise_bundle=batch["noise_bundle"],
            )
            set_layout(model, first_batch["layout"])
            set_token_skip(model, mode="refresh_all")
            with torch.no_grad():
                first_velocity_v, _ = fs.student_forward(model, first_batch["inputs"], True)
                first_pred_target = first_velocity_v[0][first_batch["source_rows"]:].float()
                first_x0_rows = x0_from_velocity(
                    first_batch["target_noisy_rows"].float(), first_pred_target,
                    first_batch["t_v"],
                )
                shape = tuple(first_batch["source_clean_latents"].shape)
                first_x0 = unpatchify_video_rows(first_x0_rows, shape)
                active_target_mask, selector_score, selector_threshold = latent_selector_mask(
                    first_batch["source_clean_latents"], first_x0
                )
                selector_raw_active_ratio = float(
                    (selector_score >= selector_threshold).float().mean()
                )
                active_target_mask, selector_mask_disagreement = (
                    _synchronize_replica_mask(active_target_mask, world, shard_size)
                )
            if batch["step_index"] >= 5:
                refresh_batch = pack_ref2va_batch(
                    sample, device, noise_generator, timestep_generator, num_steps=8,
                    step_index=4, noise_bundle=batch["noise_bundle"],
                )
                set_layout(model, refresh_batch["layout"])
                set_token_skip(model, active_target_mask, mode="refresh")
                with torch.no_grad():
                    fs.student_forward(model, refresh_batch["inputs"], True)
            set_layout(model, batch["layout"])
            skip_stats = set_token_skip(
                model, active_target_mask, mode="skip"
            )
        velocity_v, velocity_a = fs.student_forward(model, batch["inputs"], True)
        first_block = model.transformer_blocks[0]
        expected_query_rows = (
            batch["layout"].seq_len - skip_stats.get("stable_target_rows", 0)
            if skip_stats["mode"] == "skip" else batch["layout"].seq_len
        )
        work_gate = {
            "total_rows": int(first_block._last_total_rows),
            "query_rows": int(first_block.attn._last_query_rows),
            "kv_rows": int(first_block.attn._last_kv_rows),
            "out_proj_rows": int(first_block.attn._last_out_proj_rows),
            "ff_rows": int(first_block._last_ff_rows),
        }
        if not (
            work_gate["total_rows"] == batch["layout"].seq_len
            and work_gate["kv_rows"] == batch["layout"].seq_len
            and work_gate["query_rows"] == expected_query_rows
            and work_gate["out_proj_rows"] == expected_query_rows
            and work_gate["ff_rows"] == expected_query_rows
        ):
            raise RuntimeError(
                f"token-skip work gate failed: expected active rows {expected_query_rows}, "
                f"got {work_gate}"
            )
        pred_target_v = velocity_v[0][batch["source_rows"]:].float()
        pred_target_v, fusion_alpha = _fuse_stable_velocity(
            pred_target_v, batch, active_target_mask
        )
        pred_target_a = velocity_a[0].float()
        if skip_stats["mode"] == "skip":
            active = active_target_mask
            video_loss = torch.nn.functional.mse_loss(
                pred_target_v[active], batch["target_video_rows"][active]
            )
        else:
            video_loss = torch.nn.functional.mse_loss(
                pred_target_v, batch["target_video_rows"]
            )
        # SAViE is trained for video editing. Audio is retained as a frozen silent
        # structural stream but is deliberately excluded from the objective.
        audio_loss = torch.zeros((), device=device)
        loss = video_loss
        require_finite_on_all_ranks(loss, "loss", device)
        loss.backward()
        grad_norm = fs.clip_gradients(trainable, 1.0)
        require_finite_on_all_ranks(grad_norm, "gradient norm", device)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        elapsed = time.time() - step_t0
        row = {
            "step": step + 1,
            "loss": float(loss),
            "video_loss": float(video_loss.detach()),
            "audio_loss": float(audio_loss.detach()),
            "grad_norm": float(grad_norm),
            "dmd8_timestep_index": batch["step_index"],
            "seconds": elapsed,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "seq_len": batch["layout"].seq_len,
            "sample_id": sample.get("sample_id"),
            "clip_contract_sha256": sample.get("clip_contract_sha256"),
            "audio_input_policy": sample.get("audio_input_policy", "synthetic"),
            "audio_objective": False,
            "token_skip": skip_stats,
            "selector": {
                "kind": "first-x0-vs-source-channel-normalized-two-means-r1",
                "threshold": (None if selector_threshold is None
                              else float(selector_threshold)),
                "score_mean": (None if selector_score is None
                               else float(selector_score.float().mean())),
                "raw_active_ratio_before_dilation": (
                    None if selector_score is None else selector_raw_active_ratio
                ),
                "active_ratio_after_dilation": (
                    None if active_target_mask is None
                    else float(active_target_mask.float().mean())
                ),
                "fusion_alpha": fusion_alpha,
                "replica_mask_disagreement": selector_mask_disagreement,
            },
            "token_skip_work_gate": work_gate,
        }
        rows.append(row)
        if rank == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        torch.cuda.reset_peak_memory_stats()
        if args.save_every > 0 and ((step + 1) % args.save_every == 0
                                    or step + 1 == args.max_steps):
            fs.save_weights_artifact(
                model, args.output, f"savie_step{step + 1:06d}.pt", "SAViE",
                step + 1, rank, spec, lambda _name, parameter: parameter.requires_grad,
                metadata={
                    "task_base": "MiniMax-H3 Ref2VA",
                    "acceleration_checkpoint": "OpenVDN stage-dmd-step-250 DMD8",
                    "token_skip": ("validated latent selector" if args.train_token_skip
                                   else "disabled during connectivity training"),
                    "sample_count_consumed": (step + 1) * replicas,
                    "audio_input_policy": args.audio_input_policy,
                    "preprocessing_schema": sample.get("preprocessing_schema", "synthetic"),
                },
            )

    if rank == 0:
        receipt = {
            "status": "ok",
            "mode": ("real-data-stream-training" if args.sample_dir is not None
                     else "real-model-synthetic-data-smoke"),
            "task_base": "MiniMax-H3 Ref2VA",
            "acceleration_checkpoint": "OpenVDN stage-dmd-step-250 DMD8",
            "world_size": world,
            "hsdp": {
                "shard_size": shard_size,
                "replicas": world // shard_size,
                "gradient_sync": "FSDP2 replicate-mesh reduction every backward",
            },
            "base": os.path.abspath(args.base),
            "dmd8": os.path.abspath(args.dmd8),
            "architecture": {
                "tt": "VDN local softmax + VDN linear",
                "ts": "corresponding source latent frame",
                "ss": "VDN local softmax, no linear",
                "st": "removed",
                "token_skip": (
                    "latent selector; refresh forwards 1/5; stable target Q, "
                    "attention out-proj and FFN skipped on other forwards"
                ),
            },
            "training_scope": {
                "mode": "attention-lora-plus-vdn-branch",
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "trainable_parameters": sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
            },
            "rows": rows,
            "model_spec": spec,
        }
        with open(os.path.join(args.output, "smoke_receipt.json"), "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print("REF2VA_DUAL_TRAIN_SMOKE_OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
