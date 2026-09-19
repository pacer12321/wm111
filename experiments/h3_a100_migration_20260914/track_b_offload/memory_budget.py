"""Header-derived CPU/GPU budget; explicit unknowns, not a fit guarantee."""
import argparse
import json
from pathlib import Path
import re

GIB = 1024**3


def encoder_budget(entries):
    sharded = replicated_vision = replicated_norm = 0
    projections = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
                   ".self_attn.o_proj.weight", ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")
    for entry in entries:
        name, size = entry["name"], entry["numel"] * 2  # runtime encoder.to(BF16)
        if name.startswith("model.visual."):
            replicated_vision += size
        elif name.startswith("model.language_model.embed_tokens."):
            sharded += size
        elif name.startswith("model.language_model.layers.") and int(name.split(".")[3]) < 50:
            if name.endswith(projections):
                sharded += size
            elif name.endswith(("norm.weight", "layernorm.weight")):
                replicated_norm += size
            else:
                raise ValueError("Unclassified retained text parameter: " + name)
    if sharded % 2:
        raise ValueError("TP2 payload not divisible by two")
    return dict(full_shardable_bytes=sharded, vision_replicated_bytes=replicated_vision,
                norm_replicated_bytes=replicated_norm,
                per_rank_bytes=sharded // 2 + replicated_vision + replicated_norm)


def budget(manifest, aux, *, baseline_bytes=21_000_000_000, video_summary=None):
    blocks, nonblock = {}, 0
    for p in manifest["plans"]:
        size = p["numel"] * {"BF16": 2, "F32": 4}[p["dtype"]]
        if re.fullmatch(r"blocks\.\d+", p["offload_unit"]):
            blocks[p["offload_unit"]] = blocks.get(p["offload_unit"], 0) + size
        else:
            nonblock += size
    if len(blocks) != 50 or any(size % 2 for size in blocks.values()):
        raise ValueError("Unexpected main-block shard layout")
    main = sum(blocks.values())
    largest = max(blocks.values())
    gpu_buffers = 2 * largest + 2 * (largest // 2)
    encoder = encoder_budget(aux["components"]["text_encoder"])
    audio = sum(p["numel"] * 4 for p in aux["components"]["audio_vae"])
    video_entries = aux["components"]["video_vae"]
    video_exact = sum(p["numel"] * 4 for p in video_entries) if video_entries else None
    if video_summary is not None:
        if video_summary["dtype"] != "F32" or video_summary["numel"] * 4 != video_summary["runtime_fp32_bytes"]:
            raise ValueError("Invalid verified video VAE summary")
        if video_summary["runtime_fp32_bytes"] + video_summary["header_size"] + 8 != video_summary["file_size"]:
            raise ValueError("Video VAE header/payload extent mismatch")
        video_exact = video_summary["runtime_fp32_bytes"]
    known_video_file_size = 10_415_548_320  # read-only stat, excludes no header yet
    video_cases = {"verified_fp32_runtime_payload": video_exact} if video_exact is not None else {
        "UNVERIFIED_if_checkpoint_F32_approx_includes_header": known_video_file_size,
        "UNVERIFIED_if_checkpoint_16bit_approx_includes_header": known_video_file_size * 2,
    }
    cases = {}
    for label, video in video_cases.items():
        vae = video + audio
        # CPU upper-residency envelope retains both nonblock, TE and VAE
        # CPU copies even if a phase actually moves some exclusively to GPU.
        cpu_model = main + 2 * nonblock + 2 * encoder["per_rank_bytes"] + 2 * vae
        gpu_denoise = gpu_buffers + nonblock + vae
        cases[label] = dict(
            video_vae_bytes_per_rank=video,
            cpu_model_payload_both_ranks=cpu_model,
            cpu_payload_plus_provided_baseline=cpu_model + baseline_bytes,
            cpu_hard_limit_headroom_before_unknowns=250_999_996_416 - cpu_model - baseline_bytes,
            gpu_per_rank_static_TE_offloaded=gpu_denoise,
            gpu_per_rank_static_TE_resident=gpu_denoise + encoder["per_rank_bytes"],
            gpu0_headroom_TE_resident_before_unknowns=80 * GIB - gpu_denoise - encoder["per_rank_bytes"] - 850 * 1024**2,
        )
    return dict(
        status="budget_gate_incomplete" if video_exact is None else "static_payload_budget_only_not_fit_proof",
        units="bytes (GB=1e9; A100 80 GiB=85899345920 bytes)",
        baseline=dict(cpu_bytes=baseline_bytes, provenance="root supplied latest approx 21GB; not live measured by this script",
                      gpu0_old_residual_bytes=850*1024**2, gpu1_old_residual_bytes=0,
                      cgroup_limit_bytes=250_999_996_416, gpu_total_bytes_each=80*GIB),
        dit=dict(main_blocks_bytes=main, main_pinned_shard_bytes_each_rank=main//2,
                 non_main_bytes_each_rank=nonblock, largest_full_layer_bytes=largest,
                 gpu_two_full_output_buffers_bytes=2*largest, gpu_two_half_input_buffers_bytes=largest,
                 gpu_allgather_buffers_total_each_rank=gpu_buffers,
                 pinned_warning="Pinned shard storage is the main weight term itself; do NOT add it twice."),
        encoder=encoder, audio_vae_runtime_fp32_bytes_each_rank=audio,
        video_vae_runtime_fp32_bytes_each_rank=video_exact, cases=cases,
        illustrative_activation_formula=dict(
            local_tokens="L=ceil(total packed tokens / 2); real request layout must supply N",
            hidden_BF16_bytes="L*5376*2 per live hidden tensor",
            qkv_BF16_bytes="L*(3*7168)*2; Ulysses may simultaneously retain send/receive copies",
            ffn_gate_up_BF16_bytes="L*(2*14336)*2 before SiLU/multiply/output temporaries",
            example_L_40000_hidden_bytes=40000*5376*2,
            example_L_40000_qkv_bytes=40000*3*7168*2,
            example_L_40000_ffn_gate_up_bytes=40000*2*14336*2,
            warning="These are individual tensor payloads, not peak sums or measured activation bounds."),
        unresolved=[
            "Video VAE nested checkpoint header dtype/numel missing" if video_exact is None else "Non-parameter VAE buffers/decoder workspaces",
            "TE load_file maps whole source file before TP copy; largest observed shard 4932328944 bytes; mappings/page cache not private payload",
            "File cache/readahead, old allocator retained CPU pages, process/runtime heaps, pinned allocator metadata",
            "CUDA contexts, allocator fragmentation, cuBLAS/FlashAttention/NCCL workspaces and attention local gather scratch",
            "VAE encode/decode activations, text encoder attention/vision activations, packed source/target tensors and branch scans",
            "External tasks can consume resources after admission; current xuchubo CPU recovery process may later use GPU",
        ],
        proposed_admission_not_installed=dict(
            before_model="No new foreign GPU process; fresh cgroup/host memory and /proc rank RSS samples; agree available GPUs",
            cpu_after_load="At least 32 GiB headroom using cgroup usage minus inactive_file (working-set estimate), plus stable pin/offload peaks; never equate all file cache with guaranteed reclaimable memory",
            gpu_before_first_denoise="After real buffers/nonblock placement, at least 20 GiB free per GPU as a provisional gate; measure first-block peak before committing long generation",
            first_denoise="Aim peak total usage <=70 GiB/GPU (10 GiB margin); this is a proposed safety margin, not a proven model requirement or change to existing guard",
            continuous="Monitor through encoder, pinning, AllGather, every denoise step and VAE decode; stage markers + allocated/reserved + device-used + cgroup counters",
        ),
        comparison="Final B/D timing first, then quality; both use identical offload implementation/settings. Equal-weight and runtime validation gates remain.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("real_manifest.json"))
    parser.add_argument("--aux", type=Path, default=Path(__file__).with_name("aux_budget_headers.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("memory_budget.json"))
    parser.add_argument("--video-verified-summary", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    video_summary = json.loads(args.video_verified_summary.read_text(encoding="utf-8")) if args.video_verified_summary else None
    result = budget(json.loads(args.manifest.read_text(encoding="utf-8")), json.loads(args.aux.read_text(encoding="utf-8")), video_summary=video_summary)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status", "dit", "encoder", "audio_vae_runtime_fp32_bytes_each_rank", "cases")}, indent=2))


if __name__ == "__main__":
    main()
