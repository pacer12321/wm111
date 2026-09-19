"""Build a fixed H3 SpotEdit selector from native Video-VAE decoder features.

This intentionally changes only the selector metric.  The active-token budget,
source-token rows, warmup/fusion/reset logic, and partial-query implementation
remain identical to the latent-distance experiment-4 run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class _FeatureReady(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--spatial-scale", type=int, default=4)
    parser.add_argument("--temporal-chunk", type=int, default=5)
    parser.add_argument("--active-ratio", type=float, default=0.323145)
    return parser.parse_args()


def load_native_vae(component_path: Path, device: torch.device):
    config = json.loads((component_path / "config.json").read_text())
    class_reference = config["auto_map"]["AutoModel"]
    component_cls = get_class_from_dynamic_module(class_reference, str(component_path))
    remote = component_cls.from_pretrained(str(component_path))
    remote.eval().to(device=device, dtype=torch.float32)
    return remote.model, config


@torch.inference_mode()
def decoder_feature(
    model,
    normalized_latent: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    block_index: int,
    spatial_scale: int,
    temporal_chunk: int,
) -> torch.Tensor:
    """Return channel-first decoder features on a reduced spatial grid."""
    device = next(model.parameters()).device
    b, _, total_t, h, w = normalized_latent.shape
    reduced_h = max(1, h // spatial_scale)
    reduced_w = max(1, w // spatial_scale)
    latent = F.interpolate(
        normalized_latent.float(),
        size=(total_t, reduced_h, reduced_w),
        mode="trilinear",
        align_corners=False,
    )
    latent = latent * std + mean
    chunks: list[torch.Tensor] = []

    for start in range(0, total_t, temporal_chunk):
        stop = min(total_t, start + temporal_chunk)
        z = latent[:, :, start:stop].to(device=device, non_blocking=True)
        z = model.post_quant_conv(z)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["value"] = output[:, : (stop - start) * reduced_h * reduced_w].detach()
            raise _FeatureReady()

        handle = model.decoder.transformer_blocks[block_index].register_forward_hook(hook)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model.decoder(z)
        except _FeatureReady:
            pass
        finally:
            handle.remove()
        hidden = captured["value"]
        hidden = hidden.reshape(b, stop - start, reduced_h, reduced_w, -1)
        hidden = hidden.permute(0, 4, 1, 2, 3).contiguous().float().cpu()
        chunks.append(hidden)
        del z, hidden
        torch.cuda.empty_cache()

    return torch.cat(chunks, dim=2)


def exact_topk_mask(score: torch.Tensor, active_ratio: float) -> tuple[torch.Tensor, float]:
    flat = score.reshape(-1)
    count = max(1, min(flat.numel(), round(flat.numel() * active_ratio)))
    indices = torch.topk(flat, k=count, largest=True, sorted=False).indices
    mask = torch.zeros(flat.numel(), dtype=torch.bool)
    mask[indices] = True
    threshold = float(flat[indices].min())
    return mask, threshold


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    template = torch.load(args.template_payload, map_location="cpu", weights_only=True)
    target = capture["target_first_x0_normalized_latent"].float()
    source = capture["source_clean_normalized_latent"].float()
    if target.shape != source.shape:
        raise ValueError((target.shape, source.shape))

    model, config = load_native_vae(args.vae, device)
    channels = target.shape[1]
    mean = torch.tensor(config["latents_mean"]).reshape(1, channels, 1, 1, 1)
    std = torch.tensor(config["latents_std"]).reshape(1, channels, 1, 1, 1)

    target_feature = decoder_feature(
        model, target, mean, std, args.block, args.spatial_scale, args.temporal_chunk
    )
    source_feature = decoder_feature(
        model, source, mean, std, args.block, args.spatial_scale, args.temporal_chunk
    )

    target_feature = F.normalize(target_feature, dim=1, eps=1e-6)
    source_feature = F.normalize(source_feature, dim=1, eps=1e-6)
    distance = 1.0 - (target_feature * source_feature).sum(dim=1, keepdim=True)
    token_grid = tuple(int(x) for x in template["token_grid"])
    distance = F.interpolate(distance, size=token_grid, mode="trilinear", align_corners=False)
    # A light spatial smoothing suppresses isolated one-token selector noise.
    distance = F.avg_pool3d(distance, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    mask, threshold = exact_topk_mask(distance, args.active_ratio)

    payload = {
        "schema": "h3_fixed_selector_payload_v1",
        "active_target_mask": mask,
        "source_clean_normalized_rows": template["source_clean_normalized_rows"],
        "token_grid": token_grid,
        "metric": "h3_native_video_vae_decoder_cosine_distance",
        "threshold": threshold,
        "vae_decoder_block": args.block,
        "feature_spatial_scale": args.spatial_scale,
        "active_ratio": float(mask.float().mean()),
        "selector_budget_control": "matched_to_latent_selector_for_clean_quality_comparison",
    }
    payload_path = args.output_dir / "fixed_selector_payload.pt"
    torch.save(payload, payload_path)
    torch.save(distance.cpu(), args.output_dir / "perceptual_distance.pt")
    summary = {
        "schema": payload["schema"],
        "capture": str(args.capture),
        "template_payload": str(args.template_payload),
        "vae": str(args.vae),
        "target_shape": list(target.shape),
        "feature_shape": list(target_feature.shape),
        "token_grid": list(token_grid),
        "decoder_block": args.block,
        "spatial_scale": args.spatial_scale,
        "temporal_chunk": args.temporal_chunk,
        "active_tokens": int(mask.sum()),
        "total_tokens": mask.numel(),
        "active_ratio": float(mask.float().mean()),
        "threshold": threshold,
        "score_min": float(distance.min()),
        "score_mean": float(distance.mean()),
        "score_max": float(distance.max()),
        "cuda_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
