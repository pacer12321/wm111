"""Analyze a captured first-x0/source pair without changing model execution.

The latent-space map is a fast sanity check only.  The paper-faithful H3
perceptual map is computed separately from intermediate video-VAE decoder
features because raw latent distance is known to over-weight low frequencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def optimal_two_means_threshold(values: torch.Tensor) -> tuple[float, dict[str, float | int]]:
    flat = values.detach().float().reshape(-1).cpu()
    ordered, _ = torch.sort(flat)
    n = int(ordered.numel())
    if n < 2:
        raise ValueError("at least two scores are required")
    prefix = ordered.cumsum(0)
    prefix_sq = ordered.square().cumsum(0)
    counts_left = torch.arange(1, n, dtype=torch.float64)
    counts_right = n - counts_left
    left_sum = prefix[:-1].double()
    left_sq = prefix_sq[:-1].double()
    total_sum = prefix[-1].double()
    total_sq = prefix_sq[-1].double()
    right_sum = total_sum - left_sum
    right_sq = total_sq - left_sq
    sse_left = left_sq - left_sum.square() / counts_left
    sse_right = right_sq - right_sum.square() / counts_right
    split = int(torch.argmin(sse_left + sse_right).item()) + 1
    low_max = float(ordered[split - 1])
    high_min = float(ordered[split])
    threshold = (low_max + high_min) / 2.0
    return threshold, {
        "n": n,
        "split_index": split,
        "low_cluster_count": split,
        "high_cluster_count": n - split,
        "low_cluster_mean": float(ordered[:split].mean()),
        "high_cluster_mean": float(ordered[split:].mean()),
        "low_max": low_max,
        "high_min": high_min,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    target = payload["target_first_x0_normalized_latent"].float()
    source = payload["source_clean_normalized_latent"].float()
    if target.shape != source.shape:
        raise ValueError(f"source/target latent shapes differ: {source.shape} vs {target.shape}")
    if target.ndim != 5 or target.shape[0] != 1:
        raise ValueError(f"expected [1,C,T,H,W], got {tuple(target.shape)}")

    # Channel-normalized distance is much less scale-sensitive than plain L2,
    # but remains a diagnostic rather than the final perceptual criterion.
    target_unit = F.normalize(target, dim=1, eps=1e-6)
    source_unit = F.normalize(source, dim=1, eps=1e-6)
    latent_score = (target_unit - source_unit).square().sum(dim=1, keepdim=True)
    token_score = F.avg_pool3d(latent_score, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    threshold, clustering = optimal_two_means_threshold(token_score)
    active = token_score >= threshold
    active_dilated = F.max_pool3d(
        active.float(), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    ).bool()

    batch, channels, latent_t, latent_h, latent_w = source.shape
    source_rows = source.reshape(batch, channels, latent_t, 1, latent_h // 2, 2, latent_w // 2, 2)
    source_rows = torch.einsum("nctrhpwq->nthwcrpq", source_rows)
    source_rows = source_rows.reshape(-1, channels * 4).contiguous().half()

    torch.save(
        {
            "metric": "channel_normalized_latent_distance_diagnostic",
            "token_score": token_score.half(),
            "active_mask": active,
            "active_mask_spatial_dilation_r1": active_dilated,
            "threshold": threshold,
        },
        args.output / "latent_selector_map.pt",
    )
    torch.save(
        {
            "schema": "h3_fixed_selector_payload_v1",
            "active_target_mask": active_dilated.reshape(-1).cpu(),
            "source_clean_normalized_rows": source_rows.cpu(),
            "token_grid": (latent_t, latent_h // 2, latent_w // 2),
            "metric": "channel_normalized_latent_distance_diagnostic",
            "threshold": threshold,
            "warning": "engineering proxy; replace with H3 perceptual feature selector",
        },
        args.output / "fixed_selector_payload.pt",
    )
    frame_ratios = active.float().mean(dim=(0, 1, 3, 4))
    frame_ratios_dilated = active_dilated.float().mean(dim=(0, 1, 3, 4))
    summary = {
        "schema": "h3_selector_latent_sanity_v1",
        "capture_schema": payload.get("schema"),
        "latent_shape": list(target.shape),
        "dit_token_grid": list(token_score.shape),
        "metric": "channel-normalized latent distance (sanity check, not final VAE perceptual metric)",
        "threshold_method": "global exact optimal 1D two-means SSE split",
        "threshold": threshold,
        "clustering": clustering,
        "active_token_ratio": float(active.float().mean()),
        "active_token_ratio_after_spatial_dilation_r1": float(active_dilated.float().mean()),
        "per_latent_frame_active_ratio": [float(v) for v in frame_ratios],
        "per_latent_frame_active_ratio_after_dilation": [float(v) for v in frame_ratios_dilated],
        "attention_policy_for_next_experiment": {
            "stable_target_tokens": "reuse/fuse; skip target query computation",
            "active_target_queries_to_source": "all source KV visible; no strict same-frame mask",
            "target_target": "VDN local plus linear",
            "source_source": "unchanged by selector; dense in ablation 4, local plus linear in ablation 5",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
