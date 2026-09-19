from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def dilated_mask(score: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    raw = score >= threshold
    dilated = F.max_pool3d(
        raw.float(), kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    ).bool()
    return raw, dilated


def threshold_for_ratio(score: torch.Tensor, target_ratio: float) -> tuple[float, torch.Tensor, torch.Tensor]:
    low = float(score.min()) - 1e-7
    high = float(score.max()) + 1e-7
    candidates: list[tuple[float, float, torch.Tensor, torch.Tensor]] = []
    for _ in range(48):
        threshold = (low + high) / 2.0
        raw, dilated = dilated_mask(score, threshold)
        ratio = float(dilated.float().mean())
        candidates.append((abs(ratio - target_ratio), threshold, raw, dilated))
        if ratio > target_ratio:
            low = threshold
        else:
            high = threshold
    _, threshold, raw, dilated = min(candidates, key=lambda item: item[0])
    return threshold, raw, dilated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector_map", type=Path)
    parser.add_argument("template_payload", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratios", type=float, nargs="+", default=(0.28, 0.24, 0.20))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    selector = torch.load(args.selector_map, map_location="cpu", weights_only=False)
    template = torch.load(args.template_payload, map_location="cpu", weights_only=False)
    score = selector["token_score"].float()

    catalogue = []
    for target_ratio in args.ratios:
        threshold, raw, dilated = threshold_for_ratio(score, target_ratio)
        actual_ratio = float(dilated.float().mean())
        raw_ratio = float(raw.float().mean())
        label = f"active_{round(target_ratio * 100):02d}pct"
        destination = args.output / label
        destination.mkdir()
        payload = {
            **template,
            "active_target_mask": dilated.reshape(-1).cpu(),
            "metric": "channel_normalized_latent_distance_threshold_sweep_r1",
            "threshold": threshold,
            "target_active_ratio_after_dilation": target_ratio,
            "raw_active_ratio": raw_ratio,
            "active_ratio_after_dilation": actual_ratio,
        }
        torch.save(payload, destination / "fixed_selector_payload.pt")
        summary = {
            "schema": "h3_latent_selector_threshold_sweep_v1",
            "label": label,
            "threshold": threshold,
            "target_active_ratio_after_dilation": target_ratio,
            "raw_active_ratio": raw_ratio,
            "active_ratio_after_dilation": actual_ratio,
            "active_tokens": int(dilated.sum()),
            "total_tokens": int(dilated.numel()),
            "dilation": "spatial radius 1 (3x3), temporal radius 0",
        }
        (destination / "summary.json").write_text(json.dumps(summary, indent=2))
        catalogue.append(summary)

    (args.output / "catalogue.json").write_text(json.dumps(catalogue, indent=2))
    print(json.dumps(catalogue, indent=2), flush=True)


if __name__ == "__main__":
    main()
