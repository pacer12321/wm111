"""Combine per-rank/per-layer H3 temporal sums into one global mean map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=49)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    total = None
    records = []
    for layer in range(args.layers):
        for rank in range(args.ranks):
            path = args.capture_dir / f"layer_{layer:02d}_rank_{rank}.pt"
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["schema"] != "h3_dense_global_temporal_sum_v1":
                raise ValueError(f"Unexpected schema in {path}")
            expected = {
                "layer": layer,
                "rank": rank,
                "world_size": args.ranks,
                "timesteps": args.timesteps,
                "frames": 37,
                "samples_per_target_frame": 16,
            }
            for key, value in expected.items():
                if int(record[key]) != value:
                    raise ValueError(f"{path}: {key}={record[key]} != {value}")
            value = record["temporal_conditional_sum"].double()
            if value.shape != (37, 37) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid map in {path}")
            total = value.clone() if total is None else total + value
            records.append(record)

    denominator = args.layers * args.ranks * args.timesteps
    temporal = total / denominator
    row_error = (temporal.sum(1) - 1).abs().max().item()
    if row_error > 1e-5:
        raise ValueError(f"Temporal rows do not sum to one: max error {row_error}")
    frames = temporal.shape[0]
    diagonal = temporal.diag()
    target = torch.arange(frames)[:, None]
    source = torch.arange(frames)[None, :]
    expected_offset = (temporal * (target - source).abs()).sum(1)
    top = temporal.argmax(1)
    local_heads = {int(record["local_heads"]) for record in records}
    if len(local_heads) != 1:
        raise ValueError(f"Unequal local head counts: {local_heads}")
    total_heads = args.ranks * next(iter(local_heads))
    summary = {
        "schema": "h3_dense_global_temporal_average_summary_v1",
        "attention_mode": "Ref2VA base dense attention",
        "aggregation": (
            "probability-level arithmetic mean over all 56 heads, 50 DiT "
            "layers, 49 denoising forwards, and 16 sampled spatial target "
            "queries per frame; conditioned on attention landing on source-video tokens"
        ),
        "heads": total_heads,
        "layers": args.layers,
        "timesteps": args.timesteps,
        "samples_per_target_frame": 16,
        "target_frames": frames,
        "source_frames": frames,
        "component_maps_averaged": denominator,
        "mean_same_frame_fraction_within_source_attention": float(diagonal.mean()),
        "median_same_frame_fraction_within_source_attention": float(diagonal.median()),
        "same_frame_is_argmax_fraction": float((top == torch.arange(frames)).float().mean()),
        "mean_expected_absolute_frame_offset": float(expected_offset.mean()),
        "uniform_same_frame_fraction": 1.0 / frames,
        "max_row_sum_error": row_error,
        "top_source_frame_by_target": top.tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / "attention_maps.npz",
        temporal_conditional=temporal.numpy().astype(np.float32),
        diagonal_conditional=diagonal.numpy().astype(np.float32),
        expected_abs_offset=expected_offset.numpy().astype(np.float32),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
