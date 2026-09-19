from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from analyze_ss_spatial_oracle import _load_layer, _relative_errors, _summary


def _parse_grids(value: str) -> list[tuple[int, int]]:
    grids: list[tuple[int, int]] = []
    for item in value.split(","):
        rows, columns = item.lower().split("x", maxsplit=1)
        grid = (int(rows), int(columns))
        if min(grid) <= 0:
            raise ValueError(f"invalid summary grid: {item}")
        grids.append(grid)
    return grids


def _target_group_ids(
    payload: dict[str, Any],
    grid_rows: int,
    grid_columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = payload["layout"]
    frames = int(layout["num_frames"])
    height = int(layout["frame_height"])
    width = int(layout["frame_width"])
    if grid_rows > height or grid_columns > width:
        raise ValueError(
            f"summary grid {grid_rows}x{grid_columns} exceeds "
            f"target grid {height}x{width}"
        )
    expected = frames * height * width
    # H3's RoPE coordinates are continuous, camera-projected coordinates rather
    # than integer latent-grid indices.  Pool the fixed target tensor layout,
    # which capture construction guarantees is frame-major [T, H, W].
    ordinal = torch.arange(expected, dtype=torch.long)
    frame = torch.div(ordinal, height * width, rounding_mode="floor")
    within_frame = ordinal.remainder(height * width)
    source_row = torch.div(within_frame, width, rounding_mode="floor")
    source_column = within_frame.remainder(width)
    row = torch.div(source_row * grid_rows, height, rounding_mode="floor")
    column = torch.div(source_column * grid_columns, width, rounding_mode="floor")
    group_ids = (frame * grid_rows + row) * grid_columns + column
    num_groups = frames * grid_rows * grid_columns
    counts = torch.bincount(group_ids, minlength=num_groups)
    if torch.any(counts == 0):
        raise ValueError("every pooled target group must be non-empty")
    return group_ids, counts


def _pool_target(
    values: torch.Tensor,
    target_positions: torch.Tensor,
    group_ids: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    selected = values.index_select(0, target_positions)
    pooled = torch.zeros(
        (counts.numel(), *selected.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )
    pooled.index_add_(0, group_ids, selected)
    divisor = counts.to(device=values.device, dtype=values.dtype)
    divisor = divisor.view(-1, *([1] * (selected.ndim - 1)))
    return pooled / divisor


def _attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    pooled_count: int = 0,
    pooled_log_multiplicity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("qhd,khd->qhk", q, k) * scale
    if pooled_log_multiplicity is not None:
        if pooled_count != pooled_log_multiplicity.numel():
            raise ValueError("pooled multiplicity shape mismatch")
        scores[..., -pooled_count:] += pooled_log_multiplicity.view(1, 1, -1)
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("qhk,khd->qhd", weights, v)
    pooled_mass = (
        weights[..., -pooled_count:].sum(dim=-1)
        if pooled_count
        else torch.zeros(weights.shape[:-1], device=weights.device)
    )
    return output, pooled_mass


def _global_relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((candidate - reference).float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)
    return float((numerator / denominator).cpu())


def analyze_layer(
    payload: dict[str, Any],
    device: torch.device,
    grids: list[tuple[int, int]],
    query_batch: int,
) -> dict[str, Any]:
    q = payload["q"].to(device=device, dtype=torch.float32)
    k = payload["k"].to(device=device, dtype=torch.float32)
    v = payload["v"].to(device=device, dtype=torch.float32)
    target_positions = payload["target_positions"].to(device=device, dtype=torch.long)
    target_mask = torch.zeros(k.shape[0], dtype=torch.bool, device=device)
    target_mask[target_positions] = True
    non_target_positions = torch.nonzero(~target_mask, as_tuple=False).view(-1)
    non_target_k = k.index_select(0, non_target_positions)
    non_target_v = v.index_select(0, non_target_positions)
    scale = float(payload["softmax_scale"])

    candidates: list[dict[str, Any]] = []
    for grid_rows, grid_columns in grids:
        group_ids_cpu, counts_cpu = _target_group_ids(
            payload, grid_rows, grid_columns
        )
        group_ids = group_ids_cpu.to(device)
        counts = counts_cpu.to(device)
        pooled_k = _pool_target(k, target_positions, group_ids, counts)
        pooled_v = _pool_target(v, target_positions, group_ids, counts)
        candidates.append(
            {
                "grid": (grid_rows, grid_columns),
                "counts": counts,
                "k": torch.cat((non_target_k, pooled_k), dim=0),
                "v": torch.cat((non_target_v, pooled_v), dim=0),
                "pooled_count": int(pooled_k.shape[0]),
                "naive_errors": [],
                "corrected_errors": [],
                "naive_cosines": [],
                "corrected_cosines": [],
                "naive_target_mass": [],
                "corrected_target_mass": [],
            }
        )

    dense_outputs = []
    dense_target_masses = []
    candidate_outputs: dict[tuple[int, int, str], list[torch.Tensor]] = {}
    for start in range(0, q.shape[0], query_batch):
        q_batch = q[start : start + query_batch]
        dense_output, dense_target_mass = _attention_output(
            q_batch,
            k,
            v,
            scale,
            pooled_count=int(target_positions.numel()),
        )
        dense_outputs.append(dense_output.cpu())
        dense_target_masses.append(dense_target_mass.cpu())
        for candidate in candidates:
            pooled_count = candidate["pooled_count"]
            for mode in ("naive", "multiplicity_corrected"):
                bias = None
                if mode == "multiplicity_corrected":
                    bias = torch.log(candidate["counts"].to(torch.float32))
                output, target_mass = _attention_output(
                    q_batch,
                    candidate["k"],
                    candidate["v"],
                    scale,
                    pooled_count=pooled_count,
                    pooled_log_multiplicity=bias,
                )
                errors = _relative_errors(dense_output, output).cpu()
                cosines = torch.nn.functional.cosine_similarity(
                    dense_output, output, dim=-1
                ).cpu()
                prefix = "corrected" if mode == "multiplicity_corrected" else "naive"
                candidate[f"{prefix}_errors"].append(errors)
                candidate[f"{prefix}_cosines"].append(cosines)
                candidate[f"{prefix}_target_mass"].append(target_mass.cpu())
                candidate_outputs.setdefault(
                    (*candidate["grid"], mode), []
                ).append(output.cpu())

    dense_output_all = torch.cat(dense_outputs)
    dense_target_mass_all = torch.cat(dense_target_masses)
    results = []
    for candidate in candidates:
        grid_rows, grid_columns = candidate["grid"]
        row: dict[str, Any] = {
            "summary_grid_per_target_frame": f"{grid_rows}x{grid_columns}",
            "target_summary_keys": candidate["pooled_count"],
            "dense_target_keys": int(target_positions.numel()),
            "target_key_reduction_fraction": 1.0
            - candidate["pooled_count"] / target_positions.numel(),
            "all_keys_after_pooling": int(
                non_target_positions.numel() + candidate["pooled_count"]
            ),
            "all_key_reduction_fraction": 1.0
            - (non_target_positions.numel() + candidate["pooled_count"]) / k.shape[0],
        }
        for mode, prefix in (
            ("naive", "naive"),
            ("multiplicity_corrected", "corrected"),
        ):
            output = torch.cat(candidate_outputs[(grid_rows, grid_columns, mode)])
            row[mode] = {
                "global_relative_output_l2": _global_relative_l2(
                    dense_output_all, output
                ),
                "per_query_head_relative_output_error": _summary(
                    torch.cat(candidate[f"{prefix}_errors"])
                ),
                "output_cosine": _summary(
                    torch.cat(candidate[f"{prefix}_cosines"])
                ),
                "target_attention_mass": _summary(
                    torch.cat(candidate[f"{prefix}_target_mass"])
                ),
            }
        results.append(row)

    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "sampled_source_queries": int(q.shape[0]),
        "heads": int(q.shape[1]),
        "dense_keys_per_query": int(k.shape[0]),
        "dense_target_keys": int(target_positions.numel()),
        "dense_target_attention_mass": _summary(dense_target_mass_all),
        "pooling_location": "post-QK-norm, post-RoPE K; projected V",
        "grids": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--layers", default="8,24,41")
    parser.add_argument("--grids", default="1x1,2x2,4x4,7x9,14x18")
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    grids = _parse_grids(args.grids)
    device = torch.device(args.device)
    step_dir = args.capture_dir / f"step_{args.step:02d}"
    results = []
    for layer in layers:
        payload = _load_layer(step_dir, layer)
        results.append(
            analyze_layer(payload, device, grids, args.query_batch)
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "h3_fastest_st_pooled_oracle_v1",
        "definition": (
            "For sampled source queries, preserve every non-target key and replace "
            "all target spatial tokens with fixed-grid mean-pooled post-RoPE K/V "
            "summaries in every target frame. Naive pooling and a log(group-size) "
            "softmax multiplicity correction are reported separately. This is a "
            "layer-output screening oracle, not an end-to-end quality certificate."
        ),
        "capture_dir": str(args.capture_dir),
        "layers": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
