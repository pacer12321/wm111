from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_layer(step_dir: Path, layer: int) -> dict[str, Any]:
    paths = sorted(step_dir.glob(f"layer_{layer:02d}_rank_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no rank captures for layer {layer} in {step_dir}")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    first = payloads[0]
    if any(item["schema"] != "h3_fastest_ss_source_qkv_v1" for item in payloads):
        raise ValueError("unexpected capture schema")
    if len(payloads) != int(first["world_size"]):
        raise ValueError(f"expected {first['world_size']} ranks, found {len(payloads)}")

    def merge(value_name: str, index_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([item[value_name] for item in payloads], dim=0)
        indices = torch.cat([item[index_name] for item in payloads]).to(torch.long)
        order = torch.argsort(indices)
        indices = indices[order]
        if indices.numel() and torch.unique_consecutive(indices).numel() != indices.numel():
            raise ValueError(f"duplicate logical indices in {index_name}")
        return values[order], indices

    q, q_indices = merge("q_selected", "query_indices")
    k, k_indices = merge("k_local", "key_indices")
    v, v_indices = merge("v_local", "key_indices")
    if not torch.equal(k_indices, v_indices):
        raise ValueError("K/V logical indices differ")
    used_len = int(first["layout"]["used_len"])
    if not torch.equal(k_indices, torch.arange(used_len)):
        raise ValueError("captured K/V rows do not reconstruct [0, used_len)")
    expected_q = first["all_query_indices"].to(torch.long)
    if not torch.equal(q_indices, expected_q.sort().values):
        raise ValueError("captured source queries do not match the requested sample")
    return {
        **first,
        "q": q,
        "q_indices": q_indices,
        "k": k,
        "v": v,
        "k_indices": k_indices,
    }


def _shifted_window(center: int, size: int, extent: int) -> range:
    if size > extent:
        raise ValueError(f"window {size} exceeds axis extent {extent}")
    start = min(max(center - size // 2, 0), extent - size)
    return range(start, start + size)


def _source_grid(payload: dict[str, Any]) -> tuple[torch.Tensor, dict[int, tuple[int, int, int]]]:
    layout = payload["layout"]
    frames = int(layout["num_frames"])
    height = int(layout["frame_height"])
    width = int(layout["frame_width"])
    positions = payload["source_positions"].to(torch.long)
    coords = payload["source_coords"].to(torch.long)
    if positions.numel() != frames * height * width:
        raise ValueError("source positions do not form the advertised grid")
    grid = positions.view(frames, height, width)
    logical_to_coord = {
        int(position): tuple(int(value) for value in coord)
        for position, coord in zip(positions.tolist(), coords.tolist(), strict=True)
    }
    return grid, logical_to_coord


def _local_source_keys(
    grid: torch.Tensor,
    y: int,
    x: int,
    window: int,
) -> torch.Tensor:
    _, height, width = grid.shape
    ys = list(_shifted_window(y, window, height))
    xs = list(_shifted_window(x, window, width))
    selected = grid[:, ys][:, :, xs].reshape(-1)
    expected = grid.shape[0] * window * window
    if selected.numel() != expected or torch.unique(selected).numel() != expected:
        raise ValueError("spatial window must contain a fixed number of unique source keys")
    return selected.sort().values


def _random_source_keys(
    grid: torch.Tensor,
    seed: int,
    count_per_frame: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    tokens_per_frame = grid.shape[1] * grid.shape[2]
    for frame in range(grid.shape[0]):
        offsets = torch.randperm(tokens_per_frame, generator=generator)[:count_per_frame]
        selected.append(grid[frame].reshape(-1).index_select(0, offsets))
    return torch.cat(selected).sort().values


def _attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("qhd,khd->qhk", q, k) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("qhk,khd->qhd", weights, v)
    return output, weights


def _relative_errors(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(candidate - reference, dim=-1)
    denominator = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-8)
    return numerator / denominator


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def analyze_layer(
    payload: dict[str, Any],
    device: torch.device,
    window: int,
    random_seed: int,
) -> dict[str, Any]:
    q = payload["q"].to(device=device, dtype=torch.float32)
    k = payload["k"].to(device=device, dtype=torch.float32)
    v = payload["v"].to(device=device, dtype=torch.float32)
    q_indices = payload["q_indices"].to(torch.long)
    grid, logical_to_coord = _source_grid(payload)
    source_positions = payload["source_positions"].to(torch.long)
    source_mask = torch.zeros(k.shape[0], dtype=torch.bool)
    source_mask[source_positions] = True
    non_source_positions = torch.nonzero(~source_mask, as_tuple=False).view(-1)

    grouped: dict[tuple[int, int], list[int]] = {}
    for row, logical in enumerate(q_indices.tolist()):
        _, y, x = logical_to_coord[int(logical)]
        grouped.setdefault((y, x), []).append(row)

    local_errors = []
    random_errors = []
    local_source_mass = []
    random_source_mass = []
    local_total_mass = []
    random_total_mass = []
    cosine_values = []
    scale = float(payload["softmax_scale"])
    count_per_frame = window * window

    for group_index, ((y, x), rows) in enumerate(sorted(grouped.items())):
        row_tensor = torch.tensor(rows, device=device, dtype=torch.long)
        q_group = q.index_select(0, row_tensor)
        dense_output, dense_weights = _attention_output(q_group, k, v, scale)

        local_source = _local_source_keys(grid, y, x, window)
        random_source = _random_source_keys(
            grid,
            seed=random_seed + group_index,
            count_per_frame=count_per_frame,
        )
        local_keep = torch.cat((non_source_positions, local_source)).sort().values
        random_keep = torch.cat((non_source_positions, random_source)).sort().values
        local_keep_device = local_keep.to(device)
        random_keep_device = random_keep.to(device)

        local_output, _ = _attention_output(
            q_group,
            k.index_select(0, local_keep_device),
            v.index_select(0, local_keep_device),
            scale,
        )
        random_output, _ = _attention_output(
            q_group,
            k.index_select(0, random_keep_device),
            v.index_select(0, random_keep_device),
            scale,
        )
        local_errors.append(_relative_errors(dense_output, local_output).cpu())
        random_errors.append(_relative_errors(dense_output, random_output).cpu())
        cosine_values.append(
            torch.nn.functional.cosine_similarity(
                dense_output,
                local_output,
                dim=-1,
            ).cpu()
        )

        source_mass = dense_weights.index_select(2, source_positions.to(device)).sum(dim=-1)
        local_mass = dense_weights.index_select(2, local_source.to(device)).sum(dim=-1)
        random_mass = dense_weights.index_select(2, random_source.to(device)).sum(dim=-1)
        local_source_mass.append((local_mass / source_mass.clamp_min(1e-8)).cpu())
        random_source_mass.append((random_mass / source_mass.clamp_min(1e-8)).cpu())
        local_total_mass.append(local_mass.cpu())
        random_total_mass.append(random_mass.cpu())

    local_error = torch.cat(local_errors)
    random_error = torch.cat(random_errors)
    local_retained_source = torch.cat(local_source_mass)
    random_retained_source = torch.cat(random_source_mass)
    local_retained_total = torch.cat(local_total_mass)
    random_retained_total = torch.cat(random_total_mass)
    cosine = torch.cat(cosine_values)
    keys_per_query = int(non_source_positions.numel() + grid.shape[0] * count_per_frame)

    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "spotedit_refresh": bool(payload["spotedit_refresh"]),
        "active_target_ratio": float(payload["active_target_ratio"]),
        "sampled_source_queries": int(q.shape[0]),
        "heads": int(q.shape[1]),
        "dense_keys_per_query": int(k.shape[0]),
        "source_keys_dense": int(source_positions.numel()),
        "source_keys_spatial_local": int(grid.shape[0] * count_per_frame),
        "all_keys_spatial_local": keys_per_query,
        "ss_key_reduction_fraction": 1.0 - (
            grid.shape[0] * count_per_frame / source_positions.numel()
        ),
        "all_key_reduction_fraction": 1.0 - keys_per_query / k.shape[0],
        "spatial_local_relative_output_error": _summary(local_error),
        "random_relative_output_error": _summary(random_error),
        "spatial_local_output_cosine": _summary(cosine),
        "spatial_local_retained_source_mass": _summary(local_retained_source),
        "random_retained_source_mass": _summary(random_retained_source),
        "spatial_local_retained_total_attention_mass": _summary(local_retained_total),
        "random_retained_total_attention_mass": _summary(random_retained_total),
        "local_vs_random_mean_error_ratio": float(
            local_error.float().mean() / random_error.float().mean().clamp_min(1e-8)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--layers", default="8,24,41")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.window % 2 != 1:
        raise ValueError("spatial window must be odd")
    device = torch.device(args.device)
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    step_dir = args.capture_dir / f"step_{args.step:02d}"
    results = []
    for layer in layers:
        payload = _load_layer(step_dir, layer)
        results.append(
            analyze_layer(payload, device, args.window, args.random_seed)
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "h3_fastest_ss_spatial_oracle_analysis_v1",
        "definition": (
            "For each sampled source query, preserve every non-source key and "
            "replace dense S-S with a shifted same-coordinate spatial window in "
            "every source frame. The 5x5 setting therefore keeps exactly 25 "
            "source keys per source frame, not 25 for the whole video."
        ),
        "capture_dir": str(args.capture_dir),
        "window": args.window,
        "layers": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
