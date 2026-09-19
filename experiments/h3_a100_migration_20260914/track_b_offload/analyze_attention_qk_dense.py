"""Analyze sampled Ref2VA-base dense attention from local SP Q/K shards."""

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.capture_dir.glob("layer_24_rank_*.pt"))
    if len(paths) != 2:
        raise RuntimeError(f"expected two rank files, found {paths}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    first = rows[0]
    if first.get("schema") != "h3_dense_local_post_rope_qk_v1":
        raise RuntimeError(f"unexpected schema: {first.get('schema')!r}")
    for row in rows[1:]:
        if row["layout"] != first["layout"] or row["step"] != first["step"]:
            raise RuntimeError("rank metadata mismatch")

    layout = first["layout"]
    frames = int(layout["num_frames"])
    per = int(layout["tokens_per_frame"])
    height = int(layout["frame_height"])
    width = int(layout["frame_width"])
    used = int(layout["used_len"])
    heads = int(first["num_heads"])
    device = torch.device("cuda:0")

    query_indices = torch.cat([row["query_indices"].long() for row in rows])
    q = torch.cat([row["q_selected"] for row in rows])
    q_order = query_indices.argsort()
    query_indices = query_indices[q_order]
    q = q[q_order].to(device)
    expected_queries = first["all_query_indices"].long().sort().values
    if not torch.equal(query_indices, expected_queries):
        raise RuntimeError("rank query shards do not reconstruct sampled queries")

    key_indices = torch.cat([row["key_indices"].long() for row in rows])
    k = torch.cat([row["k_local"] for row in rows])
    key_order = key_indices.argsort()
    key_indices = key_indices[key_order]
    k = k[key_order]
    if not torch.equal(key_indices, torch.arange(used)):
        raise RuntimeError("rank key shards do not reconstruct all used dense keys")
    k = k.to(device)

    target_positions = first["target_positions"].long()
    video_start = int(layout["video_start"])
    target_lookup = torch.full((used,), -1, dtype=torch.long)
    target_lookup[target_positions] = torch.arange(target_positions.numel())
    q_target = target_lookup[query_indices]
    if bool(torch.any(q_target < 0)):
        raise RuntimeError("sampled query is not a target-video row")
    q_frame = q_target // per
    q_spatial = q_target % per
    if query_indices.numel() % frames:
        raise RuntimeError("sampled queries are not frame-balanced")

    source_positions = first["source_positions"].long()
    source_coords = first["source_coords"].float()
    valid = (source_positions >= 0) & (source_positions < used)
    source_positions = source_positions[valid]
    source_coords = source_coords[valid]
    source_times, source_frame = torch.unique(
        source_coords[:, 0], sorted=True, return_inverse=True
    )
    source_ys, source_y = torch.unique(
        source_coords[:, 1], sorted=True, return_inverse=True
    )
    source_xs, source_x = torch.unique(
        source_coords[:, 2], sorted=True, return_inverse=True
    )
    if (
        source_times.numel() != frames
        or source_ys.numel() != height
        or source_xs.numel() != width
        or source_positions.numel() != frames * per
    ):
        raise RuntimeError("unexpected source video grid")

    source_positions_gpu = source_positions.to(device)
    source_frame_gpu = source_frame.to(device)
    source_y_gpu = source_y.to(device)
    source_x_gpu = source_x.to(device)
    scale = float(first["softmax_scale"])
    representative_frames = torch.tensor(
        [0, frames // 4, frames // 2, 3 * frames // 4, frames - 1]
    )
    roi_yx = torch.tensor([max(0, height * 2 // 5), max(0, width * 3 // 4)])

    temporal_numerator = torch.zeros((frames, frames), dtype=torch.float64)
    denominator = torch.zeros(frames, dtype=torch.float64)
    same_frame_numerator = torch.zeros(frames, dtype=torch.float64)
    exact_numerator = torch.zeros(frames, dtype=torch.float64)
    neighbor_numerator = torch.zeros(frames, dtype=torch.float64)
    roi_cube = torch.zeros(
        (representative_frames.numel(), frames, height, width), dtype=torch.float64
    )

    for frame in range(frames):
        mask = q_frame == frame
        q_rows = q[mask.to(device)]
        q_sp = q_spatial[mask].to(device)
        logits = torch.einsum("qhd,khd->hqk", q_rows, k) * scale
        probs = torch.softmax(logits.float(), dim=-1)
        source_probs = probs.index_select(2, source_positions_gpu)
        temporal = torch.zeros(
            (heads, q_rows.shape[0], frames), dtype=torch.float32, device=device
        )
        temporal.scatter_add_(
            2,
            source_frame_gpu.view(1, 1, -1).expand(heads, q_rows.shape[0], -1),
            source_probs,
        )
        temporal_numerator[frame] = temporal.sum((0, 1)).double().cpu()
        denominator[frame] = heads * q_rows.shape[0]
        same_frame_numerator[frame] = temporal[:, :, frame].sum().double().cpu()

        for query_row in range(q_rows.shape[0]):
            qy = q_sp[query_row] // width
            qx = q_sp[query_row] % width
            same = source_frame_gpu == frame
            exact = same & (source_y_gpu == qy) & (source_x_gpu == qx)
            near = same & ((source_y_gpu - qy).abs() <= 1) & ((source_x_gpu - qx).abs() <= 1)
            exact_numerator[frame] += source_probs[:, query_row, exact].sum().double().cpu()
            neighbor_numerator[frame] += source_probs[:, query_row, near].sum().double().cpu()

        representative = torch.nonzero(
            representative_frames == frame, as_tuple=False
        ).view(-1)
        if representative.numel():
            qy_all = q_sp // width
            qx_all = q_sp % width
            nearest = (
                (qy_all - roi_yx[0].to(device)).square()
                + (qx_all - roi_yx[1].to(device)).square()
            ).argmin()
            one = source_probs[:, nearest].sum(0)
            cube = torch.zeros(frames * height * width, device=device)
            linear = source_frame_gpu * per + source_y_gpu * width + source_x_gpu
            cube.scatter_add_(0, linear, one)
            roi_cube[representative.item()] = (
                cube.view(frames, height, width).double().cpu() / heads
            )
        del logits, probs, source_probs, temporal

    temporal_raw = temporal_numerator / denominator[:, None]
    source_mass = temporal_raw.sum(1)
    temporal_conditional = temporal_raw / source_mass[:, None].clamp_min(1e-12)
    diagonal = temporal_conditional.diag()
    offsets = (
        temporal_conditional
        * (torch.arange(frames)[:, None] - torch.arange(frames)[None, :]).abs()
    ).sum(1)
    top_frame = temporal_conditional.argmax(1)
    exact_within_source = exact_numerator / (denominator * source_mass).clamp_min(1e-12)
    neighbor_within_source = neighbor_numerator / (denominator * source_mass).clamp_min(1e-12)
    roi_conditional = roi_cube / roi_cube.sum((1, 2, 3), keepdim=True).clamp_min(1e-12)

    summary = {
        "schema": "h3_dense_attention_map_summary_v1",
        "attention_mode": "Ref2VA base dense attention",
        "layer": int(first["layer"]),
        "spotedit_step": int(first["step"]),
        "captured_stage": (
            f"DiT denoising forward step {int(first['step'])}; "
            "post-QK-norm and post-RoPE"
        ),
        "query_sampling": f"{query_indices.numel() // frames} spatial queries per target frame",
        "heads": heads,
        "target_grid": [frames, height, width],
        "source_grid": [frames, height, width],
        "mean_total_attention_mass_to_source": float(source_mass.mean()),
        "mean_same_frame_fraction_within_source_attention": float(diagonal.mean()),
        "median_same_frame_fraction_within_source_attention": float(diagonal.median()),
        "same_frame_is_argmax_fraction": float((top_frame == torch.arange(frames)).float().mean()),
        "mean_expected_absolute_frame_offset": float(offsets.mean()),
        "mean_exact_same_position_fraction_within_all_source_attention": float(exact_within_source.mean()),
        "mean_3x3_same_frame_neighborhood_fraction_within_all_source_attention": float(neighbor_within_source.mean()),
        "mean_same_frame_fraction_under_uniform_source_attention": 1.0 / frames,
        "mean_exact_position_fraction_under_uniform_source_attention": 1.0 / (frames * per),
        "mean_3x3_fraction_upper_bound_under_uniform_source_attention": 9.0 / (frames * per),
        "top_source_frame_by_target": top_frame.tolist(),
        "representative_target_frames": representative_frames.tolist(),
        "requested_roi_yx": roi_yx.tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / "attention_maps.npz",
        temporal_raw=temporal_raw.numpy().astype(np.float32),
        temporal_conditional=temporal_conditional.numpy().astype(np.float32),
        source_mass=source_mass.numpy().astype(np.float32),
        diagonal_conditional=diagonal.numpy().astype(np.float32),
        expected_abs_offset=offsets.numpy().astype(np.float32),
        exact_position_fraction=exact_within_source.numpy().astype(np.float32),
        neighbor_position_fraction=neighbor_within_source.numpy().astype(np.float32),
        roi_attention=roi_conditional.numpy().astype(np.float32),
        representative_frames=representative_frames.numpy(),
        roi_yx=roi_yx.numpy(),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
