"""Compute exact sampled T->S softmax summaries from captured post-RoPE Q/K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def allowed_keys(layout: dict, frame: int, device: torch.device) -> torch.Tensor:
    used = int(layout["used_len"])
    start = int(layout["video_start"])
    frames = int(layout["num_frames"])
    per = int(layout["tokens_per_frame"])
    end = start + frames * per
    if frame in (0, frames - 1):
        return torch.arange(used, device=device, dtype=torch.long)
    global_idx = torch.cat(
        (
            torch.arange(0, start, device=device),
            torch.arange(end, used, device=device),
        )
    )
    lo = max(0, ((frame // 5) - 1) * 5)
    hi = min(frames - 1, ((frame // 5) + 2) * 5 - 1)
    window = torch.arange(start + lo * per, start + (hi + 1) * per, device=device)
    pieces = [global_idx, window]
    if lo > 0:
        pieces.append(torch.arange(start, start + per, device=device))
    if hi < frames - 1:
        pieces.append(torch.arange(end - per, end, device=device))
    return torch.cat(pieces).sort().values.unique_consecutive()


def main() -> None:
    a = args()
    paths = sorted(a.capture_dir.glob("layer_24_rank_*.pt"))
    if len(paths) != 2:
        raise RuntimeError(f"expected two rank files, found {paths}")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")

    first = torch.load(paths[0], map_location="cpu", weights_only=True)
    layout = first["layout"]
    frames = int(layout["num_frames"])
    per = int(layout["tokens_per_frame"])
    height = int(layout["frame_height"])
    width = int(layout["frame_width"])
    used = int(layout["used_len"])
    video_start = int(layout["video_start"])
    video_end = video_start + frames * per
    q_indices = first["query_indices"].long()
    q_active = first.get("query_active", torch.ones_like(q_indices, dtype=torch.bool)).bool()
    if not bool(torch.all(q_active)):
        raise RuntimeError("captured query set contains skipped queries")
    q_frame = (q_indices - video_start) // per
    q_spatial = (q_indices - video_start) % per
    if q_indices.numel() % frames:
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
    source_frames = int(source_times.numel())
    if source_frames != frames or source_ys.numel() != height or source_xs.numel() != width:
        raise RuntimeError(
            f"unexpected source grid: {(source_frames, source_ys.numel(), source_xs.numel())}"
        )
    counts = torch.bincount(source_frame, minlength=source_frames)
    if not bool(torch.all(counts == per)):
        raise RuntimeError(f"source frame counts are not {per}: {counts.tolist()}")

    lookup = torch.full((used,), -1, dtype=torch.long)
    lookup[source_positions] = torch.arange(source_positions.numel())
    representative_frames = torch.tensor([0, frames // 4, frames // 2, 3 * frames // 4, frames - 1])
    roi_yx = torch.tensor([max(0, height * 2 // 5), max(0, width * 3 // 4)])

    temporal_numerator = torch.zeros((frames, source_frames), dtype=torch.float64)
    temporal_denominator = torch.zeros(frames, dtype=torch.float64)
    exact_position_numerator = torch.zeros(frames, dtype=torch.float64)
    neighbor_position_numerator = torch.zeros(frames, dtype=torch.float64)
    same_frame_numerator = torch.zeros(frames, dtype=torch.float64)
    roi_cube = torch.zeros(
        (representative_frames.numel(), source_frames, height, width), dtype=torch.float64
    )
    total_heads = 0

    for path in paths:
        row = torch.load(path, map_location="cpu", weights_only=True)
        if row["layout"] != layout or not torch.equal(row["query_indices"], q_indices):
            raise RuntimeError(f"rank metadata mismatch: {path}")
        q_all = row["q_selected"].to(device=device)
        k_all = row["k_used"].to(device=device)
        scale = float(row["softmax_scale"])
        heads = int(q_all.shape[1])
        total_heads += heads
        lookup_gpu = lookup.to(device)
        sf_gpu = source_frame.to(device)
        sy_gpu = source_y.to(device)
        sx_gpu = source_x.to(device)

        for frame in range(frames):
            q_mask_cpu = q_frame == frame
            q_rows = q_all[q_mask_cpu.to(device)]
            q_sp = q_spatial[q_mask_cpu].to(device)
            keys = allowed_keys(layout, frame, device)
            key_rows = k_all.index_select(0, keys)
            logits = torch.einsum("qhd,khd->hqk", q_rows, key_rows) * scale
            probs = torch.softmax(logits.float(), dim=-1)
            src_ids = lookup_gpu.index_select(0, keys)
            src_valid = src_ids >= 0
            src_ids = src_ids[src_valid]
            source_probs = probs[:, :, src_valid]
            src_frames_for_keys = sf_gpu.index_select(0, src_ids)
            temporal = torch.zeros(
                (heads, q_rows.shape[0], source_frames), device=device, dtype=torch.float32
            )
            temporal.scatter_add_(
                2,
                src_frames_for_keys.view(1, 1, -1).expand(heads, q_rows.shape[0], -1),
                source_probs,
            )
            temporal_numerator[frame] += temporal.sum((0, 1)).double().cpu()
            temporal_denominator[frame] += heads * q_rows.shape[0]
            same_frame_numerator[frame] += temporal[:, :, frame].sum().double().cpu()

            src_y_for_keys = sy_gpu.index_select(0, src_ids)
            src_x_for_keys = sx_gpu.index_select(0, src_ids)
            for qi in range(q_rows.shape[0]):
                qy = q_sp[qi] // width
                qx = q_sp[qi] % width
                is_same_frame = src_frames_for_keys == frame
                exact = is_same_frame & (src_y_for_keys == qy) & (src_x_for_keys == qx)
                near = is_same_frame & ((src_y_for_keys - qy).abs() <= 1) & ((src_x_for_keys - qx).abs() <= 1)
                exact_position_numerator[frame] += source_probs[:, qi, exact].sum().double().cpu()
                neighbor_position_numerator[frame] += source_probs[:, qi, near].sum().double().cpu()

            rep = torch.nonzero(representative_frames == frame, as_tuple=False).view(-1)
            if rep.numel():
                qy_all = q_sp // width
                qx_all = q_sp % width
                nearest = ((qy_all - roi_yx[0].to(device)).square() + (qx_all - roi_yx[1].to(device)).square()).argmin()
                one = source_probs[:, nearest].sum(0)
                cube = torch.zeros(source_frames * height * width, device=device, dtype=torch.float32)
                linear = (
                    src_frames_for_keys * (height * width)
                    + src_y_for_keys * width
                    + src_x_for_keys
                )
                cube.scatter_add_(0, linear, one)
                roi_cube[rep.item()] += cube.view(source_frames, height, width).double().cpu()
            del logits, probs, source_probs, temporal, key_rows

        del q_all, k_all
        torch.cuda.empty_cache()

    temporal_raw = temporal_numerator / temporal_denominator[:, None]
    source_mass = temporal_raw.sum(1)
    temporal_conditional = temporal_raw / source_mass[:, None].clamp_min(1e-12)
    diagonal_conditional = temporal_conditional.diag()
    expected_abs_offset = (
        temporal_conditional
        * (torch.arange(frames)[:, None] - torch.arange(source_frames)[None, :]).abs()
    ).sum(1)
    top_frame = temporal_conditional.argmax(1)
    diagonal_argmax = top_frame == torch.arange(frames)
    exact_within_source = exact_position_numerator / (
        temporal_denominator * source_mass
    ).clamp_min(1e-12)
    neighbor_within_source = neighbor_position_numerator / (
        temporal_denominator * source_mass
    ).clamp_min(1e-12)
    roi_cube /= total_heads
    roi_conditional = roi_cube / roi_cube.sum((1, 2, 3), keepdim=True).clamp_min(1e-12)

    summary = {
        "schema": "h3_attention_map_summary_v1",
        "layer": int(first["layer"]),
        "captured_stage": (
            f"DiT denoising forward step {int(first.get('spotedit_step', 0))}; "
            "post-QK-norm and post-RoPE"
        ),
        "spotedit_step": int(first.get("spotedit_step", 0)),
        "spotedit_refresh": bool(first.get("spotedit_refresh", False)),
        "spotedit_skip_step": bool(first.get("spotedit_skip_step", False)),
        "target_active_ratio": float(first.get(
            "target_active_mask", torch.ones_like(first["target_positions"], dtype=torch.bool)
        ).float().mean()),
        "query_sampling": (
            f"{q_indices.numel() // frames} active spatial queries per target frame"
            if bool(first.get("spotedit_skip_step", False))
            else f"{q_indices.numel() // frames} spatial queries per target frame"
        ),
        "heads": total_heads,
        "target_grid": [frames, height, width],
        "source_grid": [source_frames, height, width],
        "mean_total_attention_mass_to_source": float(source_mass.mean()),
        "mean_same_frame_fraction_within_source_attention": float(diagonal_conditional.mean()),
        "median_same_frame_fraction_within_source_attention": float(diagonal_conditional.median()),
        "same_frame_is_argmax_fraction": float(diagonal_argmax.float().mean()),
        "mean_expected_absolute_frame_offset": float(expected_abs_offset.mean()),
        "mean_exact_same_position_fraction_within_all_source_attention": float(exact_within_source.mean()),
        "mean_3x3_same_frame_neighborhood_fraction_within_all_source_attention": float(neighbor_within_source.mean()),
        "mean_same_frame_fraction_under_uniform_source_attention": 1.0 / source_frames,
        "mean_exact_position_fraction_under_uniform_source_attention": 1.0 / (source_frames * per),
        "mean_3x3_fraction_upper_bound_under_uniform_source_attention": 9.0 / (source_frames * per),
        "top_source_frame_by_target": top_frame.tolist(),
        "representative_target_frames": representative_frames.tolist(),
        "requested_roi_yx": roi_yx.tolist(),
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        a.output_dir / "attention_maps.npz",
        temporal_raw=temporal_raw.numpy().astype(np.float32),
        temporal_conditional=temporal_conditional.numpy().astype(np.float32),
        source_mass=source_mass.numpy().astype(np.float32),
        diagonal_conditional=diagonal_conditional.numpy().astype(np.float32),
        expected_abs_offset=expected_abs_offset.numpy().astype(np.float32),
        exact_position_fraction=exact_within_source.numpy().astype(np.float32),
        neighbor_position_fraction=neighbor_within_source.numpy().astype(np.float32),
        roi_attention=roi_conditional.numpy().astype(np.float32),
        target_active_mask=first.get(
            "target_active_mask", torch.ones_like(first["target_positions"], dtype=torch.bool)
        ).view(frames, height, width).numpy(),
        representative_frames=representative_frames.numpy(),
        roi_yx=roi_yx.numpy(),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
