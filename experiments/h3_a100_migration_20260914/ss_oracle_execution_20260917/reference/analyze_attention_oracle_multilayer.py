"""Evaluate Oracle T-to-S frame masks from captured dense Ref2VA Q/K/V.

The mask changes only source-video visibility. Text, target-video, audio, and
all other dense keys remain untouched. Dynamic Top-K counts frames in addition
to the fixed H3 structural anchors and the corresponding-frame safety edge.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="8,24,41")
    parser.add_argument("--steps", default="0,12,24,36,48")
    parser.add_argument("--ks", default="5,8,12,16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--teacher-mode",
        choices=("dense", "b_vdn"),
        default="dense",
        help="Dense A visibility or B's exact c5/r1 T-T visibility with dense T-S.",
    )
    return parser.parse_args()


def load_capture(step_dir: Path, layer: int) -> dict:
    paths = sorted(step_dir.glob(f"layer_{layer:02d}_rank_*.pt"))
    if len(paths) != 2:
        raise RuntimeError(f"expected two rank files for layer {layer}: {paths}")
    rows = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    first = rows[0]
    if first.get("schema") not in {
        "h3_dense_local_post_rope_qkv_v2",
        "h3_b_vdn_local_post_rope_qkv_v1",
    }:
        raise RuntimeError(f"unexpected schema: {first.get('schema')!r}")
    for row in rows[1:]:
        for key in ("layout", "step", "layer", "num_heads", "head_dim"):
            if row[key] != first[key]:
                raise RuntimeError(f"rank metadata mismatch for {key}")

    query_indices = torch.cat([row["query_indices"].long() for row in rows])
    q = torch.cat([row["q_selected"] for row in rows])
    order = query_indices.argsort()
    query_indices = query_indices[order]
    q = q[order]
    expected = first["all_query_indices"].long().sort().values
    if not torch.equal(query_indices, expected):
        raise RuntimeError("rank query shards do not reconstruct sampled queries")

    key_indices = torch.cat([row["key_indices"].long() for row in rows])
    k = torch.cat([row["k_local"] for row in rows])
    v = torch.cat([row["v_local"] for row in rows])
    order = key_indices.argsort()
    key_indices = key_indices[order]
    k = k[order]
    v = v[order]
    used = int(first["layout"]["used_len"])
    if not torch.equal(key_indices, torch.arange(used)):
        raise RuntimeError("rank key shards do not reconstruct all used dense keys")
    return {"meta": first, "query_indices": query_indices, "q": q, "k": k, "v": v}


def b_vdn_allowed_keys(
    *, target_frame: int, frames: int, per: int, layout: dict, device: torch.device
) -> torch.Tensor:
    """Reproduce B's exact OpenVDN c5/r1 Softmax key set for one T frame."""
    used = int(layout["used_len"])
    video_start = int(layout["video_start"])
    video_end = video_start + frames * per
    if target_frame in (0, frames - 1):
        return torch.ones(used, device=device, dtype=torch.bool)
    chunk = int(layout.get("vdn_window_chunk", 5))
    radius = int(layout.get("vdn_window_radius", 1))
    lo = max(0, ((target_frame // chunk) - radius) * chunk)
    hi = min(frames - 1, ((target_frame // chunk) + radius + 1) * chunk - 1)
    allowed = torch.ones(used, device=device, dtype=torch.bool)
    allowed[video_start:video_end] = False
    allowed[video_start + lo * per : video_start + (hi + 1) * per] = True
    if lo > 0:
        allowed[video_start : video_start + per] = True
    if hi < frames - 1:
        allowed[video_end - per : video_end] = True
    return allowed


def analyze_one(
    payload: dict, ks: list[int], device: torch.device, teacher_mode: str
) -> dict:
    first = payload["meta"]
    layout = first["layout"]
    frames = int(layout["num_frames"])
    per = int(layout["tokens_per_frame"])
    used = int(layout["used_len"])
    heads = int(first["num_heads"])

    q = payload["q"].to(device)
    k = payload["k"].to(device)
    v = payload["v"].to(device=device, dtype=torch.float32)
    query_indices = payload["query_indices"]
    target_positions = first["target_positions"].long()
    target_lookup = torch.full((used,), -1, dtype=torch.long)
    target_lookup[target_positions] = torch.arange(target_positions.numel())
    q_target = target_lookup[query_indices]
    if bool(torch.any(q_target < 0)):
        raise RuntimeError("sampled query is not a target-video row")
    q_frame = q_target // per

    source_positions = first["source_positions"].long()
    source_coords = first["source_coords"].float()
    valid = (source_positions >= 0) & (source_positions < used)
    source_positions = source_positions[valid]
    source_coords = source_coords[valid]
    source_times, source_frame = torch.unique(
        source_coords[:, 0], sorted=True, return_inverse=True
    )
    if source_times.numel() != frames or source_positions.numel() != frames * per:
        raise RuntimeError("unexpected source grid")
    source_positions_gpu = source_positions.to(device)
    source_frame_gpu = source_frame.to(device)
    source_v = v.index_select(0, source_positions_gpu)

    anchors = sorted(set(range(0, frames, 5)) | {0, frames - 1})
    anchor_tensor = torch.tensor(anchors, device=device, dtype=torch.long)
    scale = float(first["softmax_scale"])

    source_mass_sum = 0.0
    source_mass_count = 0
    anchor_mass = 0.0
    corresponding_mass = 0.0
    local2_mass = 0.0
    ts_output_sq = 0.0
    dense_output_sq = 0.0
    teacher_frame_distributions: list[list[float]] = []
    oracle = {
        size: {
            "retained_source_mass": 0.0,
            "source_mass": 0.0,
            "error_sq": 0.0,
            "dense_sq": 0.0,
            "visible_counts": [],
            "routes": [],
            "head_error_sq": torch.zeros(heads, dtype=torch.float64),
            "head_dense_sq": torch.zeros(heads, dtype=torch.float64),
        }
        for size in ks
    }

    for target_frame in range(frames):
        local_mask = q_frame == target_frame
        q_rows = q[local_mask.to(device)]
        logits = torch.einsum("qhd,khd->hqk", q_rows, k) * scale
        if teacher_mode == "b_vdn":
            teacher_allowed = b_vdn_allowed_keys(
                target_frame=target_frame,
                frames=frames,
                per=per,
                layout=layout,
                device=device,
            )
            teacher_logits = logits.masked_fill(
                ~teacher_allowed.view(1, 1, -1), -torch.inf
            )
        else:
            teacher_allowed = torch.ones(used, device=device, dtype=torch.bool)
            teacher_logits = logits
        probs = torch.softmax(teacher_logits.float(), dim=-1)
        dense_out = torch.einsum("hqk,khd->qhd", probs, v)
        source_probs = probs.index_select(2, source_positions_gpu)
        ts_out = torch.einsum("hqs,shd->qhd", source_probs, source_v)
        dense_output_sq += float(dense_out.double().square().sum().cpu())
        ts_output_sq += float(ts_out.double().square().sum().cpu())

        frame_mass = torch.zeros(
            (heads, q_rows.shape[0], frames), dtype=torch.float32, device=device
        )
        frame_mass.scatter_add_(
            2,
            source_frame_gpu.view(1, 1, -1).expand(heads, q_rows.shape[0], -1),
            source_probs,
        )
        teacher = frame_mass.sum((0, 1))
        current_source_mass = float(teacher.sum().double().cpu())
        teacher_frame_distributions.append(
            (teacher / teacher.sum().clamp_min(1e-30)).double().cpu().tolist()
        )
        source_mass_sum += current_source_mass
        source_mass_count += heads * q_rows.shape[0]
        anchor_mass += float(teacher.index_select(0, anchor_tensor).sum().double().cpu())
        corresponding_mass += float(teacher[target_frame].double().cpu())
        lo = max(0, target_frame - 2)
        hi = min(frames, target_frame + 3)
        local2_mass += float(teacher[lo:hi].sum().double().cpu())

        base = set(anchors) | {target_frame}
        for size in ks:
            candidates = [index for index in range(frames) if index not in base]
            take = min(size, len(candidates))
            if take:
                candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.long)
                scores = teacher.index_select(0, candidate_tensor)
                selected = candidate_tensor.index_select(0, torch.topk(scores, take).indices)
                dynamic = [int(value) for value in selected.cpu().tolist()]
            else:
                dynamic = []
            visible = sorted(base | set(dynamic))
            visible_tensor = torch.tensor(visible, device=device, dtype=torch.long)
            allowed_frames = torch.zeros(frames, device=device, dtype=torch.bool)
            allowed_frames[visible_tensor] = True
            disallowed_source = source_positions_gpu[~allowed_frames[source_frame_gpu]]
            key_mask = torch.zeros(used, device=device, dtype=torch.bool)
            key_mask[disallowed_source] = True
            sparse_allowed = teacher_allowed & ~key_mask
            sparse_probs = torch.softmax(
                logits.masked_fill(
                    ~sparse_allowed.view(1, 1, -1), -torch.inf
                ).float(),
                dim=-1,
            )
            sparse_out = torch.einsum("hqk,khd->qhd", sparse_probs, v)
            diff = sparse_out - dense_out
            row = oracle[size]
            row["retained_source_mass"] += float(
                teacher.index_select(0, visible_tensor).sum().double().cpu()
            )
            row["source_mass"] += current_source_mass
            row["error_sq"] += float(diff.double().square().sum().cpu())
            row["dense_sq"] += float(dense_out.double().square().sum().cpu())
            row["visible_counts"].append(len(visible))
            row["routes"].append({
                "target_frame": target_frame,
                "dynamic_topk": dynamic,
                "visible_frames": visible,
            })
            row["head_error_sq"] += diff.double().square().sum((0, 2)).cpu()
            row["head_dense_sq"] += dense_out.double().square().sum((0, 2)).cpu()
            del sparse_probs, sparse_out, diff, key_mask
        del logits, probs, dense_out, source_probs, ts_out, frame_mass

    oracle_json = {}
    for size, row in oracle.items():
        head_error = torch.sqrt(
            row["head_error_sq"] / row["head_dense_sq"].clamp_min(1e-30)
        )
        sorted_head = head_error.sort().values
        oracle_json[str(size)] = {
            "dynamic_k_non_anchor": size,
            "mean_unique_visible_source_frames": sum(row["visible_counts"]) / frames,
            "min_unique_visible_source_frames": min(row["visible_counts"]),
            "max_unique_visible_source_frames": max(row["visible_counts"]),
            "mean_visible_source_fraction": (
                sum(row["visible_counts"]) / frames / frames
            ),
            "retained_dense_source_attention_mass_fraction": (
                row["retained_source_mass"] / max(row["source_mass"], 1e-30)
            ),
            "attention_core_relative_l2": math.sqrt(
                row["error_sq"] / max(row["dense_sq"], 1e-30)
            ),
            "per_head_relative_l2": {
                "min": float(sorted_head[0]),
                "median": float(sorted_head[len(sorted_head) // 2]),
                "p90": float(sorted_head[min(len(sorted_head) - 1, math.ceil(0.9 * len(sorted_head)) - 1)]),
                "max": float(sorted_head[-1]),
            },
            "routes": row["routes"],
        }

    return {
        "layer": int(first["layer"]),
        "step": int(first["step"]),
        "heads": heads,
        "sampled_queries_per_target_frame": int(q.shape[0] // frames),
        "source_frames": frames,
        "structural_anchors": anchors,
        "teacher_mode": teacher_mode,
        "mean_total_attention_mass_to_source": source_mass_sum / source_mass_count,
        "structural_anchor_share_within_source": anchor_mass / max(source_mass_sum, 1e-30),
        "corresponding_frame_share_within_source": corresponding_mass / max(source_mass_sum, 1e-30),
        "plus_minus_2_share_within_source": local2_mass / max(source_mass_sum, 1e-30),
        "ts_attention_core_output_norm_over_total": math.sqrt(
            ts_output_sq / max(dense_output_sq, 1e-30)
        ),
        "teacher_frame_distribution_within_source": teacher_frame_distributions,
        "oracle": oracle_json,
    }


def main() -> None:
    args = parse_args()
    layers = csv_ints(args.layers)
    steps = csv_ints(args.steps)
    ks = csv_ints(args.ks)
    if args.output_dir.exists():
        raise RuntimeError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    results = []
    for step in steps:
        for layer in layers:
            print(json.dumps({"status": "analyzing", "step": step, "layer": layer}), flush=True)
            payload = load_capture(args.capture_root / f"step_{step:02d}", layer)
            result = analyze_one(payload, ks, device, args.teacher_mode)
            results.append(result)
            (args.output_dir / f"step_{step:02d}_layer_{layer:02d}.json").write_text(
                json.dumps(result, indent=2)
            )
            del payload
            torch.cuda.empty_cache()
    summary = {
        "schema": "h3_multilayer_ts_oracle_v2",
        "attention": (
            "B teacher: VDN c5/r1 T-T plus dense T-S; only T-to-S visibility is masked"
            if args.teacher_mode == "b_vdn"
            else "Ref2VA base dense teacher; only T-to-S visibility is masked"
        ),
        "teacher_mode": args.teacher_mode,
        "layers": layers,
        "steps": steps,
        "dynamic_ks": ks,
        "dynamic_k_definition": "additional non-anchor, non-corresponding source frames",
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "completed", "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
