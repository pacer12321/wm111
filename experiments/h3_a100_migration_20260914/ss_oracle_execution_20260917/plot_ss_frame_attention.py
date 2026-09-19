#!/usr/bin/env python3
"""Plot dense S-query -> S-key frame attention from captured H3 Q/K tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_layer(capture_dir: Path, layer: int) -> dict:
    shards = [
        torch.load(
            capture_dir / f"layer_{layer:02d}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for rank in (0, 1)
    ]
    first = shards[0]

    def merge(value_name: str, index_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([item[value_name] for item in shards])
        indices = torch.cat([item[index_name] for item in shards]).long()
        order = torch.argsort(indices)
        return values.index_select(0, order), indices.index_select(0, order)

    q, q_indices = merge("q_selected", "query_indices")
    k, k_indices = merge("k_local", "key_indices")
    used_len = int(first["layout"]["used_len"])
    if not torch.equal(k_indices, torch.arange(used_len)):
        raise ValueError("K rows do not reconstruct the logical sequence")
    expected_q = torch.sort(first["all_query_indices"].long()).values
    if not torch.equal(q_indices, expected_q):
        raise ValueError("source query sample differs from capture metadata")
    return {**first, "q": q, "q_indices": q_indices, "k": k}


@torch.inference_mode()
def frame_map(payload: dict, device: torch.device, query_chunk: int) -> dict:
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    scale = float(payload["softmax_scale"])
    used_len = k.shape[0]
    frames = int(payload["layout"]["num_frames"])
    tokens_per_frame = int(payload["layout"]["tokens_per_frame"])
    source_positions = payload["source_positions"].long().to(device)
    q_indices = payload["q_indices"].long()

    logical_to_source_frame = torch.full((used_len,), -1, dtype=torch.long)
    # source_coords contains RoPE coordinates, not zero-based frame ids.
    # The source positions themselves are stored in frame-major grid order.
    logical_to_source_frame[payload["source_positions"].long()] = torch.arange(
        frames, dtype=torch.long
    ).repeat_interleave(tokens_per_frame)
    query_frames = logical_to_source_frame.index_select(0, q_indices)
    if (query_frames < 0).any():
        raise ValueError("captured query is not a source token")
    if source_positions.numel() != frames * tokens_per_frame:
        raise ValueError("source grid size mismatch")

    mass = torch.zeros((frames, frames), dtype=torch.float64)
    counts = torch.zeros(frames, dtype=torch.float64)
    k_heads = k.transpose(0, 1).contiguous()  # H,K,D
    source_positions_sorted = source_positions
    for start in range(0, q.shape[0], query_chunk):
        stop = min(start + query_chunk, q.shape[0])
        q_chunk = q[start:stop].transpose(0, 1).contiguous()  # H,Q,D
        logits = torch.bmm(q_chunk, k_heads.transpose(1, 2)) * scale
        probs = torch.softmax(logits.float(), dim=-1)
        source_probs = probs.index_select(2, source_positions_sorted)
        source_by_frame = source_probs.reshape(
            probs.shape[0], probs.shape[1], frames, tokens_per_frame
        ).sum(dim=-1)
        source_by_frame = source_by_frame.mean(dim=0).double().cpu()  # Q,F; heads averaged
        for local_row, q_frame in enumerate(query_frames[start:stop].tolist()):
            mass[q_frame] += source_by_frame[local_row]
            counts[q_frame] += 1
        del logits, probs, source_probs, source_by_frame

    if (counts == 0).any():
        raise ValueError("at least one source frame has no sampled query")
    absolute = mass / counts[:, None]
    ss_total = absolute.sum(dim=1)
    conditional = absolute / ss_total[:, None].clamp_min(1e-30)
    return {
        "absolute": absolute.numpy(),
        "conditional": conditional.numpy(),
        "ss_total": ss_total.numpy(),
        "queries_per_frame": counts.numpy(),
    }


def metrics(matrix: np.ndarray, seed: int = 20260917) -> dict:
    frames = matrix.shape[0]
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})
    periodic = float(matrix[:, anchors].sum(axis=1).mean())
    diagonal = float(np.diag(matrix).mean())
    global_share = matrix.mean(axis=0)
    top_order = np.argsort(-global_share)
    rng = np.random.default_rng(seed)
    random_coverages = []
    for _ in range(5000):
        chosen = rng.choice(frames, size=len(anchors), replace=False)
        random_coverages.append(float(matrix[:, chosen].sum(axis=1).mean()))
    shift_coverages = {}
    for shift in range(5):
        shifted = sorted(set(range(shift, frames, 5)) | {0, frames - 1})
        shift_coverages[str(shift)] = {
            "frames": shifted,
            "mean_coverage": float(matrix[:, shifted].sum(axis=1).mean()),
        }
    return {
        "fixed_anchor_frames": anchors,
        "fixed_anchor_mean_conditional_ss_coverage": periodic,
        "random_same_count_mean": float(np.mean(random_coverages)),
        "random_same_count_p95": float(np.quantile(random_coverages, 0.95)),
        "diagonal_mean_conditional_ss_share": diagonal,
        "global_key_frame_order": top_order.tolist(),
        "global_key_frame_share": global_share.tolist(),
        "shifted_periodic_sets": shift_coverages,
    }


def draw(matrices: dict[int | str, np.ndarray], output: Path) -> None:
    # PIL-only renderer so the analysis does not depend on matplotlib being
    # present in the production H3 environment.
    canvas = Image.new("RGB", (1500, 1420), "white")
    draw_ctx = ImageDraw.Draw(canvas)
    draw_ctx.text((350, 20), "H3 Dense S-S Frame Attention | all heads averaged | step 4", fill="black")
    keys = list(matrices)
    vmax = max(float(np.quantile(matrix, 0.995)) for matrix in matrices.values())
    heat_size, cell = 518, 14
    anchors = sorted(set(range(0, 37, 5)) | {36})
    for panel_index, key in enumerate(keys):
        panel_x = 45 + (panel_index % 2) * 735
        panel_y = 75 + (panel_index // 2) * 675
        matrix = matrices[key]
        normalized = np.clip(matrix / max(vmax, 1e-12), 0.0, 1.0)
        rgb = np.empty((37, 37, 3), dtype=np.uint8)
        rgb[..., 0] = (247 - 214 * normalized).astype(np.uint8)
        rgb[..., 1] = (251 - 138 * normalized).astype(np.uint8)
        rgb[..., 2] = (255 - 74 * normalized).astype(np.uint8)
        heat = Image.fromarray(rgb, mode="RGB").resize(
            (heat_size, heat_size), resample=Image.Resampling.NEAREST
        )
        hx, hy = panel_x + 105, panel_y + 55
        canvas.paste(heat, (hx, hy))
        title = f"Layer {key}" if isinstance(key, int) else "Mean: layers 8 / 24 / 41"
        draw_ctx.text((hx + 175, panel_y + 15), title, fill="black")
        draw_ctx.rectangle((hx, hy, hx + heat_size, hy + heat_size), outline="black", width=2)
        draw_ctx.line((hx, hy, hx + heat_size, hy + heat_size), fill=(30, 30, 30), width=1)
        for anchor in anchors:
            x = hx + anchor * cell + cell // 2
            draw_ctx.line((x, hy, x, hy + heat_size), fill=(220, 55, 55), width=1)
        for tick in [0, 5, 10, 15, 20, 25, 30, 35, 36]:
            x = hx + tick * cell + cell // 2
            draw_ctx.text((x - 6, hy + heat_size + 8), str(tick), fill="black")
        for tick in [0, 9, 18, 27, 36]:
            y = hy + tick * cell + cell // 2
            draw_ctx.text((hx - 30, y - 6), str(tick), fill="black")
        draw_ctx.text((hx + 180, hy + heat_size + 32), "S key frame", fill="black")
        draw_ctx.text((panel_x + 5, hy + 245), "S query frame", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    parser.add_argument("--query-chunk", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    matrices: dict[int | str, np.ndarray] = {}
    report: dict[str, object] = {
        "schema": "h3_dense_ss_frame_attention_v1",
        "definition": (
            "Rows are sampled source-query frames; columns are source-key frames. "
            "Softmax denominator contains the full logical key sequence. Values are "
            "then conditioned on S-S mass so every heatmap row sums to one."
        ),
        "layers": {},
    }
    for layer in args.layers:
        payload = load_layer(args.capture_dir, layer)
        result = frame_map(payload, device, args.query_chunk)
        matrices[layer] = result["conditional"]
        report["layers"][str(layer)] = {
            "mean_absolute_ss_mass": float(np.mean(result["ss_total"])),
            "queries_per_frame": result["queries_per_frame"].tolist(),
            "conditional_frame_matrix": result["conditional"].tolist(),
            "metrics": metrics(result["conditional"], seed=20260917 + layer),
        }
        del payload
        torch.cuda.empty_cache()
    matrices["mean"] = np.mean([matrices[layer] for layer in args.layers], axis=0)
    report["mean_layers"] = {
        "layers": args.layers,
        "conditional_frame_matrix": matrices["mean"].tolist(),
        "metrics": metrics(matrices["mean"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(matrices, args.output_dir / "ss_frame_attention_layers8_24_41_step04.png")
    (args.output_dir / "ss_frame_attention_layers8_24_41_step04.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "png": str(args.output_dir / "ss_frame_attention_layers8_24_41_step04.png"),
        "json": str(args.output_dir / "ss_frame_attention_layers8_24_41_step04.json"),
        "metrics": report["mean_layers"]["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
