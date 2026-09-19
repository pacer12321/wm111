from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_layer(step_dir: Path, layer: int) -> dict:
    shards = [
        torch.load(
            step_dir / f"layer_{layer:02d}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in (0, 1)
    ]
    first = shards[0]
    q = torch.cat([item["q_selected"] for item in shards])
    q_indices = torch.cat([item["query_indices"] for item in shards]).long()
    q_order = torch.argsort(q_indices)
    q = q.index_select(0, q_order)
    q_indices = q_indices.index_select(0, q_order)

    k = torch.cat([item["k_local"] for item in shards])
    k_indices = torch.cat([item["key_indices"] for item in shards]).long()
    k_order = torch.argsort(k_indices)
    k = k.index_select(0, k_order)
    k_indices = k_indices.index_select(0, k_order)
    used_len = int(first["layout"]["used_len"])
    if not torch.equal(k_indices, torch.arange(used_len)):
        raise ValueError("K rows do not reconstruct the logical sequence")

    requested_indices = first["all_query_indices"].long()
    requested_coords = first["all_query_coords"].long()
    index_to_coord = {
        int(index): coord
        for index, coord in zip(
            requested_indices.tolist(), requested_coords.tolist(), strict=True
        )
    }
    q_coords = torch.tensor([index_to_coord[int(index)] for index in q_indices])
    return {**first, "q": q, "q_indices": q_indices, "q_coords": q_coords, "k": k}


@torch.inference_mode()
def compute_frame_map(payload: dict, device: torch.device) -> dict:
    q = payload["q"].to(device)
    k = payload["k"].to(device)
    q_times = payload["q_coords"][:, 0].long()
    source_times = torch.unique(q_times, sorted=True)
    target_positions = payload["target_positions"].long()
    target_times_per_token = payload["target_coords"][:, 0].long()
    target_times = torch.unique(target_times_per_token, sorted=True)
    if source_times.numel() != target_times.numel():
        raise ValueError("source and target frame counts differ")

    target_frame_positions = [
        target_positions[target_times_per_token == time].to(device)
        for time in target_times
    ]
    absolute_rows = []
    scale = float(payload["softmax_scale"])
    k_heads = k.transpose(0, 1)
    for source_time in source_times:
        q_frame = q[q_times == source_time.item()].transpose(0, 1)
        logits = torch.bmm(q_frame, k_heads.transpose(1, 2)) * scale
        probabilities = torch.softmax(logits.float(), dim=-1)
        masses = torch.stack(
            [probabilities.index_select(2, positions).sum(dim=-1).mean() for positions in target_frame_positions]
        )
        absolute_rows.append(masses.cpu())
        del logits, probabilities
    absolute = torch.stack(absolute_rows)
    total_st = absolute.sum(dim=1)
    conditional = absolute / total_st[:, None].clamp_min(1e-12)
    diagonal = torch.arange(absolute.shape[0])
    diagonal_conditional = conditional[diagonal, diagonal]
    diagonal_absolute = absolute[diagonal, diagonal]
    target_frame_share = absolute.sum(dim=0)
    target_frame_share = target_frame_share / target_frame_share.sum().clamp_min(1e-12)
    target_order = torch.argsort(target_frame_share, descending=True)
    anchor_coverages = {
        str(k): float(target_frame_share.index_select(0, target_order[:k]).sum())
        for k in (1, 3, 5, 8, 12, 16)
    }
    return {
        "absolute": absolute,
        "conditional": conditional,
        "mean_total_st_mass": float(total_st.mean()),
        "mean_corresponding_absolute_mass": float(diagonal_absolute.mean()),
        "mean_corresponding_share_within_st": float(diagonal_conditional.mean()),
        "median_corresponding_share_within_st": float(diagonal_conditional.median()),
        "top1_corresponding_rate": float(
            (conditional.argmax(dim=1) == diagonal).float().mean()
        ),
        "uniform_share": 1.0 / absolute.shape[0],
        "target_frame_share": target_frame_share,
        "target_anchor_order": target_order,
        "anchor_coverages": anchor_coverages,
    }


def _heat_color(value: float) -> tuple[int, int, int]:
    stops = (
        (0.00, (4, 5, 30)),
        (0.25, (65, 15, 110)),
        (0.50, (180, 45, 95)),
        (0.75, (245, 125, 45)),
        (1.00, (255, 245, 180)),
    )
    value = min(max(value, 0.0), 1.0)
    for (left_x, left), (right_x, right) in zip(stops, stops[1:], strict=True):
        if value <= right_x:
            alpha = (value - left_x) / (right_x - left_x)
            return tuple(int(a + alpha * (b - a)) for a, b in zip(left, right, strict=True))
    return stops[-1][1]


def _draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    matrix: np.ndarray,
    origin_x: int,
    origin_y: int,
    cell: int,
    title: str,
) -> None:
    rows, cols = matrix.shape
    low = float(matrix.min())
    high = float(matrix.max())
    scale = max(high - low, 1e-12)
    draw.text((origin_x, origin_y - 28), title, fill="black")
    for row in range(rows):
        for col in range(cols):
            normalized = (float(matrix[row, col]) - low) / scale
            x0 = origin_x + col * cell
            y0 = origin_y + row * cell
            draw.rectangle(
                (x0, y0, x0 + cell - 1, y0 + cell - 1),
                fill=_heat_color(normalized),
            )
            if row == col:
                draw.rectangle(
                    (x0, y0, x0 + cell - 1, y0 + cell - 1),
                    outline=(0, 255, 255),
                    width=1,
                )
    for tick in range(0, rows, 5):
        draw.text((origin_x - 24, origin_y + tick * cell), str(tick), fill="black")
        draw.text((origin_x + tick * cell, origin_y + rows * cell + 4), str(tick), fill="black")
    draw.text((origin_x, origin_y + rows * cell + 24), "Target key frame", fill="black")
    draw.text((origin_x, origin_y - 48), f"min={low:.5f} max={high:.5f}", fill="black")


def save_figure(result: dict, output: Path, layer: int, step: int) -> None:
    absolute = result["absolute"].numpy()
    conditional = result["conditional"].numpy()
    cell = 12
    panel = absolute.shape[0] * cell
    width = panel * 2 + 150
    height = panel + 145
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (24, 12),
        f"Fastest pipeline | step {step} | layer {layer} | heads + 16 spatial source queries/frame averaged",
        fill="black",
    )
    draw.text(
        (24, 30),
        f"mean S->T mass={result['mean_total_st_mass']:.4f} | corresponding share={result['mean_corresponding_share_within_st']:.4f} | diagonal top1={result['top1_corresponding_rate']:.3f}",
        fill="black",
    )
    _draw_panel(image, draw, absolute, 45, 92, cell, "Absolute S-query -> T-key attention mass")
    _draw_panel(
        image,
        draw,
        conditional,
        panel + 105,
        92,
        cell,
        "Conditional distribution within S->T mass",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step_dir", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--layer", type=int, default=24)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-size attention map")
    payload = load_layer(args.step_dir, args.layer)
    result = compute_frame_map(payload, torch.device("cuda:0"))
    png = args.output_prefix.with_suffix(".png")
    save_figure(result, png, int(payload["layer"]), int(payload["step"]))
    report = {
        key: value
        for key, value in result.items()
        if key not in {
            "absolute",
            "conditional",
            "target_frame_share",
            "target_anchor_order",
        }
    }
    target_frame_share = result["target_frame_share"]
    target_anchor_order = result["target_anchor_order"]
    report["target_frame_share"] = [float(value) for value in target_frame_share]
    report["target_anchor_order"] = [int(value) for value in target_anchor_order]
    report["top_target_anchors"] = [
        {
            "frame": int(frame),
            "share_within_st": float(target_frame_share[frame]),
        }
        for frame in target_anchor_order[:16]
    ]
    report.update(
        {
            "layer": int(payload["layer"]),
            "step": int(payload["step"]),
            "source_queries": int(payload["q"].shape[0]),
            "heads": int(payload["q"].shape[1]),
            "png": str(png),
        }
    )
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
