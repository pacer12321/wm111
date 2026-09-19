from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


BOUNDARIES = [0, 6158, 6159, 43454, 43455, 80750, 80751, 81215]


def load_logical(directory: Path) -> torch.Tensor:
    rows = []
    values = []
    for path in sorted(directory.glob("attention_block0_rank*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "h3_attention_equivalence_capture_v1":
            raise ValueError(f"unexpected capture schema in {path}")
        rows.append(payload["logical_indices"].long())
        values.append(payload["attention_output"])
    if len(rows) != 2:
        raise ValueError(f"expected two rank captures in {directory}, got {len(rows)}")
    logical_rows = torch.cat(rows)
    physical_values = torch.cat(values)
    order = torch.argsort(logical_rows)
    logical_rows = logical_rows.index_select(0, order)
    expected = torch.arange(logical_rows.numel())
    if not torch.equal(logical_rows, expected):
        raise ValueError(f"capture does not cover every logical row exactly once: {directory}")
    return physical_values.index_select(0, order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contiguous", type=Path)
    parser.add_argument("interleaved", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    contiguous = load_logical(args.contiguous)
    interleaved = load_logical(args.interleaved)
    if contiguous.shape != interleaved.shape:
        raise ValueError(f"capture shape mismatch: {contiguous.shape} vs {interleaved.shape}")

    max_abs = 0.0
    abs_sum = 0.0
    element_count = 0
    exact_count = 0
    all_close = True
    reference_square_sum = 0.0
    difference_square_sum = 0.0
    threshold_counts = {"gt_1e-3": 0, "gt_1e-2": 0, "gt_1e-1": 0, "gt_1": 0}
    worst = {"value": -1.0, "row": -1, "channel": -1, "reference": 0.0, "candidate": 0.0}
    chunk_rows = 512
    for start in range(0, contiguous.shape[0], chunk_rows):
        stop = min(start + chunk_rows, contiguous.shape[0])
        left = contiguous[start:stop]
        right = interleaved[start:stop]
        difference = (left.float() - right.float()).abs()
        max_abs = max(max_abs, float(difference.max()))
        abs_sum += float(difference.sum())
        element_count += difference.numel()
        exact_count += int((left == right).sum())
        all_close = all_close and torch.allclose(left.float(), right.float(), rtol=1e-3, atol=1e-3)
        reference_square_sum += float(left.float().square().sum())
        difference_square_sum += float(difference.square().sum())
        threshold_counts["gt_1e-3"] += int((difference > 1e-3).sum())
        threshold_counts["gt_1e-2"] += int((difference > 1e-2).sum())
        threshold_counts["gt_1e-1"] += int((difference > 1e-1).sum())
        threshold_counts["gt_1"] += int((difference > 1.0).sum())
        chunk_max = float(difference.max())
        if chunk_max > worst["value"]:
            flat = int(torch.argmax(difference))
            local_row = flat // difference.shape[1]
            channel = flat % difference.shape[1]
            worst = {
                "value": chunk_max,
                "row": start + local_row,
                "channel": channel,
                "reference": float(left[local_row, channel]),
                "candidate": float(right[local_row, channel]),
            }

    boundary_results = {}
    for row in BOUNDARIES:
        difference = (contiguous[row].float() - interleaved[row].float()).abs()
        boundary_results[str(row)] = {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "exact_fraction": float((contiguous[row] == interleaved[row]).float().mean()),
        }

    result = {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "max_abs": max_abs,
        "mean_abs": abs_sum / element_count,
        "exact_fraction": exact_count / element_count,
        "relative_l2": (difference_square_sum / reference_square_sum) ** 0.5,
        "threshold_fractions": {
            key: value / element_count for key, value in threshold_counts.items()
        },
        "worst_element": worst,
        "allclose_rtol_1e-3_atol_1e-3": all_close,
        "boundaries": boundary_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not all_close:
        raise SystemExit("attention captures are not numerically equivalent")


if __name__ == "__main__":
    main()
