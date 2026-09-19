"""Export compact timestep-separated T-to-S attention matrices for visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="0,12,24,36,48")
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]

    records = []
    global_max = 0.0
    for step in steps:
        directory = args.analysis_root / f"analysis_step_{step:02d}"
        matrix = np.load(directory / "attention_maps.npz")["temporal_conditional"]
        summary = json.loads((directory / "summary.json").read_text())
        target = np.arange(matrix.shape[0])[:, None]
        source = np.arange(matrix.shape[1])[None, :]
        record = {
            "step": step,
            "matrix": np.round(matrix, 8).tolist(),
            "same_frame_fraction": summary[
                "mean_same_frame_fraction_within_source_attention"
            ],
            "same_frame_top1_fraction": summary["same_frame_is_argmax_fraction"],
            "expected_absolute_frame_offset": summary[
                "mean_expected_absolute_frame_offset"
            ],
            "source_attention_mass": summary["mean_total_attention_mass_to_source"],
            "within_1_fraction": float(
                matrix[np.abs(target - source) <= 1].sum() / matrix.shape[0]
            ),
            "within_2_fraction": float(
                matrix[np.abs(target - source) <= 2].sum() / matrix.shape[0]
            ),
        }
        records.append(record)
        global_max = max(global_max, float(np.percentile(matrix, 99.5)))

    payload = {
        "schema": "h3_ts_attention_by_timestep_v1",
        "attention": "Ref2VA base dense T-to-S, source-normalized",
        "layer": 24,
        "heads_averaged": 56,
        "spatial_queries_per_target_frame": 16,
        "frames": 37,
        "shared_color_ceiling_p995": global_max,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
