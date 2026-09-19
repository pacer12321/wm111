"""Render comparable T-to-S temporal attention heatmaps across denoising steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="0,12,24,36,48")
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]

    matrices = []
    summaries = []
    for step in steps:
        directory = args.analysis_root / f"analysis_step_{step:02d}"
        matrices.append(np.load(directory / "attention_maps.npz")["temporal_conditional"])
        summaries.append(json.loads((directory / "summary.json").read_text()))

    vmax = max(float(np.percentile(matrix, 99.5)) for matrix in matrices)
    fig, axes = plt.subplots(1, len(steps), figsize=(4.1 * len(steps), 4.25), constrained_layout=True)
    if len(steps) == 1:
        axes = [axes]
    image = None
    for axis, step, matrix, summary in zip(axes, steps, matrices, summaries):
        image = axis.imshow(
            matrix,
            origin="upper",
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        axis.plot(np.arange(matrix.shape[0]), np.arange(matrix.shape[0]), color="black", linewidth=0.7)
        axis.set_title(
            f"Step {step}\n"
            f"diag={summary['mean_same_frame_fraction_within_source_attention']:.3f}, "
            f"Top-1={summary['same_frame_is_argmax_fraction']:.3f}"
        )
        axis.set_xlabel("Source frame / Source帧")
        axis.set_ylabel("Target frame / Target帧")
        ticks = [0, 9, 18, 27, 36]
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.83, label="T→S attention probability (source-normalized)")
    fig.suptitle(
        "Ref2VA dense T→S attention across denoising timesteps\n"
        "Fixed DiT layer 24; mean over all 56 heads and sampled spatial target queries",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    rows = []
    for step, matrix, summary in zip(steps, matrices, summaries):
        target = np.arange(matrix.shape[0])[:, None]
        source = np.arange(matrix.shape[1])[None, :]
        within1 = float(matrix[np.abs(target - source) <= 1].sum() / matrix.shape[0])
        within2 = float(matrix[np.abs(target - source) <= 2].sum() / matrix.shape[0])
        rows.append(
            {
                "step": step,
                "same_frame_fraction": summary["mean_same_frame_fraction_within_source_attention"],
                "same_frame_top1_fraction": summary["same_frame_is_argmax_fraction"],
                "expected_absolute_frame_offset": summary["mean_expected_absolute_frame_offset"],
                "within_plus_minus_1_frame_fraction": within1,
                "within_plus_minus_2_frames_fraction": within2,
            }
        )
    (args.output.parent / "timestep_metrics.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
