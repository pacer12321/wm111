from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


LAYERS = (8, 24, 41)
STAGES = {
    "Early steps 0-2": (0, 1, 2),
    "Middle steps 3-5": (3, 4, 5),
    "Late steps 6-7": (6, 7),
}


def load_capture(root: Path, step: int, layer: int) -> dict:
    paths = sorted((root / f"step_{step:02d}").glob(f"layer_{layer:02d}_rank_*.pt"))
    if len(paths) != 2:
        raise RuntimeError(f"expected two shards for step={step} layer={layer}: {paths}")
    shards = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    first = shards[0]
    qi = torch.cat([row["query_indices"].long() for row in shards])
    q = torch.cat([row["q_selected"] for row in shards])
    order = qi.argsort()
    qi, q = qi[order], q[order]
    ki = torch.cat([row["key_indices"].long() for row in shards])
    k = torch.cat([row["k_local"] for row in shards])
    order = ki.argsort()
    ki, k = ki[order], k[order]
    used = int(first["layout"]["used_len"])
    if not torch.equal(ki, torch.arange(used)):
        raise RuntimeError("key shards do not reconstruct logical sequence")
    return {"meta": first, "qi": qi, "q": q, "k": k}


def frame_ids(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    times, ids = torch.unique(coords[:, 0], sorted=True, return_inverse=True)
    return times, ids


@torch.inference_mode()
def branch_maps(payload: dict, device: torch.device) -> dict[str, np.ndarray]:
    meta = payload["meta"]
    q = payload["q"].to(device)
    k = payload["k"].to(device)
    qi = payload["qi"]
    qcoords_all = meta["all_query_coords"].float()
    qindex_all = meta["all_query_indices"].long()
    lookup = {int(index): row for row, index in enumerate(qindex_all.tolist())}
    qcoords = torch.stack([qcoords_all[lookup[int(index)]] for index in qi.tolist()])
    qtimes, qframe = frame_ids(qcoords)

    branches = {}
    for name, pos_key, coord_key in (
        ("ss", "source_positions", "source_coords"),
        ("st", "target_positions", "target_coords"),
    ):
        positions = meta[pos_key].long()
        coords = meta[coord_key].float()
        ktimes, kframe = frame_ids(coords)
        if qtimes.numel() != ktimes.numel():
            raise RuntimeError("query/key frame count mismatch")
        frames = int(qtimes.numel())
        keys = k.index_select(0, positions.to(device))
        kframe = kframe.to(device)
        result = torch.zeros((frames, frames), dtype=torch.float64)
        for frame in range(frames):
            rows = qframe == frame
            qrows = q[rows.to(device)]
            logits = torch.einsum("qhd,khd->hqk", qrows, keys) * float(meta["softmax_scale"])
            probs = torch.softmax(logits.float(), dim=-1)
            mass = torch.zeros(
                (probs.shape[0], probs.shape[1], frames),
                dtype=torch.float32,
                device=device,
            )
            mass.scatter_add_(
                2,
                kframe.view(1, 1, -1).expand(probs.shape[0], probs.shape[1], -1),
                probs,
            )
            result[frame] = mass.mean((0, 1)).double().cpu()
            del logits, probs, mass
        branches[name] = result.numpy()
    return branches


def metrics(matrix: np.ndarray) -> dict:
    frames = matrix.shape[0]
    rows = np.arange(frames)[:, None]
    cols = np.arange(frames)[None, :]
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})
    column_mean = matrix.mean(axis=0)
    top = np.argsort(column_mean)[::-1][:10]
    return {
        "exact_diagonal_mass": float(np.mean(np.sum(matrix * (rows == cols), axis=1))),
        "plus_minus_1_mass": float(np.mean(np.sum(matrix * (np.abs(rows - cols) <= 1), axis=1))),
        "plus_minus_2_mass": float(np.mean(np.sum(matrix * (np.abs(rows - cols) <= 2), axis=1))),
        "period5_anchor_mass": float(np.mean(np.sum(matrix[:, anchors], axis=1))),
        "period5_anchors": anchors,
        "top_column_frames": [int(x) for x in top],
        "top_column_mean_mass": [float(column_mean[x]) for x in top],
        "column_mean": column_mean.tolist(),
    }


def plot_grid(grouped: dict, branch: str, output: Path) -> None:
    if plt is None:
        return
    values = [grouped[(layer, stage)][branch] for layer in LAYERS for stage in STAGES]
    vmax = float(np.quantile(np.concatenate([x.ravel() for x in values]), 0.995))
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    image = None
    for row, layer in enumerate(LAYERS):
        for col, stage in enumerate(STAGES):
            matrix = grouped[(layer, stage)][branch]
            image = axes[row, col].imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
            axes[row, col].set_title(f"Layer {layer} / {stage}")
            axes[row, col].set_xlabel("Source key frame" if branch == "ss" else "Target key frame")
            axes[row, col].set_ylabel("Source query frame")
            axes[row, col].set_xticks([0, 9, 18, 27, 36])
            axes[row, col].set_yticks([0, 9, 18, 27, 36])
    fig.colorbar(image, ax=axes, shrink=0.85, label="Conditional attention mass (heads/spatial queries averaged)")
    title = "S-S frame attention" if branch == "ss" else "S-to-T frame attention"
    fig.suptitle(title + " | DMD8 latent-skip, dense QK probe", fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    per_capture = {}
    raw_metrics = {}
    for layer in LAYERS:
        for step in range(8):
            payload = load_capture(args.capture_root, step, layer)
            maps = branch_maps(payload, device)
            per_capture[(layer, step)] = maps
            raw_metrics[f"layer_{layer}_step_{step}"] = {
                branch: metrics(matrix) for branch, matrix in maps.items()
            }

    grouped = {}
    grouped_metrics = {}
    for layer in LAYERS:
        for stage, steps in STAGES.items():
            maps = {
                branch: np.mean([per_capture[(layer, step)][branch] for step in steps], axis=0)
                for branch in ("ss", "st")
            }
            grouped[(layer, stage)] = maps
            grouped_metrics[f"layer_{layer}_{stage}"] = {
                branch: metrics(matrix) for branch, matrix in maps.items()
            }

    np.savez_compressed(
        args.output_dir / "attention_matrices.npz",
        **{
            f"layer_{layer}_{stage.replace(' ', '_').replace('-', '_')}_{branch}": matrix
            for (layer, stage), maps in grouped.items()
            for branch, matrix in maps.items()
        },
    )

    plot_grid(grouped, "ss", args.output_dir / "ss_early_middle_late_layers8_24_41.png")
    plot_grid(grouped, "st", args.output_dir / "st_early_middle_late_layers8_24_41.png")
    (args.output_dir / "attention_metrics.json").write_text(json.dumps({
        "config": {
            "layers": list(LAYERS),
            "stages": {key: list(value) for key, value in STAGES.items()},
            "head_handling": "mean over all heads",
            "spatial_query_handling": "mean over 4x4 sampled positions per source frame",
            "normalization": "softmax conditional within the plotted key branch",
            "probe": "dense post-RoPE QK reconstruction from current DMD8 latent-skip states",
        },
        "grouped": grouped_metrics,
        "per_step": raw_metrics,
    }, indent=2))
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
