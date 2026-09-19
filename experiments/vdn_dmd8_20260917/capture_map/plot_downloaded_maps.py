from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(r"C:\Users\DZH\OneDrive\Documents\ChatGPT\wm111\experiments\vdn_dmd8_20260917\capture_map\artifacts")
LAYERS = (8, 24, 41)
STAGES = (
    ("Early_steps_0_2", "早期 / Early (steps 0–2)"),
    ("Middle_steps_3_5", "中期 / Middle (steps 3–5)"),
    ("Late_steps_6_7", "后期 / Late (steps 6–7)"),
)


def draw(branch: str) -> None:
    data = np.load(ROOT / "attention_matrices.npz")
    values = [data[f"layer_{layer}_{stage}_{branch}"] for layer in LAYERS for stage, _ in STAGES]
    vmax = float(np.quantile(np.concatenate([x.ravel() for x in values]), 0.995))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    im = None
    for row, layer in enumerate(LAYERS):
        for col, (stage, stage_label) in enumerate(STAGES):
            matrix = data[f"layer_{layer}_{stage}_{branch}"]
            im = axes[row, col].imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
            axes[row, col].plot(np.arange(37), np.arange(37), color="black", linewidth=0.7, alpha=0.7)
            axes[row, col].set_title(f"Layer {layer} · {stage_label}")
            axes[row, col].set_xlabel("Source key frame / 源关键帧" if branch == "ss" else "Target key frame / 目标关键帧")
            axes[row, col].set_ylabel("Source query frame / 源查询帧")
            axes[row, col].set_xticks([0, 9, 18, 27, 36])
            axes[row, col].set_yticks([0, 9, 18, 27, 36])
    fig.colorbar(im, ax=axes, shrink=0.85, label="Conditional attention mass / 分支内归一化注意力质量")
    title = "S→S frame attention / 源到源帧注意力" if branch == "ss" else "S→T frame attention / 源到目标帧注意力"
    fig.suptitle(title + "\nDMD8 + latent skip + interleaved SP, heads/spatial samples averaged", fontsize=15)
    fig.savefig(ROOT / f"{branch}_dmd8_early_middle_late_layers8_24_41.png", dpi=180)
    plt.close(fig)


def print_summary() -> None:
    metrics = json.loads((ROOT / "attention_metrics.json").read_text())
    for branch in ("ss", "st"):
        print(branch.upper())
        for layer in LAYERS:
            for stage, label in STAGES:
                row = metrics["grouped"][f"layer_{layer}_{stage.replace('_', ' ').replace(' 0 2', ' 0-2').replace(' 3 5', ' 3-5').replace(' 6 7', ' 6-7')}"][branch]
                print(layer, label, "diag", f'{row["exact_diagonal_mass"]:.6f}', "pm1", f'{row["plus_minus_1_mass"]:.6f}', "p5", f'{row["period5_anchor_mass"]:.6f}', "top", row["top_column_frames"][:5])


if __name__ == "__main__":
    draw("ss")
    draw("st")
    print_summary()
