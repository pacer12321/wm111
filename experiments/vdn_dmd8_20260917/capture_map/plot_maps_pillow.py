from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\DZH\OneDrive\Documents\ChatGPT\wm111\experiments\vdn_dmd8_20260917\capture_map\artifacts")
LAYERS = (8, 24, 41)
STAGES = (
    ("Early_steps_0_2", "早期 Early · steps 0–2", "Early steps 0-2"),
    ("Middle_steps_3_5", "中期 Middle · steps 3–5", "Middle steps 3-5"),
    ("Late_steps_6_7", "后期 Late · steps 6–7", "Late steps 6-7"),
)


def font(size: int):
    for name in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(30)
F_SUB = font(20)
F_AXIS = font(17)
F_TICK = font(14)


def blues(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)[..., None]
    stops = np.array([[247, 251, 255], [198, 219, 239], [107, 174, 214], [33, 113, 181], [8, 48, 107]], dtype=float)
    idx = np.minimum((x[..., 0] * (len(stops) - 1)).astype(int), len(stops) - 2)
    frac = x[..., 0] * (len(stops) - 1) - idx
    return ((1 - frac[..., None]) * stops[idx] + frac[..., None] * stops[idx + 1]).astype(np.uint8)


def draw(branch: str) -> None:
    z = np.load(ROOT / "attention_matrices.npz")
    arrays = [z[f"layer_{layer}_{stage}_{branch}"] for layer in LAYERS for stage, _, _ in STAGES]
    vmax = float(np.quantile(np.concatenate([a.ravel() for a in arrays]), 0.995))
    cell = 8
    heat = 37 * cell
    left, top, right, bottom = 82, 55, 22, 68
    panel_w, panel_h = left + heat + right, top + heat + bottom
    gap_x, gap_y = 30, 34
    canvas = Image.new("RGB", (3 * panel_w + 2 * gap_x + 60, 3 * panel_h + 2 * gap_y + 150), "white")
    draw_ctx = ImageDraw.Draw(canvas)
    title = "S→S 源到源帧注意力" if branch == "ss" else "S→T 源查询目标帧注意力"
    draw_ctx.text((canvas.width // 2, 22), title, fill="#111827", font=F_TITLE, anchor="ma")
    draw_ctx.text((canvas.width // 2, 62), "DMD8 + latent skip + 奇偶分片；所有 heads 与采样空间 query 平均", fill="#475569", font=F_SUB, anchor="ma")

    for r, layer in enumerate(LAYERS):
        for c, (stage_key, stage_label, _) in enumerate(STAGES):
            x0 = 30 + c * (panel_w + gap_x)
            y0 = 105 + r * (panel_h + gap_y)
            matrix = z[f"layer_{layer}_{stage_key}_{branch}"]
            rgb = blues(matrix / max(vmax, 1e-12))
            hm = Image.fromarray(rgb, "RGB").resize((heat, heat), Image.Resampling.NEAREST)
            hx, hy = x0 + left, y0 + top
            canvas.paste(hm, (hx, hy))
            draw_ctx.rectangle((hx, hy, hx + heat - 1, hy + heat - 1), outline="#334155", width=1)
            draw_ctx.text((hx + heat // 2, y0 + 4), f"Layer {layer} · {stage_label}", fill="#0f172a", font=F_AXIS, anchor="ma")
            for tick in (0, 9, 18, 27, 36):
                pos = int((tick + 0.5) * cell)
                draw_ctx.text((hx + pos, hy + heat + 7), str(tick), fill="#334155", font=F_TICK, anchor="ma")
                draw_ctx.text((hx - 10, hy + pos), str(tick), fill="#334155", font=F_TICK, anchor="rm")
            xlabel = "源 key 帧" if branch == "ss" else "目标 key 帧"
            draw_ctx.text((hx + heat // 2, hy + heat + 35), xlabel, fill="#334155", font=F_AXIS, anchor="ma")
            draw_ctx.text((x0 + 12, hy + heat // 2), "源 query 帧", fill="#334155", font=F_AXIS, anchor="mm")

    # Shared color bar.
    cb_x, cb_y, cb_w, cb_h = canvas.width - 38, 130, 16, canvas.height - 190
    grad = np.linspace(1, 0, cb_h)[:, None]
    grad_rgb = blues(grad)
    grad_img = Image.fromarray(np.repeat(grad_rgb, cb_w, axis=1), "RGB")
    canvas.paste(grad_img, (cb_x, cb_y))
    draw_ctx.text((cb_x - 2, cb_y - 24), f"{vmax:.3f}", fill="#334155", font=F_TICK, anchor="ra")
    draw_ctx.text((cb_x - 2, cb_y + cb_h - 5), "0", fill="#334155", font=F_TICK, anchor="ra")
    canvas.save(ROOT / f"{branch}_dmd8_early_middle_late_layers8_24_41.png")


def summary() -> None:
    m = json.loads((ROOT / "attention_metrics.json").read_text())
    for branch in ("ss", "st"):
        print(branch.upper())
        for layer in LAYERS:
            for _, label, metric_stage in STAGES:
                row = m["grouped"][f"layer_{layer}_{metric_stage}"][branch]
                print(
                    layer, label,
                    "diag", f'{row["exact_diagonal_mass"]:.6f}',
                    "pm1", f'{row["plus_minus_1_mass"]:.6f}',
                    "pm2", f'{row["plus_minus_2_mass"]:.6f}',
                    "p5", f'{row["period5_anchor_mass"]:.6f}',
                    "top", row["top_column_frames"][:5],
                )


if __name__ == "__main__":
    draw("ss")
    draw("st")
    summary()
