"""Render source, first-x0, and selector heatmap side by side."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def label(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selector", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("first_x0", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = torch.load(args.selector, map_location="cpu", weights_only=False)
    score = data["token_score"].float().squeeze(0).squeeze(0).numpy()
    mask = data["active_mask_spatial_dilation_r1"].squeeze(0).squeeze(0).numpy().astype(np.uint8)
    low, high = np.percentile(score, [5, 99])
    score = np.clip((score - low) / max(high - low, 1e-8), 0.0, 1.0)

    source_cap = cv2.VideoCapture(str(args.source))
    target_cap = cv2.VideoCapture(str(args.first_x0))
    if not source_cap.isOpened() or not target_cap.isOpened():
        raise RuntimeError("failed to open input video")
    frames = min(int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(target_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = target_cap.get(cv2.CAP_PROP_FPS) or 24.0
    panel_w, panel_h = 672, 384
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 3, panel_h)
    )
    if not writer.isOpened():
        raise RuntimeError("failed to create output video")

    for index in range(frames):
        ok_s, source = source_cap.read()
        ok_t, target = target_cap.read()
        if not ok_s or not ok_t:
            break
        latent_index = int(round(index * (score.shape[0] - 1) / max(frames - 1, 1)))
        source = cv2.resize(source, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        heat_scalar = cv2.resize(score[latent_index], (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((heat_scalar * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        active = cv2.resize(mask[latent_index], (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(source, 0.50, heat, 0.50, 0.0)
        contours, _ = cv2.findContours(active, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
        label(source, "source")
        label(target, "B first x0 (one DiT forward)")
        label(overlay, "latent selector diagnostic; white=active boundary")
        writer.write(np.concatenate([source, target, overlay], axis=1))

    source_cap.release()
    target_cap.release()
    writer.release()
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
