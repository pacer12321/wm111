#!/usr/bin/env python3
"""Diagnose latent-selector thresholds inside the red-shirt edit region.

This is a CPU-only diagnostic.  The edit ROI is a conservative proxy built
from pixels that are (a) clearly red in the dense B result and (b) changed
substantially from the source video.  Results are reported at several ROI
coverage cutoffs because the proxy is not ground-truth segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_THRESHOLDS = {
    "current_32pct": 0.7458736300468445,
    "target_28pct": 1.0090251782,
    "target_24pct": 1.2029385754,
    "target_20pct": 1.3052546896,
}


def read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return frames


def quantiles(x: np.ndarray) -> dict[str, float | None]:
    if x.size == 0:
        return {k: None for k in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    q = np.quantile(x, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {
        "min": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "p50": float(q[3]), "p75": float(q[4]), "p90": float(q[5]),
        "max": float(q[6]), "mean": float(x.mean()),
    }


def dilate_active(raw: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(raw.astype(np.float32))[None, None]
    # Spatial r=1, temporal r=0: identical to the formal latent selector.
    y = F.max_pool3d(t, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    return y[0, 0].numpy() > 0.5


def build_pixel_roi(source_bgr: np.ndarray, edited_bgr: np.ndarray) -> np.ndarray:
    if source_bgr.shape[:2] != edited_bgr.shape[:2]:
        source_bgr = cv2.resize(source_bgr, (edited_bgr.shape[1], edited_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    src = source_bgr.astype(np.int16)
    edt = edited_bgr.astype(np.int16)
    b, g, r = edt[..., 0], edt[..., 1], edt[..., 2]
    # Red dominance excludes most skin and pale background.  The edit-difference
    # term further excludes pixels that were already red in the source.
    red = (r >= 105) & (r - g >= 38) & (r - b >= 38)
    colour_delta = np.linalg.norm(edt.astype(np.float32) - src.astype(np.float32), axis=2)
    roi = (red & (colour_delta >= 32)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return roi


def make_overlay(frame: np.ndarray, roi: np.ndarray, token_roi: np.ndarray) -> np.ndarray:
    out = frame.copy()
    red_layer = np.zeros_like(out)
    red_layer[..., 2] = 255
    alpha = (roi.astype(np.float32) / 255.0 * 0.35)[..., None]
    out = (out * (1 - alpha) + red_layer * alpha).astype(np.uint8)
    h, w = out.shape[:2]
    th, tw = token_roi.shape
    for yy, xx in np.argwhere(token_roi):
        x0, x1 = round(xx * w / tw), round((xx + 1) * w / tw)
        y0, y1 = round(yy * h / th), round((yy + 1) * h / th)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector-map", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--edited", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.selector_map, map_location="cpu", weights_only=False)
    score_tensor = payload["token_score"] if isinstance(payload, dict) else payload
    score = score_tensor.detach().float().cpu().numpy().squeeze()
    if score.ndim != 3:
        raise ValueError(f"Expected [T,H,W] token score, got {score.shape}")
    nt, nh, nw = score.shape

    source_frames = read_all_frames(args.source)
    edited_frames = read_all_frames(args.edited)
    common_n = min(len(source_frames), len(edited_frames))
    mapped_indices = np.rint(np.linspace(0, common_n - 1, nt)).astype(int)

    coverage = np.zeros((nt, nh, nw), dtype=np.float32)
    pixel_rois: list[np.ndarray] = []
    edited_selected: list[np.ndarray] = []
    for ti, vi in enumerate(mapped_indices):
        src = source_frames[vi]
        edt = edited_frames[vi]
        roi = build_pixel_roi(src, edt)
        pixel_rois.append(roi)
        edited_selected.append(edt)
        coverage[ti] = cv2.resize(roi.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_AREA)

    report: dict[str, object] = {
        "inputs": {
            "selector_map": str(args.selector_map), "source": str(args.source),
            "edited": str(args.edited), "source_frames": len(source_frames),
            "edited_frames": len(edited_frames), "latent_shape": [nt, nh, nw],
            "mapped_video_indices": mapped_indices.tolist(),
        },
        "roi_definition": {
            "red": "R>=105, R-G>=38, R-B>=38",
            "change": "RGB L2(source, B)>=32",
            "morphology": "close 9x9 ellipse, open 3x3",
            "token_coverage_cutoffs": [0.05, 0.10, 0.20],
        },
        "thresholds": DEFAULT_THRESHOLDS,
        "coverage_sensitivity": {},
    }

    rows: list[dict[str, object]] = []
    for cutoff in (0.05, 0.10, 0.20):
        edit_roi = coverage >= cutoff
        outside = ~edit_roi
        item: dict[str, object] = {
            "edit_token_count": int(edit_roi.sum()),
            "edit_token_fraction": float(edit_roi.mean()),
            "score_inside": quantiles(score[edit_roi]),
            "score_outside": quantiles(score[outside]),
            "bands_inside": {},
            "variants": {},
        }
        vals = score[edit_roi]
        edges = [-np.inf, DEFAULT_THRESHOLDS["current_32pct"], DEFAULT_THRESHOLDS["target_28pct"],
                 DEFAULT_THRESHOLDS["target_24pct"], DEFAULT_THRESHOLDS["target_20pct"], np.inf]
        names = ["below_current", "current_to_28", "28_to_24", "24_to_20", "above_20"]
        for name, lo, hi in zip(names, edges[:-1], edges[1:]):
            item["bands_inside"][name] = float(((vals >= lo) & (vals < hi)).mean()) if vals.size else None

        for variant, threshold in DEFAULT_THRESHOLDS.items():
            raw = score >= threshold
            active = dilate_active(raw)
            edit_active = int((active & edit_roi).sum())
            edit_total = int(edit_roi.sum())
            background_total = int(outside.sum())
            stats = {
                "threshold": threshold,
                "global_raw_active_fraction": float(raw.mean()),
                "global_final_active_fraction": float(active.mean()),
                "edit_active_recall": float(edit_active / edit_total) if edit_total else None,
                "edit_false_stable_fraction": float(1 - edit_active / edit_total) if edit_total else None,
                "background_active_fraction": float((active & outside).sum() / background_total) if background_total else None,
                "edit_active_tokens": edit_active,
                "edit_total_tokens": edit_total,
            }
            item["variants"][variant] = stats
            rows.append({"roi_coverage_cutoff": cutoff, "variant": variant, **stats})
        report["coverage_sensitivity"][str(cutoff)] = item

    with (args.output_dir / "roi_threshold_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with (args.output_dir / "roi_threshold_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Visual sanity sheet at the main 10% token-coverage cutoff.
    main_roi = coverage >= 0.10
    picks = np.unique(np.rint(np.linspace(0, nt - 1, min(8, nt))).astype(int))
    thumbs: list[np.ndarray] = []
    for ti in picks:
        img = make_overlay(edited_selected[ti], pixel_rois[ti], main_roi[ti])
        img = cv2.resize(img, (504, 288), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"latent {ti} / video {mapped_indices[ti]}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(img)
    cols = 2
    rows_n = (len(thumbs) + cols - 1) // cols
    blank = np.zeros_like(thumbs[0])
    sheet_rows = []
    for r in range(rows_n):
        row = [thumbs[r * cols + c] if r * cols + c < len(thumbs) else blank for c in range(cols)]
        sheet_rows.append(np.hstack(row))
    cv2.imwrite(str(args.output_dir / "roi_overlay_contact.jpg"), np.vstack(sheet_rows))
    print(json.dumps(report["coverage_sensitivity"]["0.1"], indent=2))


if __name__ == "__main__":
    main()
