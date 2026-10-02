#!/usr/bin/env python3
"""Approximate-change acceptance metrics for plan section 6.2: candidate vs baseline.

All three numbers compare a CANDIDATE output against the original BASELINE output
(t-skip v4) for the same clip, seed and request settings, with the SOURCE video as
the anchor. They answer "did the speed-up keep what the baseline did", not "is the
edit correct" -- absolute edit quality is still judged by the full-video review.

Edit region M (per frame, fixed by the baseline so every candidate is scored on the
same pixels, including candidates that change the skip mask such as t-skip 4):
    dE_BS = CIELAB distance between baseline and source
    M     = open3x3(dE_BS > tau), then dilated by `dilate` pixels
Background is the complement of M.

edit_retention (plan: >= 0.99)
    R = 1 - sum_M dE(candidate, baseline) / sum_M dE(baseline, source)
    The fraction of the baseline's edit (its colour change away from the source)
    that the candidate reproduces. 1.0 = identical to baseline inside the edit.

background_psnr_drop_db (plan: <= 0.3 dB)
    PSNR(baseline, source | background) - PSNR(candidate, source | background),
    RGB, MSE pooled over all background pixels of all frames.

background_flicker_ratio (plan: "temporal not worse"; provisional bound 1.05)
    flicker(X) = mean_{t>=1, background} |(X_t - X_{t-1}) - (S_t - S_{t-1})| (gray)
    ratio = flicker(candidate) / flicker(baseline). Frame-to-frame change that the
    source does not have. Also reported: edit_temporal_residual, the mean
    |(C_t - C_{t-1}) - (B_t - B_{t-1})| inside M.

Calibrate before trusting the thresholds: pass a second run of the baseline with
--rerun. Its numbers are the run-to-run noise floor; a threshold tighter than what a
rerun achieves can never pass. In a synthetic check where baseline and rerun differ
at ~46.5 dB PSNR (just below the measured 47-52 dB floor) a rerun scores R ~ 0.988,
below 0.99. With --rerun the report adds a calibrated verdict:
    r_min_effective       = min(r_min, R_rerun - margin)
    flicker_max_effective = max(flicker_max, ratio_rerun + (flicker_max - 1))
The background drop bound stays 0.3 dB: it is a budget against the baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def lab(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(axis=-1))


def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def psnr(sse: float, count: int) -> float | None:
    if count == 0:
        return None
    mse = sse / count
    return math.inf if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def evaluate(source: Path, baseline: Path, candidate: Path, *, tau: float, dilate: int,
             min_area: float) -> dict:
    counts = dict(source=frame_count(source), baseline=frame_count(baseline),
                  candidate=frame_count(candidate))
    open_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
                     if dilate > 0 else None)

    edit_num = edit_den = 0.0
    edit_pixels = total_pixels = 0
    bg_sse_b = bg_sse_c = 0.0
    bg_count = 0
    flick_b = flick_c = 0.0
    flick_count = 0
    edit_temporal = 0.0
    edit_temporal_count = 0
    per_frame = []
    previous = None
    resized_source = False

    for t, (s, b, c) in enumerate(zip(frames(source), frames(baseline), frames(candidate))):
        if b.shape != c.shape:
            raise SystemExit(f"baseline {b.shape} and candidate {c.shape} differ in size; "
                             "they must come from the same request settings")
        if s.shape != b.shape:
            s = cv2.resize(s, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
            resized_source = True
        ls, lb, lc = lab(s), lab(b), lab(c)
        de_bs, de_cb = delta_e(lb, ls), delta_e(lc, lb)

        mask = (de_bs > tau).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        if dilate_kernel is not None:
            mask = cv2.dilate(mask, dilate_kernel)
        m = mask.astype(bool)
        bg = ~m

        frame_num, frame_den = float(de_cb[m].sum()), float(de_bs[m].sum())
        edit_num += frame_num
        edit_den += frame_den
        edit_pixels += int(m.sum())
        total_pixels += m.size
        area = float(m.mean())
        record = dict(frame=t, edit_area=area, edit_retention=None)
        if area >= min_area and frame_den > 0:
            record["edit_retention"] = 1.0 - frame_num / frame_den

        if bg.any():
            sf, bf, cf = s.astype(np.float32), b.astype(np.float32), c.astype(np.float32)
            bg_sse_b += float(((bf - sf)[bg] ** 2).sum())
            bg_sse_c += float(((cf - sf)[bg] ** 2).sum())
            bg_count += int(bg.sum()) * 3

        gs, gb, gc = gray(s), gray(b), gray(c)
        if previous is not None:
            ps, pb, pc = previous
            ds, db, dc = gs - ps, gb - pb, gc - pc
            if bg.any():
                flick_b += float(np.abs(db - ds)[bg].sum())
                flick_c += float(np.abs(dc - ds)[bg].sum())
                flick_count += int(bg.sum())
            if m.any():
                edit_temporal += float(np.abs(dc - db)[m].sum())
                edit_temporal_count += int(m.sum())
        previous = (gs, gb, gc)
        per_frame.append(record)

    compared = len(per_frame)
    retention = None if edit_den == 0 else 1.0 - edit_num / edit_den
    psnr_b, psnr_c = psnr(bg_sse_b, bg_count), psnr(bg_sse_c, bg_count)
    drop = None
    if psnr_b is not None and psnr_c is not None:
        drop = 0.0 if psnr_b == psnr_c else psnr_b - psnr_c
    fb = flick_b / flick_count if flick_count else None
    fc = flick_c / flick_count if flick_count else None
    scored = [r for r in per_frame if r["edit_retention"] is not None]
    worst = sorted(scored, key=lambda r: r["edit_retention"])[:5]
    notes = []
    if len(set(counts.values())) > 1:
        notes.append(f"frame counts differ {counts}; compared the first {compared}")
    if resized_source:
        notes.append("source resized to the baseline resolution (INTER_AREA)")
    if retention is None:
        notes.append("baseline barely differs from source at this tau: edit retention undefined")
    if edit_pixels / max(total_pixels, 1) > 0.6:
        notes.append("edit region covers >60% of pixels (global edit?): background metrics are thin")
    return dict(
        frames_compared=compared,
        frame_counts=counts,
        edit_area_fraction=edit_pixels / max(total_pixels, 1),
        edit_retention=retention,
        edit_retention_frame_min=min((r["edit_retention"] for r in scored), default=None),
        worst_edit_frames=[dict(frame=r["frame"], edit_retention=r["edit_retention"],
                                edit_area=r["edit_area"]) for r in worst],
        background_psnr_baseline_db=psnr_b,
        background_psnr_candidate_db=psnr_c,
        background_psnr_drop_db=drop,
        background_flicker_baseline=fb,
        background_flicker_candidate=fc,
        background_flicker_ratio=(fc / fb) if fb else None,
        edit_temporal_residual=(edit_temporal / edit_temporal_count) if edit_temporal_count else None,
        notes=notes,
    )


def verdict(result: dict, *, r_min: float, drop_max: float, flicker_max: float) -> dict:
    checks = {
        "edit_retention": (result["edit_retention"] is not None
                           and result["edit_retention"] >= r_min),
        "background_psnr_drop": (result["background_psnr_drop_db"] is None
                                 or result["background_psnr_drop_db"] <= drop_max),
        "background_flicker": (result["background_flicker_ratio"] is None
                               or result["background_flicker_ratio"] <= flicker_max),
    }
    return dict(checks=checks, metrics_pass=all(checks.values()),
                note="Metrics only. Plan 6.2 still requires the full-clip side-by-side review.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="original baseline output (t-skip v4), same clip/seed/settings")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rerun", type=Path,
                        help="a second baseline run, scored as if it were a candidate: noise floor")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=10.0, help="CIELAB dE for the edit mask")
    parser.add_argument("--dilate", type=int, default=7, help="edit mask dilation, pixels")
    parser.add_argument("--min-area", type=float, default=0.002,
                        help="per-frame edit retention only when the mask covers this fraction")
    parser.add_argument("--r-min", type=float, default=0.99)
    parser.add_argument("--drop-max", type=float, default=0.3)
    parser.add_argument("--flicker-max", type=float, default=1.05)
    parser.add_argument("--margin", type=float, default=0.01,
                        help="calibrated r_min = min(r_min, rerun edit retention - margin)")
    args = parser.parse_args()

    options = dict(tau=args.tau, dilate=args.dilate, min_area=args.min_area)
    thresholds = dict(r_min=args.r_min, drop_max=args.drop_max, flicker_max=args.flicker_max)
    report = dict(source=str(args.source), baseline=str(args.baseline),
                  candidate=str(args.candidate), options=options, thresholds=thresholds)
    report["candidate_metrics"] = evaluate(args.source, args.baseline, args.candidate, **options)
    report["candidate_verdict"] = verdict(report["candidate_metrics"], **thresholds)
    if args.rerun:
        report["rerun"] = str(args.rerun)
        report["rerun_metrics"] = evaluate(args.source, args.baseline, args.rerun, **options)
        report["rerun_verdict"] = verdict(report["rerun_metrics"], **thresholds)
        if not report["rerun_verdict"]["metrics_pass"]:
            report["calibration_warning"] = ("A baseline rerun fails the fixed thresholds; they "
                                             "are tighter than run-to-run noise. Use the "
                                             "calibrated verdict.")
        rerun = report["rerun_metrics"]
        calibrated = dict(thresholds)
        if rerun["edit_retention"] is not None:
            calibrated["r_min"] = min(args.r_min, rerun["edit_retention"] - args.margin)
        if rerun["background_flicker_ratio"] is not None:
            calibrated["flicker_max"] = max(args.flicker_max,
                                            rerun["background_flicker_ratio"] + args.flicker_max - 1)
        report["calibrated_thresholds"] = calibrated
        report["candidate_verdict_calibrated"] = verdict(report["candidate_metrics"], **calibrated)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
                        encoding="utf-8")
    keys = ("edit_retention", "edit_retention_frame_min", "background_psnr_drop_db",
            "background_flicker_ratio")
    summary = {name: {k: report[f"{name}_metrics"][k] for k in keys}
               for name in ("candidate", "rerun") if f"{name}_metrics" in report}
    summary["candidate_pass_fixed"] = report["candidate_verdict"]["metrics_pass"]
    if "candidate_verdict_calibrated" in report:
        summary["calibrated_thresholds"] = report["calibrated_thresholds"]
        summary["candidate_pass_calibrated"] = report["candidate_verdict_calibrated"]["metrics_pass"]
    if "calibration_warning" in report:
        summary["calibration_warning"] = report["calibration_warning"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
