#!/usr/bin/env python3
"""Check whether an edit followed the instruction WITHOUT drifting elsewhere.

Success criterion (as specified by the user): a generated edit only counts as
successful if it strictly follows the instruction. Any other change -- a
different environment/background, a different pose/stance, different framing
-- is a failure, even if the requested color/style/scenery change is present.

This gives two independent, complementary signals per pair:
  - structure_ssim: Canny-edge SSIM between source and generated frames.
    Edge maps are near-invariant to color/style/texture changes but very
    sensitive to changes in pose, composition, or object placement. LOW
    structure_ssim means something moved/changed that should not have.
  - color_shift: mean LAB color-histogram distance between source and
    generated frames. This is expected to be nonzero for a real color/style
    edit and near-zero if the model did nothing (edit not applied).

Neither metric alone proves success -- both are needed:
  - high structure_ssim + near-zero color_shift  -> edit likely NOT applied
  - low structure_ssim (regardless of color_shift) -> edit drifted into
    content it should not have touched (the failure mode this user described)
  - high structure_ssim + clear color_shift        -> looks like a real,
    contained edit (still needs a human/VLM look, this is not ground truth)

Input: a directory containing one or more pairs of
  <sample_id>_source.mp4 and <sample_id>_generated.mp4
plus an optional <sample_id>.json with {"instruction": "..."} for context in
the report (purely informational, not used in scoring).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def edge_ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    ea = cv2.Canny(a_gray, 80, 160).astype(np.float32)
    eb = cv2.Canny(b_gray, 80, 160).astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(ea), blur(eb)
    va, vb = blur(ea * ea) - ma * ma, blur(eb * eb) - mb * mb
    co = blur(ea * eb) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * ma * mb + c1) * (2 * co + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(score.mean())


def lab_hist_distance(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a_lab = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB)
    b_lab = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB)
    dists = []
    for channel in range(3):
        ha = cv2.calcHist([a_lab], [channel], None, [64], [0, 256])
        hb = cv2.calcHist([b_lab], [channel], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        dists.append(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
    return float(np.mean(dists))


def evaluate_pair(source_path: Path, generated_path: Path) -> dict:
    src_cap = cv2.VideoCapture(str(source_path))
    gen_cap = cv2.VideoCapture(str(generated_path))
    assert src_cap.isOpened() and gen_cap.isOpened(), (source_path, generated_path)
    src_n = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gen_n = int(gen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(src_n, gen_n)
    if src_n != gen_n:
        # Frame-count mismatch is itself worth surfacing: same-frame-index
        # cross-stream attention (ts) silently desyncs when this happens.
        pass
    structure_scores, color_scores = [], []
    for _ in range(n):
        ok_s, fs = src_cap.read()
        ok_g, fg = gen_cap.read()
        if not (ok_s and ok_g):
            break
        if fs.shape != fg.shape:
            fg = cv2.resize(fg, (fs.shape[1], fs.shape[0]))
        gray_s = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)
        gray_g = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        structure_scores.append(edge_ssim(gray_s, gray_g))
        color_scores.append(lab_hist_distance(fs, fg))
    structure = np.array(structure_scores)
    color = np.array(color_scores)
    verdict = "edit_not_applied" if structure.mean() > 0.75 and color.mean() < 0.03 else (
        "structure_drifted" if structure.mean() < 0.55 else "looks_contained"
    )
    return dict(
        source=str(source_path),
        generated=str(generated_path),
        frame_count_source=src_n,
        frame_count_generated=gen_n,
        frame_count_mismatch=src_n != gen_n,
        frames_compared=n,
        structure_ssim=dict(mean=float(structure.mean()), min=float(structure.min()),
                            p10=float(np.quantile(structure, .1)) if n else None),
        color_shift_lab_bhattacharyya=dict(mean=float(color.mean()), max=float(color.max())),
        worst_structure_frames=[int(i) for i in np.argsort(structure)[:5]] if n else [],
        heuristic_verdict=verdict,
        verdict_note=("Heuristic only -- not a substitute for a human/VLM look. "
                      "structure_ssim<0.55 means something likely drifted "
                      "(pose/background/composition) beyond the requested edit."),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path,
                        help="directory containing <id>_source.mp4 / <id>_generated.mp4 pairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = sorted(args.directory.glob("*_source.mp4"))
    if not sources:
        raise SystemExit(f"no *_source.mp4 files found in {args.directory}")

    results = []
    for source_path in sources:
        sample_id = source_path.name[: -len("_source.mp4")]
        generated_path = args.directory / f"{sample_id}_generated.mp4"
        if not generated_path.is_file():
            results.append(dict(sample_id=sample_id, error="missing generated video"))
            continue
        meta_path = args.directory / f"{sample_id}.json"
        instruction = None
        if meta_path.is_file():
            instruction = json.loads(meta_path.read_text()).get("instruction")
        result = evaluate_pair(source_path, generated_path)
        result["sample_id"] = sample_id
        result["instruction"] = instruction
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("worst_structure_frames",)},
                         ensure_ascii=False))

    verdict_counts: dict[str, int] = {}
    for r in results:
        v = r.get("heuristic_verdict", "error")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    report = dict(directory=str(args.directory), pairs=len(results),
                 verdict_counts=verdict_counts, results=results)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(pairs=len(results), verdict_counts=verdict_counts), indent=2))


if __name__ == "__main__":
    main()
