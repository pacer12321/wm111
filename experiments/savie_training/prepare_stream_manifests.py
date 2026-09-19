#!/usr/bin/env python3
"""Build deterministic 4K SAViE pool, then random 2K train/reserve splits.

Only source-disjoint, non-temporal Ditto pairs whose two videos are present and pass
basic container/alignment checks are eligible.  The 2K selection is deliberately a
plain fixed-seed random split of the final 4K pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import subprocess
from pathlib import Path


def probe(path: Path) -> dict | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
        fps = float(num) / max(float(den), 1.0)
        frames = int(stream.get("nb_frames") or 0)
        duration = float(payload.get("format", {}).get("duration") or 0.0)
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "frames": frames,
            "duration": duration,
        }
    except (KeyError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def inspect(item: tuple[dict, Path]) -> tuple[dict, dict, dict] | None:
    row, root = item
    source = root / row["source_relpath"]
    target = root / row["target_relpath"]
    if not source.is_file() or not target.is_file():
        return None
    source_meta, target_meta = probe(source), probe(target)
    if source_meta is None or target_meta is None:
        return None
    if min(source_meta["width"], source_meta["height"],
           target_meta["width"], target_meta["height"]) < 256:
        return None
    if min(source_meta["duration"], target_meta["duration"]) < 2.0:
        return None
    if source_meta["frames"] and source_meta["frames"] < 49:
        return None
    if target_meta["frames"] and target_meta["frames"] < 49:
        return None
    duration_gap = abs(source_meta["duration"] - target_meta["duration"])
    if duration_gap > max(0.25, 0.05 * source_meta["duration"]):
        return None
    if abs(source_meta["fps"] - target_meta["fps"]) > 1.0:
        return None
    return row, source_meta, target_meta


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--defer-qc", action="store_true",
        help="Freeze the random pool immediately; validate each pair before encoding.",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    # Source-disjointness is a hard requirement, not a post-hoc statistic.
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        source = row["source_relpath"]
        if source not in seen:
            seen.add(source)
            deduped.append(row)

    passing: list[dict] = []
    if args.defer_qc:
        passing = deduped[:args.pool_size]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(inspect, ((row, args.video_root) for row in deduped)):
                if result is None:
                    continue
                row, source_meta, target_meta = result
                enriched = dict(row)
                enriched["source_probe"] = source_meta
                enriched["target_probe"] = target_meta
                passing.append(enriched)
                if len(passing) >= args.pool_size:
                    break

    if len(passing) < args.pool_size:
        raise RuntimeError(
            f"only {len(passing)} complete/QC-passing pairs; need {args.pool_size}"
        )
    pool_rows = passing[:args.pool_size]
    random.Random(args.seed + 1).shuffle(pool_rows)
    train_rows = pool_rows[:args.train_size]
    reserve_rows = pool_rows[args.train_size:]
    if len({row["source_relpath"] for row in pool_rows}) != len(pool_rows):
        raise RuntimeError("source-disjoint invariant failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "pool_4k.jsonl", pool_rows)
    write_jsonl(args.output_dir / "train_2k.jsonl", train_rows)
    write_jsonl(args.output_dir / "reserve_2k.jsonl", reserve_rows)
    with (args.output_dir / "video_paths.txt").open("w", encoding="utf-8") as handle:
        for row in pool_rows:
            handle.write(row["source_relpath"] + "\n")
            handle.write(row["target_relpath"] + "\n")
    with (args.output_dir / "train_video_paths.txt").open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(row["source_relpath"] + "\n")
            handle.write(row["target_relpath"] + "\n")
    print(json.dumps({
        "candidate_rows": len(rows),
        "source_disjoint_candidates": len(deduped),
        "qc_passing_used": len(pool_rows),
        "train": len(train_rows),
        "reserve": len(reserve_rows),
        "seed_pool_order": args.seed,
        "seed_train_reserve_split": args.seed + 1,
    }, indent=2))


if __name__ == "__main__":
    main()
