#!/usr/bin/env python3
"""Validate extracted Ditto videos and finalize source-disjoint 20k splits."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from collections import Counter
from pathlib import Path


def probe(path: Path) -> tuple[str, dict]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
        video_streams = [
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
        valid = bool(video_streams) and duration > 0.5
        stream = video_streams[0] if video_streams else {}
        result = {
            "valid": valid,
            "duration": duration,
            "width": int(stream.get("width", 0) or 0),
            "height": int(stream.get("height", 0) or 0),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "nb_frames": stream.get("nb_frames"),
        }
    except Exception as error:  # ffprobe failures are data-quality outcomes.
        result = {"valid": False, "error": repr(error)}
    return str(path), result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    paths = sorted(
        {
            args.video_root / row[key]
            for row in rows
            for key in ("source_relpath", "target_relpath")
        }
    )

    probes: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, result in pool.map(probe, paths):
            probes[path] = result

    accepted = []
    rejected = []
    for row in sorted(rows, key=lambda item: item["candidate_rank"]):
        source = args.video_root / row["source_relpath"]
        target = args.video_root / row["target_relpath"]
        source_probe = probes.get(str(source), {"valid": False, "error": "missing probe"})
        target_probe = probes.get(str(target), {"valid": False, "error": "missing probe"})
        duration_ratio = 0.0
        if source_probe.get("duration") and target_probe.get("duration"):
            duration_ratio = target_probe["duration"] / source_probe["duration"]
        valid_pair = (
            source_probe.get("valid", False)
            and target_probe.get("valid", False)
            and 0.8 <= duration_ratio <= 1.25
        )
        output_row = {
            **row,
            "source_path": str(source),
            "target_path": str(target),
            "source_probe": source_probe,
            "target_probe": target_probe,
            "duration_ratio": duration_ratio,
        }
        (accepted if valid_pair else rejected).append(output_row)

    if len(accepted) < args.target_size:
        raise RuntimeError(f"Only {len(accepted)} valid pairs; need {args.target_size}")
    final = accepted[: args.target_size]
    train = final[:19_000]
    validation = final[19_000:19_500]
    test = final[19_500:20_000]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all_20k.jsonl", final)
    write_jsonl(args.output_dir / "train_19k.jsonl", train)
    write_jsonl(args.output_dir / "validation_500.jsonl", validation)
    write_jsonl(args.output_dir / "test_500.jsonl", test)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    with (args.output_dir / "video_probes.json").open("w", encoding="utf-8") as handle:
        json.dump(probes, handle, ensure_ascii=False)

    summary = {
        "candidate_pairs": len(rows),
        "unique_video_files": len(paths),
        "valid_pairs": len(accepted),
        "rejected_pairs": len(rejected),
        "final_pairs": len(final),
        "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
        "final_categories": dict(Counter(row["category"] for row in final)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
