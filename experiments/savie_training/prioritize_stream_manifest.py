#!/usr/bin/env python3
"""Put already materialized pairs first without changing train-set membership."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    ready, pending = [], []
    for row in rows:
        bucket = ready if (
            (args.video_root / row["source_relpath"]).is_file()
            and (args.video_root / row["target_relpath"]).is_file()
        ) else pending
        bucket.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ready + pending:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(rows), "ready_first": len(ready), "pending": len(pending)}))


if __name__ == "__main__":
    main()
