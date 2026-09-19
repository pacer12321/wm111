#!/usr/bin/env python3
"""Continuously copy newly extracted selected videos into the shared queue."""

import argparse
import json
import shutil
import time
from pathlib import Path


def sync_one(source: Path, destination: Path) -> bool:
    done = Path(str(destination) + ".copydone")
    if done.is_file():
        return True
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, destination)
    done.touch()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    pending = list(range(len(rows)))
    while pending:
        next_pending = []
        completed = 0
        for index in pending:
            row = rows[index]
            source_ok = sync_one(
                args.source_root / row["source_relpath"],
                args.shared_root / row["source_relpath"],
            )
            target_ok = sync_one(
                args.source_root / row["target_relpath"],
                args.shared_root / row["target_relpath"],
            )
            if source_ok and target_ok:
                completed += 1
            else:
                next_pending.append(index)
        print(json.dumps({
            "newly_complete_pairs": completed,
            "remaining_pairs": len(next_pending),
        }), flush=True)
        pending = next_pending
        if pending:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
