#!/usr/bin/env python3
"""Create exact tar member lists required by a Ditto candidate manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    members: dict[str, set[str]] = defaultdict(set)
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for key in ("source_relpath", "target_relpath"):
                relpath = str(row[key])
                group = PurePosixPath(relpath).parts[0]
                members[group].add(relpath)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for group, paths in sorted(members.items()):
        output = args.output_dir / f"{group}.txt"
        output.write_text("".join(f"{path}\n" for path in sorted(paths)), encoding="utf-8")
        summary[group] = len(paths)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
