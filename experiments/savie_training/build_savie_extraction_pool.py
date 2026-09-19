#!/usr/bin/env python3
"""Build exact tar member lists for available, non-temporal Ditto pairs."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


AVAILABLE_TARGET_GROUPS = (
    "global_freeform1",
    "global_freeform2",
    "global_style1",
)


def write_lines(path: Path, values: set[str]) -> None:
    # Ditto tar members are stored with a literal ``./`` prefix. GNU tar's
    # files-from matching is exact here, so omitting it scans the full archive
    # and then reports every requested member as missing.
    path.write_text("".join(f"./{value}\n" for value in sorted(values)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("category") not in AVAILABLE_TARGET_GROUPS:
                continue
            if row.get("edit_type_heuristic") != "content_edit_candidate":
                continue
            rows.append(row)

    # Keep every technically available content-edit candidate for subsequent
    # video-level quality ranking. Do not randomly truncate before QC.
    source_members = {row["source_relpath"] for row in rows}
    target_members: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        target_members[row["category"]].add(row["target_relpath"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(args.output_dir / "source.txt", source_members)
    for group in AVAILABLE_TARGET_GROUPS:
        write_lines(args.output_dir / f"{group}.txt", target_members[group])

    manifest_out = args.output_dir / "savie_available_content_candidates.jsonl"
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "pairs": len(rows),
        "unique_sources": len(source_members),
        "targets": {group: len(target_members[group]) for group in AVAILABLE_TARGET_GROUPS},
        "manifest": str(manifest_out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
