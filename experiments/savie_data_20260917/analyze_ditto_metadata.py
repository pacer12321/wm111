#!/usr/bin/env python3
"""Audit Ditto-1M metadata without loading all categories at once."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath


FILES = (
    "global+local.json",
    "global.json",
    "global_freeform3.json",
    "global_style.json",
    "local.json",
    "local_replace.json",
    "sim2real.json",
)


def audit_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    required = {"source_path", "edited_path", "instruction"}
    prefixes: Counter[str] = Counter()
    unique_sources: set[str] = set()
    triplets: set[tuple[str, str, str]] = set()
    empty_instruction = 0
    bad_schema = 0

    for row in rows:
        if not required.issubset(row):
            bad_schema += 1
            continue
        source = str(row["source_path"])
        edited = str(row["edited_path"])
        instruction = str(row["instruction"]).strip()
        parts = PurePosixPath(edited).parts
        prefixes[parts[0] if parts else ""] += 1
        unique_sources.add(source)
        triplets.add((source, edited, instruction))
        empty_instruction += not instruction

    return {
        "metadata_file": path.name,
        "rows": len(rows),
        "unique_sources": len(unique_sources),
        "unique_triplets": len(triplets),
        "duplicate_triplets": len(rows) - len(triplets),
        "empty_instruction": empty_instruction,
        "bad_schema": bad_schema,
        "edited_prefixes": dict(prefixes.most_common()),
        "example": rows[0] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = []
    for name in FILES:
        path = args.metadata_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(audit_file(path))

    result = {
        "metadata_dir": str(args.metadata_dir),
        "total_rows": sum(item["rows"] for item in reports),
        "files": reports,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
