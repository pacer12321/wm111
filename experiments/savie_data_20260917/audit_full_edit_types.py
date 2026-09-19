#!/usr/bin/env python3
"""Audit temporal-change vs content-preserving edits in canonical Ditto metadata.

``ambiguous`` is a review queue, not a third task category. Such rows must be
manually/semantically classified into one of the two task classes or excluded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from label_edit_types import classify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=200)
    args = parser.parse_args()

    with args.metadata_json.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    counts: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    unique_sources: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        instruction = str(row["instruction"]).strip()
        label, evidence = classify(instruction)
        category = PurePosixPath(str(row["edited_path"])).parts[0]
        counts[label] += 1
        by_category[category][label] += 1
        unique_sources[label].add(str(row["source_path"]))
        if len(examples[label]) < args.examples:
            examples[label].append(
                {
                    "category": category,
                    "source_path": row["source_path"],
                    "edited_path": row["edited_path"],
                    "instruction": instruction,
                    "evidence": evidence,
                }
            )

    report = {
        "total_rows": len(rows),
        "counts": dict(counts),
        "unique_sources": {key: len(value) for key, value in unique_sources.items()},
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
        "examples": examples,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
