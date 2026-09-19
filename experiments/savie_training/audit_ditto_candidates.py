#!/usr/bin/env python3
"""Print a compact audit of the downloaded Ditto candidate manifest."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    counters: dict[str, collections.Counter[object]] = {
        "category": collections.Counter(),
        "edit_type_heuristic": collections.Counter(),
        "edit_type_evidence": collections.Counter(),
    }
    first: dict | None = None
    count = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            first = first or row
            count += 1
            for key, counter in counters.items():
                value = row.get(key)
                if isinstance(value, list):
                    value = tuple(value)
                counter[value] += 1

    print(json.dumps({
        "count": count,
        "keys": sorted(first or {}),
        "counters": {
            key: {repr(item): count for item, count in value.items()}
            for key, value in counters.items()
        },
        "first": first,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
