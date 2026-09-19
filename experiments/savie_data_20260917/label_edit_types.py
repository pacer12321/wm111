#!/usr/bin/env python3
"""Add a conservative instruction-only edit taxonomy and report its balance.

This is a first-pass audit, not ground truth. High-precision temporal phrases are
flagged for review; ambiguous instructions remain explicitly unclassified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TEMPORAL_PATTERNS = [
    r"\b(change|reverse|reorder|swap|alter)\b.{0,30}\b(order|sequence|motion|movement|action|trajectory)\b",
    r"\bfirst\b.{0,100}\b(then|afterwards|followed by)\b",
    r"\b(before|after)\b.{0,80}\b(walk|run|jump|dance|wave|turn|raise|lower|sit|stand|move)\w*\b",
    r"\b(reverse|backward|backwards|rewind|time[- ]reversal)\b",
    r"\b(slow motion|slow down|speed up|time[- ]lapse|faster|slower|accelerat\w*|decelerat\w*)\b",
    r"\b(loop|repeat|pause|freeze|stop moving|start moving)\b",
    r"\bmake\b.{0,40}\b(walk|run|jump|dance|wave|turn around|raise|lower|sit|stand)\b",
]

CONTENT_PATTERNS = [
    r"\b(color|colour|red|blue|green|yellow|black|white|purple|pink|orange)\b",
    r"\b(style|stylized|painting|drawing|anime|cartoon|watercolor|comic|sketch)\b",
    r"\b(replace|change|transform|turn)\b.{0,40}\b(into|to)\b",
    r"\b(add|remove|erase|insert)\b",
    r"\b(material|texture|lighting|background|weather|atmosphere)\b",
]

TEMPORAL_RE = [re.compile(pattern, re.IGNORECASE) for pattern in TEMPORAL_PATTERNS]
CONTENT_RE = [re.compile(pattern, re.IGNORECASE) for pattern in CONTENT_PATTERNS]


def classify(instruction: str) -> tuple[str, list[str]]:
    temporal_hits = [pattern.pattern for pattern in TEMPORAL_RE if pattern.search(instruction)]
    content_hits = [pattern.pattern for pattern in CONTENT_RE if pattern.search(instruction)]
    if temporal_hits:
        return "temporal_change_candidate", temporal_hits
    if content_hits:
        return "content_edit_candidate", content_hits
    return "ambiguous", []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--examples-per-label", type=int, default=100)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            row = json.loads(line)
            label, evidence = classify(str(row["instruction"]))
            row["edit_type_heuristic"] = label
            row["edit_type_evidence"] = evidence
            counts[label] += 1
            by_category[row["category"]][label] += 1
            if len(examples[label]) < args.examples_per_label:
                examples[label].append(
                    {
                        "sample_id": row["sample_id"],
                        "category": row["category"],
                        "instruction": row["instruction"],
                        "evidence": evidence,
                    }
                )
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "warning": "Heuristic labels are audit candidates, not ground-truth annotations.",
        "counts": dict(counts),
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
        "examples": examples,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": report["counts"], "by_category": report["by_category"]}, indent=2))


if __name__ == "__main__":
    main()
