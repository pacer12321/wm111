#!/usr/bin/env python3
"""Build a deterministic, source-disjoint candidate manifest from Ditto-1M.

The canonical pool is training_metadata/global+local.json, which contains the
roughly one-million triplets corresponding to the released Ditto-1M corpus.
We oversample candidates so video-level QC can still yield 20k valid pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def allocate_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    raw = {key: total * value / population for key, value in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: raw[key] - quotas[key], reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def stable_id(source: str, edited: str, instruction: str) -> str:
    payload = "\0".join((source, edited, instruction)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=25_000)
    parser.add_argument("--target-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20270917)
    args = parser.parse_args()

    with args.metadata_json.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        source = str(row["source_path"])
        edited = str(row["edited_path"])
        instruction = str(row["instruction"]).strip()
        if not instruction:
            continue
        category = PurePosixPath(edited).parts[0]
        by_category[category].append(
            {
                "source_relpath": source,
                "target_relpath": edited,
                "instruction": instruction,
                "category": category,
            }
        )

    counts = {key: len(value) for key, value in by_category.items()}
    quotas = allocate_quotas(counts, args.candidates)
    rng = random.Random(args.seed)
    for category in sorted(by_category):
        rng.shuffle(by_category[category])

    selected: list[dict] = []
    used_sources: set[str] = set()
    selected_counts: Counter[str] = Counter()
    for category in sorted(by_category):
        for row in by_category[category]:
            if selected_counts[category] >= quotas[category]:
                break
            if row["source_relpath"] in used_sources:
                continue
            used_sources.add(row["source_relpath"])
            selected_counts[category] += 1
            row["sample_id"] = stable_id(
                row["source_relpath"], row["target_relpath"], row["instruction"]
            )
            row["source_repo_path"] = "videos/" + row["source_relpath"]
            row["target_repo_path"] = "videos/" + row["target_relpath"]
            selected.append(row)

    if len(selected) != args.candidates:
        raise RuntimeError(
            f"Could select only {len(selected)} source-disjoint rows; "
            f"expected {args.candidates}"
        )

    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for rank, row in enumerate(selected):
            record = dict(row)
            record["candidate_rank"] = rank
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source_metadata": str(args.metadata_json),
        "seed": args.seed,
        "target_size_after_qc": args.target_size,
        "candidate_size_before_qc": len(selected),
        "source_disjoint": True,
        "population_by_category": counts,
        "candidate_quota_by_category": quotas,
        "selected_by_category": dict(sorted(selected_counts.items())),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
