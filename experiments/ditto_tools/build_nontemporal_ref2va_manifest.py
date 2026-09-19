#!/usr/bin/env python3
"""Build the SAViE Ref2VA candidate set from the existing Ditto-1M audit.

The training scope is deliberately narrow: appearance/content edits whose source and
target retain the same temporal structure.  Ambiguous and temporal-change candidates
are excluded.  Archive member lists are regenerated from this filtered manifest so
the data extraction job does not unpack irrelevant videos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def source_group(row: dict) -> str:
    # Same source clip (possibly paired with several targets/instructions) must never
    # cross train/validation/test boundaries.
    return str(row["source_relpath"])


def deterministic_group_order(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_member(path: str) -> tuple[str, str]:
    clean = path.removeprefix("videos/")
    group = clean.split("/", 1)[0]
    return group, clean


def grouped_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[source_group(row)].append(row)
    ordered = sorted(groups, key=lambda key: deterministic_group_order(key, seed))
    total = len(rows)
    targets = {
        "test": round(total * 0.025),
        "validation": round(total * 0.025),
    }
    result = {"train": [], "validation": [], "test": []}
    for group in ordered:
        items = groups[group]
        candidates = ("test", "validation")
        destination = min(
            candidates,
            key=lambda split: (len(result[split]) >= targets[split], len(result[split])),
        )
        if all(len(result[split]) >= targets[split] for split in candidates):
            destination = "train"
        result[destination].extend(items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270918)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    selected = [
        row for row in rows
        if row.get("edit_type_heuristic") == "content_edit_candidate"
    ]
    selected.sort(key=lambda row: int(row["candidate_rank"]))
    for row in selected:
        row["scope"] = "temporally_aligned_content_edit"

    split = grouped_split(selected, args.seed)
    output = args.output_dir
    write_jsonl(output / "candidates_nontemporal.jsonl", selected)
    for name, split_rows in split.items():
        write_jsonl(output / f"candidates_{name}.jsonl", split_rows)

    member_sets: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        for key in ("source_repo_path", "target_repo_path"):
            group, member = repo_member(str(row[key]))
            member_sets[group].add(member)
    members_dir = output / "members_nontemporal"
    members_dir.mkdir(parents=True, exist_ok=True)
    for group, members in member_sets.items():
        (members_dir / f"{group}.txt").write_text(
            "\n".join(sorted(members)) + "\n", encoding="utf-8"
        )

    source_sets = {
        name: {source_group(row) for row in split_rows}
        for name, split_rows in split.items()
    }
    assert source_sets["train"].isdisjoint(source_sets["validation"])
    assert source_sets["train"].isdisjoint(source_sets["test"])
    assert source_sets["validation"].isdisjoint(source_sets["test"])
    summary = {
        "input_rows": len(rows),
        "selected_nontemporal_candidates": len(selected),
        "excluded": len(rows) - len(selected),
        "split_rows": {name: len(value) for name, value in split.items()},
        "split_source_groups": {name: len(value) for name, value in source_sets.items()},
        "categories": dict(Counter(row["category"] for row in selected)),
        "archive_members": {name: len(value) for name, value in sorted(member_sets.items())},
        "source_disjoint": True,
        "note": "Heuristic first pass; ffprobe and temporal-alignment QC still required.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
