#!/usr/bin/env python3
"""Select a source-disjoint 2K pool restricted to SIMPLE, non-temporal edits.

Scope and honesty notes (read before trusting the output):
  - This operates on the *instruction text* only (category + wording). It can
    reject temporal/compound/object-insertion edits and do stratified,
    source-disjoint sampling, but it CANNOT verify: actual visual quality,
    motion amount, edit extent in pixels, Qwen/VAE clip/geometry match, or
    real H3-encoded length. Those require the real video files and the model
    stack, which only exist on the training server.
  - "Simple" here means: single-clause, short, no local object add/remove/
    insert/replace, and matches a color-swap, scenery/background-swap, or
    named-style-swap pattern. Long or multi-clause instructions (the
    "Transform the scene into a surreal dream world where..." style) are
    rejected even when they came from a category previously treated as
    eligible, because they are compound edits, not simple ones.
  - Output manifest rows carry a stable ``sample_id`` (sha1 of source+target+
    instruction) as required by retrain_readiness.py's 2K uniqueness check.
    Still run the video-level QC (qc_and_finalize style) and the checks
    listed in this run's own ``coverage`` summary before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# Categories whose instructions are short, single-directive style swaps by
# construction ("Make it Pop Art style.", "Adopt the aesthetic of X.").
STYLE_ONLY_CATEGORIES = ("global_style1", "global_style2")

# Any of these verbs/nouns indicate a local object insertion, removal, or
# placement edit -- explicitly excluded even if short.
OBJECT_EDIT_RE = re.compile(
    r"\b(add|remove|erase|insert|delete|place|attach|mount|hang|hovering|"
    r"floating|suspended|spawn)\b",
    re.IGNORECASE,
)

COLOR_SWAP_RE = re.compile(
    r"\b(color|colour|hue|shade|tint)\b|"
    r"\b(red|blue|green|yellow|black|white|purple|pink|orange|golden|silver|"
    r"pastel|maroon|turquoise|beige|crimson|violet|teal|magenta)\b.{0,25}\b"
    r"(top|shirt|dress|blouse|jacket|coat|sky|wall|car|hair|skin|color|colour)\b",
    re.IGNORECASE,
)

SCENERY_SWAP_RE = re.compile(
    r"\b(background|backdrop|scenery|setting|sky|weather|season|environment|"
    r"landscape|time of day|day ?time|night ?time|sunset|sunrise|snow|rain)\b",
    re.IGNORECASE,
)

STYLE_SWAP_RE = re.compile(
    r"\b(style|aesthetic|filter|painting|drawing|anime|cartoon|watercolor|"
    r"comic|sketch|art ?style)\b",
    re.IGNORECASE,
)

# Anything with more than this many words is treated as compound/complex
# regardless of which patterns match -- short is part of the definition of
# "simple" here, matching what the style1/style2 category examples look like.
MAX_WORDS = 22
MAX_CLAUSES = 2


def clause_count(text: str) -> int:
    return len(re.split(r"[.;]|\band\b", text, flags=re.IGNORECASE))


def classify_complexity(instruction: str, category: str) -> tuple[str, str]:
    text = instruction.strip()
    words = text.split()

    if OBJECT_EDIT_RE.search(text):
        return "complex", "object_insertion_or_removal"

    if category in STYLE_ONLY_CATEGORIES:
        if len(words) <= MAX_WORDS:
            return "simple", "style_directive"
        return "complex", "style_directive_too_long"

    if len(words) > MAX_WORDS:
        return "complex", "too_long"
    if clause_count(text) > MAX_CLAUSES:
        return "complex", "multi_clause"

    if COLOR_SWAP_RE.search(text):
        return "simple", "color_swap"
    if SCENERY_SWAP_RE.search(text):
        return "simple", "scenery_swap"
    if STYLE_SWAP_RE.search(text):
        return "simple", "style_swap"
    return "complex", "unmatched_pattern"


def sample_id_for(row: dict) -> str:
    key = "|".join(
        [str(row.get("source_relpath", "")), str(row.get("target_relpath", "")), str(row["instruction"])]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def duration_bin(row: dict) -> str:
    duration = row.get("common_duration_seconds") or row.get("duration")
    if duration is None:
        return "unknown"
    duration = float(duration)
    if duration < 3:
        return "<3s"
    if duration < 4:
        return "3-4s"
    if duration < 5:
        return "4-5s"
    if duration < 6:
        return "5-6s"
    return ">=6s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="labeled candidates jsonl, e.g. liveedit_candidates_25k_labeled.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="output manifest jsonl path")
    parser.add_argument("--report", type=Path, required=True, help="output coverage/rejection report json path")
    parser.add_argument("--target-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--only-eligible",
        action="store_true",
        help="require row['metadata_eligible'] is True (pass this once you have run "
        "audit_reselect_candidates.py / build_savie_extraction_pool.py video probes "
        "and joined that eligibility flag back onto this manifest)",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    rejections: Counter[str] = Counter()
    simple_by_category: Counter[str] = Counter()
    pool: list[dict] = []
    for row in rows:
        if row.get("edit_type_heuristic") != "content_edit_candidate":
            rejections["not_content_edit_candidate"] += 1
            continue
        if args.only_eligible and not row.get("metadata_eligible", False):
            rejections["not_metadata_eligible"] += 1
            continue
        complexity, reason = classify_complexity(str(row["instruction"]), row.get("category", ""))
        if complexity != "simple":
            rejections[f"complex:{reason}"] += 1
            continue
        row = dict(row)
        row["simple_edit_reason"] = reason
        row["sample_id"] = sample_id_for(row)
        pool.append(row)
        simple_by_category[row.get("category", "unknown")] += 1

    # Source-disjoint: at most one selected edit per source video, so the 2K
    # set cannot be dominated by many edits of the same clip.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_source[row["source_relpath"]].append(row)

    rng = random.Random(args.seed)
    sources = list(by_source.keys())
    rng.shuffle(sources)

    # Stratify by category proportional to how much simple-eligible pool each
    # category contributed, so global_style1/2 (naturally simple, short
    # instructions) don't crowd out every local/freeform simple edit.
    category_targets = {
        category: max(1, round(args.target_size * count / max(1, len(pool))))
        for category, count in simple_by_category.items()
    }

    selected: list[dict] = []
    selected_sources: set[str] = set()
    category_counts: Counter[str] = Counter()
    for source in sources:
        if len(selected) >= args.target_size:
            break
        candidates = by_source[source]
        rng.shuffle(candidates)
        for row in candidates:
            category = row.get("category", "unknown")
            if category_counts[category] >= category_targets.get(category, args.target_size):
                continue
            selected.append(row)
            selected_sources.add(source)
            category_counts[category] += 1
            break

    # Top up from whatever is left (any category) if stratified quotas left
    # the pool short of target-size, still respecting source-disjointness.
    if len(selected) < args.target_size:
        for source in sources:
            if len(selected) >= args.target_size:
                break
            if source in selected_sources:
                continue
            candidates = by_source[source]
            rng.shuffle(candidates)
            row = candidates[0]
            selected.append(row)
            selected_sources.add(source)
            category_counts[row.get("category", "unknown")] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    duration_counts = Counter(duration_bin(row) for row in selected)
    report = {
        "status": "instruction_level_selection_only_not_visual_qc_certified",
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "simple_pool_size": len(pool),
        "simple_pool_unique_sources": len(by_source),
        "simple_pool_by_category": dict(simple_by_category),
        "rejections": dict(rejections),
        "selected": len(selected),
        "selected_unique_sources": len(selected_sources),
        "selected_by_category": dict(category_counts),
        "selected_duration_bins": dict(duration_counts),
        "target_size": args.target_size,
        "seed": args.seed,
        "still_required_before_training": [
            "run real ffprobe/duration-ratio QC on the selected pairs (qc_and_finalize style)",
            "visual spot-check: confirm edit really is color/scenery/style only, not object-level",
            "motion + edit-extent measurement on decoded frames",
            "Qwen/VAE clip + geometry match check",
            "actual H3 encoded length check",
            "held-out validation split disjoint from these 2000 sources",
        ],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if len(selected) < args.target_size:
        raise SystemExit(
            f"only selected {len(selected)}/{args.target_size} -- simple-edit pool "
            "too small at current thresholds, see rejections in the report"
        )


if __name__ == "__main__":
    main()
