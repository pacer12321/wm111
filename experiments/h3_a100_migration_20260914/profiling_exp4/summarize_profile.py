"""Aggregate per-rank Experiment-4 CUDA-event profiling JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


RETAINED = {
    "qkv_projection",
    "qk_norm",
    "rope",
    "adaln_projection",
    "norm1_and_modulation",
    "norm2_and_modulation",
    "ulysses_softmax_pre_attention",
    "ulysses_softmax_post_attention",
    "ulysses_active_mask_allgather",
    "ulysses_linear_pre_attention",
    "ulysses_linear_post_attention",
    "ulysses_linear_frame_reduce",
}
SKIPPABLE_HEAVY = {
    "softmax_attention_core",
    "linear_attention_core",
    "attention_out_projection",
    "linear_out_projection",
    "mlp",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.profile_dir.glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit("no profile rows found")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    result = {"source_rows": len(rows), "modes": {}}
    for mode, mode_rows in sorted(by_mode.items()):
        names = sorted({name for row in mode_rows for name in row["timings_ms"]})
        average_ms = {
            name: mean(row["timings_ms"].get(name, 0.0) for row in mode_rows)
            for name in names
        }
        block_total = average_ms["block_total"]
        retained_ms = sum(average_ms.get(name, 0.0) for name in RETAINED)
        heavy_ms = sum(average_ms.get(name, 0.0) for name in SKIPPABLE_HEAVY)
        accounted = retained_ms + heavy_ms
        result["modes"][mode] = {
            "samples_across_ranks": len(mode_rows),
            "steps": sorted({row["step"] for row in mode_rows}),
            "active_target_ratio": mean(row["active_target_ratio"] for row in mode_rows),
            "block_total_ms_per_forward": block_total,
            "average_ms_per_forward": average_ms,
            "percent_of_block_total": {
                name: 100.0 * value / block_total for name, value in average_ms.items()
                if name != "block_total"
            },
            "requested_groups": {
                "retained_qkv_norm_rope_ulysses_ms": retained_ms,
                "retained_qkv_norm_rope_ulysses_percent": 100.0 * retained_ms / block_total,
                "attention_core_out_proj_mlp_ms": heavy_ms,
                "attention_core_out_proj_mlp_percent": 100.0 * heavy_ms / block_total,
                "other_or_unattributed_ms": block_total - accounted,
                "other_or_unattributed_percent": 100.0 * (block_total - accounted) / block_total,
            },
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
