"""Analyze stable refresh and partial-skip DiT block profiles by GPU rank."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


GROUPS = {
    "qkv_projection": {"qkv_projection"},
    "norm_and_adaln": {
        "qk_norm",
        "adaln_projection",
        "norm1_and_modulation",
        "norm2_and_modulation",
    },
    "rope": {"rope"},
    "ulysses_communication": {
        "ulysses_softmax_pre_attention",
        "ulysses_softmax_post_attention",
        "ulysses_active_mask_allgather",
        "ulysses_linear_pre_attention",
        "ulysses_linear_post_attention",
        "ulysses_linear_frame_reduce",
    },
    "attention_core": {"softmax_attention_core", "linear_attention_core"},
    "out_projection": {"attention_out_projection", "linear_out_projection"},
    "mlp": {"mlp"},
}


def normalized_component(row: dict, name: str) -> float:
    """Scale partially captured component totals to all 50 DiT blocks."""
    count = row["counts"].get(name, 0)
    if not count:
        return 0.0
    return row["timings_ms"].get(name, 0.0) * row["num_blocks"] / count


def average_component(rows: list[dict], names: set[str]) -> float:
    return sum(
        statistics.mean(normalized_component(row, name) for row in rows)
        for name in names
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.profile_dir.glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())

    # Step 0 paid TorchDynamo compilation/recompilation costs and is not a
    # steady-state timing sample. Stable refresh steps are 1, 2, and 3.
    steady_rows = [
        row for row in rows
        if not (row["mode"] == "refresh" and row["step"] == 0)
    ]
    report: dict = {
        "method": {
            "source_records": len(rows),
            "steady_records": len(steady_rows),
            "excluded": "refresh step 0 (TorchDynamo cold start)",
            "refresh_steps": [1, 2, 3],
            "partial_skip_steps": [4, 5, 6, 7, 8, 9, 10],
            "block_count_per_forward": 50,
            "active_target_ratio": steady_rows[0]["active_target_ratio"],
        },
        "by_mode_and_rank": {},
    }

    for mode in ("refresh", "partial_skip"):
        report["by_mode_and_rank"][mode] = {}
        for rank in (0, 1):
            selected = [
                row for row in steady_rows
                if row["mode"] == mode and row["rank"] == rank
            ]
            block_total = statistics.mean(
                row["timings_ms"]["block_total"] for row in selected
            )
            group_ms = {
                name: average_component(selected, components)
                for name, components in GROUPS.items()
            }
            retained = sum(
                group_ms[name]
                for name in (
                    "qkv_projection",
                    "norm_and_adaln",
                    "rope",
                    "ulysses_communication",
                )
            )
            heavy = sum(
                group_ms[name]
                for name in ("attention_core", "out_projection", "mlp")
            )
            report["by_mode_and_rank"][mode][f"rank{rank}"] = {
                "sample_count": len(selected),
                "block_total_ms_per_forward": block_total,
                "group_ms_per_forward": group_ms,
                "group_percent_of_block_total": {
                    name: 100.0 * value / block_total
                    for name, value in group_ms.items()
                },
                "requested_comparison": {
                    "qkv_norm_rope_ulysses_ms": retained,
                    "qkv_norm_rope_ulysses_percent": 100.0 * retained / block_total,
                    "attention_core_out_proj_mlp_ms": heavy,
                    "attention_core_out_proj_mlp_percent": 100.0 * heavy / block_total,
                    "other_ms": block_total - retained - heavy,
                    "other_percent": 100.0 * (block_total - retained - heavy) / block_total,
                },
                "event_error_count": sum(len(row.get("event_errors", [])) for row in selected),
                "component_totals_normalized_to_50_blocks": True,
            }

    critical_by_mode: dict[str, list[float]] = {}
    for mode in ("refresh", "partial_skip"):
        values = []
        for step in sorted({row["step"] for row in steady_rows if row["mode"] == mode}):
            rank_values = [
                row["timings_ms"]["block_total"] for row in steady_rows
                if row["mode"] == mode and row["step"] == step
            ]
            values.append(max(rank_values))
        critical_by_mode[mode] = values
    refresh_mean = statistics.mean(critical_by_mode["refresh"])
    partial_mean = statistics.mean(critical_by_mode["partial_skip"])
    report["critical_path"] = {
        "refresh_block_ms_per_forward": refresh_mean,
        "partial_skip_block_ms_per_forward": partial_mean,
        "block_speedup": refresh_mean / partial_mean,
        "block_time_reduction_percent": 100.0 * (refresh_mean - partial_mean) / refresh_mean,
        "per_step_ms": critical_by_mode,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
