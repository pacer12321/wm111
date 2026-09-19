"""Expand the sampled B-teacher route grid into per-step/per-layer schedules.

This produces a nearest-grid Oracle proxy for generation experiments.  It is
explicitly not the deployable VAE/X0 router and its lookup overhead is excluded
from claims about router cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def nearest(value: int, candidates: list[int]) -> int:
    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", default="8,12,16")
    parser.add_argument("--actual-steps", type=int, default=49)
    parser.add_argument("--actual-layers", type=int, default=50)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    summary = json.loads(args.summary.read_text())
    if summary.get("teacher_mode") != "b_vdn":
        raise RuntimeError("route schedules require the B-aware teacher summary")
    sampled_steps = [int(value) for value in summary["steps"]]
    sampled_layers = [int(value) for value in summary["layers"]]
    indexed = {
        (int(row["step"]), int(row["layer"])): row for row in summary["results"]
    }
    frames = int(summary["results"][0]["source_frames"])
    anchors = sorted(set(range(0, frames, 5)) | {0, frames - 1})
    manifest = {
        "schema": "h3_b_nearest_grid_oracle_proxy_routes_v1",
        "teacher_summary": str(args.summary),
        "sampled_steps": sampled_steps,
        "sampled_layers": sampled_layers,
        "actual_steps": args.actual_steps,
        "actual_layers": args.actual_layers,
        "source_frames": frames,
        "structural_anchors": anchors,
        "mapping": "nearest sampled timestep and nearest sampled layer; ties choose lower",
        "warning": "quality/speed proxy with precomputed teacher routes; not deployable-router timing",
        "files": {},
    }
    for size in csv_ints(args.ks):
        routes: dict[str, dict[str, list[list[int]]]] = {}
        for step in range(args.actual_steps):
            sampled_step = nearest(step, sampled_steps)
            routes[str(step)] = {}
            for layer in range(args.actual_layers):
                sampled_layer = nearest(layer, sampled_layers)
                if size == 0:
                    # Fixed-only ablation: structural anchors plus the
                    # corresponding source frame.  There is no dynamic Top-K.
                    visible = [
                        sorted(set(anchors) | {target_frame})
                        for target_frame in range(frames)
                    ]
                else:
                    row = indexed[(sampled_step, sampled_layer)]
                    visible = [
                        list(item["visible_frames"])
                        for item in row["oracle"][str(size)]["routes"]
                    ]
                if len(visible) != frames:
                    raise RuntimeError("route count does not match source frame count")
                for target_frame, values in enumerate(visible):
                    value_set = set(values)
                    if target_frame not in value_set:
                        raise RuntimeError(f"K={size} T{target_frame} omitted S_i")
                    if not set(anchors).issubset(value_set):
                        raise RuntimeError(f"K={size} T{target_frame} omitted an anchor")
                routes[str(step)][str(layer)] = visible
        payload = {
            "schema": manifest["schema"],
            "dynamic_k": size,
            "source_frames": frames,
            "structural_anchors": anchors,
            "actual_steps": args.actual_steps,
            "actual_layers": args.actual_layers,
            "mapping": manifest["mapping"],
            "warning": manifest["warning"],
            "routes": routes,
        }
        filename = f"routes_k{size}.json"
        (args.output_dir / filename).write_text(json.dumps(payload, indent=2))
        manifest["files"][str(size)] = filename
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "completed", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
