from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from analyze_ss_spatial_oracle import _load_layer, _relative_errors, _summary


def _parse_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError("ratios must lie in [0, 1]")
    return result


def _global_relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((candidate - reference).float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)
    return float((numerator / denominator).cpu())


def _pad_branch_blocks(
    values: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = values.index_select(0, positions)
    length = int(selected.shape[0])
    blocks = math.ceil(length / block_size)
    padded_length = blocks * block_size
    padded = torch.zeros(
        (padded_length, *selected.shape[1:]),
        dtype=selected.dtype,
        device=selected.device,
    )
    padded[:length] = selected
    valid = torch.arange(padded_length, device=values.device) < length
    blocked = padded.view(blocks, block_size, *selected.shape[1:])
    valid = valid.view(blocks, block_size)
    counts = valid.sum(dim=1)
    return blocked, valid, counts


def _block_means(blocked: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    divisor = counts.to(blocked.dtype).view(-1, *([1] * (blocked.ndim - 1)))
    return blocked.sum(dim=1) / divisor.squeeze(1)


def _attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.einsum("qhd,khd->qhk", q, k) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,khd->qhd", weights, v)


def _sharp_full_mask(sharpness: torch.Tensor, full_fraction: float) -> torch.Tensor:
    queries, heads = sharpness.shape
    count = int(round(queries * full_fraction))
    if count <= 0:
        return torch.zeros_like(sharpness, dtype=torch.bool)
    if count >= queries:
        return torch.ones_like(sharpness, dtype=torch.bool)
    top = torch.topk(sharpness, k=count, dim=0, largest=True).indices
    mask = torch.zeros_like(sharpness, dtype=torch.bool)
    mask.scatter_(0, top, True)
    return mask


def _hybrid_output(
    q: torch.Tensor,
    non_branch_k: torch.Tensor,
    non_branch_v: torch.Tensor,
    blocked_k: torch.Tensor,
    blocked_v: torch.Tensor,
    block_valid: torch.Tensor,
    block_counts: torch.Tensor,
    centroid_k: torch.Tensor,
    centroid_v: torch.Tensor,
    coarse_scores: torch.Tensor,
    exact_ratio: float,
    scale: float,
) -> torch.Tensor:
    queries, heads, _ = q.shape
    blocks, block_size = block_valid.shape
    exact_count = int(math.ceil(blocks * exact_ratio))
    exact_count = min(max(exact_count, 0), blocks)

    non_scores = torch.einsum("qhd,khd->qhk", q, non_branch_k) * scale
    approximate_mask = torch.ones(
        (queries, heads, blocks), dtype=torch.bool, device=q.device
    )

    exact_scores: torch.Tensor | None = None
    exact_values: torch.Tensor | None = None
    exact_valid: torch.Tensor | None = None
    if exact_count:
        exact_indices = torch.topk(
            coarse_scores, k=exact_count, dim=-1, largest=True
        ).indices
        approximate_mask.scatter_(2, exact_indices, False)
        by_head_k = blocked_k.permute(2, 0, 1, 3)
        by_head_v = blocked_v.permute(2, 0, 1, 3)
        head_indices = torch.arange(heads, device=q.device).view(1, heads, 1)
        exact_keys = by_head_k[head_indices, exact_indices]
        exact_values = by_head_v[head_indices, exact_indices]
        exact_valid = block_valid[exact_indices]
        exact_scores = torch.einsum("qhd,qhebd->qheb", q, exact_keys) * scale
        exact_scores = exact_scores.masked_fill(~exact_valid, -torch.inf)

    non_max = non_scores.amax(dim=-1)
    approximate_scores = coarse_scores.masked_fill(~approximate_mask, -torch.inf)
    approximate_max = approximate_scores.amax(dim=-1)
    maximum = torch.maximum(non_max, approximate_max)
    if exact_scores is not None:
        maximum = torch.maximum(maximum, exact_scores.amax(dim=(-1, -2)))

    non_weights = torch.exp(non_scores - maximum.unsqueeze(-1))
    denominator = non_weights.sum(dim=-1)
    numerator = torch.einsum("qhk,khd->qhd", non_weights, non_branch_v)

    approximate_weights = torch.exp(
        approximate_scores - maximum.unsqueeze(-1)
    ).masked_fill(~approximate_mask, 0.0)
    approximate_weights = approximate_weights * block_counts.view(1, 1, -1)
    denominator = denominator + approximate_weights.sum(dim=-1)
    numerator = numerator + torch.einsum(
        "qhb,bhd->qhd", approximate_weights, centroid_v
    )

    if exact_scores is not None and exact_values is not None and exact_valid is not None:
        exact_weights = torch.exp(
            exact_scores - maximum.unsqueeze(-1).unsqueeze(-1)
        ).masked_fill(~exact_valid, 0.0)
        denominator = denominator + exact_weights.sum(dim=(-1, -2))
        numerator = numerator + torch.einsum(
            "qheb,qhebd->qhd", exact_weights, exact_values
        )
    return numerator / denominator.unsqueeze(-1).clamp_min(1e-12)


def analyze_layer(
    payload: dict[str, Any],
    branch: str,
    device: torch.device,
    block_size: int,
    exact_ratios: list[float],
    full_query_fractions: list[float],
    query_batch: int,
) -> dict[str, Any]:
    q = payload["q"].to(device=device, dtype=torch.float32)
    k = payload["k"].to(device=device, dtype=torch.float32)
    v = payload["v"].to(device=device, dtype=torch.float32)
    branch_positions = payload[
        "target_positions" if branch == "qskt" else "source_positions"
    ].to(device=device, dtype=torch.long)
    branch_mask = torch.zeros(k.shape[0], dtype=torch.bool, device=device)
    branch_mask[branch_positions] = True
    non_branch_positions = torch.nonzero(~branch_mask, as_tuple=False).view(-1)
    non_branch_k = k.index_select(0, non_branch_positions)
    non_branch_v = v.index_select(0, non_branch_positions)
    blocked_k, block_valid, block_counts = _pad_branch_blocks(
        k, branch_positions, block_size
    )
    blocked_v, _, _ = _pad_branch_blocks(v, branch_positions, block_size)
    centroid_k = _block_means(blocked_k, block_counts)
    centroid_v = _block_means(blocked_v, block_counts)
    scale = float(payload["softmax_scale"])

    coarse_scores_all = torch.einsum("qhd,bhd->qhb", q, centroid_k) * scale
    coarse_distribution = torch.softmax(coarse_scores_all, dim=-1)
    sharpness = torch.sum(coarse_distribution.square(), dim=-1)
    full_masks = {
        fraction: _sharp_full_mask(sharpness, fraction)
        for fraction in full_query_fractions
    }
    records: dict[tuple[float, float], dict[str, list[torch.Tensor]]] = {
        (exact_ratio, full_fraction): {
            "dense": [],
            "candidate": [],
            "errors": [],
            "cosines": [],
        }
        for exact_ratio in exact_ratios
        for full_fraction in full_query_fractions
    }

    for start in range(0, q.shape[0], query_batch):
        stop = min(start + query_batch, q.shape[0])
        q_batch = q[start:stop]
        dense = _attention_output(q_batch, k, v, scale)
        coarse_batch = coarse_scores_all[start:stop]
        for exact_ratio in exact_ratios:
            approximate = _hybrid_output(
                q_batch,
                non_branch_k,
                non_branch_v,
                blocked_k,
                blocked_v,
                block_valid,
                block_counts,
                centroid_k,
                centroid_v,
                coarse_batch,
                exact_ratio,
                scale,
            )
            for full_fraction in full_query_fractions:
                full_mask = full_masks[full_fraction][start:stop]
                candidate = torch.where(full_mask.unsqueeze(-1), dense, approximate)
                record = records[(exact_ratio, full_fraction)]
                record["dense"].append(dense.cpu())
                record["candidate"].append(candidate.cpu())
                record["errors"].append(
                    _relative_errors(dense, candidate).cpu()
                )
                record["cosines"].append(
                    torch.nn.functional.cosine_similarity(
                        dense, candidate, dim=-1
                    ).cpu()
                )

    results = []
    blocks = int(blocked_k.shape[0])
    for (exact_ratio, full_fraction), record in records.items():
        dense = torch.cat(record["dense"])
        candidate = torch.cat(record["candidate"])
        exact_blocks = min(int(math.ceil(blocks * exact_ratio)), blocks)
        flat_fraction = 1.0 - full_fraction
        approximate_branch_cost = (
            exact_blocks * block_size + (blocks - exact_blocks)
        ) / (blocks * block_size)
        ideal_branch_compute_fraction = (
            full_fraction + flat_fraction * approximate_branch_cost
        )
        results.append(
            {
                "exact_block_ratio_requested": exact_ratio,
                "exact_blocks": exact_blocks,
                "exact_block_ratio_realized": exact_blocks / blocks,
                "full_query_fraction": full_fraction,
                "ideal_branch_compute_fraction": ideal_branch_compute_fraction,
                "ideal_branch_compute_reduction": 1.0
                - ideal_branch_compute_fraction,
                "global_relative_output_l2": _global_relative_l2(
                    dense, candidate
                ),
                "per_query_head_relative_output_error": _summary(
                    torch.cat(record["errors"])
                ),
                "output_cosine": _summary(torch.cat(record["cosines"])),
            }
        )

    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "branch": branch,
        "sampled_source_queries": int(q.shape[0]),
        "heads": int(q.shape[1]),
        "branch_keys": int(branch_positions.numel()),
        "non_branch_keys_preserved_exactly": int(non_branch_positions.numel()),
        "block_size": block_size,
        "blocks": blocks,
        "sharpness": _summary(sharpness),
        "configs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--layers", default="8,24,41")
    parser.add_argument("--branch", choices=("qskt", "qsks"), required=True)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--exact-ratios", default="0.0625,0.125,0.25")
    parser.add_argument("--full-query-fractions", default="0.0,0.5")
    parser.add_argument("--query-batch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    exact_ratios = _parse_floats(args.exact_ratios)
    full_query_fractions = _parse_floats(args.full_query_fractions)
    device = torch.device(args.device)
    step_dir = args.capture_dir / f"step_{args.step:02d}"
    layer_results = []
    for layer in layers:
        payload = _load_layer(step_dir, layer)
        layer_results.append(
            analyze_layer(
                payload,
                args.branch,
                device,
                args.block_size,
                exact_ratios,
                full_query_fractions,
                args.query_batch,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema": "h3_svoo_isa_branch_screening_oracle_v1",
        "definition": (
            "Screening oracle on sampled source queries. The selected branch is "
            "partitioned into contiguous fixed-size blocks. Coarse Q-centroid-K "
            "scores select exact blocks; every unselected block remains visible "
            "through ISA-style zeroth-order Taylor mean-K/V approximation with "
            "the correct block multiplicity. High-sharpness query/head pairs can "
            "fall back to the exact dense output. All other attention quadrants "
            "are preserved exactly. SVOO cross-input profiling and online QK "
            "co-clustering are intentionally not included in this first screen."
        ),
        "capture_dir": str(args.capture_dir),
        "branch": args.branch,
        "layers": layer_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
