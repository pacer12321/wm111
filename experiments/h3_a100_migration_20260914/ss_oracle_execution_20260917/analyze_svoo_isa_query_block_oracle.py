from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from analyze_ss_spatial_oracle import _load_layer, _relative_errors, _summary
from analyze_svoo_isa_branch_oracle import (
    _attention_output,
    _block_means,
    _global_relative_l2,
    _pad_branch_blocks,
)


def _parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("query block sizes must be positive")
    return result


def _shared_exact_hybrid_output(
    q: torch.Tensor,
    non_branch_k: torch.Tensor,
    non_branch_v: torch.Tensor,
    blocked_k: torch.Tensor,
    blocked_v: torch.Tensor,
    block_valid: torch.Tensor,
    block_counts: torch.Tensor,
    centroid_v: torch.Tensor,
    per_query_coarse_scores: torch.Tensor,
    shared_exact_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """ISA approximation with one exact-key-block route per query-block/head.

    Args:
        q: [Qb, H, D].
        per_query_coarse_scores: [Qb, H, B].
        shared_exact_indices: [H, E], shared by every query in this Q block.
    """
    queries, heads, _ = q.shape
    blocks, _ = block_valid.shape
    exact_count = int(shared_exact_indices.shape[-1])

    non_scores = torch.einsum("qhd,khd->qhk", q, non_branch_k) * scale
    approximate_mask = torch.ones(
        (heads, blocks), dtype=torch.bool, device=q.device
    )
    if exact_count:
        approximate_mask.scatter_(1, shared_exact_indices, False)

    exact_scores: torch.Tensor | None = None
    exact_values: torch.Tensor | None = None
    exact_valid: torch.Tensor | None = None
    if exact_count:
        by_head_k = blocked_k.permute(2, 0, 1, 3)
        by_head_v = blocked_v.permute(2, 0, 1, 3)
        head_indices = torch.arange(heads, device=q.device).view(heads, 1)
        exact_keys = by_head_k[head_indices, shared_exact_indices]
        exact_values = by_head_v[head_indices, shared_exact_indices]
        exact_valid = block_valid[shared_exact_indices]
        exact_scores = torch.einsum("qhd,hebd->qheb", q, exact_keys) * scale
        exact_scores = exact_scores.masked_fill(
            ~exact_valid.unsqueeze(0), -torch.inf
        )

    approximate_scores = per_query_coarse_scores.masked_fill(
        ~approximate_mask.unsqueeze(0), -torch.inf
    )
    maximum = torch.maximum(
        non_scores.amax(dim=-1), approximate_scores.amax(dim=-1)
    )
    if exact_scores is not None:
        maximum = torch.maximum(maximum, exact_scores.amax(dim=(-1, -2)))

    non_weights = torch.exp(non_scores - maximum.unsqueeze(-1))
    denominator = non_weights.sum(dim=-1)
    numerator = torch.einsum("qhk,khd->qhd", non_weights, non_branch_v)

    approximate_weights = torch.exp(
        approximate_scores - maximum.unsqueeze(-1)
    ).masked_fill(~approximate_mask.unsqueeze(0), 0.0)
    approximate_weights = approximate_weights * block_counts.view(1, 1, -1)
    denominator = denominator + approximate_weights.sum(dim=-1)
    numerator = numerator + torch.einsum(
        "qhb,bhd->qhd", approximate_weights, centroid_v
    )

    if exact_scores is not None and exact_values is not None and exact_valid is not None:
        exact_weights = torch.exp(
            exact_scores - maximum.unsqueeze(-1).unsqueeze(-1)
        ).masked_fill(~exact_valid.unsqueeze(0), 0.0)
        denominator = denominator + exact_weights.sum(dim=(-1, -2))
        numerator = numerator + torch.einsum(
            "qheb,hebd->qhd", exact_weights, exact_values
        )
    return numerator / denominator.unsqueeze(-1).clamp_min(1e-12)


def analyze_layer(
    payload: dict[str, Any],
    branch: str,
    device: torch.device,
    key_block_size: int,
    query_block_size: int,
    exact_ratio: float,
    full_query_fraction: float,
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
        k, branch_positions, key_block_size
    )
    blocked_v, _, _ = _pad_branch_blocks(v, branch_positions, key_block_size)
    centroid_k = _block_means(blocked_k, block_counts)
    centroid_v = _block_means(blocked_v, block_counts)
    scale = float(payload["softmax_scale"])

    queries, heads, _ = q.shape
    key_blocks = int(blocked_k.shape[0])
    exact_count = min(max(int(math.ceil(key_blocks * exact_ratio)), 0), key_blocks)
    query_blocks = math.ceil(queries / query_block_size)

    # Routing is deliberately deployable: a contiguous query block shares one
    # route for each head.  Its representative is the mean query vector.
    representative_scores: list[torch.Tensor] = []
    for start in range(0, queries, query_block_size):
        stop = min(start + query_block_size, queries)
        representative = q[start:stop].mean(dim=0)
        representative_scores.append(
            torch.einsum("hd,bhd->hb", representative, centroid_k) * scale
        )
    route_scores = torch.stack(representative_scores)  # [QB, H, KB]
    route_distribution = torch.softmax(route_scores, dim=-1)
    route_sharpness = torch.sum(route_distribution.square(), dim=-1)
    dense_block_count = min(
        max(int(round(query_blocks * full_query_fraction)), 0), query_blocks
    )
    dense_query_blocks = torch.zeros(
        (query_blocks, heads), dtype=torch.bool, device=device
    )
    if dense_block_count:
        dense_indices = torch.topk(
            route_sharpness, k=dense_block_count, dim=0, largest=True
        ).indices
        dense_query_blocks.scatter_(0, dense_indices, True)

    dense_outputs: list[torch.Tensor] = []
    candidate_outputs: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    cosines: list[torch.Tensor] = []
    for block_index, start in enumerate(range(0, queries, query_block_size)):
        stop = min(start + query_block_size, queries)
        q_block = q[start:stop]
        dense = _attention_output(q_block, k, v, scale)
        per_query_scores = torch.einsum(
            "qhd,bhd->qhb", q_block, centroid_k
        ) * scale
        if exact_count:
            exact_indices = torch.topk(
                route_scores[block_index], k=exact_count, dim=-1, largest=True
            ).indices
        else:
            exact_indices = torch.empty(
                (heads, 0), dtype=torch.long, device=device
            )
        approximate = _shared_exact_hybrid_output(
            q_block,
            non_branch_k,
            non_branch_v,
            blocked_k,
            blocked_v,
            block_valid,
            block_counts,
            centroid_v,
            per_query_scores,
            exact_indices,
            scale,
        )
        full_mask = dense_query_blocks[block_index].view(1, heads, 1)
        candidate = torch.where(full_mask, dense, approximate)
        dense_outputs.append(dense.cpu())
        candidate_outputs.append(candidate.cpu())
        errors.append(_relative_errors(dense, candidate).cpu())
        cosines.append(
            torch.nn.functional.cosine_similarity(dense, candidate, dim=-1).cpu()
        )

    dense_all = torch.cat(dense_outputs)
    candidate_all = torch.cat(candidate_outputs)
    flat_fraction = 1.0 - full_query_fraction
    approximate_branch_cost = (
        exact_count * key_block_size + (key_blocks - exact_count)
    ) / (key_blocks * key_block_size)
    ideal_branch_compute_fraction = (
        full_query_fraction + flat_fraction * approximate_branch_cost
    )
    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "branch": branch,
        "sampled_source_queries": queries,
        "heads": heads,
        "query_block_size": query_block_size,
        "query_blocks": query_blocks,
        "key_block_size": key_block_size,
        "key_blocks": key_blocks,
        "exact_block_ratio_requested": exact_ratio,
        "exact_blocks": exact_count,
        "exact_block_ratio_realized": exact_count / key_blocks,
        "dense_query_block_fraction_requested": full_query_fraction,
        "dense_query_blocks_per_head": dense_block_count,
        "dense_query_block_fraction_realized": dense_block_count / query_blocks,
        "ideal_branch_compute_fraction": ideal_branch_compute_fraction,
        "ideal_branch_compute_reduction": 1.0 - ideal_branch_compute_fraction,
        "global_relative_output_l2": _global_relative_l2(
            dense_all, candidate_all
        ),
        "per_query_head_relative_output_error": _summary(torch.cat(errors)),
        "output_cosine": _summary(torch.cat(cosines)),
        "route_sharpness": _summary(route_sharpness),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--layers", default="8,24,41")
    parser.add_argument("--branch", choices=("qskt", "qsks"), required=True)
    parser.add_argument("--key-block-size", type=int, default=64)
    parser.add_argument("--query-block-sizes", default="16,32,64")
    parser.add_argument("--exact-ratio", type=float, default=0.5)
    parser.add_argument("--full-query-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0.0 <= args.exact_ratio <= 1.0:
        raise ValueError("exact ratio must lie in [0, 1]")
    if not 0.0 <= args.full_query_fraction <= 1.0:
        raise ValueError("full query fraction must lie in [0, 1]")

    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    query_block_sizes = _parse_ints(args.query_block_sizes)
    device = torch.device(args.device)
    step_dir = args.capture_dir / f"step_{args.step:02d}"
    results = []
    for query_block_size in query_block_sizes:
        for layer in layers:
            payload = _load_layer(step_dir, layer)
            results.append(
                analyze_layer(
                    payload,
                    args.branch,
                    device,
                    args.key_block_size,
                    query_block_size,
                    args.exact_ratio,
                    args.full_query_fraction,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "schema": "h3_svoo_isa_query_block_oracle_v1",
        "definition": (
            "Deployment-granularity calibration on the existing sampled source "
            "queries. Contiguous query blocks share one exact-key-block route per "
            "head, and dense fallback is selected for whole query blocks. Key "
            "blocks remain contiguous and unselected blocks remain visible via "
            "the ISA zeroth-order Taylor mean-K/V remainder."
        ),
        "capture_dir": str(args.capture_dir),
        "branch": args.branch,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
