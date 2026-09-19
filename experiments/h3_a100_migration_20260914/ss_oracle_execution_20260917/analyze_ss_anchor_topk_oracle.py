#!/usr/bin/env python3
"""Fast quality upper bound for S-S anchor + high-TopK sparsity.

All non-source keys remain visible.  Within S-S, every source query/head keeps
the fixed five-frame anchors, its corresponding source frame, and an oracle
Top-K fraction of the remaining source frames ranked by dense attention mass.
This is intentionally optimistic: failure disproves the route quickly, while
success still requires validation with the real Sparge block selector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_layer(capture_dir: Path, layer: int) -> dict:
    shards = [
        torch.load(
            capture_dir / f"layer_{layer:02d}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for rank in (0, 1)
    ]
    first = shards[0]

    def merge(value_name: str, index_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([item[value_name] for item in shards])
        indices = torch.cat([item[index_name] for item in shards]).long()
        order = torch.argsort(indices)
        return values.index_select(0, order), indices.index_select(0, order)

    q, q_indices = merge("q_selected", "query_indices")
    k, k_indices = merge("k_local", "key_indices")
    v, v_indices = merge("v_local", "key_indices")
    used = int(first["layout"]["used_len"])
    if not torch.equal(k_indices, torch.arange(used)) or not torch.equal(k_indices, v_indices):
        raise ValueError("K/V capture does not reconstruct the logical sequence")
    return {**first, "q": q, "q_indices": q_indices, "k": k, "v": v}


@torch.inference_mode()
def analyze(payload: dict, fractions: list[float], chunk: int, device: torch.device) -> dict:
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    v = payload["v"].to(device=device, dtype=torch.bfloat16)
    q_indices = payload["q_indices"].long()
    source_positions_cpu = payload["source_positions"].long()
    source_positions = source_positions_cpu.to(device)
    used = k.shape[0]
    frames = int(payload["layout"]["num_frames"])
    tokens_per_frame = int(payload["layout"]["tokens_per_frame"])
    scale = float(payload["softmax_scale"])
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})

    source_mask = torch.zeros(used, dtype=torch.bool)
    source_mask[source_positions_cpu] = True
    non_source_positions = torch.nonzero(~source_mask, as_tuple=False).view(-1).to(device)
    logical_to_frame = torch.full((used,), -1, dtype=torch.long)
    logical_to_frame[source_positions_cpu] = torch.arange(frames).repeat_interleave(tokens_per_frame)
    query_frames = logical_to_frame.index_select(0, q_indices).to(device)
    if (query_frames < 0).any():
        raise ValueError("query sample contains non-source rows")

    states = {
        fraction: {
            "sq_error": 0.0,
            "sq_dense": 0.0,
            "cos_sum": 0.0,
            "count": 0,
            "visible_frames_sum": 0.0,
        }
        for fraction in fractions
    }
    k_heads = k.transpose(0, 1).contiguous()
    v_heads = v.transpose(0, 1).contiguous()
    anchor_mask = torch.zeros(frames, dtype=torch.bool, device=device)
    anchor_mask[torch.tensor(anchors, device=device)] = True

    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        qh = q[start:stop].transpose(0, 1).contiguous()  # H,Q,D
        logits = torch.bmm(qh, k_heads.transpose(1, 2)).float() * scale
        dense_weights = torch.softmax(logits, dim=-1)
        dense_out = torch.bmm(dense_weights.to(v_heads.dtype), v_heads).float()  # H,Q,D
        source_logits = logits.index_select(2, source_positions).reshape(
            logits.shape[0], logits.shape[1], frames, tokens_per_frame
        )
        source_frame_lse = torch.logsumexp(source_logits, dim=-1)  # H,Q,F

        mandatory = anchor_mask.view(1, 1, frames).expand(
            logits.shape[0], logits.shape[1], frames
        ).clone()
        mandatory.scatter_(
            2,
            query_frames[start:stop].view(1, -1, 1).expand(logits.shape[0], -1, 1),
            True,
        )
        remaining_score = source_frame_lse.masked_fill(mandatory, float("-inf"))
        remaining_count = frames - mandatory.sum(dim=-1)

        for fraction in fractions:
            keep = mandatory.clone()
            # The count differs only for query frames that are themselves anchors.
            for rem in torch.unique(remaining_count).tolist():
                rows = remaining_count == int(rem)
                k_extra = int(math.ceil(float(fraction) * int(rem)))
                if k_extra:
                    selected = torch.topk(remaining_score[rows], k=k_extra, dim=-1).indices
                    selected_mask = torch.zeros((rows.sum(), frames), dtype=torch.bool, device=device)
                    selected_mask.scatter_(1, selected, True)
                    keep[rows] |= selected_mask

            token_keep = keep.repeat_interleave(tokens_per_frame, dim=-1)
            sparse_logits_source = source_logits.reshape(
                logits.shape[0], logits.shape[1], -1
            ).masked_fill(~token_keep, float("-inf"))
            combined_logits = torch.cat(
                (logits.index_select(2, non_source_positions), sparse_logits_source), dim=-1
            )
            combined_v = torch.cat(
                (v_heads.index_select(1, non_source_positions), v_heads.index_select(1, source_positions)),
                dim=1,
            )
            sparse_weights = torch.softmax(combined_logits, dim=-1)
            sparse_out = torch.bmm(sparse_weights.to(combined_v.dtype), combined_v).float()
            error = sparse_out - dense_out
            state = states[fraction]
            state["sq_error"] += float(error.square().sum().cpu())
            state["sq_dense"] += float(dense_out.square().sum().cpu())
            cosine = F.cosine_similarity(sparse_out, dense_out, dim=-1)
            state["cos_sum"] += float(cosine.sum().cpu())
            state["count"] += int(cosine.numel())
            state["visible_frames_sum"] += float(keep.sum().cpu())
        del logits, dense_weights, dense_out, source_logits, source_frame_lse

    result = {}
    total_query_heads = q.shape[0] * q.shape[1]
    for fraction, state in states.items():
        result[str(fraction)] = {
            "remaining_topk_fraction": fraction,
            "relative_l2": math.sqrt(state["sq_error"] / max(state["sq_dense"], 1e-30)),
            "mean_cosine": state["cos_sum"] / state["count"],
            "mean_visible_source_frames": state["visible_frames_sum"] / total_query_heads,
            "mean_visible_source_frame_fraction": (
                state["visible_frames_sum"] / total_query_heads / frames
            ),
        }
    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "queries": int(q.shape[0]),
        "heads": int(q.shape[1]),
        "frames": frames,
        "fixed_anchors": anchors,
        "variants": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.875, 0.75, 0.625, 0.5])
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "h3_ss_anchor_oracle_topk_quality_gate_v1",
        "definition": (
            "Optimistic teacher oracle: all non-source keys remain dense; S-S keeps "
            "fixed five-frame anchors, the corresponding frame, and the specified "
            "fraction of remaining source frames ranked by dense per-query/head mass."
        ),
        "warning": (
            "Passing is necessary but not sufficient. The actual Sparge block selector "
            "may be less accurate than this dense-teacher oracle."
        ),
        "layers": [],
    }
    device = torch.device("cuda:0")
    for layer in args.layers:
        payload = load_layer(args.capture_dir, layer)
        report["layers"].append(analyze(payload, args.fractions, args.chunk, device))
        del payload
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
