#!/usr/bin/env python3
"""Quality gate for static S-S local + bidirectional global-anchor attention.

Only S-S is changed.  Every non-source key (including every target token) stays
visible, so S->T remains exactly dense.  Anchor source-query frames see every
source key.  Non-anchor source-query frames see local source frames, all fixed
anchor source frames, and optionally a dense-teacher oracle Top-K of remaining
source frames.  The optional Top-K is diagnostic only; K=0 is the deployable
fully-static proposal.
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
def analyze(
    payload: dict,
    radii: list[int],
    remote_topks: list[int],
    device: torch.device,
) -> dict:
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    v = payload["v"].to(device=device, dtype=torch.bfloat16)
    q_indices = payload["q_indices"].long()
    source_positions_cpu = payload["source_positions"].long()
    source_positions = source_positions_cpu.to(device)
    frames = int(payload["layout"]["num_frames"])
    tokens_per_frame = int(payload["layout"]["tokens_per_frame"])
    scale = float(payload["softmax_scale"])
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})
    q_per_frame = q.shape[0] // frames
    if q_per_frame * frames != q.shape[0]:
        raise ValueError("sampled source queries are not frame-balanced")

    source_mask = torch.zeros(k.shape[0], dtype=torch.bool)
    source_mask[source_positions_cpu] = True
    logical_to_frame = torch.full((k.shape[0],), -1, dtype=torch.long)
    logical_to_frame[source_positions_cpu] = torch.arange(frames).repeat_interleave(tokens_per_frame)
    query_frames = logical_to_frame.index_select(0, q_indices).to(device)
    if (query_frames < 0).any():
        raise ValueError("query sample contains non-source rows")

    anchor_frames = torch.tensor(anchors, device=device)
    source_frame_ids = torch.arange(frames, device=device).repeat_interleave(tokens_per_frame)
    k_heads = k.transpose(0, 1).contiguous()
    v_heads = v.transpose(0, 1).contiguous()
    variants = {
        (radius, remote_topk): {
            "sq_error": 0.0,
            "sq_dense": 0.0,
            "cos_sum": 0.0,
            "count": 0,
            "visible_sum": 0.0,
            "source_mass_kept": 0.0,
            "source_mass_total": 0.0,
        }
        for radius in radii
        for remote_topk in remote_topks
    }

    for start in range(0, q.shape[0], q_per_frame):
        stop = min(start + q_per_frame, q.shape[0])
        qh = q[start:stop].transpose(0, 1).contiguous()  # H,Q,D
        logits = torch.bmm(qh, k_heads.transpose(1, 2)).float() * scale
        dense_weights = torch.softmax(logits, dim=-1)
        dense_out = torch.bmm(dense_weights.to(v_heads.dtype), v_heads).float()
        source_logits = logits.index_select(2, source_positions).reshape(
            logits.shape[0], logits.shape[1], frames, tokens_per_frame
        )
        source_frame_lse = torch.logsumexp(source_logits, dim=-1)
        source_dense_mass = dense_weights.index_select(2, source_positions).reshape(
            logits.shape[0], logits.shape[1], frames, tokens_per_frame
        ).sum(dim=-1)
        q_frames = query_frames[start:stop]

        for radius in radii:
            mandatory = torch.zeros(
                (logits.shape[0], logits.shape[1], frames),
                dtype=torch.bool,
                device=device,
            )
            mandatory[..., anchor_frames] = True
            anchor_query = torch.isin(q_frames, anchor_frames)
            for row, frame in enumerate(q_frames.tolist()):
                lo = max(0, int(frame) - radius)
                hi = min(frames, int(frame) + radius + 1)
                mandatory[:, row, lo:hi] = True
                if bool(anchor_query[row]):
                    mandatory[:, row, :] = True

            for remote_topk in remote_topks:
                keep = mandatory.clone()
                if remote_topk:
                    remaining_score = source_frame_lse.masked_fill(keep, float("-inf"))
                    available = (~keep).sum(dim=-1)
                    for count in torch.unique(available).tolist():
                        rows = available == int(count)
                        selected_count = min(remote_topk, int(count))
                        if selected_count:
                            selected = torch.topk(
                                remaining_score[rows], k=selected_count, dim=-1
                            ).indices
                            selected_mask = torch.zeros(
                                (int(rows.sum()), frames), dtype=torch.bool, device=device
                            )
                            selected_mask.scatter_(1, selected, True)
                            keep[rows] |= selected_mask

                token_keep = keep.repeat_interleave(tokens_per_frame, dim=-1)
                sparse_logits = logits.clone()
                sparse_source_logits = source_logits.reshape(
                    logits.shape[0], logits.shape[1], -1
                ).masked_fill(~token_keep, float("-inf"))
                sparse_logits[:, :, source_positions] = sparse_source_logits
                sparse_weights = torch.softmax(sparse_logits, dim=-1)
                sparse_out = torch.bmm(sparse_weights.to(v_heads.dtype), v_heads).float()
                error = sparse_out - dense_out
                state = variants[(radius, remote_topk)]
                state["sq_error"] += float(error.square().sum().cpu())
                state["sq_dense"] += float(dense_out.square().sum().cpu())
                state["cos_sum"] += float(
                    F.cosine_similarity(sparse_out, dense_out, dim=-1).sum().cpu()
                )
                state["count"] += int(sparse_out.shape[0] * sparse_out.shape[1])
                state["visible_sum"] += float(keep.sum().cpu())
                state["source_mass_kept"] += float((source_dense_mass * keep).sum().cpu())
                state["source_mass_total"] += float(source_dense_mass.sum().cpu())
                del sparse_logits, sparse_source_logits, sparse_weights, sparse_out
        del logits, dense_weights, dense_out, source_logits, source_frame_lse, source_dense_mass

    results = {}
    for (radius, remote_topk), state in variants.items():
        mean_visible = state["visible_sum"] / state["count"]
        results[f"r{radius}_k{remote_topk}"] = {
            "local_radius": radius,
            "local_window_nominal_frames": 2 * radius + 1,
            "oracle_remote_topk_frames": remote_topk,
            "relative_l2": math.sqrt(state["sq_error"] / max(state["sq_dense"], 1e-30)),
            "mean_cosine": state["cos_sum"] / state["count"],
            "mean_visible_source_frames": mean_visible,
            "mean_visible_source_frame_fraction": mean_visible / frames,
            "estimated_ss_pair_reduction": 1.0 - mean_visible / frames,
            "retained_dense_source_attention_mass": (
                state["source_mass_kept"] / max(state["source_mass_total"], 1e-30)
            ),
        }
    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "queries": int(q.shape[0]),
        "heads": int(q.shape[1]),
        "frames": frames,
        "fixed_anchor_frames": anchors,
        "anchor_query_policy": "anchor S queries see every S key",
        "non_source_policy": "all non-source keys remain dense; S->T is unchanged",
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--remote-topks", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "h3_ss_bidirectional_global_anchor_local_oracle_v1",
        "scope": (
            "Real H3 Q/K/V. Only S-S is sparsified. S->T and every other non-source "
            "connection remain dense. Anchor source-query frames are global; other "
            "source-query frames see local source frames plus every anchor. Optional "
            "remote Top-K is a dense-teacher oracle diagnostic, not a deployable router."
        ),
        "layers": [],
    }
    device = torch.device("cuda:0")
    for layer in args.layers:
        payload = load_layer(args.capture_dir, layer)
        report["layers"].append(
            analyze(payload, args.radii, args.remote_topks, device)
        )
        del payload
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
