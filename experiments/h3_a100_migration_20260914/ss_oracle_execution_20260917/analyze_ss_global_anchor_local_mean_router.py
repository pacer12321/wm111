#!/usr/bin/env python3
"""Real low-cost frame-router gate for S-S local + global-anchor sparsity.

S->T and every non-source key remain dense. Anchor S query frames see all S.
For each non-anchor source frame and head, remote S frames are selected using a
cheap dot product between mean-pooled sampled frame queries and mean-pooled
frame keys. This is a deployable proxy rather than a dense-attention teacher.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_layer(capture_dir: Path, layer: int) -> dict:
    shards = [torch.load(
        capture_dir / f"layer_{layer:02d}_rank_{rank}.pt",
        map_location="cpu", weights_only=True, mmap=True,
    ) for rank in (0, 1)]
    first = shards[0]
    def merge(name: str, index: str):
        values = torch.cat([x[name] for x in shards])
        indices = torch.cat([x[index] for x in shards]).long()
        order = torch.argsort(indices)
        return values[order], indices[order]
    q, qi = merge("q_selected", "query_indices")
    k, ki = merge("k_local", "key_indices")
    v, vi = merge("v_local", "key_indices")
    if not torch.equal(ki, vi) or not torch.equal(ki, torch.arange(k.shape[0])):
        raise ValueError("invalid logical K/V layout")
    return {**first, "q": q, "q_indices": qi, "k": k, "v": v}


@torch.inference_mode()
def analyze(payload: dict, radii: list[int], topks: list[int], device: torch.device) -> dict:
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    v = payload["v"].to(device=device, dtype=torch.bfloat16)
    source_cpu = payload["source_positions"].long()
    source = source_cpu.to(device)
    qi = payload["q_indices"].long()
    frames = int(payload["layout"]["num_frames"])
    tpf = int(payload["layout"]["tokens_per_frame"])
    qpf = q.shape[0] // frames
    scale = float(payload["softmax_scale"])
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})
    anchor_ids = torch.tensor(anchors, device=device)
    logical_to_frame = torch.full((k.shape[0],), -1, dtype=torch.long)
    logical_to_frame[source_cpu] = torch.arange(frames).repeat_interleave(tpf)
    q_frames = logical_to_frame[qi].to(device)
    if qpf * frames != q.shape[0] or (q_frames < 0).any():
        raise ValueError("invalid source-query sampling")

    # One low-cost score per (head, query frame, source frame).
    q_frame = q.reshape(frames, qpf, q.shape[1], q.shape[2]).mean(dim=1)  # F,H,D
    k_source = k.index_select(0, source).reshape(frames, tpf, k.shape[1], k.shape[2])
    k_frame = k_source.mean(dim=1)  # F,H,D
    router_score = torch.einsum("fhd,ghd->hfg", q_frame, k_frame).float() * scale

    kh = k.transpose(0, 1).contiguous()
    vh = v.transpose(0, 1).contiguous()
    source_frame_ids = torch.arange(frames, device=device).repeat_interleave(tpf)
    states = {(r, topk): {"se": 0.0, "sd": 0.0, "cos": 0.0, "n": 0,
                           "visible": 0.0, "mass_keep": 0.0, "mass_total": 0.0}
              for r in radii for topk in topks}

    for frame in range(frames):
        start, stop = frame * qpf, (frame + 1) * qpf
        qh = q[start:stop].transpose(0, 1).contiguous()
        logits = torch.bmm(qh, kh.transpose(1, 2)).float() * scale
        dense_w = torch.softmax(logits, dim=-1)
        dense_o = torch.bmm(dense_w.to(vh.dtype), vh).float()
        source_logits = logits.index_select(2, source)
        source_mass = dense_w.index_select(2, source).reshape(
            dense_w.shape[0], dense_w.shape[1], frames, tpf
        ).sum(dim=-1)

        for radius in radii:
            base = torch.zeros((q.shape[1], frames), dtype=torch.bool, device=device)
            base[:, anchor_ids] = True
            base[:, max(0, frame-radius):min(frames, frame+radius+1)] = True
            if frame in anchors:
                base[:] = True
            for topk in topks:
                keep_frame = base.clone()
                if topk and frame not in anchors:
                    scores = router_score[:, frame].masked_fill(keep_frame, float("-inf"))
                    available = (~keep_frame).sum(dim=-1)
                    for count in torch.unique(available).tolist():
                        rows = available == int(count)
                        chosen = min(topk, int(count))
                        if chosen:
                            selected = torch.topk(scores[rows], k=chosen, dim=-1).indices
                            selected_mask = torch.zeros(
                                (int(rows.sum()), frames), dtype=torch.bool, device=device
                            )
                            selected_mask.scatter_(1, selected, True)
                            keep_frame[rows] |= selected_mask
                token_keep = keep_frame.repeat_interleave(tpf, dim=-1)[:, None, :]
                sparse_logits = logits.clone()
                sparse_logits[:, :, source] = source_logits.masked_fill(~token_keep, -torch.inf)
                sparse_w = torch.softmax(sparse_logits, dim=-1)
                sparse_o = torch.bmm(sparse_w.to(vh.dtype), vh).float()
                err = sparse_o - dense_o
                state = states[(radius, topk)]
                state["se"] += float(err.square().sum().cpu())
                state["sd"] += float(dense_o.square().sum().cpu())
                state["cos"] += float(F.cosine_similarity(sparse_o, dense_o, dim=-1).sum().cpu())
                state["n"] += int(sparse_o.shape[0] * sparse_o.shape[1])
                state["visible"] += float(keep_frame.sum().cpu()) * qpf
                state["mass_keep"] += float((source_mass * keep_frame[:, None, :]).sum().cpu())
                state["mass_total"] += float(source_mass.sum().cpu())

    result = {}
    for (radius, topk), state in states.items():
        visible = state["visible"] / state["n"]
        result[f"r{radius}_k{topk}"] = {
            "local_radius": radius,
            "mean_router_remote_topk_frames": topk,
            "relative_l2": math.sqrt(state["se"] / max(state["sd"], 1e-30)),
            "mean_cosine": state["cos"] / state["n"],
            "mean_visible_source_frames": visible,
            "mean_visible_source_frame_fraction": visible / frames,
            "estimated_ss_pair_reduction": 1.0 - visible / frames,
            "retained_dense_source_attention_mass": state["mass_keep"] / max(state["mass_total"], 1e-30),
        }
    return {"layer": int(payload["layer"]), "step": int(payload["step"]),
            "fixed_anchor_frames": anchors, "variants": result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    ap.add_argument("--radii", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--topks", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "schema": "h3_ss_global_anchor_local_mean_router_v1",
        "scope": "Only S-S is sparse; S->T remains dense. Remote frames use mean-pooled real-Q/K frame router, not dense teacher.",
        "layers": [],
    }
    for layer in args.layers:
        payload = load_layer(args.capture_dir, layer)
        report["layers"].append(analyze(payload, args.radii, args.topks, torch.device("cuda:0")))
        del payload
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
