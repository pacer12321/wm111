#!/usr/bin/env python3
"""Evaluate a step-4 exact S-S frame cache unchanged at later denoising steps.

Only S-S is masked. Every non-source key, including every target key, remains
dense. One remote-frame Top-K set is selected at the reference step for each
layer/head/source-query-frame and then reused without change at all evaluation
steps. Local frames and bidirectional global anchors are always retained.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_layer(step_dir: Path, layer: int) -> dict:
    items = [torch.load(step_dir / f"layer_{layer:02d}_rank_{rank}.pt",
                        map_location="cpu", weights_only=True, mmap=True)
             for rank in (0, 1)]
    first = items[0]
    def merge(name: str, index: str):
        values = torch.cat([x[name] for x in items])
        indices = torch.cat([x[index] for x in items]).long()
        order = torch.argsort(indices)
        return values[order], indices[order]
    q, qi = merge("q_selected", "query_indices")
    k, ki = merge("k_local", "key_indices")
    v, vi = merge("v_local", "key_indices")
    if not torch.equal(ki, vi) or not torch.equal(ki, torch.arange(k.shape[0])):
        raise ValueError("invalid K/V logical layout")
    return {**first, "q": q, "q_indices": qi, "k": k, "v": v}


def metadata(payload: dict) -> tuple[int, int, int, list[int], torch.Tensor]:
    frames = int(payload["layout"]["num_frames"])
    tpf = int(payload["layout"]["tokens_per_frame"])
    qpf = payload["q"].shape[0] // frames
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})
    source_cpu = payload["source_positions"].long()
    logical_to_frame = torch.full((payload["k"].shape[0],), -1, dtype=torch.long)
    logical_to_frame[source_cpu] = torch.arange(frames).repeat_interleave(tpf)
    q_frames = logical_to_frame[payload["q_indices"].long()]
    if qpf * frames != payload["q"].shape[0] or (q_frames < 0).any():
        raise ValueError("invalid source query sample")
    return frames, tpf, qpf, anchors, q_frames


@torch.inference_mode()
def build_reference_cache(payload: dict, radii: list[int], topks: list[int], device: torch.device):
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    source = payload["source_positions"].long().to(device)
    frames, tpf, qpf, anchors, _ = metadata(payload)
    anchor_ids = torch.tensor(anchors, device=device)
    kh = k.transpose(0, 1).contiguous()
    scale = float(payload["softmax_scale"])
    caches = {(r, z): [] for r in radii for z in topks}
    for frame in range(frames):
        qh = q[frame*qpf:(frame+1)*qpf].transpose(0, 1).contiguous()
        logits = torch.bmm(qh, kh.transpose(1, 2)).float() * scale
        dense_w = torch.softmax(logits, dim=-1)
        frame_score = dense_w.index_select(2, source).reshape(
            dense_w.shape[0], dense_w.shape[1], frames, tpf
        ).sum(dim=-1).sum(dim=1)  # H,F
        for radius in radii:
            base = torch.zeros((q.shape[1], frames), dtype=torch.bool, device=device)
            base[:, anchor_ids] = True
            base[:, max(0, frame-radius):min(frames, frame+radius+1)] = True
            if frame in anchors:
                base[:] = True
            for topk in topks:
                keep = base.clone()
                if topk and frame not in anchors:
                    score = frame_score.masked_fill(keep, -torch.inf)
                    available = (~keep).sum(dim=-1)
                    for count in torch.unique(available).tolist():
                        rows = available == int(count)
                        chosen = min(topk, int(count))
                        if chosen:
                            selected = torch.topk(score[rows], k=chosen, dim=-1).indices
                            add = torch.zeros((int(rows.sum()), frames), dtype=torch.bool, device=device)
                            add.scatter_(1, selected, True)
                            keep[rows] |= add
                caches[(radius, topk)].append(keep.cpu())
    return {key: torch.stack(value) for key, value in caches.items()}


@torch.inference_mode()
def evaluate(payload: dict, caches: dict, device: torch.device) -> dict:
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    v = payload["v"].to(device=device, dtype=torch.bfloat16)
    source = payload["source_positions"].long().to(device)
    frames, tpf, qpf, _anchors, _ = metadata(payload)
    kh = k.transpose(0, 1).contiguous()
    vh = v.transpose(0, 1).contiguous()
    scale = float(payload["softmax_scale"])
    states = {key: {"se": 0.0, "sd": 0.0, "cos": 0.0, "n": 0,
                    "mass_keep": 0.0, "mass_total": 0.0}
              for key in caches}
    for frame in range(frames):
        qh = q[frame*qpf:(frame+1)*qpf].transpose(0, 1).contiguous()
        logits = torch.bmm(qh, kh.transpose(1, 2)).float() * scale
        dense_w = torch.softmax(logits, dim=-1)
        dense_o = torch.bmm(dense_w.to(vh.dtype), vh).float()
        source_logits = logits.index_select(2, source)
        source_mass = dense_w.index_select(2, source).reshape(
            dense_w.shape[0], dense_w.shape[1], frames, tpf
        ).sum(dim=-1)
        for key, cache in caches.items():
            keep = cache[frame].to(device)
            token_keep = keep.repeat_interleave(tpf, dim=-1)[:, None, :]
            sparse_logits = logits.clone()
            sparse_logits[:, :, source] = source_logits.masked_fill(~token_keep, -torch.inf)
            sparse_w = torch.softmax(sparse_logits, dim=-1)
            sparse_o = torch.bmm(sparse_w.to(vh.dtype), vh).float()
            err = sparse_o - dense_o
            state = states[key]
            state["se"] += float(err.square().sum().cpu())
            state["sd"] += float(dense_o.square().sum().cpu())
            state["cos"] += float(F.cosine_similarity(sparse_o, dense_o, dim=-1).sum().cpu())
            state["n"] += int(sparse_o.shape[0] * sparse_o.shape[1])
            state["mass_keep"] += float((source_mass * keep[:, None, :]).sum().cpu())
            state["mass_total"] += float(source_mass.sum().cpu())
    result = {}
    for (radius, topk), state in states.items():
        visible = float(caches[(radius, topk)].float().sum()) / (frames * q.shape[1])
        result[f"r{radius}_k{topk}"] = {
            "local_radius": radius,
            "cached_remote_topk_frames": topk,
            "relative_l2": math.sqrt(state["se"] / max(state["sd"], 1e-30)),
            "mean_cosine": state["cos"] / state["n"],
            "mean_visible_source_frames": visible,
            "mean_visible_source_frame_fraction": visible / frames,
            "estimated_ss_pair_reduction": 1.0 - visible / frames,
            "retained_dense_source_attention_mass": state["mass_keep"] / max(state["mass_total"], 1e-30),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", type=Path, required=True)
    ap.add_argument("--reference-step", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, nargs="+", default=[4, 25, 45])
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    ap.add_argument("--radii", type=int, nargs="+", default=[3])
    ap.add_argument("--topks", type=int, nargs="+", default=[6, 8, 12])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    report = {
        "schema": "h3_ss_exact_step4_cached_topk_cross_step_v1",
        "scope": "Only S-S is sparse and S->T stays dense. Exact step-4 frame/head Top-K cache is reused unchanged at later steps.",
        "reference_step": args.reference_step,
        "layers": [],
    }
    for layer in args.layers:
        reference = load_layer(args.capture_root / f"step_{args.reference_step:02d}", layer)
        caches = build_reference_cache(reference, args.radii, args.topks, device)
        layer_result = {"layer": layer, "steps": {}}
        for step in args.eval_steps:
            payload = load_layer(args.capture_root / f"step_{step:02d}", layer)
            if not torch.equal(payload["q_indices"], reference["q_indices"]):
                raise ValueError(f"query logical indices changed at step {step}")
            if not torch.equal(payload["source_positions"], reference["source_positions"]):
                raise ValueError(f"source logical positions changed at step {step}")
            layer_result["steps"][str(step)] = evaluate(payload, caches, device)
            del payload
            torch.cuda.empty_cache()
        report["layers"].append(layer_result)
        del reference, caches
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
