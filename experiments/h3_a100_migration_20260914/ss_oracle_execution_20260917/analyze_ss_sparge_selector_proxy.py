#!/usr/bin/env python3
"""Fast real-QKV proxy for Sparge S-S block selection quality.

The capture contains 16 spatial source queries per frame.  They are grouped as
one query block per frame, while source keys use Sparge's A100 64-token blocks.
All non-source keys stay dense.  Fixed five-frame anchors and the corresponding
source frame are forced visible in addition to Sparge's coarse Top-K mask.
Quantized kernel error is excluded; this isolates the actual coarse selector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from spas_sage_attn.utils import get_block_map_meansim


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
def analyze(payload: dict, topks: list[float], device: torch.device) -> dict:
    q = payload["q"].to(device=device, dtype=torch.bfloat16)
    k = payload["k"].to(device=device, dtype=torch.bfloat16)
    v = payload["v"].to(device=device, dtype=torch.bfloat16)
    source_pos_cpu = payload["source_positions"].long()
    source_pos = source_pos_cpu.to(device)
    q_indices = payload["q_indices"].long()
    frames = int(payload["layout"]["num_frames"])
    tpf = int(payload["layout"]["tokens_per_frame"])
    used = k.shape[0]
    anchors = sorted(set(range(0, frames, 5)) | {frames - 1})
    q_per_frame = q.shape[0] // frames
    if q_per_frame * frames != q.shape[0]:
        raise ValueError("sampled source queries are not frame-balanced")

    source_mask = torch.zeros(used, dtype=torch.bool)
    source_mask[source_pos_cpu] = True
    non_source = torch.nonzero(~source_mask, as_tuple=False).view(-1).to(device)
    logical_to_frame = torch.full((used,), -1, dtype=torch.long)
    logical_to_frame[source_pos_cpu] = torch.arange(frames).repeat_interleave(tpf)
    q_frames = logical_to_frame[q_indices].to(device)
    if not torch.equal(q_frames, torch.arange(frames, device=device).repeat_interleave(q_per_frame)):
        raise ValueError("query ordering is not frame-major")

    qh = q.transpose(0, 1).unsqueeze(0).contiguous()  # B,H,Q,D
    ks = k.index_select(0, source_pos).transpose(0, 1).unsqueeze(0).contiguous()
    scale = float(payload["softmax_scale"])
    dense_o_cpu = torch.empty(q.shape, dtype=torch.float32, device="cpu")
    for start in range(0, q.shape[0], q_per_frame):
        stop = min(start + q_per_frame, q.shape[0])
        logits = torch.einsum("qhd,khd->qhk", q[start:stop], k).float() * scale
        dense_w = torch.softmax(logits, dim=-1)
        dense_o_cpu[start:stop] = torch.einsum(
            "qhk,khd->qhd", dense_w.to(v.dtype), v
        ).float().cpu()
        del logits, dense_w
    results = {}

    mandatory_token = torch.zeros((q.shape[0], source_pos.numel()), dtype=torch.bool, device=device)
    anchor_frames = torch.tensor(anchors, device=device)
    source_frame_ids = torch.arange(frames, device=device).repeat_interleave(tpf)
    mandatory_token |= torch.isin(source_frame_ids, anchor_frames).view(1, -1)
    mandatory_token |= source_frame_ids.view(1, -1) == q_frames.view(-1, 1)

    for topk in topks:
        block_map = get_block_map_meansim(
            qh, ks, is_causal=False, BLKQ=q_per_frame, BLKK=64,
            simthreshd1=-0.1, cdfthreshd=None, topk=float(topk),
            return_lut=False,
        )  # B,H,F,Kblocks
        sq_error = sq_dense = cosine_sum = visible_sum = 0.0
        count = 0
        for start in range(0, q.shape[0], q_per_frame):
            stop = min(start + q_per_frame, q.shape[0])
            query_frame_ids = q_frames[start:stop]
            frame_block_map = block_map[0].index_select(1, query_frame_ids)  # H,Q,Kblocks
            token_map = frame_block_map.repeat_interleave(64, dim=2)[..., :source_pos.numel()]
            token_keep = token_map.permute(1, 0, 2) | mandatory_token[start:stop, None, :]
            logits = torch.einsum("qhd,khd->qhk", q[start:stop], k).float() * scale
            sparse_logits = logits.clone()
            sparse_source_logits = logits.index_select(2, source_pos).masked_fill(
                ~token_keep, -torch.inf
            )
            sparse_logits[:, :, source_pos] = sparse_source_logits
            sparse_w = torch.softmax(sparse_logits, dim=-1)
            sparse_o = torch.einsum("qhk,khd->qhd", sparse_w.to(v.dtype), v).float()
            dense_o = dense_o_cpu[start:stop].to(device)
            error = sparse_o - dense_o
            sq_error += float(error.square().sum().cpu())
            sq_dense += float(dense_o.square().sum().cpu())
            cosine_sum += float(F.cosine_similarity(sparse_o, dense_o, dim=-1).sum().cpu())
            count += int(sparse_o.shape[0] * sparse_o.shape[1])
            visible_sum += float(token_keep.float().mean(dim=-1).sum().cpu())
            del logits, sparse_logits, sparse_source_logits, sparse_w, sparse_o, dense_o
        rel_l2 = math.sqrt(sq_error / max(sq_dense, 1e-30))
        cosine = cosine_sum / count
        actual_density = visible_sum / count
        results[str(topk)] = {
            "requested_sparge_topk": topk,
            "actual_source_token_density_after_forced_edges": actual_density,
            "relative_l2": rel_l2,
            "mean_cosine": cosine,
            "query_block_size_proxy": q_per_frame,
            "key_block_size": 64,
        }
    return {"layer": int(payload["layer"]), "variants": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 24, 41])
    ap.add_argument("--topks", nargs="+", type=float, default=[0.75, 0.625, 0.5])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "schema": "h3_ss_real_sparge_selector_proxy_v1",
        "scope": (
            "Real H3 Q/K/V. Sparge mean-similarity key-block selector with one sampled "
            "query block per source frame; fixed anchors and corresponding frame forced. "
            "This excludes int8 kernel numerical error and is not yet end-to-end video."
        ),
        "layers": [],
    }
    for layer in args.layers:
        payload = load_layer(args.capture_dir, layer)
        report["layers"].append(analyze(payload, args.topks, torch.device("cuda:0")))
        del payload
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
