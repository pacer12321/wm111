#!/usr/bin/env python3
"""Kernel-only H3 T->S benchmark for the official SpargeAttention API.

This deliberately excludes Ulysses, model loading, anchor-mask construction,
and the rest of the DiT block.  It answers only whether the complete online
Sparge path (coarse scoring + mask + quantization + sparse kernel) beats a
dense Flash-SDPA call at the post-Ulysses H3 T->S shape.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda


def timed_cuda(fn, warmup: int, repeats: int) -> tuple[list[float], torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    assert out is not None
    return samples, out


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-tokens", type=int, default=12052)
    parser.add_argument("--kv-tokens", type=int, default=37296)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=float, nargs="+", default=[0.25, 0.375, 0.5])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    shape_q = (1, args.heads, args.q_tokens, args.head_dim)
    shape_kv = (1, args.heads, args.kv_tokens, args.head_dim)
    q = torch.randn(shape_q, device=device, dtype=dtype)
    k = torch.randn(shape_kv, device=device, dtype=dtype)
    v = torch.randn(shape_kv, device=device, dtype=dtype)
    scale = args.head_dim**-0.5

    def dense_call() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=False)

    torch.cuda.reset_peak_memory_stats(device)
    dense_samples, dense_out = timed_cuda(dense_call, args.warmup, args.repeats)
    dense_stats = summarize(dense_samples)
    dense_peak = int(torch.cuda.max_memory_allocated(device))

    payload: dict[str, object] = {
        "schema": "h3_ts_sparge_kernel_microbench_v1",
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(dtype),
        "shape_q": list(shape_q),
        "shape_kv": list(shape_kv),
        "softmax_scale": scale,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dense": {**dense_stats, "peak_allocated_bytes": dense_peak},
        "sparge": [],
        "scope": (
            "Kernel-only post-Ulysses T->S. Sparge timing includes online coarse "
            "scoring, mask construction, Q/K quantization, and sparse attention; "
            "it excludes Ulysses and the future fixed-anchor mixed mask."
        ),
    }

    dense_f32 = dense_out.float()
    dense_norm = torch.linalg.vector_norm(dense_f32).item()
    for topk in args.topk:
        def sparge_call() -> torch.Tensor:
            return spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                topk=float(topk),
                is_causal=False,
                scale=scale,
                tensor_layout="HND",
                output_dtype=torch.bfloat16,
            )

        torch.cuda.reset_peak_memory_stats(device)
        samples, out = timed_cuda(sparge_call, args.warmup, args.repeats)
        stats = summarize(samples)
        peak = int(torch.cuda.max_memory_allocated(device))
        rel_l2 = torch.linalg.vector_norm(out.float() - dense_f32).item() / max(dense_norm, 1e-12)
        speedup = dense_stats["median_ms"] / stats["median_ms"]
        payload["sparge"].append(
            {
                "requested_topk": float(topk),
                **stats,
                "peak_allocated_bytes": peak,
                "speedup_x_vs_dense_median": speedup,
                "latency_reduction_pct_vs_dense_median": 100.0 * (1.0 - 1.0 / speedup),
                "random_qkv_relative_l2_vs_dense": rel_l2,
                "quality_note": "Random-QKV error is a sanity check, not a video-quality oracle.",
            }
        )
        del out
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
