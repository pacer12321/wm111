from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_official_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("liveditor_triton_kernels", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SparsePiecewiseAttn1stIntervalsFunction


def _time_ms(fn, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2], min(samples), max(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_kernel", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-tokens", type=int, default=37296)
    parser.add_argument("--kv-tokens", type=int, default=81165)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--sparsities", default="0.5,0.75")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    sparse_function = _load_official_kernel(args.official_kernel)
    block = args.block_size
    tq = math.ceil(args.query_tokens / block) * block
    tkv = math.ceil(args.kv_tokens / block) * block
    query_blocks = tq // block
    dense_blocks = math.ceil(query_blocks / 2)
    dense_q = dense_blocks * block
    sparse_q = tq - dense_q
    device = torch.device("cuda")
    torch.manual_seed(17)
    q = torch.randn(
        1, args.heads, tq, args.head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        1, args.heads, tkv, args.head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    q_dense = q[:, :, :dense_q].contiguous()
    q_sparse = q[:, :, dense_q:].contiguous()

    def dense_full():
        return F.scaled_dot_product_attention(q, k, v)

    dense_median, dense_min, dense_max = _time_ms(
        dense_full, args.warmup, args.repeats
    )
    results = []
    for sparsity in [float(item) for item in args.sparsities.split(",")]:
        def sparse_half():
            return sparse_function.apply(q_sparse, k, v, sparsity, block)

        def split_path():
            dense_out = F.scaled_dot_product_attention(q_dense, k, v)
            sparse_out = sparse_function.apply(q_sparse, k, v, sparsity, block)
            return torch.cat((dense_out, sparse_out), dim=2)

        sparse_median, sparse_min, sparse_max = _time_ms(
            sparse_half, args.warmup, args.repeats
        )
        split_median, split_min, split_max = _time_ms(
            split_path, args.warmup, args.repeats
        )
        results.append(
            {
                "sparsity_exact_block_fraction": sparsity,
                "sparse_half_query_ms": {
                    "median": sparse_median,
                    "min": sparse_min,
                    "max": sparse_max,
                },
                "dense_half_plus_sparse_half_ms": {
                    "median": split_median,
                    "min": split_min,
                    "max": split_max,
                },
                "speedup_vs_dense_full": dense_median / split_median,
                "latency_reduction_vs_dense_full": 1.0 - split_median / dense_median,
            }
        )

    # A small all-exact correctness check exercises the imported official kernel
    # without allocating another H3-sized dense reference.
    qs = torch.randn(1, 2, 128, 128, device=device, dtype=torch.bfloat16)
    ks = torch.randn(1, 2, 256, 128, device=device, dtype=torch.bfloat16)
    vs = torch.randn_like(ks)
    dense_small = F.scaled_dot_product_attention(qs, ks, vs).float()
    exact_small = sparse_function.apply(qs, ks, vs, 1.0, block).float()
    exact_relative_l2 = float(
        torch.linalg.vector_norm(exact_small - dense_small)
        / torch.linalg.vector_norm(dense_small).clamp_min(1e-8)
    )

    report = {
        "schema": "h3_liveditor_official_kernel_shape_benchmark_v1",
        "scope": (
            "Kernel feasibility/upper-bound benchmark only. It includes official "
            "ISA reductions, Top-K and Taylor kernel plus a dense/sparse query split, "
            "but not H3 branch-mandatory exact masking, Ulysses, or model integration."
        ),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "shape": {
            "batch": 1,
            "heads_per_ulysses_rank": args.heads,
            "head_dim": args.head_dim,
            "query_tokens_requested": args.query_tokens,
            "query_tokens_padded": tq,
            "kv_tokens_requested": args.kv_tokens,
            "kv_tokens_padded": tkv,
            "dense_query_tokens": dense_q,
            "sparse_query_tokens": sparse_q,
            "block_size": block,
        },
        "dense_full_ms": {
            "median": dense_median,
            "min": dense_min,
            "max": dense_max,
        },
        "official_kernel_all_exact_small_relative_l2": exact_relative_l2,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
