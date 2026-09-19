from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_layer(capture_dir: Path, layer: int) -> dict:
    shards = [
        torch.load(capture_dir / f"layer_{layer:02d}_rank_{rank}.pt", map_location="cpu")
        for rank in (0, 1)
    ]
    first = shards[0]
    for shard in shards[1:]:
        for key in ("layer", "step", "softmax_scale", "layout"):
            if shard[key] != first[key]:
                raise ValueError(f"capture mismatch for {key}")

    query_indices = torch.cat([s["query_indices"] for s in shards]).long()
    q = torch.cat([s["q_selected"] for s in shards])
    q_order = torch.argsort(query_indices)
    query_indices = query_indices.index_select(0, q_order)
    q = q.index_select(0, q_order)

    key_indices = torch.cat([s["key_indices"] for s in shards]).long()
    k = torch.cat([s["k_local"] for s in shards])
    v = torch.cat([s["v_local"] for s in shards])
    k_order = torch.argsort(key_indices)
    key_indices = key_indices.index_select(0, k_order)
    k = k.index_select(0, k_order)
    v = v.index_select(0, k_order)
    if not torch.equal(key_indices, torch.arange(int(first["layout"]["used_len"]))):
        raise ValueError("captured K/V rows do not reconstruct the logical used sequence")
    sampled_source_positions = torch.sort(first["all_query_indices"].long()).values
    if not torch.equal(query_indices, sampled_source_positions):
        missing = sampled_source_positions[
            ~torch.isin(sampled_source_positions, query_indices)
        ]
        extra = query_indices[
            ~torch.isin(query_indices, sampled_source_positions)
        ]
        raise ValueError(
            "captured queries are not exactly the requested source sample: "
            f"queries={query_indices.numel()} sample={sampled_source_positions.numel()} "
            f"missing={missing.numel()} extra={extra.numel()}"
        )
    return {**first, "q": q, "k": k, "v": v, "query_indices": query_indices}


@torch.inference_mode()
def compare(payload: dict, device: torch.device, chunk: int) -> dict:
    q = payload["q"].to(device)
    k = payload["k"].to(device)
    v = payload["v"].to(device)
    target_positions = payload["target_positions"].long()
    keep = torch.ones(k.shape[0], dtype=torch.bool)
    keep[target_positions] = False
    keep = torch.nonzero(keep, as_tuple=False).view(-1).to(device)
    k_without_target = k.index_select(0, keep)
    v_without_target = v.index_select(0, keep)

    dense_norm_sq = 0.0
    error_norm_sq = 0.0
    cosine_sum = 0.0
    cosine_count = 0
    per_query_relative: list[torch.Tensor] = []
    for start in range(0, q.shape[0], chunk):
        q_chunk = q[start : start + chunk].transpose(0, 1).unsqueeze(0)
        dense = F.scaled_dot_product_attention(
            q_chunk,
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=float(payload["softmax_scale"]),
        ).squeeze(0).transpose(0, 1).float()
        masked = F.scaled_dot_product_attention(
            q_chunk,
            k_without_target.transpose(0, 1).unsqueeze(0),
            v_without_target.transpose(0, 1).unsqueeze(0),
            scale=float(payload["softmax_scale"]),
        ).squeeze(0).transpose(0, 1).float()
        error = masked - dense
        dense_norm_sq += float(dense.square().sum().cpu())
        error_norm_sq += float(error.square().sum().cpu())
        dense_flat = dense.flatten(1)
        masked_flat = masked.flatten(1)
        cosine_sum += float(F.cosine_similarity(dense_flat, masked_flat, dim=1).sum().cpu())
        cosine_count += dense.shape[0]
        per_query_relative.append(
            (error_flat_norm := error.flatten(1).norm(dim=1))
            .div(dense_flat.norm(dim=1).clamp_min(1e-12))
            .cpu()
        )
        del dense, masked, error, q_chunk, dense_flat, masked_flat, error_flat_norm

    relative = torch.cat(per_query_relative)
    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "source_queries": int(q.shape[0]),
        "all_keys": int(k.shape[0]),
        "keys_after_dropping_target": int(k_without_target.shape[0]),
        "global_relative_l2": (error_norm_sq / max(dense_norm_sq, 1e-30)) ** 0.5,
        "mean_query_relative_l2": float(relative.mean()),
        "p50_query_relative_l2": float(relative.quantile(0.50)),
        "p95_query_relative_l2": float(relative.quantile(0.95)),
        "mean_query_cosine": cosine_sum / max(cosine_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    parser.add_argument("--chunk", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-size attention comparison")
    results = []
    for layer in args.layers:
        payload = load_layer(args.capture_dir, layer)
        results.append(compare(payload, torch.device("cuda:0"), args.chunk))
        torch.cuda.empty_cache()
    report = {
        "definition": "source-query attention output: dense all-K/V vs target K/V removed",
        "capture_dir": str(args.capture_dir),
        "layers": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
