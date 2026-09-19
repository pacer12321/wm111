from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_layer(step_dir: Path, layer: int) -> dict:
    shards = [
        torch.load(
            step_dir / f"layer_{layer:02d}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
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
    if not torch.equal(k_indices, v_indices):
        raise ValueError("K/V logical indices differ")
    if not torch.equal(k_indices, torch.arange(int(first["layout"]["used_len"]))):
        raise ValueError("K/V do not reconstruct the logical used sequence")
    expected_q = torch.sort(first["all_query_indices"].long()).values
    if not torch.equal(q_indices, expected_q):
        raise ValueError("source queries do not match the requested spatial sample")
    return {**first, "q": q, "k": k, "v": v}


def accumulate_error(
    totals: dict,
    label: str,
    dense: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    error = (candidate - dense).float()
    dense_float = dense.float()
    state = totals.setdefault(
        label,
        {"dense_sq": 0.0, "error_sq": 0.0, "cos_sum": 0.0, "rel": []},
    )
    state["dense_sq"] += float(dense_float.square().sum().cpu())
    state["error_sq"] += float(error.square().sum().cpu())
    dense_flat = dense_float.flatten(1)
    candidate_flat = candidate.float().flatten(1)
    state["cos_sum"] += float(
        F.cosine_similarity(dense_flat, candidate_flat, dim=1).sum().cpu()
    )
    state["rel"].append(
        error.flatten(1)
        .norm(dim=1)
        .div(dense_flat.norm(dim=1).clamp_min(1e-12))
        .cpu()
    )


@torch.inference_mode()
def analyze_layer(payload: dict, device: torch.device, chunk: int, ks: list[int]) -> dict:
    q = payload["q"].to(device)
    k = payload["k"].to(device)
    v = payload["v"].to(device)
    scale = float(payload["softmax_scale"])
    source_positions = payload["source_positions"].long().to(device)
    target_positions = payload["target_positions"].long().to(device)
    target_times_per_token = payload["target_coords"][:, 0].long()
    target_times = torch.unique(target_times_per_token, sorted=True)
    target_frames = [
        target_positions[target_times_per_token.to(device) == time.item()]
        for time in target_times
    ]
    used = k.shape[0]
    source_mask = torch.zeros(used, dtype=torch.bool, device=device)
    target_mask = torch.zeros_like(source_mask)
    source_mask[source_positions] = True
    target_mask[target_positions] = True
    other_positions = torch.nonzero(~source_mask & ~target_mask, as_tuple=False).view(-1)

    frame_mass = torch.zeros(len(target_frames), dtype=torch.float64)
    ss_mass = other_mass = st_mass = 0.0
    normalizer = 0
    k_heads = k.transpose(0, 1)
    for start in range(0, q.shape[0], chunk):
        q_chunk = q[start : start + chunk].transpose(0, 1)
        logits = torch.bmm(q_chunk, k_heads.transpose(1, 2)) * scale
        probabilities = torch.softmax(logits.float(), dim=-1)
        count = probabilities.shape[0] * probabilities.shape[1]
        normalizer += count
        ss_mass += float(probabilities.index_select(2, source_positions).sum().cpu())
        st_mass += float(probabilities.index_select(2, target_positions).sum().cpu())
        other_mass += float(probabilities.index_select(2, other_positions).sum().cpu())
        for frame, positions in enumerate(target_frames):
            frame_mass[frame] += float(probabilities.index_select(2, positions).sum().cpu())
        del logits, probabilities
    ss_mass /= normalizer
    st_mass /= normalizer
    other_mass /= normalizer
    frame_share = frame_mass / frame_mass.sum().clamp_min(1e-30)
    order = torch.argsort(frame_share, descending=True)

    base_positions = torch.cat((source_positions, other_positions)).sort().values
    allowed: dict[int, torch.Tensor] = {}
    for topk in ks:
        selected_target = torch.cat(
            [target_frames[int(frame)] for frame in order[:topk]]
        )
        allowed[topk] = torch.cat((base_positions, selected_target)).sort().values

    totals: dict = {}
    for start in range(0, q.shape[0], chunk):
        q_chunk = q[start : start + chunk].transpose(0, 1).unsqueeze(0)
        dense = F.scaled_dot_product_attention(
            q_chunk,
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=scale,
        ).squeeze(0).transpose(0, 1)
        for topk in ks:
            indices = allowed[topk]
            candidate = F.scaled_dot_product_attention(
                q_chunk,
                k.index_select(0, indices).transpose(0, 1).unsqueeze(0),
                v.index_select(0, indices).transpose(0, 1).unsqueeze(0),
                scale=scale,
            ).squeeze(0).transpose(0, 1)
            accumulate_error(totals, str(topk), dense, candidate)
        del dense, q_chunk

    variants = {}
    for label, state in totals.items():
        relative = torch.cat(state["rel"])
        variants[label] = {
            "global_relative_l2": (
                state["error_sq"] / max(state["dense_sq"], 1e-30)
            ) ** 0.5,
            "mean_query_relative_l2": float(relative.mean()),
            "p95_query_relative_l2": float(relative.quantile(0.95)),
            "mean_query_cosine": state["cos_sum"] / q.shape[0],
            "target_mass_coverage": float(
                frame_share.index_select(0, order[: int(label)]).sum()
            ),
            "anchors": [int(frame) for frame in order[: int(label)]],
            "keys_per_source_query": int(allowed[int(label)].numel()),
        }
    return {
        "layer": int(payload["layer"]),
        "step": int(payload["step"]),
        "sampled_source_queries": int(q.shape[0]),
        "attention_mass": {
            "S_to_S": ss_mass,
            "S_to_T": st_mass,
            "S_to_other": other_mass,
            "sum": ss_mass + st_mass + other_mass,
        },
        "target_frame_order": [int(frame) for frame in order],
        "target_frame_share": [float(value) for value in frame_share],
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 24, 41])
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 8, 12, 16])
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    report = {
        "definition": (
            "Oracle fixed target-frame anchors for S->T; all S and non-target "
            "condition keys remain visible"
        ),
        "layers": [],
    }
    for layer in args.layers:
        payload = load_layer(args.step_dir, layer)
        report["layers"].append(
            analyze_layer(payload, torch.device("cuda:0"), args.chunk, args.ks)
        )
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
