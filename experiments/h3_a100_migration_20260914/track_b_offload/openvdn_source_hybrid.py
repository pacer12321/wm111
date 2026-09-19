"""Two-video OpenVDN routing for the clean S-S ablation.

Only within-source long-range Softmax edges are replaced by the released
OpenVDN local-window/anchor policy.  All non-S keys visible to S in B remain
visible, and T keeps B's exact policy: local T plus all source/auxiliary keys.
The complementary long-range S-S path is supplied by the caller's existing
OpenVDN linear branch.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from .openvdn_npu import OpenVDNLayout, _fusion_attention_tnd, window_bounds


def _frame(layout: OpenVDNLayout, first: int, stop: int, device: torch.device) -> torch.Tensor:
    lo = layout.video_start + first * layout.tokens_per_frame
    hi = layout.video_start + stop * layout.tokens_per_frame
    return torch.arange(lo, hi, device=device, dtype=torch.long)


def _outside(layout: OpenVDNLayout, device: torch.device) -> torch.Tensor:
    parts = []
    if layout.video_start:
        parts.append(torch.arange(layout.video_start, device=device, dtype=torch.long))
    if layout.video_end < layout.used_len:
        parts.append(torch.arange(layout.video_end, layout.used_len, device=device, dtype=torch.long))
    return torch.cat(parts) if parts else torch.empty(0, device=device, dtype=torch.long)


def _interior_groups(
    layout: OpenVDNLayout,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    outside = _outside(layout, device)
    first_anchor = _frame(layout, 0, 1, device)
    last_anchor = _frame(layout, layout.num_frames - 1, layout.num_frames, device)
    bounds = window_bounds(layout.num_frames)
    grouped: dict[tuple[int, int], list[int]] = {}
    for frame_index in range(1, layout.num_frames - 1):
        lo = max(0, bounds[frame_index][0])
        hi = min(layout.num_frames - 1, bounds[frame_index][1])
        grouped.setdefault((lo, hi), []).append(frame_index)
    result = []
    for (lo, hi), frame_indices in grouped.items():
        q_idx = torch.cat([_frame(layout, index, index + 1, device) for index in frame_indices])
        pieces = [outside, _frame(layout, lo, hi + 1, device)]
        if lo > 0:
            pieces.append(first_anchor)
        if hi < layout.num_frames - 1:
            pieces.append(last_anchor)
        k_idx = torch.cat(pieces).sort().values.unique_consecutive()
        result.append((q_idx, k_idx))
    return tuple(result)


@lru_cache(maxsize=16)
def _plan(
    source: OpenVDNLayout,
    target: OpenVDNLayout,
    device_string: str,
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    source.validate(source.used_len)
    target.validate(target.used_len)
    if source.used_len != target.used_len:
        raise ValueError("source and target layouts must cover the same packed request")
    if not (source.video_end <= target.video_start or target.video_end <= source.video_start):
        raise ValueError("source and target video spans overlap")
    device = torch.device(device_string)
    dense = torch.ones(source.used_len, device=device, dtype=torch.bool)
    # Only visual interior-frame queries become sparse.  Both streams' endpoint
    # frames, text/audio rows, and any other real rows retain B's dense route.
    for layout in (source, target):
        lo = layout.video_start + layout.tokens_per_frame
        hi = layout.video_end - layout.tokens_per_frame
        dense[lo:hi] = False
    dense_q = torch.nonzero(dense, as_tuple=False).view(-1)
    return dense_q, _interior_groups(source, device) + _interior_groups(target, device)


def source_target_hybrid_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    source_layout: OpenVDNLayout,
    target_layout: OpenVDNLayout,
    scale: float,
    groups_per_call: int = 4,
) -> torch.Tensor:
    """Apply local Softmax inside S and T while preserving all cross keys."""
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("S-S hybrid attention expects equal TND Q/K/V tensors")
    if source_layout.used_len > query.shape[0] or target_layout.used_len > query.shape[0]:
        raise ValueError("layout exceeds packed Q/K/V rows")
    dense_q, groups = _plan(source_layout, target_layout, str(query.device))
    out = torch.zeros_like(query)
    dense_out = _fusion_attention_tnd(
        query.index_select(0, dense_q),
        key[: source_layout.used_len],
        value[: source_layout.used_len],
        [dense_q.numel()],
        [source_layout.used_len],
        scale,
    )
    out.index_copy_(0, dense_q, dense_out)
    for start in range(0, len(groups), groups_per_call):
        batch = groups[start : start + groups_per_call]
        q_indices, k_indices = zip(*batch)
        q_pack = torch.cat([query.index_select(0, idx) for idx in q_indices])
        k_pack = torch.cat([key.index_select(0, idx) for idx in k_indices])
        v_pack = torch.cat([value.index_select(0, idx) for idx in k_indices])
        q_cu, kv_cu, q_total, kv_total = [], [], 0, 0
        for q_idx, k_idx in batch:
            q_total += q_idx.numel()
            kv_total += k_idx.numel()
            q_cu.append(q_total)
            kv_cu.append(kv_total)
        group_out = _fusion_attention_tnd(q_pack, k_pack, v_pack, q_cu, kv_cu, scale)
        out.index_copy_(0, torch.cat(q_indices), group_out)
    return out


__all__ = ["source_target_hybrid_softmax_attention"]
