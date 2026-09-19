"""B-preserving T-to-S sparse Softmax for Ref2VA + OpenVDN.

Queries outside the target-video span retain B's original dense route.  Every
target frame retains B's c5/r1 target-target visibility, while its source keys
are restricted to the supplied per-target visible-frame set.  The caller is
responsible for including structural anchors, the corresponding S_i safety
edge, and the dynamic Top-K frames in every route.
"""

from __future__ import annotations

import torch

from .openvdn_npu import OpenVDNLayout, _fusion_attention_tnd, window_bounds


def _frame(layout: OpenVDNLayout, frame: int, device: torch.device) -> torch.Tensor:
    start = layout.video_start + frame * layout.tokens_per_frame
    return torch.arange(
        start, start + layout.tokens_per_frame, device=device, dtype=torch.long
    )


def _outside_target(layout: OpenVDNLayout, device: torch.device) -> torch.Tensor:
    parts = []
    if layout.video_start:
        parts.append(torch.arange(layout.video_start, device=device, dtype=torch.long))
    if layout.video_end < layout.used_len:
        parts.append(
            torch.arange(layout.video_end, layout.used_len, device=device, dtype=torch.long)
        )
    return torch.cat(parts) if parts else torch.empty(0, device=device, dtype=torch.long)


def _target_frames(
    layout: OpenVDNLayout, frames: list[int], device: torch.device
) -> torch.Tensor:
    return torch.cat([_frame(layout, frame, device) for frame in frames])


def _validate_routes(
    source: OpenVDNLayout,
    target: OpenVDNLayout,
    visible_source_frames: list[list[int]],
) -> None:
    source.validate(source.used_len)
    target.validate(target.used_len)
    if source.used_len != target.used_len:
        raise ValueError("source and target layouts must share used_len")
    if source.num_frames != target.num_frames:
        raise ValueError("source and target frame counts differ")
    if source.tokens_per_frame != target.tokens_per_frame:
        raise ValueError("source and target spatial grids differ")
    if not (source.video_end <= target.video_start or target.video_end <= source.video_start):
        raise ValueError("source and target video spans overlap")
    if len(visible_source_frames) != target.num_frames:
        raise ValueError("one source route is required for every target frame")
    structural = set(range(0, source.num_frames, 5)) | {0, source.num_frames - 1}
    for target_frame, visible in enumerate(visible_source_frames):
        values = set(int(frame) for frame in visible)
        if any(frame < 0 or frame >= source.num_frames for frame in values):
            raise ValueError(f"T{target_frame} route contains an out-of-range source frame")
        if target_frame not in values:
            raise ValueError(f"T{target_frame} route omitted mandatory S_i safety edge")
        missing = structural - values
        if missing:
            raise ValueError(f"T{target_frame} route omitted structural anchors: {sorted(missing)}")


def source_target_sparse_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    source_layout: OpenVDNLayout,
    target_layout: OpenVDNLayout,
    visible_source_frames: list[list[int]],
    scale: float,
    groups_per_call: int = 4,
) -> torch.Tensor:
    """Run B's exact Softmax branch with only T-to-S visibility reduced."""
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("T-S sparse attention expects equal TND Q/K/V tensors")
    _validate_routes(source_layout, target_layout, visible_source_frames)
    device = query.device
    used = target_layout.used_len
    if used > query.shape[0]:
        raise ValueError("layout exceeds packed Q/K/V rows")

    out = torch.zeros_like(query)
    global_queries = _outside_target(target_layout, device)
    if global_queries.numel():
        dense_out = _fusion_attention_tnd(
            query.index_select(0, global_queries),
            key[:used],
            value[:used],
            [global_queries.numel()],
            [used],
            scale,
        )
        out.index_copy_(0, global_queries, dense_out)

    bounds = window_bounds(target_layout.num_frames)
    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    outside = _outside_target(target_layout, device)
    source_all = torch.arange(
        source_layout.video_start, source_layout.video_end, device=device, dtype=torch.long
    )
    outside_non_source = outside[
        (outside < source_layout.video_start) | (outside >= source_layout.video_end)
    ]
    first_target = _frame(target_layout, 0, device)
    last_target = _frame(target_layout, target_layout.num_frames - 1, device)
    for target_frame in range(target_layout.num_frames):
        q_idx = _frame(target_layout, target_frame, device)
        if target_frame in (0, target_layout.num_frames - 1):
            target_keys = torch.arange(
                target_layout.video_start,
                target_layout.video_end,
                device=device,
                dtype=torch.long,
            )
        else:
            lo = max(0, bounds[target_frame][0])
            hi = min(target_layout.num_frames - 1, bounds[target_frame][1])
            target_keys = torch.arange(
                target_layout.video_start + lo * target_layout.tokens_per_frame,
                target_layout.video_start + (hi + 1) * target_layout.tokens_per_frame,
                device=device,
                dtype=torch.long,
            )
            pieces = [target_keys]
            if lo > 0:
                pieces.append(first_target)
            if hi < target_layout.num_frames - 1:
                pieces.append(last_target)
            target_keys = torch.cat(pieces)
        source_keys = _target_frames(
            source_layout, sorted(set(visible_source_frames[target_frame])), device
        )
        k_idx = torch.cat((outside_non_source, source_keys, target_keys)).sort().values
        k_idx = k_idx.unique_consecutive()
        groups.append((q_idx, k_idx))

    for start in range(0, len(groups), groups_per_call):
        batch = groups[start : start + groups_per_call]
        q_indices, k_indices = zip(*batch)
        q_pack = torch.cat([query.index_select(0, index) for index in q_indices])
        k_pack = torch.cat([key.index_select(0, index) for index in k_indices])
        v_pack = torch.cat([value.index_select(0, index) for index in k_indices])
        q_cu, kv_cu, q_total, kv_total = [], [], 0, 0
        for q_idx, k_idx in batch:
            q_total += q_idx.numel()
            kv_total += k_idx.numel()
            q_cu.append(q_total)
            kv_cu.append(kv_total)
        group_out = _fusion_attention_tnd(q_pack, k_pack, v_pack, q_cu, kv_cu, scale)
        out.index_copy_(0, torch.cat(q_indices), group_out)
    return out


__all__ = ["source_target_sparse_softmax_attention"]
