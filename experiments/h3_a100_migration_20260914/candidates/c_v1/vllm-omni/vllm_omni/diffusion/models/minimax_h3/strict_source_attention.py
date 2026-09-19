"""Torch/NPU adapter for isolated C; uses B's unchanged fusion kernel.

Correctness-first candidate: full layout validation each forward may synchronize
CPU metadata. Do not interpret its toy tests as full-model speed measurements.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .openvdn_npu import _fusion_attention_tnd
from .strict_source_layout import infer_layout, strict_plan

_DEVICE_PLAN_CACHE = OrderedDict()


def _rows(value, name):
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"C {name} must contain integer row indices")
    return value.detach().reshape(-1).to(device="cpu").tolist()


def infer_strict_source_layout(*, img_pos, update_mask, text_pos, audio_pos,
                               img_position_ids, cu_seqlens, metadata):
    # Called before RoPE, token refiner, sp_prepare or any attention collective.
    # Full metadata is replicated by the existing pipeline on every SP worker.
    # This is not a protocol for recovering arbitrarily divergent worker inputs.
    if update_mask.dtype != torch.bool:
        raise ValueError("C requires a boolean visual update_mask")
    shape = tuple(img_position_ids.shape)
    if len(shape) not in (2, 3) or shape[-1] != 3 or (len(shape) == 3 and shape[0] != 1):
        raise ValueError("C requires complete global coordinates [S,3] or [1,S,3]")
    return infer_layout(
        img_pos=_rows(img_pos, "img_pos"),
        update_mask=update_mask.detach().reshape(-1).to(device="cpu").tolist(),
        text_pos=_rows(text_pos, "text_pos"), audio_pos=_rows(audio_pos, "audio_pos"),
        coords=img_position_ids.detach().reshape(-1, 3).to(device="cpu", dtype=torch.float64).tolist(),
        cu_seqlens=_rows(cu_seqlens, "cu_seqlens"), metadata=metadata,
    )


def _indices(spans, device):
    parts = [torch.arange(lo, hi, dtype=torch.long, device=device) for lo, hi in spans if lo < hi]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)


def _device_plan(layout, device):
    # Independent namespace: B cache is never touched. Source geometry/mode are
    # part of the frozen layout, not inferred from target shapes alone.
    cache_key = (layout, str(device))
    if cache_key in _DEVICE_PLAN_CACHE:
        _DEVICE_PLAN_CACHE.move_to_end(cache_key)
        return _DEVICE_PLAN_CACHE[cache_key]
    dense_spans, frame_groups = strict_plan(layout)
    plan = (_indices(dense_spans, device), tuple(
        (_indices((group.query_span,), device), _indices(group.key_spans, device))
        for group in frame_groups))
    _DEVICE_PLAN_CACHE[cache_key] = plan
    if len(_DEVICE_PLAN_CACHE) > 8:
        _DEVICE_PLAN_CACHE.popitem(last=False)
    return plan


def strict_source_softmax_attention(query, key, value, layout, scale, groups_per_call=4):
    if type(groups_per_call) is not int or groups_per_call < 1:
        raise ValueError("C groups_per_call must be positive")
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("C expects equal global TND Q/K/V tensors after Ulysses exchange")
    if query.device != key.device or key.device != value.device or query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("C Q/K/V devices and dtypes must match")
    layout.validate(query.shape[0])
    dense_queries, groups = _device_plan(layout, query.device)
    out = torch.zeros_like(query)
    if dense_queries.numel():
        dense = _fusion_attention_tnd(query.index_select(0, dense_queries),
                                      key[:layout.used_len], value[:layout.used_len],
                                      [dense_queries.numel()], [layout.used_len], scale)
        out.index_copy_(0, dense_queries, dense)
    for start in range(0, len(groups), groups_per_call):
        batch = groups[start:start + groups_per_call]
        q_indices, k_indices = zip(*batch)
        q_pack = torch.cat([query.index_select(0, idx) for idx in q_indices])
        k_pack = torch.cat([key.index_select(0, idx) for idx in k_indices])
        v_pack = torch.cat([value.index_select(0, idx) for idx in k_indices])
        q_cu, kv_cu = [], []
        q_total = kv_total = 0
        for q_idx, k_idx in batch:
            q_total += q_idx.numel()
            kv_total += k_idx.numel()
            q_cu.append(q_total)
            kv_cu.append(kv_total)
        result = _fusion_attention_tnd(q_pack, k_pack, v_pack, q_cu, kv_cu, scale)
        out.index_copy_(0, torch.cat(q_indices), result)
    return out
