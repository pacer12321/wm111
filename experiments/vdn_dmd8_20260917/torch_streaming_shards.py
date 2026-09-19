"""Unintegrated torch CPU counterpart of the shard-at-read prototype.

No distributed/GPU calls. LoRA batching is aligned to the original 256-row
merge chunks, even when rank ownership starts/ends in the middle of a row.
Tests require torch and must be run before use; no GPU equivalence implied.
"""
from dataclasses import replace
import math

import torch

from streaming_shards import _validate_plan, read_model_interval

TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def read_torch(ref, start, stop):
    raw = ref.reader.read_flat(ref.key, start, stop)
    tensor = torch.from_numpy(raw)
    if ref.info["dtype"] == "BF16":
        tensor = tensor.view(torch.bfloat16)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Non-finite checkpoint values")
    return tensor


def merge_owned_interval_(target, *, tensor_start, shape, loras, chunk_rows=256):
    """Merge into a flat owned tensor interval, matching original chunk grid.

    Target is already in model (not checkpoint grouped-QKV) layout. Only
    bounded FP32 deltas and the LoRA A matrix exist in addition to the shard.
    """
    if target.device.type != "cpu" or target.dtype != torch.bfloat16:
        raise ValueError("Released Stage-B merge requires CPU BF16 target")
    if target.ndim != 1 or len(shape) != 2 or chunk_rows <= 0:
        raise ValueError("Invalid merge target shape/chunk size")
    rows, columns = shape
    tensor_stop = tensor_start + target.numel()
    if not 0 <= tensor_start <= tensor_stop <= rows * columns:
        raise ValueError("Owned interval out of bounds")
    # These tensors become persistent Parameter/buffer storage. They must be
    # ordinary tensors with a version counter even when the caller is under
    # inference_mode; F.linear consults that counter outside inference mode.
    with torch.inference_mode(False), torch.no_grad():
        for lora in loras:
            lo, hi = max(tensor_start, lora.start_row * columns), min(tensor_stop, lora.end_row * columns)
            if lo >= hi:
                continue
            rank = lora.a.info["shape"][0]
            a = read_torch(lora.a, 0, rank * columns).reshape(rank, columns).float()
            first_row = lo // columns - lora.start_row
            last_row = (hi - 1) // columns - lora.start_row
            first_chunk = (first_row // chunk_rows) * chunk_rows
            for start in range(first_chunk, last_row + 1, chunk_rows):
                stop = min(start + chunk_rows, lora.end_row - lora.start_row)
                source_start = lora.b_row_offset + start
                source_stop = lora.b_row_offset + stop
                b = read_torch(lora.b, source_start * rank, source_stop * rank).reshape(stop - start, rank).float()
                # Same GEMM dimensions/grid as the production merge_lora_pair_.
                # Released alpha/rank=1; multiplication kept for lifecycle parity.
                delta = (b @ a) * 1.0
                if not bool(torch.isfinite(delta).all()):
                    raise ValueError("Non-finite LoRA delta")
                delta = delta.to(target.dtype).flatten()
                delta_start = (lora.start_row + start) * columns
                copy_lo, copy_hi = max(lo, delta_start), min(hi, delta_start + delta.numel())
                dst = target[copy_lo - tensor_start:copy_hi - tensor_start]
                dst.add_(delta[copy_lo - delta_start:copy_hi - delta_start])
                if not bool(torch.isfinite(dst).all()):
                    raise ValueError("Non-finite merged weights")


def build_torch_block_shard(plans, world_size, rank, *, chunk_elements=1 << 18, pin_memory=False):
    """Allocate ONLY local shards; ready for a future DLO prepared-shard API."""
    plans = list(plans)
    if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid group/rank")
    if len({p.name for p in plans}) != len(plans):
        raise ValueError("Duplicate tensor names")
    grouped, metadata, shards = {}, {}, {}
    for plan in plans:
        _validate_plan(plan)
        grouped.setdefault(plan.source.info["dtype"], []).append(plan)
    # Shards become persistent model storage, so explicitly override an outer
    # loader inference context and allocate tensors with version counters.
    with torch.inference_mode(False), torch.no_grad():
        for dtype, members in grouped.items():
            total = sum(math.prod(p.source.info["shape"]) for p in members)
            size = (total + world_size - 1) // world_size
            start, stop = rank * size, min((rank + 1) * size, total)
            # No intermediate full-block buffer; pin at allocation rather
            # than creating an unpinned shard then pin_memory() duplicating it.
            shard = torch.zeros(size, dtype=TORCH_DTYPES[dtype], device="cpu", pin_memory=pin_memory)
            metadata[dtype] = []
            offset = 0
            for plan in members:
                shape = plan.source.info["shape"]
                count = math.prod(shape)
                metadata[dtype].append(dict(name=plan.name, offset=offset, numel=count, shape=tuple(shape)))
                lo, hi = max(start, offset), min(stop, offset + count)
                if lo < hi:
                    identity_merge_plan = replace(plan, loras=())
                    for tensor_offset, raw in read_model_interval(identity_merge_plan, lo - offset, hi - offset, chunk_elements):
                        source = torch.from_numpy(raw)
                        if dtype == "BF16":
                            source = source.view(torch.bfloat16)
                        destination = offset + tensor_offset - start
                        shard[destination:destination + source.numel()].copy_(source)
                    if plan.loras:
                        merge_owned_interval_(shard[lo - start:hi - start], tensor_start=lo - offset,
                                              shape=shape, loras=plan.loras)
                offset += count
            shards[dtype] = shard
    return shards, metadata
