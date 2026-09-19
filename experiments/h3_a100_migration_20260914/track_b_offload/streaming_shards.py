"""Isolated CPU prototype; NOT wired into any serving candidate.

Plans the same per-block/per-dtype flat shards as DLO, but reads only the
overlap from safetensors before allocating any full model. QKV is converted
by source-row addressing, not by first copying the full fused tensor.

The NumPy LoRA implementation is a reference prototype, not a replacement
for the production torch FP32 matmul. A torch/GPU equivalence gate is still
required before integrating this with the offloader.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPES = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4")}


def unique_pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError(f"Duplicate key: {name}")
        result[name] = value
    return result


class RangeReader:
    """Read-only, bounded reads; records payload ranges for auditing."""

    def __init__(self, path):
        self.path = Path(path)
        self.payload_reads = []
        with self.path.open("rb") as stream:
            field = stream.read(8)
            if len(field) != 8:
                raise ValueError("Truncated safetensors header size")
            size = struct.unpack("<Q", field)[0]
            if not 2 <= size <= 16 * 1024 * 1024:
                raise ValueError("Invalid safetensors header size")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError("Truncated safetensors header")
        self.header = json.loads(raw, object_pairs_hook=unique_pairs)
        self.header.pop("__metadata__", None)
        self.data_offset = 8 + size
        end = 0
        for name, info in sorted(self.header.items(), key=lambda pair: pair[1]["data_offsets"][0]):
            shape, dtype = info["shape"], info["dtype"]
            if dtype not in DTYPES or any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError(f"Unsupported dtype/shape: {name}")
            lo, hi = info["data_offsets"]
            if lo != end or hi - lo != math.prod(shape) * DTYPES[dtype].itemsize:
                raise ValueError(f"Invalid/overlapping extent: {name}")
            end = hi
        if self.path.stat().st_size != self.data_offset + end:
            raise ValueError("Truncated or trailing tensor payload")

    def read_flat(self, key, start, stop):
        info = self.header[key]
        count = math.prod(info["shape"])
        if not 0 <= start <= stop <= count:
            raise ValueError("Tensor range out of bounds")
        dtype = DTYPES[info["dtype"]]
        if start == stop:
            return np.empty(0, dtype=dtype)
        nbytes = (stop - start) * dtype.itemsize
        offset = self.data_offset + info["data_offsets"][0] + start * dtype.itemsize
        with self.path.open("rb", buffering=0) as stream:
            stream.seek(offset)
            raw = stream.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError("Tensor payload changed or truncated during read")
        self.payload_reads.append((key, start, stop))
        return np.frombuffer(raw, dtype=dtype).copy()


@dataclass(frozen=True)
class TensorRef:
    reader: RangeReader
    key: str

    @property
    def info(self):
        return self.reader.header[self.key]


@dataclass(frozen=True)
class LoRA:
    """Target row interval is in the already reordered model tensor."""

    start_row: int
    end_row: int
    a: TensorRef
    b: TensorRef


@dataclass(frozen=True)
class TensorPlan:
    name: str
    source: TensorRef
    # H3 uses MHA. None means identity (including TP=1 FC1 gate/up).
    qkv_heads: int | None = None
    head_dim: int | None = None
    loras: tuple[LoRA, ...] = ()


def fp32_to_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values")
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def bf16_to_fp32(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def as_fp32(ref, start, stop):
    raw = ref.reader.read_flat(ref.key, start, stop)
    out = bf16_to_fp32(raw) if ref.info["dtype"] == "BF16" else raw
    if not np.isfinite(out).all():
        raise ValueError("Non-finite checkpoint values")
    return out


def _validate_plan(plan):
    info = plan.source.info
    shape = info["shape"]
    if plan.qkv_heads is not None:
        if len(shape) != 2 or not plan.head_dim or plan.qkv_heads <= 0:
            raise ValueError("Invalid QKV layout")
        if shape[0] != 3 * plan.qkv_heads * plan.head_dim:
            raise ValueError("QKV shape mismatch")
    elif plan.head_dim is not None:
        raise ValueError("head_dim without qkv_heads")
    previous_end = 0
    for lora in sorted(plan.loras, key=lambda item: item.start_row):
        if len(shape) != 2 or not previous_end <= lora.start_row < lora.end_row <= shape[0]:
            raise ValueError("Invalid or overlapping LoRA row ranges")
        a_shape, b_shape = lora.a.info["shape"], lora.b.info["shape"]
        if len(a_shape) != 2 or a_shape[1] != shape[1] or b_shape != [lora.end_row - lora.start_row, a_shape[0]]:
            raise ValueError("LoRA shape mismatch")
        if info["dtype"] != "BF16" or lora.a.info["dtype"] != "BF16" or lora.b.info["dtype"] != "BF16":
            raise ValueError("Prototype LoRA supports released BF16 Stage-B only")
        previous_end = lora.end_row


def read_model_interval(plan, start, stop, chunk_elements=1 << 18):
    """Yield model-layout fragments without reading a full model tensor."""
    _validate_plan(plan)
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    info, ref = plan.source.info, plan.source
    if not 0 <= start <= stop <= math.prod(info["shape"]):
        raise ValueError("Invalid model interval")
    columns = info["shape"][1] if len(info["shape"]) == 2 else None
    cursor = start
    while cursor < stop:
        end = min(stop, cursor + chunk_elements)
        source_start = cursor
        if plan.qkv_heads is not None:
            row, col = divmod(cursor, columns)
            rows_per_projection = plan.qkv_heads * plan.head_dim
            projection, within_projection = divmod(row, rows_per_projection)
            head, dim = divmod(within_projection, plan.head_dim)
            source_row = head * 3 * plan.head_dim + projection * plan.head_dim + dim
            source_start = source_row * columns + col
            # Never cross a model row in a mapped read.
            end = min(end, (row + 1) * columns)
        raw = ref.reader.read_flat(ref.key, source_start, source_start + end - cursor)
        if plan.loras:
            # Deliberately only row-local temporary deltas; implementation
            # optimization may batch rows after the torch equivalence gate.
            for lora in plan.loras:
                overlap_lo = max(cursor, lora.start_row * columns)
                overlap_hi = min(end, lora.end_row * columns)
                if overlap_lo >= overlap_hi:
                    continue
                rank = lora.a.info["shape"][0]
                a = as_fp32(lora.a, 0, rank * columns).reshape(rank, columns)
                first_row = overlap_lo // columns
                last_row = (overlap_hi - 1) // columns
                for row in range(first_row, last_row + 1):
                    b_row = row - lora.start_row
                    b = as_fp32(lora.b, b_row * rank, (b_row + 1) * rank)
                    delta = fp32_to_bf16(b @ a)  # alpha/rank = 1
                    lo, hi = max(overlap_lo, row * columns), min(overlap_hi, (row + 1) * columns)
                    dst = slice(lo - cursor, hi - cursor)
                    src = slice(lo - row * columns, hi - row * columns)
                    raw[dst] = fp32_to_bf16(bf16_to_fp32(raw[dst]) + bf16_to_fp32(delta[src]))
        finite = bf16_to_fp32(raw) if info["dtype"] == "BF16" else raw
        if not np.isfinite(finite).all():
            raise ValueError("Non-finite model values")
        yield cursor, raw
        cursor = end


def build_block_shard(plans, world_size, rank, *, chunk_elements=1 << 18):
    """Produce DLO-compatible layout metadata and private local shards.

    Plans MUST be supplied in the production named_parameters + named_buffers
    iteration order. This function does not discover/guess that order.
    """
    plans = list(plans)
    if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid shard group/rank")
    if len({p.name for p in plans}) != len(plans):
        raise ValueError("Duplicate model tensor names")
    grouped, metadata = {}, {}
    for plan in plans:
        _validate_plan(plan)
        dtype = plan.source.info["dtype"]
        grouped.setdefault(dtype, []).append(plan)
    shards = {}
    for dtype, members in grouped.items():
        total = sum(math.prod(p.source.info["shape"]) for p in members)
        size = (total + world_size - 1) // world_size
        shard_start, shard_stop = rank * size, min((rank + 1) * size, total)
        shard = np.zeros(size, dtype=DTYPES[dtype])
        offset = 0
        metadata[dtype] = []
        for plan in members:
            shape = plan.source.info["shape"]
            numel = math.prod(shape)
            metadata[dtype].append(dict(name=plan.name, offset=offset, numel=numel, shape=tuple(shape)))
            lo, hi = max(offset, shard_start), min(offset + numel, shard_stop)
            if lo < hi:
                for tensor_offset, fragment in read_model_interval(plan, lo - offset, hi - offset, chunk_elements):
                    destination = offset + tensor_offset - shard_start
                    shard[destination:destination + len(fragment)] = fragment
            offset += numel
        shards[dtype] = shard
    return shards, metadata
