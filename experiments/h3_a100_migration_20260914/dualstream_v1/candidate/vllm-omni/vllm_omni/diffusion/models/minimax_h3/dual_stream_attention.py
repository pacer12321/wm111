"""Forward-only, parameter-sharing dual-video OpenVDN candidate.

This module does not instantiate parameters, edit RoPE, perform collectives, or
load a model. Install it beside the existing ``openvdn_npu`` implementation.
Softmax accepts post-RoPE Q/K; linear accepts the original pre-RoPE raw Q/K/V.
Both USP1 and Ulysses callers reuse their existing per-layer linear branch.

All auxiliary visibility and endpoint choices are explicit. In particular,
``enforce_source_independence=False`` allows indirect T -> auxiliary -> S paths
and must NOT be described as independent source encoding. When enabled, the
dependency check includes both Softmax visibility and linear text-state seeds.
That check covers this module's fixed routing, not the entire H3 model: callers
must ensure initial source/allowed-auxiliary hidden states, timestep modulation
and other updates are not target-dependent, and retain the same closure across
layers. Naming a span "source_audio" does not certify its provenance.

The VDN solver is reused verbatim, including its forward-only ``out=`` scan.
No autograd, training, hardware, speed, or quality validation is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import (
    _activate,
    _delta_factor_apply,
    _frame_statistics,
    _fusion_attention_tnd,
    _gather_linear_state,
    _scan_states,
    window_bounds,
)


MODE = "dual_stream_shared_vdn_forward_v1"
D_MODE = "D_source_hybrid_same_frame_v1"
_ENDPOINT_POLICIES = {"vdn_anchors", "strict_local"}


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class VideoSpan:
    start: int
    num_frames: int
    frame_height: int
    frame_width: int

    @property
    def tokens_per_frame(self) -> int:
        return self.frame_height * self.frame_width

    @property
    def end(self) -> int:
        return self.start + self.num_frames * self.tokens_per_frame

    def frame_range(self, lo: int, hi: int) -> tuple[int, int]:
        """Half-open ordinal frame range, independent of absolute RoPE time."""
        if not 0 <= lo < hi <= self.num_frames:
            raise ValueError("Frame range must be nonempty and inside its stream")
        return self.start + lo * self.tokens_per_frame, self.start + hi * self.tokens_per_frame

    def validate(self, used_len: int, name: str) -> None:
        _integer(self.start, f"{name}.start")
        for field in ("num_frames", "frame_height", "frame_width"):
            _integer(getattr(self, field), f"{name}.{field}", 1)
        if self.end > used_len:
            raise ValueError(f"{name} exceeds the real packed document")


@dataclass(frozen=True)
class NamedSpan:
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class DualStreamLayout:
    used_len: int
    source: VideoSpan
    target: VideoSpan
    endpoint_policy: str
    text_spans: tuple[NamedSpan, ...] = ()
    audio_spans: tuple[NamedSpan, ...] = ()

    @property
    def total_frames(self) -> int:
        return self.source.num_frames + self.target.num_frames

    @property
    def auxiliary_spans(self) -> tuple[NamedSpan, ...]:
        return self.text_spans + self.audio_spans

    def validate(self, packed_len: int) -> None:
        _integer(packed_len, "packed_len", 1)
        _integer(self.used_len, "used_len", 1)
        if self.used_len > packed_len or self.endpoint_policy not in _ENDPOINT_POLICIES:
            raise ValueError("Invalid used_len or explicit endpoint_policy")
        if not isinstance(self.source, VideoSpan) or not isinstance(self.target, VideoSpan):
            raise ValueError("source and target must be explicit VideoSpan objects")
        self.source.validate(self.used_len, "source")
        self.target.validate(self.used_len, "target")
        if self.source.num_frames != self.target.num_frames:
            raise ValueError("Same-ordinal source access requires Fs == Ft; no interpolation fallback")
        if self.endpoint_policy == "vdn_anchors" and self.source.num_frames < 3:
            raise ValueError("vdn_anchors needs two endpoints and at least one interior frame")
        if type(self.text_spans) is not tuple or type(self.audio_spans) is not tuple:
            raise ValueError("Auxiliary spans must be immutable tuples")
        intervals = [(self.source.start, self.source.end), (self.target.start, self.target.end)]
        names = {"source", "target"}
        for span in self.auxiliary_spans:
            if not isinstance(span, NamedSpan) or not isinstance(span.name, str) or not span.name:
                raise ValueError("Each auxiliary block needs an explicit nonempty name")
            if span.name in names:
                raise ValueError("Auxiliary names must be unique and not source/target")
            names.add(span.name)
            _integer(span.start, f"{span.name}.start")
            _integer(span.length, f"{span.name}.length", 1)
            if span.end > self.used_len:
                raise ValueError("Auxiliary span exceeds the real packed document")
            intervals.append((span.start, span.end))
        cursor = 0
        for start, stop in sorted(intervals):
            if start != cursor:
                raise ValueError("Explicit source/target/text/audio spans must cover all real rows without overlap")
            cursor = stop
        if cursor != self.used_len:
            raise ValueError("Unclassified packed rows are forbidden; declare their text/audio spans")


@dataclass(frozen=True)
class DualStreamRouting:
    # Required rather than an implicit claim of independence.
    enforce_source_independence: bool
    source_aux_keys: tuple[str, ...] = ()
    target_aux_keys: tuple[str, ...] = ()
    # One entry for EVERY auxiliary query block; values are whole named blocks.
    # Permitted key labels are source, target, and any declared auxiliary name.
    auxiliary_reads: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Seeds use the original TEXT_STATE_SCALE and VDN-solve text-state formula.
    # Empty means explicitly no text seed, not silently all text.
    source_linear_text: tuple[str, ...] = ()
    target_linear_text: tuple[str, ...] = ()

    def dependency_graph(self, layout: DualStreamLayout) -> dict[str, set[str]]:
        graph = {name: set(keys) for name, keys in self.auxiliary_reads}
        graph["source"] = {"source", *self.source_aux_keys, *self.source_linear_text}
        graph["target"] = {"target", "source", *self.target_aux_keys, *self.target_linear_text}
        return graph

    def source_depends_on_target(self, layout: DualStreamLayout) -> bool:
        graph = self.dependency_graph(layout)
        visited, pending = set(), ["source"]
        while pending:
            name = pending.pop()
            if name == "target":
                return True
            if name not in visited:
                visited.add(name)
                pending.extend(graph.get(name, ()))
        return False

    def validate(self, layout: DualStreamLayout) -> None:
        if type(self.enforce_source_independence) is not bool:
            raise ValueError("enforce_source_independence must be explicitly bool")
        aux_names = {span.name for span in layout.auxiliary_spans}
        text_names = {span.name for span in layout.text_spans}
        all_names = aux_names | {"source", "target"}

        def check_keys(keys, allowed, label, *, nonempty=False):
            if type(keys) is not tuple or any(type(key) is not str for key in keys):
                raise ValueError(f"{label} must be an immutable tuple of block names")
            if len(set(keys)) != len(keys) or not set(keys) <= allowed or (nonempty and not keys):
                raise ValueError(f"Invalid, duplicated or missing keys in {label}")

        check_keys(self.source_aux_keys, aux_names, "source_aux_keys")
        check_keys(self.target_aux_keys, aux_names, "target_aux_keys")
        check_keys(self.source_linear_text, text_names, "source_linear_text")
        check_keys(self.target_linear_text, text_names, "target_linear_text")
        if type(self.auxiliary_reads) is not tuple:
            raise ValueError("auxiliary_reads must be an immutable tuple")
        seen = set()
        for entry in self.auxiliary_reads:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("Each auxiliary route must be (query_name, key_names)")
            name, keys = entry
            if type(name) is not str or name not in aux_names or name in seen:
                raise ValueError("Each auxiliary query block must have exactly one explicit route")
            seen.add(name)
            check_keys(keys, all_names, f"auxiliary_reads[{name}]", nonempty=True)
        if seen != aux_names:
            raise ValueError("Missing explicit auxiliary query routes")
        if self.enforce_source_independence and self.source_depends_on_target(layout):
            raise ValueError("Source can depend on target through auxiliary Softmax/text-state paths")


def d_configuration_from_c_layout(c_layout):
    """Freeze experiment D: only replace C's source visual attention.

    Input is the already validated actual C layout, not requested pixel shapes.
    T-T, T-S and text/audio routes retain C's exact visible rows; original text
    seeds are retained for T and reused separately for the new S linear scan.
    Source anchors see their own stream and all original auxiliaries, never T
    visual tokens directly. This is NOT a source-independent architecture.
    """
    c_layout.validate(c_layout.used_len)
    if not (c_layout.text_start == 0 and c_layout.video_end == c_layout.used_len
            and c_layout.text_len <= c_layout.source_start < c_layout.source_end <= c_layout.video_start):
        raise ValueError("D requires the original C text/source-audio/source/target-audio/target layout")
    text = ((NamedSpan("text", 0, c_layout.text_len),) if c_layout.text_len else ())
    audio = []
    if c_layout.source_start > c_layout.text_len:
        audio.append(NamedSpan("source_audio", c_layout.text_len, c_layout.source_start - c_layout.text_len))
    if c_layout.video_start > c_layout.source_end:
        audio.append(NamedSpan("target_audio", c_layout.source_end, c_layout.video_start - c_layout.source_end))
    layout = DualStreamLayout(
        c_layout.used_len,
        VideoSpan(c_layout.source_start, c_layout.num_frames, c_layout.source_frame_height, c_layout.source_frame_width),
        VideoSpan(c_layout.video_start, c_layout.num_frames, c_layout.frame_height, c_layout.frame_width),
        "vdn_anchors", text, tuple(audio),
    )
    auxiliary_names = tuple(span.name for span in layout.auxiliary_spans)
    all_names = ("source", "target") + auxiliary_names
    text_names = tuple(span.name for span in text)
    routing = DualStreamRouting(
        False, auxiliary_names, auxiliary_names,
        tuple((name, all_names) for name in auxiliary_names), text_names, text_names,
    )
    layout.validate(c_layout.used_len)
    routing.validate(layout)
    return layout, routing


def _union_ranges(ranges) -> tuple[tuple[int, int], ...]:
    merged = []
    for start, stop in sorted(ranges):
        if not start < stop:
            raise ValueError("Attention ranges must be nonempty")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return tuple(merged)


@lru_cache(maxsize=32)
def dual_stream_softmax_plan(layout: DualStreamLayout, routing: DualStreamRouting) -> tuple:
    """CPU plan of (query_ranges, key_ranges), with one owner per real query.

    c5/r1 is a chunk window (at most three five-frame chunks), NOT +/-1 frame.
    Anchors, when selected, are global only WITHIN their own visual stream.
    Even target endpoint queries see only their same-ordinal source frame.
    "Far" in the linear complement refers to frame-state exclusion. The
    unchanged +/-2 temporal short-convolution can mix neighboring raw contents.
    """
    layout.validate(layout.used_len)
    routing.validate(layout)
    blocks = {"source": (layout.source.start, layout.source.end),
              "target": (layout.target.start, layout.target.end)}
    blocks.update({span.name: (span.start, span.end) for span in layout.auxiliary_spans})
    grouped = {}
    for stream_name, stream, aux_names in (
        ("source", layout.source, routing.source_aux_keys),
        ("target", layout.target, routing.target_aux_keys),
    ):
        for frame, (lo, hi) in enumerate(window_bounds(stream.num_frames, radius=1, chunk=5)):
            pieces = [blocks[name] for name in aux_names]
            if layout.endpoint_policy == "vdn_anchors" and frame in (0, stream.num_frames - 1):
                pieces.append((stream.start, stream.end))
            else:
                pieces.append(stream.frame_range(max(0, lo), min(stream.num_frames, hi + 1)))
                if layout.endpoint_policy == "vdn_anchors":
                    pieces.extend((stream.frame_range(0, 1), stream.frame_range(stream.num_frames - 1, stream.num_frames)))
            if stream_name == "target":
                pieces.append(layout.source.frame_range(frame, frame + 1))
            grouped.setdefault(_union_ranges(pieces), []).append(stream.frame_range(frame, frame + 1))
    for name, key_names in routing.auxiliary_reads:
        grouped.setdefault(_union_ranges(blocks[key] for key in key_names), []).append(blocks[name])
    return tuple((_union_ranges(queries), keys) for keys, queries in grouped.items())


@lru_cache(maxsize=16)
def _device_plan(layout, routing, device_string):
    device = torch.device(device_string)
    return tuple(
        (torch.cat([torch.arange(lo, hi, device=device) for lo, hi in qr]),
         torch.cat([torch.arange(lo, hi, device=device) for lo, hi in kr]))
        for qr, kr in dual_stream_softmax_plan(layout, routing)
    )


def _validate_qkv(qkv, layout, *, branch=None, head_start=0):
    if not isinstance(qkv, (tuple, list)) or len(qkv) != 3:
        raise ValueError("Q/K/V must be a tuple of three tensors")
    q = qkv[0]
    if q.ndim != 3 or any(t.shape != q.shape or t.device != q.device or t.dtype != q.dtype for t in qkv):
        raise ValueError("Q/K/V must have matching [packed_rows, heads, dim], dtype and device")
    layout.validate(q.shape[0])
    if q.shape[1] < 1 or q.shape[2] < 1 or not q.is_floating_point():
        raise ValueError("Attention heads and channels must be nonempty floating point tensors")
    if branch is not None:
        _integer(head_start, "head_start")
        if branch.delta_rule != "vdn_solve":
            raise ValueError("Dual-stream candidate preserves vdn_solve only")
        if q.shape[2] != branch.head_dim or head_start + q.shape[1] > branch.num_heads:
            raise ValueError("Invalid linear head shard")


def dual_stream_softmax_attention(query, key, value, layout, routing, scale, groups_per_call=4):
    """Exact specified visibility, using existing TND fusion (CPU SDPA fallback).

    Works for all heads on one device or global-sequence/local-head Ulysses
    tensors. Trailing packed padding is zero. Caller owns gates/projections.
    """
    _validate_qkv((query, key, value), layout)
    routing.validate(layout)
    _integer(groups_per_call, "groups_per_call", 1)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be finite and positive")
    groups = _device_plan(layout, routing, str(query.device))
    out = torch.zeros_like(query)
    for start in range(0, len(groups), groups_per_call):
        batch = groups[start:start + groups_per_call]
        q_cu, kv_cu, q_total, kv_total = [], [], 0, 0
        for q_idx, k_idx in batch:
            q_total += q_idx.numel()
            kv_total += k_idx.numel()
            q_cu.append(q_total)
            kv_cu.append(kv_total)
        packed = _fusion_attention_tnd(
            torch.cat([query.index_select(0, q_idx) for q_idx, _ in batch]),
            torch.cat([key.index_select(0, k_idx) for _, k_idx in batch]),
            torch.cat([value.index_select(0, k_idx) for _, k_idx in batch]),
            q_cu, kv_cu, scale,
        )
        out.index_copy_(0, torch.cat([q_idx for q_idx, _ in batch]), packed)
    return out


def dual_stream_local_frame_sums_counts(x_local, layout, row_start):
    """FP32 statistics [Fs+Ft, hidden], [Fs+Ft], ordered all S then all T.

    Include endpoints for BOTH policies. The caller SUM-all-reduces these two
    tensors over Ulysses and checks counts against
    ``dual_stream_expected_frame_counts`` before dividing. No full-x gather.
    Calling with the whole packed sequence and row_start=0 needs no reduction.
    """
    if x_local.ndim != 2 or not x_local.is_floating_point():
        raise ValueError("Local hidden states must be floating point [rows, hidden]")
    _integer(row_start, "row_start")
    layout.validate(max(layout.used_len, row_start + x_local.shape[0]))
    sums = torch.zeros((layout.total_frames, x_local.shape[-1]), device=x_local.device, dtype=torch.float32)
    counts = torch.zeros(layout.total_frames, device=x_local.device, dtype=torch.float32)
    row_end, offset = row_start + x_local.shape[0], 0
    for stream in (layout.source, layout.target):
        for frame in range(stream.num_frames):
            lo, hi = stream.frame_range(frame, frame + 1)
            lo, hi = max(lo, row_start), min(hi, row_end)
            if lo < hi:
                sums[offset + frame] = x_local[lo - row_start:hi - row_start].sum(dim=0, dtype=torch.float32)
                counts[offset + frame] = hi - lo
        offset += stream.num_frames
    return sums, counts


def dual_stream_expected_frame_counts(layout, *, device=None):
    layout.validate(layout.used_len)
    return torch.tensor(
        [layout.source.tokens_per_frame] * layout.source.num_frames
        + [layout.target.tokens_per_frame] * layout.target.num_frames,
        device=device, dtype=torch.float32,
    )


def _linear_text_state(branch, qkv, beta_logits, layout, text_names):
    if not text_names:
        return None
    spans = {span.name: span for span in layout.text_spans}
    indices = torch.cat([
        torch.arange(spans[name].start, spans[name].end, device=qkv[0].device)
        for name in text_names
    ])
    key = _activate(qkv[1].index_select(0, indices), True).transpose(0, 1).unsqueeze(0)
    value = _activate(qkv[2].index_select(0, indices), False).transpose(0, 1).unsqueeze(0)
    beta = torch.sigmoid(beta_logits.index_select(0, indices)).transpose(0, 1).unsqueeze(0)
    A, B = _frame_statistics(key, value, beta)
    ones = torch.ones((1, qkv[0].shape[1], qkv[0].shape[2]), device=key.device, dtype=torch.float32)
    _, injection = _delta_factor_apply("vdn_solve", ones, A, B, indices.numel())
    return branch.TEXT_STATE_SCALE * injection[0]


def dual_stream_linear_head_shard(branch, qkv_raw_global, beta_logits_global,
                                  frame_means, layout, routing, head_start=0):
    """Independent S/T scans with shared released weights, before output gate.

    qkv: [packed_rows, local_heads, head_dim]; beta: [packed_rows, local_heads].
    frame_means: FP32 [Fs+Ft, complete_hidden], S all frames followed by T.
    Returns full-packed [rows, local_heads, dim]. Nonvisual rows/padding are
    zero. vdn_anchors also zeros visual endpoints; strict_local includes them.
    Inverse all-to-all MUST precede output_gate(local_x) and to_out_linear.
    """
    _validate_qkv(qkv_raw_global, layout, branch=branch, head_start=head_start)
    routing.validate(layout)
    raw_q = qkv_raw_global[0]
    packed_len, heads, dim = raw_q.shape
    if beta_logits_global.shape != (packed_len, heads) or beta_logits_global.device != raw_q.device:
        raise ValueError("Beta logits must match global packed rows and local heads")
    if frame_means.shape != (layout.total_frames, branch.alpha.down.in_features) or frame_means.device != raw_q.device:
        raise ValueError("Frame means must cover all S/T frames and every hidden channel on the QKV device")
    out, frame_offset = torch.zeros_like(raw_q), 0
    for stream, text_names in ((layout.source, routing.source_linear_text),
                               (layout.target, routing.target_linear_text)):
        first = 1 if layout.endpoint_policy == "vdn_anchors" else 0
        stop = stream.num_frames - first
        frames, per_frame = stop - first, stream.tokens_per_frame
        lo, hi = stream.frame_range(first, stop)
        selected = slice(lo, hi)
        # Each invocation pads its OWN temporal volume with zeros. Never
        # concatenate S/T before short_conv: that would leak across streams.
        features = tuple(
            _activate(branch.short_conv.apply(name, raw[selected], frames,
                      (stream.frame_height, stream.frame_width), head_start), name != "v")
            for name, raw in zip(("q", "k", "v"), qkv_raw_global)
        )
        query, key, value = (
            t.reshape(frames, per_frame, heads, dim).permute(0, 2, 1, 3).contiguous()
            for t in features
        )
        beta = torch.sigmoid(beta_logits_global[selected]).reshape(frames, per_frame, heads)
        beta = beta.permute(0, 2, 1).contiguous()
        A, B = _frame_statistics(key, value, beta)
        alpha = branch.alpha.forward_head_shard(frame_means[frame_offset + first:frame_offset + stop], head_start, heads)
        text_state = _linear_text_state(branch, qkv_raw_global, beta_logits_global, layout, text_names)
        prefix, suffix = _scan_states(alpha, A, B, text_state,
                                      delta_rule="vdn_solve", tokens_per_frame=per_frame)
        bounds = [(left - first, right - first)
                  for left, right in window_bounds(stream.num_frames, radius=1, chunk=5)[first:stop]]
        linear_state = _gather_linear_state(prefix, suffix, alpha, bounds, text_state).to(raw_q.dtype)
        readout = torch.matmul(query, linear_state.transpose(-1, -2))
        readout = readout.permute(0, 2, 1, 3).reshape(frames * per_frame, heads, dim)
        out[selected] = branch.norm(readout)
        frame_offset += stream.num_frames
    return out


def dual_stream_linear_forward(branch, x, qkv_raw, layout, routing):
    """USP1 full-packed, gated [rows, heads*dim]; output projection is external."""
    _validate_qkv(qkv_raw, layout, branch=branch)
    if x.ndim != 2 or x.shape != (qkv_raw[0].shape[0], branch.beta_proj.in_features):
        raise ValueError("Single-device x must contain the complete packed hidden sequence")
    if qkv_raw[0].shape[1] != branch.num_heads or x.device != qkv_raw[0].device:
        raise ValueError("Single-device forward requires every head and colocated hidden states")
    sums, counts = dual_stream_local_frame_sums_counts(x, layout, 0)
    expected = dual_stream_expected_frame_counts(layout, device=x.device)
    if not torch.equal(counts, expected):
        raise ValueError("Single-device rows do not cover each source and target frame exactly once")
    readout = dual_stream_linear_head_shard(
        branch, qkv_raw, branch.beta_proj(x), sums / counts[:, None], layout, routing,
    )
    readout = readout * branch.output_gate(x).to(readout.dtype)
    return readout.reshape(x.shape[0], branch.num_heads * branch.head_dim)


__all__ = [
    "MODE", "D_MODE", "VideoSpan", "NamedSpan", "DualStreamLayout", "DualStreamRouting",
    "d_configuration_from_c_layout",
    "dual_stream_softmax_plan", "dual_stream_softmax_attention",
    "dual_stream_local_frame_sums_counts", "dual_stream_expected_frame_counts",
    "dual_stream_linear_head_shard", "dual_stream_linear_forward",
]
