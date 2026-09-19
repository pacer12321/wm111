"""D-only inference adapter: add source hybrid attention; preserve C target math.

No weights are created here. Source and target reuse the same released per-layer
linear module in two separate calls with separate layouts and scan state. Text
and audio routes remain those of C; this does NOT claim an independent encoder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from . import dual_stream_attention as kernel
from .openvdn_npu import OpenVDNLayout

MODE = "D_source_hybrid_same_frame_v1"
_LOGGER = logging.getLogger(__name__)
_REQUEST_INDEX = 0
_FORWARD_INDEX = 0


@lru_cache(maxsize=1)
def reviewed_policy():
    path_text = os.environ.get("D_REVIEWED_POLICY_PATH", "")
    expected = os.environ.get("D_REVIEWED_POLICY_SHA256", "")
    path = Path(path_text)
    if not path_text or not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("D requires an explicit regular reviewed policy file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(expected) != 64 or digest != expected:
        raise RuntimeError("D reviewed policy digest mismatch")
    policy = json.loads(raw)
    if policy.get("mode") != MODE or policy.get("review_status") != "approved_for_inference":
        raise RuntimeError("D refuses an unreviewed or different experiment policy")
    wanted = dict(endpoint_policy="vdn_anchors", auxiliary_policy="preserve_existing",
                  source_linear_text="text", target_linear_text="text", chunk_frames=5,
                  chunk_radius=1, share_parameters=True, independent_stream_states=True)
    attention = policy.get("attention")
    if not isinstance(attention, dict) or set(attention) != set(wanted):
        raise RuntimeError("D attention contract must be exact and complete")
    if any(type(attention[name]) is not type(value) or attention[name] != value
           for name, value in wanted.items()):
        raise RuntimeError("D cannot change target rules, endpoints, text state or parameters")
    return policy, digest


def begin_request():
    global _REQUEST_INDEX, _FORWARD_INDEX
    reviewed_policy()
    _REQUEST_INDEX += 1
    _FORWARD_INDEX = 0
    return _REQUEST_INDEX


def begin_forward():
    global _FORWARD_INDEX
    if _REQUEST_INDEX < 1:
        raise RuntimeError("D forward has no pipeline request binding")
    _FORWARD_INDEX += 1


@lru_cache(maxsize=8)
def kernel_layout_and_routing(layout):
    """Translate already-validated C packing, without altering positions/RoPE."""
    reviewed_policy()
    layout.validate(layout.used_len)
    if layout.mode != MODE:
        raise RuntimeError("D refuses a legacy C source-layout record")
    if layout.text_len <= 0:
        raise RuntimeError("D requires the existing nonempty text conditioning")
    text = (kernel.NamedSpan("text", layout.text_start, layout.text_len),)
    audio = []
    if layout.source_start > layout.text_len:
        audio.append(kernel.NamedSpan("source_audio", layout.text_len,
                                     layout.source_start - layout.text_len))
    if layout.video_start > layout.source_end:
        audio.append(kernel.NamedSpan("target_audio", layout.source_end,
                                     layout.video_start - layout.source_end))
    dual = kernel.DualStreamLayout(
        used_len=layout.used_len,
        source=kernel.VideoSpan(layout.source_start, layout.num_frames,
                                layout.source_frame_height, layout.source_frame_width),
        target=kernel.VideoSpan(layout.video_start, layout.num_frames,
                                layout.frame_height, layout.frame_width),
        endpoint_policy="vdn_anchors", text_spans=text, audio_spans=tuple(audio),
    )
    names = tuple(span.name for span in dual.auxiliary_spans)
    all_names = ("source", "target") + names
    routing = kernel.DualStreamRouting(
        enforce_source_independence=False,
        source_aux_keys=names, target_aux_keys=names,
        auxiliary_reads=tuple((name, all_names) for name in names),
        source_linear_text=("text",), target_linear_text=("text",),
    )
    dual.validate(layout.used_len)
    routing.validate(dual)
    return dual, routing


@lru_cache(maxsize=8)
def source_linear_layout(layout):
    """Apply the ORIGINAL VDN branch to S, without changing its solver/formula."""
    kernel_layout_and_routing(layout)
    source = OpenVDNLayout(
        used_len=layout.used_len, video_start=layout.source_start,
        num_frames=layout.num_frames, tokens_per_frame=layout.source_tokens_per_frame,
        frame_height=layout.source_frame_height, frame_width=layout.source_frame_width,
        text_start=layout.text_start, text_len=layout.text_len,
    )
    source.validate(layout.used_len)
    return source


def source_hybrid_softmax_attention(q, k, v, layout, scale, groups_per_call=4):
    dual, routing = kernel_layout_and_routing(layout)
    return kernel.dual_stream_softmax_attention(q, k, v, dual, routing, scale, groups_per_call)


@lru_cache(maxsize=1)
def _kernel_digest():
    path = Path(kernel.__file__).resolve()
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def record_execution(*, layer_index, rank, world, layout, tensor, heads):
    """Called only after actual Softmax and both original linear calls return.

First forward of each request is logged for every DiT layer/rank. Device work
may be asynchronous here; successful request completion is a separate gate.
"""
    if _FORWARD_INDEX != 1:
        return
    if not 0 <= layer_index < 50 or not 0 <= rank < world or world not in (1, 8):
        raise RuntimeError("D execution record has invalid layer/rank binding")
    _, policy_sha = reviewed_policy()
    module_path, module_sha = _kernel_digest()
    row = dict(
        mode=MODE, request_index=_REQUEST_INDEX, forward_index=_FORWARD_INDEX,
        layer_index=layer_index, rank=rank, world=world, endpoint_policy="vdn_anchors",
        auxiliary_policy="preserve_existing", policy_sha256=policy_sha,
        module_path=module_path, module_sha256=module_sha,
        source_frames=layout.num_frames, target_frames=layout.num_frames,
        source_tokens_per_frame=layout.source_tokens_per_frame,
        target_tokens_per_frame=layout.tokens_per_frame, heads=heads,
        device=str(tensor.device), dtype=str(tensor.dtype),
        softmax_source_completed=True, softmax_target_completed=True,
        linear_source_completed=True, linear_target_completed=True,
    )
    _LOGGER.info("D_EXECUTION_RECORD %s", json.dumps(row, sort_keys=True))


__all__ = ["MODE", "reviewed_policy", "begin_request", "begin_forward",
           "kernel_layout_and_routing", "source_linear_layout",
           "source_hybrid_softmax_attention", "record_execution"]
