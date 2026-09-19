"""CPU-only, fail-closed C layout and exact same-latent-frame Softmax plan.

No torch dependency, network, model parameters, diffusion math or runtime state.
Coordinates determine frame *ordinal* independently in each visual block.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

MODE = "C_strict_visual_source_same_latent_frame_v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"C: {name} must be an integer >= {minimum}")
    return value


def validate_raw_request(task, videos, image=None, audio=None):
    """Run on every worker before reference preparation/encoding collectives."""
    refs = videos if isinstance(videos, (tuple, list)) else [videos]
    if task != "ref2va" or image is not None or audio is not None:
        raise ValueError("C supports only single-video Ref2VA with its optional soundtrack")
    if len(refs) != 1 or not isinstance(refs[0], (str, os.PathLike)) or not str(refs[0]):
        raise ValueError("C requires exactly one source video path (no images/multiple references)")


def validate_arch(arch):
    if not arch.openvdn_enabled:
        raise ValueError("C requires enabled B/OpenVDN hybrid attention; no dense fallback")
    if tuple(arch.patch_size) != (1, 2, 2):
        raise ValueError("C supports only H3 patch_size=(1,2,2)")
    if arch.openvdn_interior_group_size != 0:
        raise ValueError("C implements only B c5/r1, not alternative interior-group masks")
    _integer(arch.openvdn_groups_per_call, "groups_per_call", 1)


def make_metadata(*, task, ref_blocks, visual_condition_shapes, target_shape,
                  text_len, audio_t):
    """Use actual VAE-returned shapes, not requested/truncated pixel frame counts."""
    if task != "ref2va" or not isinstance(ref_blocks, (list, tuple)) or len(ref_blocks) != 1:
        raise ValueError("C requires exactly one explicit video reference block")
    block = ref_blocks[0]
    if not isinstance(block, dict) or block.get("kind") != "video":
        raise ValueError("C rejects missing, image, audio-only, or ambiguous reference metadata")
    source_shape = tuple(block.get(key) for key in ("latent_t", "latent_h", "latent_w"))
    if not isinstance(visual_condition_shapes, (list, tuple)) or len(visual_condition_shapes) != 1:
        raise ValueError("C requires one actual VAE visual shape")
    if source_shape != tuple(visual_condition_shapes[0]):
        raise ValueError("C reference block does not match the actual VAE visual shape")
    meta = dict(mode=MODE, reference_count=1, reference_kind="video", patch_size=(1, 2, 2),
                source_shape=source_shape, target_shape=tuple(target_shape),
                text_len=text_len, source_audio_t=block.get("ref_audio_t"), target_audio_t=audio_t)
    validate_metadata(meta)
    return meta


def validate_metadata(meta):
    required = {"mode", "reference_count", "reference_kind", "patch_size", "source_shape",
                "target_shape", "text_len", "source_audio_t", "target_audio_t"}
    if not isinstance(meta, dict) or set(meta) != required:
        raise ValueError("C requires complete explicit source-layout metadata, with no unknown fields")
    if meta["mode"] != MODE or type(meta["reference_count"]) is not int or meta["reference_count"] != 1:
        raise ValueError("C mode/reference_count mismatch; no B/C fallback")
    if meta["reference_kind"] != "video" or tuple(meta["patch_size"]) != (1, 2, 2):
        raise ValueError("C requires one video and patch_size=(1,2,2)")
    for name in ("source_shape", "target_shape"):
        shape = meta[name]
        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError(f"C: invalid {name}")
        for value in shape:
            _integer(value, name, 1)
        if shape[0] < 3 or shape[1] % 2 or shape[2] % 2:
            raise ValueError(f"C: {name} needs >=3 frames and an exact [1,2,2] patch grid")
    if meta["source_shape"][0] != meta["target_shape"][0]:
        raise ValueError("C requires actual source Fs == target Ft; no truncation/padding/fallback")
    _integer(meta["text_len"], "text_len")
    _integer(meta["source_audio_t"], "source_audio_t")
    _integer(meta["target_audio_t"], "target_audio_t", 1)


def _contiguous(rows, start, length, name):
    if len(rows) != length or any(type(v) is not int or v != start + i for i, v in enumerate(rows)):
        raise ValueError(f"C {name} rows do not match the explicit packed block boundary")


def validate_grid(coords, rows, shape, name):
    """Independent unique-consecutive(t) + ordered, complete spatial-grid check."""
    frames, latent_h, latent_w = shape
    height, width = latent_h // 2, latent_w // 2
    per_frame = height * width
    if len(rows) != frames * per_frame:
        raise ValueError(f"C {name}: row count != VAE shape")
    selected = [tuple(coords[row]) for row in rows]
    if any(len(c) != 3 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in c)
           for c in selected):
        raise ValueError(f"C {name}: coordinates must be finite triples")
    times, counts = [], []
    for t, _, _ in selected:
        if not times or t != times[-1]:
            times.append(t)
            counts.append(0)
        counts[-1] += 1
    if len(times) != frames or counts != [per_frame] * frames or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"C {name}: incomplete/unordered latent frame grid")
    spatial = [c[1:] for c in selected[:per_frame]]
    hs, ws = sorted({c[0] for c in spatial}), sorted({c[1] for c in spatial})
    if len(hs) != height or len(ws) != width or spatial != [(h, w) for h in hs for w in ws]:
        raise ValueError(f"C {name}: incomplete or non-row-major spatial grid")
    if any(c[1:] != spatial[i % per_frame] for i, c in enumerate(selected)):
        raise ValueError(f"C {name}: spatial grid changes between frames")
    return tuple(times)


@dataclass(frozen=True)
class StrictSourceLayout:
    used_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int
    text_start: int
    text_len: int
    source_start: int
    source_tokens_per_frame: int
    source_frame_height: int
    source_frame_width: int
    source_times: tuple
    target_times: tuple
    mode: str = MODE

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def source_end(self):
        return self.source_start + self.num_frames * self.source_tokens_per_frame

    def validate(self, packed_len):
        if self.mode != MODE or self.num_frames < 3:
            raise ValueError("C invalid mode/frame count")
        if not (0 <= self.text_start == 0 <= self.text_len <= self.source_start < self.source_end
                <= self.video_start < self.video_end == self.used_len <= packed_len):
            raise ValueError("C invalid/overlapping packed boundaries")
        if min(self.frame_height, self.frame_width, self.source_frame_height, self.source_frame_width) < 1:
            raise ValueError("C empty spatial grid")
        if self.tokens_per_frame != self.frame_height * self.frame_width or self.source_tokens_per_frame != self.source_frame_height * self.source_frame_width:
            raise ValueError("C invalid source/target grid area")
        for axis in (self.source_times, self.target_times):
            if len(axis) != self.num_frames or any(not math.isfinite(t) for t in axis) or any(b <= a for a, b in zip(axis, axis[1:])):
                raise ValueError("C invalid temporal frame ordinal axis")


def infer_layout(*, img_pos, update_mask, text_pos, audio_pos, coords, cu_seqlens, metadata):
    """Validate full unsharded packed metadata before any DiT SP collective."""
    validate_metadata(metadata)
    if len(cu_seqlens) not in (2, 3) or cu_seqlens[0] != 0 or any(type(v) is not int for v in cu_seqlens):
        raise ValueError("C requires one real document plus optional trailing padding")
    used, packed_len = cu_seqlens[1], cu_seqlens[-1]
    if not 0 < used <= packed_len or len(coords) != packed_len:
        raise ValueError("C coordinates/cu_seqlens mismatch")
    if len(img_pos) != len(update_mask) or any(type(v) is not bool for v in update_mask):
        raise ValueError("C update_mask must be boolean and cover exactly visual rows")
    if any(type(v) is not int or not 0 <= v < used for v in img_pos):
        raise ValueError("C invalid visual row index")
    source = [row for row, update in zip(img_pos, update_mask) if not update]
    target = [row for row, update in zip(img_pos, update_mask) if update]
    st, sh, sw = metadata["source_shape"]
    tt, th, tw = metadata["target_shape"]
    sp, tp = (sh // 2) * (sw // 2), (th // 2) * (tw // 2)
    text_len = metadata["text_len"]
    ra, ta = metadata["source_audio_t"] * 2, metadata["target_audio_t"] * 2
    source_start = text_len + ra
    source_end = source_start + st * sp
    target_start = source_end + ta
    if target_start + tt * tp != used:
        raise ValueError("C actual packed used length differs from explicit single-reference blocks")
    _contiguous(source, source_start, st * sp, "source")
    _contiguous(target, target_start, tt * tp, "target")
    if img_pos != source + target:
        raise ValueError("C img_pos must contain source then target, without nonvisual rows")
    _contiguous(text_pos, 0, text_len, "text")
    if audio_pos != list(range(text_len, source_start)) + list(range(source_end, target_start)):
        raise ValueError("C audio rows must retain their exact reference/target audio blocks")
    source_times = validate_grid(coords, source, metadata["source_shape"], "source")
    target_times = validate_grid(coords, target, metadata["target_shape"], "target")
    layout = StrictSourceLayout(used, target_start, tt, tp, th // 2, tw // 2, 0, text_len,
                                source_start, sp, sh // 2, sw // 2, source_times, target_times)
    layout.validate(packed_len)
    return layout


@dataclass(frozen=True)
class Group:
    frame: int
    anchor: bool
    query_span: tuple[int, int]
    key_spans: tuple[tuple[int, int], ...]


def _merge(spans):
    out = []
    for lo, hi in sorted(spans):
        if lo == hi:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
        else:
            out.append((lo, hi))
    return tuple(out)


def strict_plan(layout):
    """Non-target queries stay dense; each target frame gets its own key set.

    A whole frozen layout is the cache key, including source start/grid/time,
    mode, target grid/time, and strict anchor policy (fixed by MODE).
    """
    if not isinstance(layout, StrictSourceLayout):
        raise ValueError("C plan requires explicitly validated C layout, not a B layout")
    layout.validate(layout.used_len)
    return _strict_plan_cached(layout)


@lru_cache(maxsize=8)
def _strict_plan_cached(layout):
    nonvisual = ((0, layout.source_start), (layout.source_end, layout.video_start))
    dense_query_spans = _merge(((0, layout.video_start), (layout.video_end, layout.used_len)))
    groups = []
    for frame in range(layout.num_frames):
        anchor = frame in (0, layout.num_frames - 1)
        q0 = layout.video_start + frame * layout.tokens_per_frame
        s0 = layout.source_start + frame * layout.source_tokens_per_frame
        if anchor:
            target_keys = ((layout.video_start, layout.video_end),)
        else:
            lo = max(0, (frame // 5 - 1) * 5)
            hi = min(layout.num_frames, (frame // 5 + 2) * 5)
            target_keys = ((layout.video_start + lo * layout.tokens_per_frame,
                            layout.video_start + hi * layout.tokens_per_frame),
                           (layout.video_start, layout.video_start + layout.tokens_per_frame),
                           (layout.video_end - layout.tokens_per_frame, layout.video_end))
        keys = _merge(nonvisual + ((s0, s0 + layout.source_tokens_per_frame),) + target_keys)
        groups.append(Group(frame, anchor, (q0, q0 + layout.tokens_per_frame), keys))
    return dense_query_spans, tuple(groups)


strict_plan.cache_clear = _strict_plan_cached.cache_clear
strict_plan.cache_info = _strict_plan_cached.cache_info


def expand_spans(spans):
    """CPU toy/oracle utility; runtime constructs index tensors from spans."""
    return [i for lo, hi in spans for i in range(lo, hi)]
