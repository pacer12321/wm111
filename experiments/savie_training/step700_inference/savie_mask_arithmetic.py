"""Exact arithmetic version of the step700 dual-stream mask.

Inference-only. Preserves c5/r1, first/last row+column anchors, same-frame
T->S, absent S->T, and context isolation. No change to token ownership.
"""
import torch


def make_arithmetic_mask(layout, source, length, permutation,
                         active_indices=None, world_size=2):
    if world_size < 1 or length < 1 or length % world_size:
        raise ValueError('Equal-row interleaved shards are required')
    if permutation.ndim != 1 or permutation.shape[0] != length or permutation.dtype != torch.long:
        raise ValueError('Invalid physical-to-logical permutation')
    rows = length // world_size
    physical = torch.arange(length, device=permutation.device)
    expected = physical.remainder(rows) * world_size + physical.div(rows, rounding_mode='floor')
    # Run at mask construction, never inside the attention kernel.
    if not torch.equal(permutation, expected):
        raise ValueError('Permutation is not equal-row round-robin interleaving')
    if source.num_frames != layout.num_frames or source.used_len != layout.used_len:
        raise ValueError('Source and target layouts must have matching frames/used length')
    if not 0 < layout.used_len <= length:
        raise ValueError('Invalid used sequence length')
    frames = layout.num_frames
    ss, se = source.video_start, source.video_end
    ts, te = layout.video_start, layout.video_end
    sf, tf = source.tokens_per_frame, layout.tokens_per_frame
    text_start, text_end = layout.text_start, layout.text_start + layout.text_len
    used = layout.used_len
    logical_queries = None
    qlen = length
    if active_indices is not None:
        if active_indices.ndim != 1 or active_indices.dtype != torch.long:
            raise ValueError('active_indices must be a one-dimensional int64 tensor')
        if active_indices.device != permutation.device:
            raise ValueError('active indices and permutation must share a device')
        qlen = active_indices.numel()
        if not qlen or bool(((active_indices < 0) | (active_indices >= length)).any()):
            raise ValueError('Active query indices must be nonempty and in bounds')
        # Arbitrary active subsets cannot be replaced with a regular formula.
        # Collapse active->physical->logical to one cached query lookup.
        logical_queries = (active_indices.remainder(rows) * world_size
                           + active_indices.div(rows, rounding_mode='floor'))

    def mask(batch, head, qi, ki):
        if logical_queries is None:
            q = (qi % rows) * world_size + qi // rows
        else:
            q = logical_queries[qi.clamp(min=0, max=qlen-1)]
        k = (ki % rows) * world_size + ki // rows
        q_s, k_s = (q >= ss) & (q < se), (k >= ss) & (k < se)
        q_t, k_t = (q >= ts) & (q < te), (k >= ts) & (k < te)
        q_video, k_video = q_s | q_t, k_s | k_t
        q_text = (q >= text_start) & (q < text_end)
        k_text = (k >= text_start) & (k < text_end)
        qsf = ((q-ss)//sf).clamp(min=0, max=frames-1)
        ksf = ((k-ss)//sf).clamp(min=0, max=frames-1)
        qtf = ((q-ts)//tf).clamp(min=0, max=frames-1)
        ktf = ((k-ts)//tf).clamp(min=0, max=frames-1)
        # EXACT checkpoint c5/r1 bounds, not a +/-1 latent-frame window.
        ss_local = ((ksf >= (qsf//5-1)*5) & (ksf <= (qsf//5+2)*5-1))
        tt_local = ((ktf >= (qtf//5-1)*5) & (ktf <= (qtf//5+2)*5-1))
        ss_local = ss_local | (ksf == 0) | (ksf == frames-1) | (qsf == 0) | (qsf == frames-1)
        tt_local = tt_local | (ktf == 0) | (ktf == frames-1) | (qtf == 0) | (qtf == frames-1)
        video_allowed = ((q_s & k_s & ss_local) | (q_t & k_t & tt_local)
                         | (q_t & k_s & (qtf == ksf)))
        ordinary_or_video = (~(q_video & k_video)) | video_allowed
        target_side = k_t | ((~k_video) & (~k_text))
        isolated = ordinary_or_video & (~((q_s | q_text) & target_side))
        real = (q < used) & (k < used)
        padding = (q >= used) & (q == k)
        valid = (qi >= 0) & (qi < qlen) & (ki >= 0) & (ki < length)
        return valid & ((real & isolated) | padding)

    return mask
