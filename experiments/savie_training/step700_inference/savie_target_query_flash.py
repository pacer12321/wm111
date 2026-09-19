"""Candidate: exact per-frame T query grouping; no new sparsity or routing.

Every document normalizes jointly over legal TT, matching-frame S, and
global conditioning keys. No independently normalized branch summation.
Plans may be cached for an unchanged active mask, but K/V are always fresh.
"""
import torch


def target_pairs(t, s, n, mapping, active_mask=None):
    ids = torch.arange(n, device=mapping.physical_to_logical.device)
    source = (ids >= s.video_start) & (ids < s.video_end)
    target = (ids >= t.video_start) & (ids < t.video_end)
    valid = ids < t.used_len
    global_keys = valid & ~source & ~target
    tf = (ids - t.video_start) // t.tokens_per_frame
    sf = (ids - s.video_start) // s.tokens_per_frame
    keep = torch.ones(n, dtype=torch.bool, device=ids.device) if active_mask is None else active_mask[mapping.logical_to_physical]
    pairs = []
    for frame in range(t.num_frames):
        qids = ids[target & (tf == frame) & keep]
        if not qids.numel():
            continue
        local = (tf >= (frame // 5 - 1) * 5) & (tf < (frame // 5 + 2) * 5)
        local |= (tf == 0) | (tf == t.num_frames - 1)
        if frame in (0, t.num_frames - 1):
            local = torch.ones_like(local)
        keys = ids[global_keys | (target & local) | (source & (sf == frame))]
        pairs.append((qids, keys))
    return pairs


def prepare_target_batches(t, s, n, mapping, active_mask=None, group_size=4):
    if group_size < 1:
        raise ValueError('Positive batch size required')
    pairs = target_pairs(t, s, n, mapping, active_mask)
    batches = []
    for begin in range(0, len(pairs), group_size):
        current = pairs[begin:begin + group_size]
        qidx = torch.cat([q for q, _ in current])
        kidx = torch.cat([k for _, k in current])
        qc, kc, qn, kn = [], [], 0, 0
        for q, k in current:
            qn += q.numel()
            kn += k.numel()
            qc.append(qn)
            kc.append(kn)
        batches.append((mapping.logical_to_physical[qidx], kidx, qc, kc))
    return batches


def fill_target_queries(result, q, k_logical, v_logical, batches, scale):
    from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import _fusion_attention_tnd
    for qidx, kidx, qc, kc in batches:
        part = _fusion_attention_tnd(q.index_select(0, qidx),
                                    k_logical.index_select(0, kidx),
                                    v_logical.index_select(0, kidx), qc, kc, scale)
        result.index_copy_(0, qidx, part)
