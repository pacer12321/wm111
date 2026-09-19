"""Consolidated inference-only KV transport candidates; unchanged attention graph.

No weights, selectors, schedules, RoPE or collectives are changed. Plans cache
indices only; every invocation consumes this layer/step's real Q/K/V.
"""
from collections import OrderedDict
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import create_block_mask
from savie_target_kv_reuse import pairs as target_pairs


@dataclass(frozen=True)
class Options:
    physical_kv: bool = False
    fused_pack: bool = False
    source_split: bool = False
    source_batch: int = 64
    fused_merge: bool = True


@triton.jit
def _pack(Q, K, V, QI, KI, OQ, OK, OV, NQ: tl.constexpr, NK: tl.constexpr,
          WIDTH: tl.constexpr, QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
          TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    qr, kr, col = i // WIDTH, i // WIDTH, i % WIDTH
    qi = tl.load(QI + qr, qr < NQ, other=0)
    ki = tl.load(KI + kr, kr < NK, other=0)
    a = tl.load(Q + qi * QS + col, qr < NQ, other=0)
    b = tl.load(K + ki * KS + col, kr < NK, other=0)
    c = tl.load(V + ki * VS + col, kr < NK, other=0)
    tl.store(OQ + i, a, qr < NQ)
    tl.store(OK + i, b, kr < NK)
    tl.store(OV + i, c, kr < NK)


@triton.jit
def _merge_scatter(A, B, LA, LB, AI, BI, QI, OUT, N: tl.constexpr,
                   H: tl.constexpr, D: tl.constexpr, LAS: tl.constexpr,
                   LAH: tl.constexpr, LBS: tl.constexpr, LBH: tl.constexpr,
                   TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row, col = i // (H * D), i % (H * D)
    head = col // D
    ar = tl.load(AI + row, row < N, other=0)
    br = tl.load(BI + row, row < N, other=0)
    dst = tl.load(QI + row, row < N, other=0)
    la = tl.load(LA + ar * LAS + head * LAH, row < N, other=0)
    lb = tl.load(LB + br * LBS + head * LBH, row < N, other=0)
    a = tl.load(A + ar * H * D + col, row < N, other=0).to(tl.float32)
    b = tl.load(B + br * H * D + col, row < N, other=0).to(tl.float32)
    w = 1.0 / (1.0 + tl.exp(lb - la))
    tl.store(OUT + dst * H * D + col, a * w + b * (1.0 - w), row < N)


def source_pairs(t, s, n, device, split, logical_keep=None):
    ids = torch.arange(n, device=device)
    src = (ids >= s.video_start) & (ids < s.video_end)
    text = (ids >= t.text_start) & (ids < t.text_start + t.text_len)
    frame = (ids - s.video_start) // s.tokens_per_frame
    anchor = src & ((frame == 0) | (frame == s.num_frames - 1))
    interior = src & ~anchor
    live = torch.ones_like(src) if logical_keep is None else logical_keep
    primary = [(ids[(text | anchor) & live], ids[text | src])]
    local_groups = []
    for chunk in range((s.num_frames + 4) // 5):
        qi = ids[interior & (frame // 5 == chunk) & live]
        if not qi.numel():
            continue
        local = (frame >= (chunk - 1) * 5) & (frame < (chunk + 2) * 5)
        if split:
            local_groups.append((qi, ids[interior & local]))
        else:
            primary.append((qi, ids[text | anchor | (src & local)]))
    if not split or not bool((interior & live).any()):
        return primary, None
    queries = ids[interior & live]
    return primary, (queries, [(queries, ids[text | anchor])], local_groups)


def _plan_batches(pairs, inv, physical, group_size):
    # Drop empty QUERY groups only. Cached S rows remain real K/V consumers.
    pairs = [(q, k) for q, k in pairs if q.numel()]
    batches = []
    for begin in range(0, len(pairs), group_size):
        part = pairs[begin:begin + group_size]
        qi = torch.cat([q for q, _ in part]); logical_ki = torch.cat([k for _, k in part])
        ki = inv[logical_ki] if physical else logical_ki
        qlens = [q.numel() for q, _ in part]; klens = [k.numel() for _, k in part]
        cq = torch.tensor([0] + qlens, device=inv.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        ck = torch.tensor([0] + klens, device=inv.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        start = int(ki[0])
        contiguous = torch.equal(ki, torch.arange(start, start + ki.numel(), device=ki.device))
        batches.append(dict(q=inv[qi], k=ki, cq=cq, ck=ck, mq=max(qlens), mk=max(klens),
                            view=(start, ki.numel()) if contiguous else None))
    return batches


def _split_plan(qids, a, b, m, options):
    if not qids.numel():
        return None
    def branch(groups):
        qi = torch.cat([q for q, _ in groups])
        assert qi.numel() == qids.numel() and torch.equal(qi.sort().values, qids)
        # qids is sorted. Each attention branch covers it exactly once.
        order = torch.argsort(qi)
        batches = _plan_batches(groups, m.logical_to_physical, options.physical_kv, len(groups))
        assert len(batches) == 1
        return batches[0], order
    return m.logical_to_physical[qids], branch(a), branch(b)


def build_plan(t, s, n, m, keep, options):
    ids = torch.arange(n, device=m.physical_to_logical.device)
    p, inv = m.physical_to_logical, m.logical_to_physical
    rows = n // m.world_size
    assert n % m.world_size == 0
    assert torch.equal(p, (ids % rows) * m.world_size + ids // rows)
    assert torch.equal(p[inv], ids)
    if keep is not None:
        assert keep.dtype == torch.bool and keep.shape == (n,)
        text = (p >= t.text_start) & (p < t.text_start + t.text_len)
        assert bool(keep[text].all()), 'Global condition queries must stay live'
    logical_keep = torch.ones(n, device=p.device, dtype=torch.bool) if keep is None else keep[inv]
    primary, split = source_pairs(t, s, n, p.device, options.source_split, logical_keep)
    source = _plan_batches(primary, inv, options.physical_kv, options.source_batch)
    source_split = None if split is None else _split_plan(*split, m, options)
    if t.num_frames == 1:
        # First and last anchor coincide; never duplicate the query in a branch.
        tq = ids[(ids >= t.video_start) & (ids < t.video_end) & logical_keep]
        sk = (ids >= s.video_start) & (ids < s.video_end)
        parts = (tq, [(tq, ids[(ids < t.used_len) & ~sk])], [(tq, ids[sk])])
    else:
        parts = target_pairs(t, s, n, m, keep)
    target = _split_plan(*parts, m, options)
    source_rows = (ids >= s.video_start) & (ids < s.video_end)
    target_rows = (ids >= t.video_start) & (ids < t.video_end)
    text_rows = (ids >= t.text_start) & (ids < t.text_start + t.text_len)
    other = inv[ids[~source_rows & ~target_rows & ~text_rows & (ids < t.used_len) & logical_keep]]
    padding = inv[ids[(ids >= t.used_len) & logical_keep]]
    used, world = t.used_len, m.world_size
    nq = other.numel()
    def mask(b, h, qi, ki):
        logical = (ki % rows) * world + ki // rows if options.physical_kv else ki
        return (qi >= 0) & (qi < nq) & (ki >= 0) & (ki < n) & (logical < used)
    bm = None if not nq else create_block_mask(mask, B=None, H=None, Q_LEN=nq, KV_LEN=n, device=p.device, BLOCK_SIZE=128, _compile=True)
    padding_keys = padding if options.physical_kv else p[padding]
    return dict(source=source, source_split=source_split, target=target, other=other,
                other_mask=bm, padding=padding, padding_keys=padding_keys,
                inv=inv, options=options)


def run_batch(q, k, v, batch, scale, options, lse=False):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    qi, ki, span = batch['q'], batch['k'], batch['view']
    if options.fused_pack and span is None:
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        assert q.stride(-2) == k.stride(-2) == v.stride(-2) == q.shape[-1]
        qq = q.new_empty((qi.numel(),) + q.shape[1:])
        kk, vv = [q.new_empty((ki.numel(),) + q.shape[1:]) for _ in range(2)]
        width = q.shape[1] * q.shape[2]
        _pack[(triton.cdiv(max(qi.numel(), ki.numel()) * width, 1024),)](
            q, k, v, qi, ki, qq, kk, vv, qi.numel(), ki.numel(), width,
            q.stride(0), k.stride(0), v.stride(0), 1024)
    else:
        qq = q.index_select(0, qi)
        kk = k.index_select(0, ki) if span is None else k.narrow(0, *span)
        vv = v.index_select(0, ki) if span is None else v.narrow(0, *span)
    return flash_attn_varlen_func(q=qq, k=kk, v=vv, cu_seqlens_q=batch['cq'], cu_seqlens_k=batch['ck'],
                                 max_seqlen_q=batch['mq'], max_seqlen_k=batch['mk'],
                                 softmax_scale=scale, causal=False, fa_version=2, return_softmax_lse=lse)


def fill_split(result, q, k, v, plan, scale, options):
    if plan is None:
        return
    qi, (a, ai), (b, bi) = plan
    av, la = run_batch(q, k, v, a, scale, options, True)
    bv, lb = run_batch(q, k, v, b, scale, options, True)
    if options.fused_merge:
        _merge_scatter[(triton.cdiv(qi.numel() * q.shape[1] * q.shape[2], 1024),)](
            av, bv, la, lb, ai, bi, qi, result, qi.numel(), q.shape[1], q.shape[2],
            la.stride(1), la.stride(0), lb.stride(1), lb.stride(0), 1024)
    else:
        from savie_target_lse_candidate import merge_outputs
        out = merge_outputs(av[ai], bv[bi], la.transpose(0, 1)[ai], lb.transpose(0, 1)[bi])
        result.index_copy_(0, qi, out)


def execute(q, k, v, plan, scale, flex):
    if torch.is_grad_enabled():
        raise RuntimeError('KV pipeline candidate is inference-only')
    options = plan['options']
    if not options.physical_kv:
        k, v = k.index_select(0, plan['inv']), v.index_select(0, plan['inv'])
    result = torch.zeros_like(q)
    for batch in plan['source']:
        result.index_copy_(0, batch['q'], run_batch(q, k, v, batch, scale, options))
    fill_split(result, q, k, v, plan['source_split'], scale, options)
    fill_split(result, q, k, v, plan['target'], scale, options)
    if plan['other'].numel():
        qq = q.index_select(0, plan['other'])
        out = flex(qq.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                   block_mask=plan['other_mask'], scale=scale)[0].transpose(0, 1)
        result.index_copy_(0, plan['other'], out)
    if plan['padding'].numel():
        # Padding's single allowed key is itself in PHYSICAL order.
        result.index_copy_(0, plan['padding'], v.index_select(0, plan['padding_keys']))
    return result


_PLANS = OrderedDict()


def install(module):
    import json, os
    import savie_grouped_queries
    options = Options(**json.loads(os.environ['SAVIE_KV_PIPELINE_OPTIONS']))
    def grouped(self, q, k, v, t, s, active_mask, mapping, flex):
        key = (t, s, q.shape[0], str(q.device), mapping.world_size, active_mask is None, options)
        cached = _PLANS.get(key)
        matching = (cached is not None and cached[0] is mapping.physical_to_logical and
                    (active_mask is None or torch.equal(cached[1], active_mask)))
        if not matching:
            cached = (mapping.physical_to_logical, None if active_mask is None else active_mask.clone(),
                      build_plan(t, s, q.shape[0], mapping, active_mask, options))
            _PLANS[key] = cached
            if len(_PLANS) > 8: _PLANS.popitem(last=False)
        _PLANS.move_to_end(key)
        return execute(q, k, v, cached[2], self.softmax_scale, flex)
    savie_grouped_queries.grouped_softmax = grouped
    print('SAVIE_KV_PIPELINE_INSTALLED ' + json.dumps(options.__dict__, sort_keys=True), flush=True)
