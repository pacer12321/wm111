"""Offline candidate: pack each distinct KV page once, share by block table.

Pages contain current layer/current step K/V, NOT a cross-step value cache.
Only integer page geometry is cached. All original attention edges remain.
"""
import torch
import savie_target_kv_reuse as reuse
import savie_target_lse_candidate as reference


def prepare(t, s, n, m, active=None, page_size=16):
    qids, shared, same = reuse.pairs(t, s, n, m, active)
    output_index = torch.full((n,), -1, device=qids.device, dtype=torch.long)
    output_index[qids] = torch.arange(qids.numel(), device=qids.device)
    page_ids = {}
    pages = []
    def branch(groups):
        if not groups:
            return None
        tables = []
        qlens, klens = [], []
        for qs, ks in groups:
            rows = ks.cpu().tolist()
            table = []
            for start in range(0, len(rows), page_size):
                page = tuple(rows[start:start + page_size])
                page = page + (-1,) * (page_size - len(page))
                if page not in page_ids:
                    page_ids[page] = len(pages)
                    pages.append(page)
                table.append(page_ids[page])
            tables.append(table)
            qlens.append(qs.numel()); klens.append(len(rows))
        table = torch.zeros(len(groups), max(map(len, tables)), dtype=torch.int32, device=qids.device)
        for i, values in enumerate(tables):
            table[i, :len(values)] = torch.tensor(values, dtype=torch.int32, device=qids.device)
        qi = torch.cat([qs for qs, _ in groups]); oi = output_index[qi]
        cq = torch.tensor([0] + qlens, device=qids.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        lengths = torch.tensor(klens, device=qids.device, dtype=torch.int32)
        ordered = torch.equal(oi, torch.arange(qids.numel(), device=qids.device))
        return m.logical_to_physical[qi], oi, cq, lengths, table, max(qlens), max(klens), ordered
    a, b = branch(shared), branch(same)
    indices = torch.tensor(pages, device=qids.device, dtype=torch.long).reshape(-1).clamp_min(0)
    return m.logical_to_physical[qids], indices, page_size, a, b


def fill(result, q, k, v, plan, scale):
    from vllm.vllm_flash_attn import flash_attn_varlen_func as flash
    qids, indices, page_size, shared, same = plan
    if not qids.numel():
        return result
    if reference._MERGE is None:
        reference._MERGE = torch.compile(reference.merge_outputs, fullgraph=True, dynamic=True)
    kk = k.index_select(0, indices).view(-1, page_size, *k.shape[1:])
    vv = v.index_select(0, indices).view(-1, page_size, *v.shape[1:])
    def run(branch):
        qi, oi, cq, lengths, table, mq, mk, ordered = branch
        out, lse = flash(q=q.index_select(0, qi), k=kk, v=vv, cu_seqlens_q=cq,
                         seqused_k=lengths, block_table=table, max_seqlen_q=mq,
                         max_seqlen_k=mk, softmax_scale=scale, causal=False,
                         return_softmax_lse=True, fa_version=2)
        if ordered:
            return out, lse.transpose(0, 1)
        return (torch.empty_like(out).index_copy_(0, oi, out),
                torch.empty(qids.numel(), q.shape[1], device=q.device, dtype=torch.float32)
                .index_copy_(0, oi, lse.transpose(0, 1)))
    a, la = run(shared); b, lb = run(same)
    return result.index_copy_(0, qids, reference._MERGE(a, b, la, lb))
