"""Exact target attention: merge equal-key groups and use contiguous KV views.

No changes to weights, connectivity, RoPE, selector or cross-step KV freshness.
The frozen target-LSE implementation is the reference, not edited in place.
"""
import torch
import savie_target_lse_candidate as reference


def pairs(t, s, n, m, active=None):
    qids, shared, same = reference.pairs(t, s, n, m, active)
    # All anchor query frames have exactly the same TT/condition keys. Also
    # merge any small-video edge chunks whose key sets happen to coincide.
    merged = []
    for qi, ki in shared:
        for index, (previous_q, previous_k) in enumerate(merged):
            if torch.equal(ki, previous_k):
                merged[index] = (torch.cat((previous_q, qi)), previous_k)
                break
        else:
            merged.append((qi, ki))
    return qids, merged, same


def prepare(t, s, n, m, active=None, group_size=8):
    qids, shared, same = pairs(t, s, n, m, active)
    output_index = torch.full((n,), -1, device=qids.device, dtype=torch.long)
    output_index[qids] = torch.arange(qids.numel(), device=qids.device)

    def pack(groups):
        batches = []
        for begin in range(0, len(groups), group_size):
            part = groups[begin:begin + group_size]
            qi = torch.cat([q for q, _ in part])
            ki = torch.cat([k for _, k in part])
            oi = output_index[qi]
            qlens = [q.numel() for q, _ in part]
            klens = [k.numel() for _, k in part]
            cq = torch.tensor([0] + qlens, device=qids.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
            ck = torch.tensor([0] + klens, device=qids.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
            # Metadata is computed once per cached plan, not per layer/step.
            start = int(ki[0])
            contiguous = torch.equal(ki, torch.arange(start, start + ki.numel(), device=ki.device))
            ordered = torch.equal(oi, torch.arange(qids.numel(), device=oi.device))
            batches.append((m.logical_to_physical[qi], ki, oi, cq, ck, max(qlens), max(klens),
                            (start, ki.numel()) if contiguous else None, ordered))
        return batches

    return m.logical_to_physical[qids], pack(shared), pack(same)


def fill(result, q, k, v, plan, scale):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    if reference._MERGE is None:
        reference._MERGE = torch.compile(reference.merge_outputs, fullgraph=True, dynamic=True)
    qids, shared, same = plan
    if not qids.numel():
        return result

    def run(batches):
        out = lse = None
        for qi, ki, oi, cq, ck, mq, mk, span, ordered in batches:
            if span is None:
                kk, vv = k.index_select(0, ki), v.index_select(0, ki)
            else:
                kk, vv = k.narrow(0, *span), v.narrow(0, *span)
            partial, normalizer = flash_attn_varlen_func(
                q=q.index_select(0, qi), k=kk, v=vv,
                cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=mq, max_seqlen_k=mk,
                softmax_scale=scale, causal=False, return_softmax_lse=True, fa_version=2)
            assert normalizer.shape == (q.shape[1], qi.numel())
            if len(batches) == 1 and ordered:
                return partial, normalizer.transpose(0, 1)
            if out is None:
                out = q.new_empty((qids.numel(),) + q.shape[1:])
                lse = torch.empty(qids.numel(), q.shape[1], device=q.device, dtype=torch.float32)
            out.index_copy_(0, oi, partial)
            lse.index_copy_(0, oi, normalizer.transpose(0, 1))
        return out, lse

    a, la = run(shared)
    b, lb = run(same)
    result.index_copy_(0, qids, reference._MERGE(a, b, la, lb))
    return result


def install(module):
    import savie_target_query_flash as target
    import os
    selected_size = int(os.environ.get('SAVIE_KV_REUSE_GROUP_SIZE', '64'))
    def configured_prepare(t, s, n, m, active=None, group_size=8):
        return prepare(t, s, n, m, active, selected_size)
    target.prepare_target_batches = configured_prepare
    target.fill_target_queries = fill
    print('SAVIE_TARGET_KV_REUSE_INSTALLED group_size=' + str(selected_size), flush=True)
