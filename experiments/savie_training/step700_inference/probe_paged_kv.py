"""Isolated capability/correctness probe; no serving modifications."""
import json
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func as flash

torch.manual_seed(4101)
with torch.inference_mode():
    for page_size in (16, 32, 64, 128, 256):
        try:
            q = torch.randn(17, 4, 128, device='cuda', dtype=torch.bfloat16)
            k, v = [torch.randn(4, page_size, 4, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            table = torch.tensor([[0, 1, 2], [0, 1, 3]], device='cuda', dtype=torch.int32)
            lengths = torch.tensor([3 * page_size - 7, 3 * page_size - 11], device='cuda', dtype=torch.int32)
            cq = torch.tensor([0, 8, 17], device='cuda', dtype=torch.int32)
            ck = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
            kk = torch.cat((k[table[0]].flatten(0, 1)[:int(lengths[0])], k[table[1]].flatten(0, 1)[:int(lengths[1])]))
            vv = torch.cat((v[table[0]].flatten(0, 1)[:int(lengths[0])], v[table[1]].flatten(0, 1)[:int(lengths[1])]))
            expected = flash(q=q, k=kk, v=vv, cu_seqlens_q=cq, cu_seqlens_k=ck,
                             max_seqlen_q=9, max_seqlen_k=3 * page_size, fa_version=2)
            actual = flash(q=q, k=k, v=v, cu_seqlens_q=cq, seqused_k=lengths, block_table=table,
                           max_seqlen_q=9, max_seqlen_k=3 * page_size, fa_version=2)
            torch.testing.assert_close(actual, expected, atol=.003, rtol=.02)
            print(json.dumps(dict(page_size=page_size, supported=True, exact=bool(torch.equal(actual, expected)))), flush=True)
        except (RuntimeError, AssertionError) as exc:
            print(json.dumps(dict(page_size=page_size, supported=False, error=str(exc)[:350])), flush=True)
