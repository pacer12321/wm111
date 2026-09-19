"""Isolated post-Ulysses diagnostics; never changes the deployed model.

Same active set, connectivity and odd/even ownership in every variant.
Synthetic QKV have the production shape. Packing costs are included.
"""
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/cache/zhonghao/h3/savie_step700_eval')
sys.path[:0] = [str(ROOT), str(ROOT/'loader'), str(ROOT/'candidate'), str(ROOT/'repo')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT', None)
import torch
from torch.nn.attention.flex_attention import create_block_mask
import diagnose_noskip_attention as diag
import savie_overlay as overlay
from savie_mask_arithmetic import make_arithmetic_mask
from audit_contract import module

RECORDS = []
def emit(kind, **data):
    row = dict(kind=kind, **data)
    RECORDS.append(row)
    print(json.dumps(row), flush=True)
    Path(__file__).with_suffix('.results.json').write_text(json.dumps(RECORDS, indent=2))
diag.emit = emit

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(4101)
    op = module('partial_layout', ROOT/'candidate/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py')
    frames, per, text, audio = 37, 1008, 6159, 250
    used = text + 2*frames*per + audio
    n = math.ceil(used/128)*128
    s = op.OpenVDNLayout(used, text, frames, per, 24, 42, 0, text)
    t = op.OpenVDNLayout(used, text+frames*per+audio, frames, per, 24, 42, 0, text)
    perm = torch.cat([torch.arange(r,n,2,device='cuda') for r in range(2)])
    inverse = torch.argsort(perm)
    mapping = SimpleNamespace(physical_to_logical=perm, logical_to_physical=inverse, world_size=2)
    q,k,v = [torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    scale = 128**-.5
    self = SimpleNamespace(softmax_scale=scale)
    active = torch.load(ROOT/'selector.pt',map_location='cpu',weights_only=True)['active_target_mask'].to('cuda').bool()
    logical_keep = torch.ones(n,device='cuda',dtype=torch.bool)
    logical_keep[t.video_start:t.video_end] = active
    physical_keep = logical_keep[perm]
    ids = torch.nonzero(physical_keep).flatten()
    emit('contract', shape=list(q.shape), target_active_ratio=float(active.float().mean()),
         total_queries=n, remaining_queries=ids.numel(), query_reduction=1-ids.numel()/n,
         actual_checkpoint_activations=False, token_ownership='unchanged odd/even', communication='none; post-all-to-all benchmark')
    full = lambda: overlay.savie_softmax(self,q,k,v,t,s,None,mapping)
    partial = lambda: overlay.savie_softmax(self,q,k,v,t,s,physical_keep,mapping)
    reference = diag.measure('current_partial_total',partial)
    full_out = diag.measure('current_full_total',full)
    emit('full_vs_partial_active_rows',max_abs=float((reference[ids]-full_out[ids]).abs().max()),
         relative_l2=float((reference[ids].float()-full_out[ids].float()).norm()/full_out[ids].float().norm()))
    for label, keep in [('full',None),('partial',physical_keep)]:
        key = (t,s,n,str(q.device),keep is None,2)
        bm = overlay._MASKS[key][1]
        qq = q if keep is None else q.index_select(0,ids)
        core = lambda: overlay._FLEX(qq.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],block_mask=bm,scale=scale)
        diag.block_stats(label,bm)
        diag.measure(label+'_kernel_only',core)
        diag.profile(label+'_total',full if keep is None else partial)
    # Sorting only Q retains K/V order and all SP ownership; reorder is local.
    logical_ids = torch.nonzero(logical_keep).flatten()
    physical_qids = inverse[logical_ids]
    for label, reorder_kv in [('logical_Q_only',False),('logical_QKV',True)]:
        logical_perm = torch.arange(n,device='cuda') if reorder_kv else perm
        query_ids = logical_ids if reorder_kv else physical_qids
        pred = make_arithmetic_mask(t,s,n,logical_perm,query_ids,1 if reorder_kv else 2)
        bm = create_block_mask(pred,B=None,H=None,Q_LEN=query_ids.numel(),KV_LEN=n,device='cuda',BLOCK_SIZE=128,_compile=True)
        diag.block_stats(label,bm)
        def packed_path():
            qq = q.index_select(0,physical_qids)
            kk,vv = (k.index_select(0,inverse),v.index_select(0,inverse)) if reorder_kv else (k,v)
            out = overlay._FLEX(qq.transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],block_mask=bm,scale=scale)[0].transpose(0,1)
            return torch.zeros_like(q).index_copy_(0,physical_qids,out)
        out = diag.measure(label+'_including_pack_scatter',packed_path)
        emit('equivalence',variant=label,max_abs=float((out-reference).abs().max()),
             relative_l2=float((out.float()-reference.float()).norm()/reference.float().norm()),
             stable_rows_zero=bool((out[~physical_keep]==0).all()))
        diag.profile(label+'_including_pack_scatter',packed_path)
    emit('complete',peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)

if __name__ == '__main__':
    main()
