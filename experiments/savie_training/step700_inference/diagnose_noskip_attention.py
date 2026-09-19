"""Diagnostic only: same-size production softmax paths, no skip, no model edits.

Synthetic Q/K/V isolate execution cost; not an eight-step generation measurement.
Both paths use one Ulysses rank's 28 heads from the actual 56-head H3 model.
"""
import ast
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/cache/zhonghao/h3')
RUN = ROOT / 'savie_step700_eval'
PORT = RUN / 'candidate/vllm_omni/diffusion/models/minimax_h3'
sys.path[:0] = [str(RUN), str(RUN/'loader'), str(RUN/'candidate'), str(RUN/'repo')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT', None)
import torch
from audit_contract import module
from savie_overlay import make_mask, savie_softmax, training_mask_predicate
import savie_overlay as overlay
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from src.models.sequence_layout import DualStreamSequenceLayout

torch.set_num_threads(4)
torch.manual_seed(4101)
RESULTS = []


def emit(kind, **data):
    obj = dict(kind=kind, **data)
    RESULTS.append(obj)
    print(json.dumps(obj), flush=True)
    Path(__file__).with_suffix('.results.json').write_text(json.dumps(RESULTS,indent=2))


def measure(name, fn, repeats=5):
    for _ in range(2):
        y = fn()
    torch.cuda.synchronize()
    wall, device = [], []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        begin.record()
        y = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-t)*1000)
        device.append(begin.elapsed_time(end))
    emit('timing', name=name, median_wall_ms=statistics.median(wall),
         median_device_ms=statistics.median(device), wall_samples_ms=wall,
         device_samples_ms=device, repeats=repeats)
    return y


def block_stats(name, bm):
    partial = int(bm.kv_num_blocks.sum())
    full = int(bm.full_kv_num_blocks.sum()) if bm.full_kv_num_blocks is not None else 0
    all_blocks = math.ceil(bm.shape[-2]/128)*math.ceil(bm.shape[-1]/128)
    emit('block_mask', name=name, shape=list(bm.shape), partial_blocks=partial,
         full_blocks=full, visited_blocks=partial+full,
         visited_fraction=(partial+full)/all_blocks,
         partial_of_visited=partial/max(1, partial+full))


def profile(name, fn):
    from torch.profiler import profile as prof, ProfilerActivity
    with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as p:
        fn()
        torch.cuda.synchronize()
    rows=[]
    for x in p.key_averages():
        # Exclude duplicate raw CUDA kernel records from operator attribution.
        if str(x.device_type) != 'DeviceType.CPU':
            continue
        ms=getattr(x, 'self_device_time_total', 0)/1000
        if ms:
            rows.append(dict(op=x.key, count=x.count, self_device_ms=ms,
                             cpu_ms=x.self_cpu_time_total/1000))
    rows.sort(key=lambda x:x['self_device_ms'],reverse=True)
    emit('profile', name=name, top_ops=rows[:18], total_self_device_ms=sum(x['self_device_ms'] for x in rows))


@torch.inference_mode()
def main():
    op=module('diag_openvdn',PORT/'openvdn_npu.py')
    tree=ast.parse((PORT/'minimax_h3_transformer.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_interleaved_openvdn_softmax_attention')
    ns=dict(op.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PORT/'minimax_h3_transformer.py'),'exec'),ns)
    baseline_fn=ns['_interleaved_openvdn_softmax_attention']
    text,frames,height,width,audio=6159,37,24,42,250
    per=height*width
    source_start=text
    target_start=text+frames*per+audio
    used=target_start+frames*per
    length=math.ceil(used/128)*128
    heads,dim=28,128
    layout=op.OpenVDNLayout(used,target_start,frames,per,height,width,0,text)
    source=op.OpenVDNLayout(used,source_start,frames,per,height,width,0,text)
    device=torch.device('cuda',0)
    perm=torch.cat([torch.arange(r,length,2,device=device) for r in range(2)])
    inv=torch.argsort(perm)
    mapping=SimpleNamespace(physical_to_logical=perm,logical_to_physical=inv)
    attn=SimpleNamespace(softmax_scale=dim**-.5)
    q,k,v=[torch.randn(length,heads,dim,device=device,dtype=torch.bfloat16) for _ in range(3)]
    emit('contract', scope='production softmax only; synthetic QKV; no selector or token skip; no weight loading',
         torch=torch.__version__,gpu=torch.cuda.get_device_name(),heads_per_rank=heads,
         qkv_shape=list(q.shape),used=used,text=text,source_tokens=frames*per,target_tokens=frames*per,
         interleaved_world=2,window_bounds=op.window_bounds(frames),
         source_geometry=[frames,height,width],groups_per_call=4)
    dense_q,_,groups=op._softmax_plan(layout,device)
    baseline_pairs=int(dense_q.numel())*used+sum(int(qi.numel())*int(ki.numel()) for qi,ki in groups)
    emit('baseline_plan',dense_q=int(dense_q.numel()),group_sizes=[[int(qi.numel()),int(ki.numel())] for qi,ki in groups],
         allowed_pairs=baseline_pairs,allowed_fraction=baseline_pairs/(used*used))
    baseline=lambda:baseline_fn(q,k,v,layout,dim**-.5,inv,None,groups_per_call=4)
    measure('B_DMD8_FA_varlen_no_skip',baseline)
    profile('B_DMD8_FA_varlen_no_skip',baseline)
    sparse=lambda:savie_softmax(attn,q,k,v,layout,source,None,mapping)
    t=time.perf_counter()
    a=sparse()
    torch.cuda.synchronize()
    emit('compile_and_mask_once',seconds=time.perf_counter()-t)
    bm=next(iter(overlay._MASKS.values()))[1]
    block_stats('SAViE_current_interleaved',bm)
    measure('SAViE_current_Flex_no_skip',sparse)
    profile('SAViE_current_Flex_no_skip',sparse)

    # Semantic-preserving diagnostic: restore original logical order for softmax.
    # Not deployed. Reports packing/scattering cost as well as the core kernel.
    dual=DualStreamSequenceLayout(seq_len=used,source_start=source_start,source_frames=frames,
        source_tokens_per_frame=per,target_start=target_start,target_frames=frames,
        target_tokens_per_frame=per,source_height=height,source_width=width,
        target_height=height,target_width=width,text_start=0,text_len=text)
    inner=training_mask_predicate()(dual,op.window_bounds(frames),device,'both')
    def logical_pred(b,h,qi,ki):
        return ((qi<used)&(ki<used)&inner(b,h,qi,ki))|((qi>=used)&(qi==ki)&(qi<length)&(ki<length))
    logical_bm=create_block_mask(logical_pred,B=None,H=None,Q_LEN=length,KV_LEN=length,
                                device=device,BLOCK_SIZE=128,_compile=True)
    block_stats('SAViE_logical_order',logical_bm)
    ql,kl,vl=[z.index_select(0,inv) for z in (q,k,v)]
    flex=torch.compile(flex_attention,dynamic=True)
    def core():
        return flex(ql.transpose(0,1)[None],kl.transpose(0,1)[None],vl.transpose(0,1)[None],
                    block_mask=logical_bm,scale=dim**-.5)[0].transpose(0,1)
    out=measure('SAViE_logical_order_kernel_only',core)
    b=out.index_select(0,perm)
    emit('same_math_layout_check',max_abs=float((a-b).abs().max()),
         relative_l2=float((a.float()-b.float()).norm()/a.float().norm()))
    def reordered():
        rq,rk,rv=[z.index_select(0,inv) for z in (q,k,v)]
        out=flex(rq.transpose(0,1)[None],rk.transpose(0,1)[None],rv.transpose(0,1)[None],
                 block_mask=logical_bm,scale=dim**-.5)[0].transpose(0,1)
        return out.index_select(0,perm)
    measure('SAViE_logical_order_including_reorder',reordered)
    profile('SAViE_logical_order_including_reorder',reordered)
    # Keep physical odd/even order exactly unchanged, but express the
    # permutation arithmetically instead of loading three index tensors.
    half=length//2
    def arithmetic_pred(batch,head,qi,ki):
        logical_q=(qi%half)*2+qi//half
        logical_k=(ki%half)*2+ki//half
        return ((qi<length)&(ki<length)&(logical_q<used)&(logical_k<used)&inner(batch,head,logical_q,logical_k))|((qi<length)&(ki<length)&(logical_q>=used)&(logical_q==logical_k))
    arithmetic_bm=create_block_mask(arithmetic_pred,B=None,H=None,Q_LEN=length,KV_LEN=length,
                                   device=device,BLOCK_SIZE=128,_compile=True)
    block_stats('SAViE_same_physical_order_arithmetic_mask',arithmetic_bm)
    arithmetic=lambda:flex(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],
                           block_mask=arithmetic_bm,scale=dim**-.5)[0].transpose(0,1)
    ar=measure('SAViE_same_physical_order_arithmetic_mask',arithmetic)
    emit('same_math_arithmetic_check',max_abs=float((a-ar).abs().max()),
         relative_l2=float((a.float()-ar.float()).norm()/a.float().norm()))
    profile('SAViE_same_physical_order_arithmetic_mask',arithmetic)
    emit('complete',peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    Path(__file__).with_suffix('.results.json').write_text(json.dumps(RESULTS,indent=2))


if __name__=='__main__':
    main()
