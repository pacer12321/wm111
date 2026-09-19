"""Isolated all-softmax-path KV bakeoff. No checkpoint or live server edits."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import savie_kv_pipeline as candidate
from savie_mask_arithmetic import make_arithmetic_mask
from savie_source_query_flash import prepare_source_batches
from savie_target_kv_reuse import prepare, fill

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--selector',type=Path,required=True);p.add_argument('--repeats',type=int,default=6)
p.add_argument('--source-skip',action='store_true')
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(4101)
torch._dynamo.config.recompile_limit=128
torch._dynamo.config.accumulated_recompile_limit=512
flex=torch.compile(flex_attention,fullgraph=True,dynamic=True)
records=[]

_CU={}
def fill_source_queries(result,q,k,v,batches,scale):
    # Identical CUDA _fusion_attention_tnd work, isolated from service/API imports.
    # A100 uses the same vLLM FA2 backend on both sides of this microbenchmark.
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    for qi,ki,qc,kc in batches:
        qq=q.index_select(0,qi);kk=k.index_select(0,ki);vv=v.index_select(0,ki)
        key=(tuple(qc),tuple(kc),str(q.device))
        if key not in _CU:
            qb=(0,*qc);kb=(0,*kc)
            _CU[key]=(torch.tensor(qb,dtype=torch.int32,device=q.device),
                torch.tensor(kb,dtype=torch.int32,device=q.device),
                max(b-a for a,b in zip(qb,qb[1:])),max(b-a for a,b in zip(kb,kb[1:])))
        cq,ck,mq,mk=_CU[key]
        part=flash_attn_varlen_func(q=qq.contiguous(),k=kk.contiguous(),v=vv.contiguous(),
            cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_q=mq,max_seqlen_k=mk,
            softmax_scale=scale,causal=False,fa_version=2)
        result.index_copy_(0,qi,part[0] if isinstance(part,tuple) else part)

def emit(**record):
    records.append(record)
    (args.output/'results.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)

def layouts(frames,per,text,audio,pad,world=2):
    used=text+2*frames*per+audio;n=((used+pad+world-1)//world)*world
    common=dict(num_frames=frames,tokens_per_frame=per,used_len=used,text_start=0,text_len=text)
    s=SimpleNamespace(video_start=text,video_end=text+frames*per,**common)
    t=SimpleNamespace(video_start=text+frames*per+audio,video_end=used,**common)
    ids=torch.arange(n,device='cuda');perm=(ids%(n//world))*world+ids//(n//world)
    m=SimpleNamespace(physical_to_logical=perm,logical_to_physical=torch.argsort(perm),world_size=world)
    return t,s,n,m

variants=[candidate.Options(physical_kv=a,fused_pack=b,source_split=c)
          for a,b,c in itertools.product((False,True),repeat=3)]

@torch.inference_mode()
def small():
    count=0;start=time.monotonic()
    for frames,world in ((1,2),(12,2),(37,4)):
        t,s,n,m=layouts(frames,3,7,3,5,world)
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        ids=torch.arange(n,device='cuda');permutation=m.physical_to_logical
        target=(permutation>=t.video_start)&(permutation<t.video_end)
        mask=make_arithmetic_mask(t,s,n,permutation,world_size=world)
        dense_mask=mask(0,0,ids[:,None],ids[None,:])
        dense=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None].float(),
            k.transpose(0,1)[None].float(),v.transpose(0,1)[None].float(),attn_mask=dense_mask[None,None])[0].transpose(0,1)
        modes=[('full',None),('partial',~target|(permutation%3==0)),('all_target_stable',~target)]
        if args.source_skip:
            src=(permutation>=s.video_start)&(permutation<s.video_end)
            modes += [('S_whole',(~target|(permutation%3==0)) & ~src),
                      ('S_selective',(~target|(permutation%3==0)) & (~src|(permutation%8==0))),
                      ('all_video_stable',~target & ~src)]
        for mode,keep in modes:
            expected=dense.clone()
            if keep is not None:expected[~keep]=0
            for options in variants:
                plan=candidate.build_plan(t,s,n,m,keep,options)
                actual=candidate.execute(q,k,v,plan,32**-.5,flex)
                rel=float((actual.float()-expected).norm()/expected.norm())
                torch.testing.assert_close(actual.float(),expected,atol=.008,rtol=.035)
                assert rel<.006,rel
                if keep is not None:assert bool((actual[~keep]==0).all())
                if keep is not None:
                    kk=k.clone();vv=v.clone();kk[target]*=50;vv[target]*=-50
                    changed=candidate.execute(q,kk,vv,plan,32**-.5,flex)
                    context=permutation<s.video_end
                    torch.testing.assert_close(actual[context],changed[context],rtol=0,atol=0)
                count+=1
        emit(stage='small_shape_passed',frames=frames,world=world,cumulative_cases=count,
             seconds=time.monotonic()-start)
    emit(stage='correctness_gate_passed',cases=count,reference='independent dense FP32 attention',
         source_target_isolation=True,includes_single_frame_padding_and_all_stable=True)

def baseline_plan(t,s,n,m,keep):
    ids=torch.arange(n,device='cuda');inv=m.logical_to_physical
    live=torch.ones(n,device='cuda',dtype=torch.bool) if keep is None else keep[inv]
    source=prepare_source_batches(t,s,n,m,group_size=4)
    if keep is not None:
        filtered=[]
        for qi,ki,qc,kc in source:
            qp=[];kp=[];newq=[];newk=[];q0=k0=0
            for q1,k1 in zip(qc,kc):
                selected=qi[q0:q1];selected=selected[keep[selected]]
                if selected.numel():
                    qp.append(selected);kp.append(ki[k0:k1])
                    newq.append((newq[-1] if newq else 0)+selected.numel())
                    newk.append((newk[-1] if newk else 0)+k1-k0)
                q0,k0=q1,k1
            if qp:filtered.append((torch.cat(qp),torch.cat(kp),newq,newk))
        source=filtered
    target=prepare(t,s,n,m,keep,group_size=64)
    other=inv[ids[(ids>=s.video_end)&(ids<t.video_start)&live]]
    padding=inv[ids[(ids>=t.used_len)&live]]
    used=t.used_len;nq=other.numel()
    def mask(b,h,q,k):return (q<nq)&(k<used)
    bm=create_block_mask(mask,None,None,nq,n,device='cuda',_compile=True)
    def run(q,k,v):
        kk=k.index_select(0,inv);vv=v.index_select(0,inv);result=torch.zeros_like(q)
        fill_source_queries(result,q,kk,vv,source,q.shape[-1]**-.5)
        fill(result,q,kk,vv,target,q.shape[-1]**-.5)
        part=flex(q[other].transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],
                  block_mask=bm,scale=q.shape[-1]**-.5)[0].transpose(0,1)
        result.index_copy_(0,other,part)
        result.index_copy_(0,padding,v.index_select(0,padding))
        return result
    return run

def timed(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();start=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
    return out,a.elapsed_time(b),(time.perf_counter()-start)*1000

@torch.inference_mode()
def full():
    t,s,n,m=layouts(37,1008,6159,250,215)
    q,k,v=[torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    payload=torch.load(args.selector,map_location='cpu',weights_only=True)
    assert payload['selector_checkpoint_step']==1000
    logical=torch.ones(n,device='cuda',dtype=torch.bool)
    logical[t.video_start:t.video_end]=payload['active_target_mask'].cuda()
    modes=[('refresh',None),('skip',logical[m.physical_to_logical])]
    if args.source_skip:
        src=(m.physical_to_logical>=s.video_start)&(m.physical_to_logical<s.video_end)
        modes=[('S_whole',logical[m.physical_to_logical] & ~src),
               ('S_selective_proxy_12p5',logical[m.physical_to_logical] & (~src|(m.physical_to_logical%8==0)))]
    for mode,keep in modes:
        base=baseline_plan(t,s,n,m,keep);old=lambda:base(q,k,v)
        reference=old()
        for options in variants:
            start=time.monotonic();plan=candidate.build_plan(t,s,n,m,keep,options)
            new=lambda:candidate.execute(q,k,v,plan,128**-.5,flex)
            actual=new();rel=float((actual.float()-reference.float()).norm()/reference.float().norm())
            assert rel<.006,rel
            torch.testing.assert_close(actual,reference,atol=.008,rtol=.035)
            del actual
            for _ in range(2):old();new()
            samples={k:[] for k in ('old_gpu','new_gpu','old_wall','new_wall')}
            for repeat in range(args.repeats):
                pairs=[('old',old),('new',new)]
                if repeat%2:pairs.reverse()
                for name,fn in pairs:
                    output,gpu,wall=timed(fn);del output
                    samples[name+'_gpu'].append(gpu);samples[name+'_wall'].append(wall)
            med={k:statistics.median(vv) for k,vv in samples.items()}
            emit(stage='real_shape_timing',mode=mode,options=asdict(options),relative_l2=rel,
                median_ms=med,speed_reduction_pct=100*(1-med['new_wall']/med['old_wall']),
                samples=samples,seconds=time.monotonic()-start,
                scope='all softmax branches incl fresh KV reorder/pack/kernel/merge/scatter; excludes one-time indices/compile, QKV projection, linear, Ulysses and whole-model runtime')
    emit(stage='completed',end_to_end_measured=False,weights_changed=False,connectivity_changed=False)

emit(stage='started',device=torch.cuda.get_device_name(0),torch=torch.__version__)
small();full()
