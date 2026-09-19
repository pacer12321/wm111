"""Regression + warmed production-size benchmark against untouched old overlay."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'candidate'),str(ROOT/'repo'),str(ROOT)]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
import torch
from torch.nn.attention.flex_attention import create_block_mask
from audit_contract import module

old=module('flex_before',ROOT/'loader/savie_overlay.py')
new=module('flex_after',HERE/'savie_overlay.py')
op=module('flex_test_layout',ROOT/'candidate/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py')
torch.set_num_threads(4)
torch.manual_seed(4101)
results=[]


def emit(name,**kw):
    record=dict(test=name,**kw)
    results.append(record)
    print(json.dumps(record),flush=True)
    (HERE/'flex_fix_tests.json').write_text(json.dumps(results,indent=2))


def layouts(frames=12,per=12,text=5,audio=7,pad=3,world=2,device='cpu'):
    used=text+2*frames*per+audio
    length=math.ceil((used+pad)/world)*world
    source=op.OpenVDNLayout(used,text,frames,per,1,per,0,text)
    target=op.OpenVDNLayout(used,text+frames*per+audio,frames,per,1,per,0,text)
    p=torch.cat([torch.arange(rank,length,world,device=device) for rank in range(world)])
    return target,source,length,SimpleNamespace(physical_to_logical=p,
                  logical_to_physical=torch.argsort(p),world_size=world)


def active_cases(layout,length,mapping):
    p=mapping.physical_to_logical
    target=(p>=layout.video_start)&(p<layout.video_end)
    keep=(~target)|((p%3)==0)
    changed=keep.clone()
    yes=torch.nonzero(target&keep).flatten()
    no=torch.nonzero(target&(~keep)).flatten()
    if yes.numel() and no.numel():
        changed[yes[0]]=False
        changed[no[-1]]=True
    return [None,keep,changed,torch.ones(length,device=p.device,dtype=torch.bool)]


def cpu_tests():
    count=0
    for frames in (3,5,6,7,12,37):
        for world in (2,4):
            t,s,n,m=layouts(frames=frames,per=4,world=world)
            for keep in active_cases(t,n,m):
                a=torch.arange(n) if keep is None else keep.nonzero().flatten()
                before=old.make_mask(t,s,n,'cpu',m.physical_to_logical,a)
                after=new.make_mask(t,s,n,'cpu',m.physical_to_logical,
                                    None if keep is None else a,world_size=world)
                qi=torch.arange(a.numel()+7)[:,None]
                ki=torch.arange(n+9)[None,:]
                assert torch.equal(before(0,0,qi,ki),after(0,0,qi,ki)),(frames,world,keep)
                count+=1
    t,s,n,m=layouts()
    # Fail closed rather than silently assuming that any layout is interleaved.
    try:
        new.make_mask(t,s,n,'cpu',torch.arange(n))
    except ValueError:
        pass
    else:
        raise AssertionError('invalid permutation accepted')
    emit('cpu_mask_equivalence',cases=count,extra_out_of_bounds_indices=True,
         cross_chunk_boundaries=True,padding=True,text_source_target_boundaries=True,
         invalid_mapping_rejected=True)


@torch.inference_mode()
def gpu_tests():
    t,s,n,m=layouts(frames=12,per=12,text=5,audio=7,pad=3,device='cuda')
    q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    self=SimpleNamespace(softmax_scale=32**-.5)
    for case,keep in enumerate(active_cases(t,n,m)):
        a=old.savie_softmax(self,q,k,v,t,s,keep,m)
        b=new.savie_softmax(self,q,k,v,t,s,keep,m)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ids=torch.arange(n,device='cuda') if keep is None else keep.nonzero().flatten()
        pred=old.make_mask(t,s,n,'cuda',m.physical_to_logical,ids)
        dense=pred(0,0,torch.arange(ids.numel(),device='cuda')[:,None],torch.arange(n,device='cuda')[None,:])
        ref=torch.nn.functional.scaled_dot_product_attention(q[ids].transpose(0,1)[None].float(),
            k.transpose(0,1)[None].float(),v.transpose(0,1)[None].float(),attn_mask=dense[None,None]).to(q.dtype)
        torch.testing.assert_close(b[ids].transpose(0,1)[None],ref,rtol=.02,atol=.02)
        changed=(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)
        k2,v2=k.clone(),v.clone()
        k2[changed]=torch.randn_like(k2[changed])*50
        v2[changed]=torch.randn_like(v2[changed])*50
        c=new.savie_softmax(self,q,k2,v2,t,s,keep,m)
        context=(m.physical_to_logical<s.video_end)
        torch.testing.assert_close(b[context],c[context],rtol=0,atol=0)
        emit('compiled_gpu_equivalence',case=case,queries=ids.numel(),max_abs=float((a-b).abs().max()),
             sdpa_max_abs=float((b[ids].transpose(0,1)[None]-ref).abs().max()),source_text_target_isolation='exact')
    empty=torch.zeros(n,device='cuda',dtype=torch.bool)
    assert torch.equal(new.savie_softmax(self,q,k,v,t,s,empty,m),torch.zeros_like(q))
    emit('empty_active_queries',safe_zero_output=True)


def timed(fn):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall=time.perf_counter()
    start.record()
    out=fn()
    end.record()
    end.synchronize()
    return out,(time.perf_counter()-wall)*1000,start.elapsed_time(end)


@torch.inference_mode()
def full_size():
    t,s,n,m=layouts(frames=37,per=1008,text=6159,audio=250,pad=23,device='cuda')
    q,k,v=[torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    self=SimpleNamespace(softmax_scale=128**-.5)
    for label,payload_path in [('no_skip',None),
          ('partial_step700',ROOT/'selector.pt'),
          ('partial_old_B_ratio',ROOT.parent/'dmd8_b_skip_20260917/selector_dmd8_latent/fixed_selector_payload.pt')]:
        keep=None
        if payload_path:
            payload=torch.load(payload_path,map_location='cpu',weights_only=True)
            mask=payload['active_target_mask'].to('cuda').bool()
            assert mask.numel()==t.num_frames*t.tokens_per_frame
            logical=torch.ones(n,device='cuda',dtype=torch.bool)
            logical[t.video_start:t.video_end]=mask
            keep=logical[m.physical_to_logical]
        before=lambda:old.savie_softmax(self,q,k,v,t,s,keep,m)
        after=lambda:new.savie_softmax(self,q,k,v,t,s,keep,m)
        for _ in range(2):
            a=before();b=after()
        torch.cuda.synchronize()
        max_abs=float((a-b).abs().max())
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        old_ms,new_ms,old_gpu,new_gpu=[],[],[],[]
        for rep in range(5):
            order=[('old',before),('new',after)] if rep%2==0 else [('new',after),('old',before)]
            for name,fn in order:
                _,wall,gpu=timed(fn)
                (old_ms if name=='old' else new_ms).append(wall)
                (old_gpu if name=='old' else new_gpu).append(gpu)
        o,r=statistics.median(old_ms),statistics.median(new_ms)
        emit('production_size_speed',variant=label,shape=list(q.shape),
             active_target_ratio=1.0 if keep is None else float(mask.float().mean()),
             old_wall_ms=o,new_wall_ms=r,latency_reduction=1-r/o,
             old_samples_ms=old_ms,new_samples_ms=new_ms,old_gpu_ms=old_gpu,new_gpu_ms=new_gpu,
             max_abs=max_abs,scope='softmax path including mask lookup, gather and scatter; excludes compilation')
        assert r<o*.95,(label,'less than five percent improvement',o,r)
    emit('peak_memory',gib=torch.cuda.max_memory_allocated()/2**30)


if __name__=='__main__':
    cpu_tests()
    gpu_tests()
    full_size()
    manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
              for name in ['savie_overlay.py','savie_mask_arithmetic.py','test_flex_arithmetic.py']}
    (HERE/'flex_fix_passed.json').write_text(json.dumps(dict(status='passed',files=manifest,
        baseline_sha256=hashlib.sha256((ROOT/'loader/savie_overlay.py').read_bytes()).hexdigest(),
        token_ownership='unchanged interleaved',checkpoint_step=700),indent=2))
    emit('ALL_PASSED',files=manifest)
