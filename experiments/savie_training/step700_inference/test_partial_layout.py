"""Regression gate for post-Ulysses packing, without editing the server code."""
import hashlib
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
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'loader'),str(ROOT/'candidate'),str(ROOT/'repo')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
import torch
from audit_contract import module
import savie_overlay as old
import savie_partial_layout as new
op=module('partial_test_layout',ROOT/'candidate/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py')

def layouts(frames=12,per=12,text=5,audio=7,pad=3,world=2,device='cpu'):
    used=text+2*frames*per+audio
    n=math.ceil((used+pad)/world)*world
    s=op.OpenVDNLayout(used,text,frames,per,1,per,0,text)
    t=op.OpenVDNLayout(used,text+frames*per+audio,frames,per,1,per,0,text)
    p=torch.cat([torch.arange(rank,n,world,device=device) for rank in range(world)])
    return t,s,n,SimpleNamespace(physical_to_logical=p,logical_to_physical=torch.argsort(p),world_size=world)

def active_cases(t,n,m):
    p=m.physical_to_logical
    target=(p>=t.video_start)&(p<t.video_end)
    keep=(~target)|((p%3)==0)
    changed=keep.clone()
    yes=torch.nonzero(target&keep).flatten()
    no=torch.nonzero(target&~keep).flatten()
    if yes.numel() and no.numel():
        changed[yes[0]]=False
        changed[no[-1]]=True
    return [None,keep,changed,torch.ones(n,device=p.device,dtype=torch.bool)]

def timed(fn):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin=time.perf_counter()
    start.record()
    out=fn()
    end.record()
    end.synchronize()
    return out,(time.perf_counter()-begin)*1000,start.elapsed_time(end)

torch.set_num_threads(4)
torch.manual_seed(4101)
records=[]
def emit(test,**data):
    row=dict(test=test,**data)
    records.append(row)
    print(json.dumps(row),flush=True)
    (HERE/'partial_layout_tests.json').write_text(json.dumps(records,indent=2))

@torch.inference_mode()
def small_tests():
    cases=0
    for world in (2,4):
        t,s,n,m=layouts(frames=12,per=12,text=5,audio=7,pad=3,world=world,device='cuda')
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        self=SimpleNamespace(softmax_scale=32**-.5)
        masks=active_cases(t,n,m)[1:]
        all_stable=torch.ones(n,device='cuda',dtype=torch.bool)
        all_stable[(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)]=False
        masks.extend([all_stable,torch.zeros(n,device='cuda',dtype=torch.bool)])
        for keep in masks:
            expected=old.savie_softmax(self,q,k,v,t,s,keep,m)
            out=new.partial_softmax(q,k,v,t,s,keep,m,old._FLEX,self.softmax_scale)
            torch.testing.assert_close(out,expected,rtol=.015,atol=.008)
            assert bool((out[~keep]==0).all())
            ids=torch.nonzero(keep).flatten()
            if ids.numel():
                # Independently verify every logical/physical mask pair.
                plan=new._PLANS[(t,s,n,str(q.device),world)][3]
                logical_q=plan['logical_queries']
                original=old.make_mask(t,s,n,q.device,m.physical_to_logical,plan['physical_queries'],world_size=world)
                original_mask=original(0,0,torch.arange(ids.numel(),device='cuda')[:,None],m.logical_to_physical[None,:])
                new_mask=plan['predicate'](0,0,torch.arange(ids.numel(),device='cuda')[:,None],torch.arange(n,device='cuda')[None,:])
                assert torch.equal(original_mask,new_mask)
                physical_q=plan['physical_queries']
                ref=torch.nn.functional.scaled_dot_product_attention(q[physical_q].transpose(0,1)[None].float(),
                    k[m.logical_to_physical].transpose(0,1)[None].float(),v[m.logical_to_physical].transpose(0,1)[None].float(),
                    attn_mask=new_mask[None,None]).to(q.dtype)[0].transpose(0,1)
                torch.testing.assert_close(out[physical_q],ref,rtol=.015,atol=.008)
                changed=(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)
                kk,vv=k.clone(),v.clone()
                kk[changed]=torch.randn_like(kk[changed])*50
                vv[changed]=torch.randn_like(vv[changed])*50
                other=new.partial_softmax(q,kk,vv,t,s,keep,m,old._FLEX,self.softmax_scale)
                context=m.physical_to_logical<s.video_end
                torch.testing.assert_close(out[context],other[context],rtol=0,atol=0)
            cases+=1
    emit('small_regression',cases=cases,world_sizes=[2,4],mask_exact=True,
         stable_queries_not_computed=True,source_text_target_isolation='exact',
         cache_active_set_changes=True,padding=True,bf16_atol=.008,bf16_rtol=.015)
    try:
        new.build_plan(t,s,n,m.physical_to_logical,torch.arange(n,device='cuda'),masks[0],world)
    except ValueError:
        emit('invalid_inverse_rejected',passed=True)
    else:
        raise AssertionError('Invalid inverse accepted')

@torch.inference_mode()
def production_tests():
    t,s,n,m=layouts(frames=37,per=1008,text=6159,audio=250,pad=23,device='cuda')
    q,k,v=[torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    self=SimpleNamespace(softmax_scale=128**-.5)
    for name,path in [('step700',ROOT/'selector.pt'),('old_B_active_ratio',ROOT.parent/'dmd8_b_skip_20260917/selector_dmd8_latent/fixed_selector_payload.pt')]:
        active=torch.load(path,map_location='cpu',weights_only=True)['active_target_mask'].to('cuda').bool()
        logical=torch.ones(n,device='cuda',dtype=torch.bool)
        logical[t.video_start:t.video_end]=active
        keep=logical[m.physical_to_logical]
        before=lambda:old.savie_softmax(self,q,k,v,t,s,keep,m)
        after=lambda:new.partial_softmax(q,k,v,t,s,keep,m,old._FLEX,self.softmax_scale)
        for _ in range(2):
            a,b=before(),after()
        rel=float((a.float()-b.float()).norm()/a.float().norm())
        assert rel<.003,(name,rel)
        torch.testing.assert_close(a,b,rtol=.015,atol=.002)
        samples={'before':[],'after':[]}
        for repeat in range(5):
            funcs=[('before',before),('after',after)]
            if repeat%2:funcs.reverse()
            for label,fn in funcs:
                _,ms,_=timed(fn)
                samples[label].append(ms)
        old_ms,new_ms=[statistics.median(samples[x]) for x in ('before','after')]
        emit('production_speed_and_equivalence',case=name,old_ms=old_ms,new_ms=new_ms,
             reduction=1-new_ms/old_ms,relative_l2=rel,max_abs=float((a-b).abs().max()),samples=samples,
             scope='including index planning cache lookup + Q/K/V pack + Flex + scatter; excludes compile',
             extra_communication=False,active_mask_unchanged=True)
        assert new_ms<old_ms*.95,(name,'less than five percent improvement')
    gate=dict(passed=True,checkpoint_step=700,token_ownership='unchanged interleaved',
        sha256=hashlib.sha256((HERE/'savie_partial_layout.py').read_bytes()).hexdigest(),
        actual_checkpoint_activations=False,bf16_rounding_allowed=True)
    (HERE/'partial_layout_passed.json').write_text(json.dumps(gate,indent=2))
    emit('ALL_PASSED',**gate)

if __name__=='__main__':
    small_tests()
    production_tests()
