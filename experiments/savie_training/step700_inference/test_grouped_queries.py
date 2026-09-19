"""Isolated gate for role-specialized partial attention; no server changes."""
import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'partial_fix_candidate'),str(ROOT),str(ROOT/'loader'),str(ROOT/'candidate'),str(ROOT/'repo')]
from test_partial_layout import layouts,active_cases,timed
import torch
import savie_overlay as baseline
import savie_grouped_queries as candidate
from savie_mask_arithmetic import make_arithmetic_mask

torch.set_num_threads(4)
torch.manual_seed(4101)
# This finite test matrix compiles several roles and toy/real shapes.
# Prevent its compile-cache limit from silently falling back to dense eager.
torch._dynamo.config.recompile_limit=64
torch._dynamo.config.accumulated_recompile_limit=256
records=[]
def emit(test,**data):
    row=dict(test=test,**data)
    records.append(row)
    print(json.dumps(row),flush=True)
    (HERE/'grouped_queries_results.json').write_text(json.dumps(records,indent=2))

def group_masks(t,s,n,m,keep):
    logical=torch.arange(n,device=keep.device)
    ids=logical[keep[m.logical_to_physical]]
    conditions=[('source',(ids>=s.video_start)&(ids<s.video_end)),
                ('target',(ids>=t.video_start)&(ids<t.video_end)),
                ('text',(ids>=t.text_start)&(ids<t.text_start+t.text_len))]
    covered=torch.zeros_like(ids,dtype=torch.bool)
    for _,condition in conditions:covered|=condition
    conditions.append(('other',(~covered)&(ids<t.used_len)))
    for kind,selected in conditions:
        qids=ids[selected]
        if not qids.numel():continue
        new=candidate.predicate_for(kind,qids,t,s,n)
        old=make_arithmetic_mask(t,s,n,logical,qids,world_size=1)
        qi=torch.arange(qids.numel()+3,device=keep.device)[:,None]
        ki=torch.arange(n+3,device=keep.device)[None,:]
        torch.testing.assert_close(new(0,0,qi,ki),old(0,0,qi,ki),rtol=0,atol=0)

def semantics():
    cases=0
    for world in (2,4):
        for frames in (1,6,12,17):
            t,s,n,m=layouts(frames=frames,per=5,text=7,audio=3,pad=7,world=world)
            masks=active_cases(t,n,m)[1:]
            masks.append(torch.zeros(n,dtype=torch.bool))
            for keep in masks:
                group_masks(t,s,n,m,keep)
                cases+=1
    emit('cpu_semantics',cases=cases,all_mask_pairs_exact=True,
         includes_invalid_block_tail=True,world_sizes=[2,4])

@torch.inference_mode()
def small_gpu():
    from torch.nn.attention.flex_attention import flex_attention
    flex=torch.compile(flex_attention,dynamic=True,fullgraph=True)
    self=SimpleNamespace(softmax_scale=32**-.5)
    tests=0
    for world in (2,4):
        t,s,n,m=layouts(frames=12,per=12,text=5,audio=7,pad=3,world=world,device='cuda')
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        for keep in active_cases(t,n,m):
            active=torch.ones(n,device='cuda',dtype=torch.bool) if keep is None else keep
            out=candidate.grouped_softmax(self,q,k,v,t,s,keep,m,flex)
            pred=baseline.make_mask(t,s,n,q.device,m.physical_to_logical,None,world_size=world)
            index=torch.arange(n,device='cuda')
            mask=pred(0,0,index[:,None],index[None,:])
            ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None].float(),
                k.transpose(0,1)[None].float(),v.transpose(0,1)[None].float(),attn_mask=mask[None,None])[0].transpose(0,1).to(q.dtype)
            ref[~active]=0
            torch.testing.assert_close(out,ref,rtol=.015,atol=.008)
            kk,vv=k.clone(),v.clone()
            target=(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)
            kk[target]=torch.randn_like(kk[target])*50
            vv[target]=torch.randn_like(vv[target])*50
            other=candidate.grouped_softmax(self,q,kk,vv,t,s,keep,m,flex)
            context=m.physical_to_logical<s.video_end
            torch.testing.assert_close(out[context],other[context],rtol=0,atol=0)
            assert bool((out[~active]==0).all())
            tests+=1
    emit('gpu_semantics',cases=tests,dense_reference=True,source_target_isolation='exact')
    return flex

@torch.inference_mode()
def production(flex):
    t,s,n,m=layouts(frames=37,per=1008,text=6159,audio=250,pad=23,device='cuda')
    self=SimpleNamespace(softmax_scale=128**-.5)
    q,k,v=[torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    masks=[('no_skip',None)]
    for name,path in [('step700',ROOT/'selector.pt'),
                      ('old_B_mask_diagnostic',ROOT.parent/'dmd8_b_skip_20260917/selector_dmd8_latent/fixed_selector_payload.pt')]:
        active=torch.load(path,map_location='cpu',weights_only=True)['active_target_mask'].to('cuda').bool()
        keep=torch.ones(n,device='cuda',dtype=torch.bool)
        keep[t.video_start:t.video_end]=active
        masks.append((name,keep[m.physical_to_logical]))
    for name,keep in masks:
        before=lambda:baseline.savie_softmax(self,q,k,v,t,s,keep,m)
        after=lambda:candidate.grouped_softmax(self,q,k,v,t,s,keep,m,flex)
        for _ in range(2):a,b=before(),after()
        rel=float((a.float()-b.float()).norm()/a.float().norm())
        assert rel<.003,(name,rel)
        torch.testing.assert_close(a,b,rtol=.015,atol=.002)
        samples={'baseline':[],'candidate':[]}
        for repeat in range(5):
            funcs=[('baseline',before),('candidate',after)]
            if repeat%2:funcs.reverse()
            for label,fn in funcs:
                _,ms,_=timed(fn)
                samples[label].append(ms)
        medians={key:statistics.median(value) for key,value in samples.items()}
        emit('production',case=name,**medians,samples=samples,relative_l2=rel,
             max_abs=float((a-b).abs().max()),total_Q=n,
             active_Q=n if keep is None else int(keep.sum()),
             scope='GPU idle, compile excluded, includes Q/K/V packing + all query groups + scatter; synthetic activations')
    emit('COMPLETE',deployment_authorized_by_test=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    semantics()
    if not args.cpu_only:
        production(small_gpu())
