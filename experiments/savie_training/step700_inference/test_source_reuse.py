import argparse,copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval');OLD=ROOT.parent/'savie_step700_eval'
sys.path[:0]=[str(ROOT/'source_reuse_loader'),str(ROOT/'speedops_candidate'),str(OLD/'partial_fix_candidate'),str(OLD),str(OLD/'loader'),str(OLD/'candidate'),str(OLD/'repo')]
from savie_source_reuse import Controller,choose_active,drift_score,reused_block,partial_source_batches
from test_active_post_block import inputs,module
from savie_step_token_plan import attach_step_plan
import savie_source_query_flash as source
source._source_reuse_original_prepare=source.prepare_source_batches
source.prepare_source_batches=partial_source_batches
from savie_target_kv_reuse import install as install_target
install_target(None)

@torch.inference_mode()
def cpu_tests():
    cases=0
    for mode in ('whole','selective'):
        c=Controller(mode)
        actual=[]
        for t in (0,.1,.2,.3,.4,.5,.6,.7):
            c.begin(t);actual.append(c.refresh)
        assert actual==[True,True,False,False,True,False,False,False]
        c.cache[0]=torch.ones(1);c.mask=torch.ones(1,dtype=torch.bool)
        c.begin(0);assert not c.cache and c.mask is None and c.step==1
    active,threshold=choose_active(torch.zeros(2*5*5),2,5,5)
    assert active.all() and threshold is None
    scores=torch.zeros(2,5,5);scores[0,2,2]=1
    active,threshold=choose_active(scores.flatten(),2,5,5)
    assert int(active.sum())==9 and not active.reshape(2,5,5)[1].any()
    a=torch.ones(6,8);assert torch.equal(drift_score(a,a),torch.zeros(6))
    assert torch.allclose(drift_score(a,2*a),torch.ones(6))
    for source_mode in ('none','half','all'):
        model,x,kw=inputs(35,32,'cpu',True)
        source_rows=torch.arange(3,15)
        keep_t=torch.ones(35,dtype=torch.bool);keep_t[20:30:2]=False
        kw.update(spotedit_active_mask=keep_t,spotedit_refresh=True)
        module.reference_forward(model,x,**kw)
        t_cache=model._spotedit_stable_output.clone()
        cached_source=torch.randn(12,32,dtype=x.dtype)
        stable_pos=torch.arange(12) if source_mode=='all' else torch.arange(0,12,2) if source_mode=='half' else torch.empty(0,dtype=torch.long)
        live=keep_t.clone();live[source_rows[stable_pos]]=False
        attach_step_plan(live,live,0)
        p=dict(live=live,stable_t=torch.nonzero(~keep_t).flatten(),source_rows=source_rows,
               stable_s=source_rows[stable_pos],stable_s_in_source=stable_pos)
        for _ in range(2):
            x=torch.randn_like(x);kw['spotedit_refresh']=False
            expected=module.reference_forward(model,x,**kw)
            expected[p['stable_s']]=cached_source[stable_pos]
            actual=reused_block(model,x,module=module,kwargs=kw,plan=p,source_cache=cached_source)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(model._spotedit_stable_output,t_cache,rtol=0,atol=0)
            cases+=1
    return cases

@torch.inference_mode()
def gpu_tests():
    from test_partial_layout import layouts
    from savie_grouped_queries import grouped_softmax
    from torch.nn.attention.flex_attention import flex_attention
    flex=torch.compile(flex_attention,dynamic=True,fullgraph=True)
    os.environ.update(SAVIE_SOURCE_QUERY_FLASH='1',SAVIE_TARGET_QUERY_FLASH='1')
    count=0
    for world in (2,4):
        t,s,n,m=layouts(frames=12,per=12,text=5,audio=7,pad=3,world=world,device='cuda')
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        obj=SimpleNamespace(softmax_scale=32**-.5)
        ids=m.physical_to_logical
        is_s=(ids>=s.video_start)&(ids<s.video_end)
        is_t=(ids>=t.video_start)&(ids<t.video_end)
        for t_partial in (False,True):
            keep_t=~is_t | (ids%3==0) if t_partial else torch.ones_like(ids,dtype=torch.bool)
            expected=grouped_softmax(obj,q,k,v,t,s,keep_t,m,flex)
            for source_mode in ('none','half','all'):
                keep=keep_t.clone()
                if source_mode=='half': keep[is_s & (ids%2==0)]=False
                if source_mode=='all': keep[is_s]=False
                actual=grouped_softmax(obj,q,k,v,t,s,keep,m,flex)
                wanted=expected.clone();wanted[~keep]=0
                torch.testing.assert_close(actual,wanted,rtol=.025,atol=.006)
                assert bool((actual[~keep]==0).all())
                # Changing target KV never leaks into active source/context output.
                k2,v2=k.clone(),v.clone();k2[is_t]=torch.randn_like(k2[is_t])*30;v2[is_t]=torch.randn_like(v2[is_t])*30
                changed=grouped_softmax(obj,q,k2,v2,t,s,keep,m,flex)
                context=(is_s|((ids>=t.text_start)&(ids<t.text_start+t.text_len)))&keep
                torch.testing.assert_close(actual[context],changed[context],rtol=0,atol=0)
                count+=1
    return count

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(4101)
    torch._dynamo.config.recompile_limit=64;torch._dynamo.config.accumulated_recompile_limit=256
    cpu=cpu_tests();print(json.dumps(dict(cpu_cases=cpu,passed=True)),flush=True)
    if not args.cpu_only:
        gpu=gpu_tests()
        path=ROOT/'source_reuse_candidate/prepared.json';gate=json.loads(path.read_text())
        gate.update(integration_passed=True,source_reuse_cpu_cases=cpu,source_reuse_gpu_cases=gpu,
                    t_cache_unchanged=True,source_kv_not_removed=True,source_target_isolation_exact=True)
        path.write_text(json.dumps(gate,indent=2));print(json.dumps(gate),flush=True)
