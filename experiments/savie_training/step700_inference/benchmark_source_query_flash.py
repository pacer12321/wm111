"""Keep SS/text connectivity EXACT; test static query groups with FA2.

Not Top-K and not a new mask. Query rows with identical allowed key sets
share one FlashAttention document. Packing is included in all speed results.
"""
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'partial_fix_candidate'),str(ROOT),str(ROOT/'loader'),str(ROOT/'candidate'),str(ROOT/'repo')]
from test_partial_layout import layouts,timed,op
from savie_grouped_queries import build_plan
from torch.nn.attention.flex_attention import flex_attention

torch.manual_seed(4101)
torch.set_num_threads(4)
torch._dynamo.config.recompile_limit=32
torch._dynamo.config.accumulated_recompile_limit=128
records=[]


def static_groups(t,s,n,m):
    device=m.physical_to_logical.device
    ids=torch.arange(n,device=device)
    qs=(ids>=s.video_start)&(ids<s.video_end)
    txt=(ids>=t.text_start)&(ids<t.text_start+t.text_len)
    sf=(ids-s.video_start)//s.tokens_per_frame
    anchor=qs&((sf==0)|(sf==s.num_frames-1))
    pairs=[(ids[txt|anchor],ids[txt|qs])]
    for chunk in range((s.num_frames+4)//5):
        qids=ids[qs&(~anchor)&(sf//5==chunk)]
        if not qids.numel():continue
        local=(sf>=(chunk-1)*5)&(sf<(chunk+2)*5)
        keys=ids[txt|(qs&(local|(sf==0)|(sf==s.num_frames-1)))]
        pairs.append((qids,keys))
    # Query partition and exact agreement with grouped Flex's legal mask.
    allq=torch.cat([q for q,k in pairs])
    assert allq.unique().numel()==int((txt|qs).sum())==allq.numel()
    return pairs


def prepare_batches(pairs,m,size):
    batches=[]
    for begin in range(0,len(pairs),size):
        current=pairs[begin:begin+size]
        qidx=torch.cat([q for q,k in current])
        kidx=torch.cat([k for q,k in current])
        qc=[];kc=[];qn=kn=0
        for q,k in current:
            qn+=q.numel();kn+=k.numel();qc.append(qn);kc.append(kn)
        batches.append((m.logical_to_physical[qidx],kidx,qc,kc))
    return batches


def source_flash(q,k_logical,v_logical,batches,scale):
    out=torch.zeros_like(q)
    for qidx,kidx,qc,kc in batches:
        part=op._fusion_attention_tnd(q[qidx],k_logical[kidx],v_logical[kidx],qc,kc,scale)
        out.index_copy_(0,qidx,part)
    return out


@torch.inference_mode()
def test(frames,per,text,audio,heads,dim,full):
    t,s,n,m=layouts(frames=frames,per=per,text=text,audio=audio,pad=215 if full else 3,device='cuda')
    pairs=static_groups(t,s,n,m)
    keep=torch.ones(n,device='cuda',dtype=torch.bool)
    groups,_=build_plan(t,s,n,m,keep)
    groups=[g for g in groups if g['kind'] in ('source','text')]
    if not full:
        for qids,kids in pairs:
            qf=(qids-s.video_start)//per
            kf=(kids-s.video_start)//per
            for qid in qids.tolist():
                group=next(g for g in groups if bool((g['logical_queries']==qid).any()))
                qi=torch.nonzero(group['logical_queries']==qid).flatten()[0]
                legal=group['predicate'](0,0,qi,torch.arange(n,device='cuda'))
                assert torch.equal(torch.nonzero(legal).flatten(),kids)
    q,k,v=[torch.randn(n,heads,dim,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    # KV logical packing is identical in both paths, outside this subpath test.
    kk,vv=k[m.logical_to_physical],v[m.logical_to_physical]
    scale=dim**-.5
    flex=torch.compile(flex_attention,dynamic=True,fullgraph=True)
    def reference():
        out=torch.zeros_like(q)
        for g in groups:
            ids=g['physical_queries']
            part=flex(q[ids].transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],
                      block_mask=g['block_mask'],scale=scale)[0].transpose(0,1)
            out.index_copy_(0,ids,part)
        return out
    expected=reference()
    for size in (2,4,8) if full else (4,):
        batches=prepare_batches(pairs,m,size)
        candidate=lambda:source_flash(q,kk,vv,batches,scale)
        actual=candidate()
        rel=float((actual.float()-expected.float()).norm()/expected.float().norm())
        assert rel<.004,rel
        torch.testing.assert_close(actual,expected,rtol=.02,atol=.005)
        result=dict(real_shape=full,n=n,heads=heads,dim=dim,batch_groups=size,relative_l2=rel,
                    scope='S/text query subpath only; synthetic activations; packing+kernel+scatter included; common KV reorder excluded')
        if full:
            for _ in range(2):reference();candidate()
            samples={'flex':[],'flash':[]}
            for repeat in range(5):
                funcs=[('flex',reference),('flash',candidate)]
                if repeat%2:funcs.reverse()
                for label,fn in funcs:samples[label].append(timed(fn)[1])
            result.update(median_ms={k:statistics.median(v) for k,v in samples.items()},samples_ms=samples)
        records.append(result)
        (HERE/'source_query_flash_results.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    test(7,8,7,3,4,32,False)
    test(12,8,7,3,4,32,False)
    test(37,1008,6159,250,28,128,True)
