"""Offline experiment only; never installed in inference from this script."""
import argparse,json,statistics
from pathlib import Path
import torch
from benchmark_target_query_flash import layouts,ROOT,HERE
from savie_target_query_flash import target_pairs,prepare_target_batches,fill_target_queries
from savie_target_lse_candidate import pairs,prepare,fill,merge_outputs

def cpu_test():
    count=0
    for world in (2,4):
     for frames in (3,5,7,12,37):
      t,s,n,m=layouts(frames,6,7,3,5,world)
      is_t=(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)
      for keep in (None,~is_t,~is_t|(m.physical_to_logical%3==0)):
        qids,a,b=pairs(t,s,n,m,keep)
        dictionaries=[]
        for part in (a,b,target_pairs(t,s,n,m,keep)):
            table={}
            for qs,ks in part:
                for qi in qs.tolist():
                    assert qi not in table
                    table[qi]=set(ks.tolist())
            dictionaries.append(table)
        da,db,reference=dictionaries
        assert set(da)==set(db)==set(reference)==set(qids.tolist())
        for qi in reference:
            assert not da[qi]&db[qi]
            assert da[qi]|db[qi]==reference[qi]
        count+=1
    for _ in range(10):
        score=torch.randn(7,3,19);value=torch.randn(3,19,5)
        expected=torch.einsum('qhk,hkd->qhd',score.softmax(-1),value)
        a=torch.einsum('qhk,hkd->qhd',score[:,:,:8].softmax(-1),value[:,:8])
        b=torch.einsum('qhk,hkd->qhd',score[:,:,8:].softmax(-1),value[:,8:])
        actual=merge_outputs(a,b,score[:,:,:8].logsumexp(-1),score[:,:,8:].logsumexp(-1))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    print(json.dumps(dict(cpu_mask_cases=count,exact_partition=True,joint_softmax_merge_verified=True)),flush=True)

@torch.inference_mode()
def gpu_test(full,records):
    t,s,n,m=layouts(37,1008,6159,250,215,device='cuda') if full else layouts(12,8,7,3,5,device='cuda')
    h,d=(28,128) if full else (4,32)
    q,k,v=[torch.randn(n,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    # Match baseline: common logical KV reorder is outside both measured paths.
    kk,vv=k.index_select(0,m.logical_to_physical),v.index_select(0,m.logical_to_physical)
    logical=torch.ones(n,device='cuda',dtype=torch.bool)
    if full:
        payload=torch.load(ROOT/'selector_adaptive_targetflash.pt',map_location='cpu',weights_only=True)
        assert payload['selector_checkpoint_step']==1000
        logical[t.video_start:t.video_end]=payload['active_target_mask'].cuda()
    else:logical[t.video_start:t.video_end]=torch.arange(t.video_end-t.video_start,device='cuda')%3==0
    for name,keep in [('full',None),('skip',logical[m.physical_to_logical])]:
        baseline=prepare_target_batches(t,s,n,m,keep,8)
        def old():
            result=torch.zeros_like(q);fill_target_queries(result,q,kk,vv,baseline,d**-.5);return result
        reference=old()
        for size in ((2,4,8) if full else (4,)):
            plan=prepare(t,s,n,m,keep,size)
            def new():return fill(torch.zeros_like(q),q,kk,vv,plan,d**-.5)
            actual=new();error=float((actual.float()-reference.float()).norm()/reference.float().norm())
            torch.testing.assert_close(actual,reference,atol=.008,rtol=.035)
            assert error<.006,error
            row=dict(real_shape=full,mode=name,group_size=size,relative_l2=error,
                old_packed_key_rows=sum(b[1].numel() for b in baseline),
                new_packed_key_rows=sum(b[1].numel() for part in plan[1:] for b in part),
                scope='Target attention with pack/scatter/compiled LSE merge; no whole-model speed or quality claim')
            if full:
                for _ in range(3):old();new()
                samples={'joint_flash':[],'shared_kv_lse':[]}
                for repeat in range(8):
                    order=[('joint_flash',old),('shared_kv_lse',new)]
                    if repeat%2:order.reverse()
                    for label,fn in order:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();result=fn();b.record();b.synchronize()
                        samples[label].append(a.elapsed_time(b));del result
                row.update(median_ms={k:statistics.median(v) for k,v in samples.items()},samples=samples)
            records.append(row);(HERE/'target_lse_results.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(row),flush=True)

p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');args=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(4101);cpu_test()
if not args.cpu_only:
    results=[];gpu_test(False,results);gpu_test(True,results)
