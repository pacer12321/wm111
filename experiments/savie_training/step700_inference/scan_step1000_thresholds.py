"""Retain selector score/pooling/dilation; vary only scalar threshold.

Targets refer to stable TARGET rows AFTER the unchanged 3x3 dilation.
Ratios are speed-probe configurations, not calibrated quality thresholds.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch
import torch.nn.functional as F

ROOT=Path('/cache/zhonghao/h3')
STUDY=ROOT/'savie_step1000_eval/threshold_study'
HERE=Path(__file__).resolve().parent
torch.set_num_threads(4)


def observed_sequence_rows():
    profile=STUDY/'block_profile/rank0.jsonl'
    rows=[json.loads(line) for line in profile.read_text().splitlines() if line.strip()]
    assert rows and rows[-1]['interleaved_sp']
    return 2*int(rows[-1]['local_sequence_rows'])


def mask_for(score,shape,threshold):
    b,c,t,h,w=shape
    grid=score.reshape(b,1,t,h//2,w//2)
    return F.max_pool3d((grid>=threshold).float(),(1,3,3),1,(0,1,1)).bool().flatten()


def make_masks():
    payload=torch.load(STUDY/'selector_original.pt',map_location='cpu',weights_only=True)
    assert payload['selector_checkpoint_step']==1000
    score=payload['selector_score'].float()
    shape=tuple(payload['source_shape'])
    threshold=float(payload['selector_threshold'])
    original=mask_for(score,shape,threshold)
    assert torch.equal(original,payload['active_target_mask']), 'Original selector reproduction failed'
    records=[]
    candidates=[('original',None,threshold,original)]
    original_stable=float((~original).float().mean())
    # Probe increased stable fractions without removing spatial protection.
    for target in (.50,.65,.75,.85):
        if target<=original_stable:continue
        lo,hi=threshold,float(score.max())+1e-5
        for _ in range(40):
            mid=(lo+hi)/2
            ratio=float((~mask_for(score,shape,mid)).float().mean())
            if ratio<target:lo=mid
            else:hi=mid
        chosen=min((lo,hi),key=lambda x:abs(float((~mask_for(score,shape,x)).float().mean())-target))
        candidates.append((f'stable{int(target*100)}',target,chosen,mask_for(score,shape,chosen)))
    previous=original
    for name,target,threshold,active in candidates:
        assert not bool((active&(~previous)).any()),'Increasing threshold must not reactivate rows'
        previous=active
        path=STUDY/f'{name}_payload.pt'
        assert not path.exists(),f'Refusing to overwrite {path}'
        out=dict(payload,active_target_mask=active,selector_threshold=threshold,
                 threshold_experiment=name,quality_validated=False)
        torch.save(out,path)
        total=torch.ones(observed_sequence_rows(),dtype=torch.bool)
        assert active.numel()==37296
        total[43705:81001]=active
        row=dict(name=name,desired_stable_target_fraction=target,threshold=threshold,
                 active_target_fraction=float(active.float().mean()),
                 stable_target_fraction=float((~active).float().mean()),
                 active_full_sequence_fraction=float(total.float().mean()),
                 per_rank_active=[int(total[r::2].sum()) for r in (0,1)],
                 additionally_stable_vs_original=int((original&(~active)).sum()),
                 mask_path=str(path),quality_validated=False)
        records.append(row)
        print(json.dumps(row),flush=True)
    report=dict(checkpoint_step=1000,seed=4101,original_exact_reproduction=True,
                score_origin='OWN step1000 first-x0; not old B or step700',
                unchanged=['normalized latent difference','2x2 average pooling','3x3 active dilation','odd/even SP'],
                records=records)
    (STUDY/'threshold_scan.json').write_text(json.dumps(report,indent=2))
    return report


@torch.inference_mode()
def benchmark(report):
    old=ROOT/'savie_step700_eval'
    sys.path[:0]=[str(old/'partial_fix_candidate'),str(old),str(old/'loader'),str(old/'candidate'),str(old/'repo')]
    from test_partial_layout import layouts,timed
    from savie_grouped_queries import grouped_softmax
    from torch.nn.attention.flex_attention import flex_attention
    torch.manual_seed(4101)
    torch._dynamo.config.recompile_limit=64
    torch._dynamo.config.accumulated_recompile_limit=256
    flex=torch.compile(flex_attention,dynamic=True,fullgraph=True)
    total=observed_sequence_rows()
    used=6159+2*37*1008+250
    t,s,n,m=layouts(frames=37,per=1008,text=6159,audio=250,pad=total-used,device='cuda')
    assert n==total
    self=SimpleNamespace(softmax_scale=128**-.5)
    q,k,v=[torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    results=[]
    for row in [dict(name='skip_off',mask_path=None)]+report['records']:
        keep=None
        if row['mask_path']:
            active=torch.load(row['mask_path'],map_location='cpu',weights_only=True)['active_target_mask'].to('cuda')
            logical=torch.ones(n,device='cuda',dtype=torch.bool)
            logical[t.video_start:t.video_end]=active
            keep=logical[m.physical_to_logical]
        call=lambda:grouped_softmax(self,q,k,v,t,s,keep,m,flex)
        for _ in range(3):output=call()
        assert bool(torch.isfinite(output).all())
        if keep is not None:assert bool((output[~keep]==0).all())
        samples=[timed(call)[1] for _ in range(7)]
        result=dict(name=row['name'],median_ms=statistics.median(samples),samples_ms=samples,
                    active_q=n if keep is None else int(keep.sum()),
                    scope='Grouped softmax including pack/scatter; synthetic QKV, own1000 masks; NOT DiT/E2E timing')
        results.append(result)
        (STUDY/'threshold_kernel_times.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    report=(json.loads((STUDY/'threshold_scan.json').read_text())
            if (STUDY/'threshold_scan.json').exists() else make_masks())
    # Refresh only ancillary whole-sequence counts if earlier diagnostics used
    # a different padding length. Target masks/thresholds remain untouched.
    report['observed_global_sequence_rows']=observed_sequence_rows()
    for row in report['records']:
        active=torch.load(row['mask_path'],map_location='cpu',weights_only=True)['active_target_mask']
        whole=torch.ones(report['observed_global_sequence_rows'],dtype=torch.bool)
        whole[43705:81001]=active
        row['active_full_sequence_fraction']=float(whole.float().mean())
        row['per_rank_active']=[int(whole[r::2].sum()) for r in (0,1)]
    (STUDY/'threshold_scan.json').write_text(json.dumps(report,indent=2))
    if args.benchmark:benchmark(report)
