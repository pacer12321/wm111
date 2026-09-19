"""GEMM/gather/scatter microbenchmark only; does NOT change live inference.

Exact dependency: skipped SOURCE queries are consumed by neither softmax nor
TT linear. All TARGET raw Q (including skipped T) and ALL K/V stay current.
Both split variants include allocation, row gathering and reconstruction.
"""
import argparse,json,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F

def split_projection(x,w,live,stable,h,mode):
    if mode=='column_split':
        kv=F.linear(x,w[h:])
        q=x.new_zeros((x.shape[0],h))
        if live.numel():q.index_copy_(0,live,F.linear(x.index_select(0,live),w[:h]))
        return torch.cat((q,kv),dim=-1)
    if mode=='row_split':
        out=x.new_zeros((x.shape[0],3*h))
        if live.numel():out.index_copy_(0,live,F.linear(x.index_select(0,live),w))
        if stable.numel():out[:,h:].index_copy_(0,stable,F.linear(x.index_select(0,stable),w[h:]))
        return out
    raise ValueError(mode)

@torch.inference_mode()
def check(x,w,live,stable,mode):
    h=w.shape[0]//3
    expected=F.linear(x,w);expected[stable,:h]=0
    got=split_projection(x,w,live,stable,h,mode)
    rel=float((got.float()-expected.float()).norm()/expected.float().norm())
    torch.testing.assert_close(got,expected,atol=.04,rtol=.035)
    assert rel<.006 and bool((got[stable,:h]==0).all())
    return rel

def timed(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();start=time.perf_counter();a.record();v=fn();b.record();b.synchronize()
    return a.elapsed_time(b),(time.perf_counter()-start)*1000

@torch.inference_mode()
def main(args):
    torch.set_num_threads(4);torch.manual_seed(4101)
    args.output.mkdir(exist_ok=False,parents=True);records=[]
    def emit(**r):
        records.append(r);(args.output/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(r),flush=True)
    count=0
    for n,d,h in ((31,64,96),(129,128,128)):
        x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(3*h,d,device='cuda',dtype=torch.bfloat16)/d**.5
        ids=torch.arange(n,device='cuda')
        for mask in (ids<0,ids%3==0,ids>=0):
            for mode in ('column_split','row_split'):
                check(x,w,ids[~mask],ids[mask],mode);count+=1
    emit(stage='small_correctness_passed',cases=count)
    n,d,h=40608,5376,7168
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(3*h,d,device='cuda',dtype=torch.bfloat16)/d**.5
    ids=torch.arange(n,device='cuda');logical=ids*2
    src=(logical>=6159)&(logical<6159+37*1008)
    # Synthetic B mask: 87.5% of SOURCE rows stable, all other Q retained.
    mask=src&(ids%8!=0);live=ids[~mask];stable=ids[mask]
    for mode in ('column_split','row_split'):
        rel=check(x,w,live,stable,mode)
        old=lambda:F.linear(x,w)
        new=lambda:split_projection(x,w,live,stable,h,mode)
        for _ in range(3):old();new()
        samples={name:[] for name in ('old_gpu','new_gpu','old_wall','new_wall')}
        for i in range(args.repeats):
            order=[('old',old),('new',new)]
            if i%2:order.reverse()
            for label,fn in order:
                gpu,wall=timed(fn);samples[label+'_gpu'].append(gpu);samples[label+'_wall'].append(wall)
        med={k:statistics.median(v) for k,v in samples.items()}
        emit(stage='full_shape_timing',mode=mode,shape=[n,d,3*h],source_rows=int(src.sum()),
             skipped_source_q_rows=stable.numel(),relative_l2=rel,median_ms=med,samples=samples,
             speed_reduction_pct=100*(1-med['new_wall']/med['old_wall']),
             scope='BF16 F.linear GEMM proxy including gather/zero/scatter/concat; excludes Norm/RoPE/Ulysses; not service runtime')
    emit(stage='completed',inference_modified=False,extra_approximation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=10)
    main(p.parse_args())
