"""SP2 real NCCL + projection + pinned S-output transfer feasibility gate.

No live service mutation. Cached KV is a deliberate approximation; correctness
reference explicitly substitutes the SAME old K/V, not fresh dense attention.
"""
import argparse,json,os,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.nn.functional as F

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=Path('/cache/zhonghao/h3/savie_step700_eval')
sys.path[:0]=[str(root/'candidate'),str(root/'repo')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4)
dist.init_process_group('nccl');world=dist.get_world_size();assert world==2
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from savie_source_delta_transport import make_plan,pack_local,pre,post
group=SimpleNamespace(ulysses_world_size=world,ulysses_rank=rank,ring_world_size=1,ulysses_group=dist.group.WORLD)
strategy=UlyssesParallelAttention(group,2,1,False)
if rank==0:args.output.mkdir(exist_ok=False,parents=True)
dist.barrier();records=[]
def emit(**r):
    if rank==0:
        records.append(r);(args.output/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(r),flush=True)

@torch.inference_mode()
def numerical():
    count=0
    for n in (70,146,510):
        ids=torch.arange(n,device='cuda');logical=(ids%(n//world))*world+ids//(n//world)
        source=(logical>=7)&(logical<7+n//3)
        for mode in ('none','all_S','selective_uneven'):
            stable=torch.zeros_like(source) if mode=='none' else source if mode=='all_S' else source&(logical%3!=0)
            plan=make_plan(stable,world,rank)
            torch.manual_seed(400+rank+n)
            old=[torch.randn(n//world,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
            fresh=[torch.randn_like(z) for z in old]
            _,ok,ov,_,_=strategy.pre_attention(*[z[None] for z in old],None)
            ck=ok[0].index_select(0,plan.stable_global);cv=ov[0].index_select(0,plan.stable_global)
            a,b,c,_,ctx=strategy.pre_attention(*[z[None] for z in fresh],None)
            a[:,stable]=0;b[0,stable]=ck;c[0,stable]=cv
            expected=(a,b,c)
            aa,bb,cc,_,compact_ctx=pre(strategy,*fresh,plan,ck,cv)
            for x,y in zip((aa,bb,cc),expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
            result=a+b+c;result[:,stable]=0
            wanted=strategy.post_attention(result,ctx)
            actual=post(strategy,aa+bb+cc,compact_ctx,plan)
            torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
            count+=1
    emit(stage='NCCL_numerical_passed',cases=count,bitexact=True,
         reference='fresh Q/current live KV + explicitly substituted cached stable-S KV',unequal_rank_active_counts=True)

def timing(fn,repeats=6):
    for _ in range(2):fn()
    samples=[]
    for _ in range(repeats):
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        sec=torch.tensor([time.perf_counter()-start],device='cuda');dist.all_reduce(sec,op=dist.ReduceOp.MAX)
        samples.append(float(sec)*1000);del value
    return dict(median_ms=statistics.median(samples),samples_ms=samples)

@torch.inference_mode()
def full():
    n=81216;hidden=5376;heads=56;dim=128
    ids=torch.arange(n,device='cuda');logical=(ids%(n//world))*world+ids//(n//world)
    source=(logical>=6159)&(logical<6159+37*1008)
    stable=source&((logical//2)%8!=0);plan=make_plan(stable,world,rank)
    torch.manual_seed(4101+rank)
    x=torch.randn(n//world,hidden,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(3*heads*dim,hidden,device='cuda',dtype=torch.bfloat16)/hidden**.5
    ck=torch.randn(plan.stable_global.numel(),heads//world,dim,device='cuda',dtype=torch.bfloat16)
    cv=torch.randn_like(ck)
    # Full S block-output cache moves to pinned host memory to make GPU KV fit.
    host=torch.empty(plan.stable_local.numel(),hidden,dtype=torch.bfloat16,pin_memory=True)
    host.zero_();source_output=torch.randn_like(host,device='cuda')
    source_full_output=torch.randn(int(source[rank*(n//world):(rank+1)*(n//world)].sum()),hidden,device='cuda',dtype=torch.bfloat16)
    host_full=torch.empty(source_full_output.shape,dtype=torch.bfloat16,pin_memory=True)
    def baseline():
        raw=F.linear(x,w).reshape(n//world,3,heads,dim)
        q,k,v=raw.unbind(1)
        a,b,c,_,ctx=strategy.pre_attention(q[None],k[None],v[None],None)
        # Same query/output support as B; core attention omitted on BOTH sides.
        return strategy.post_attention(a,ctx)
    def delta():
        # Include source-output cache read, fresh row packing and reconstruction.
        old_s=host.to(device='cuda',non_blocking=True)
        raw=F.linear(pack_local(x,plan),w).reshape(plan.width,3,heads,dim)
        q,k,v=raw.unbind(1)
        a,b,c,_,ctx=pre(strategy,q,k,v,plan,ck,cv,already_packed=True)
        result=post(strategy,a,ctx,plan)
        return result,old_s
    base=timing(baseline);new=timing(delta)
    # Refresh overhead must be charged, never hidden outside the timed request.
    def d2h():host_full.copy_(source_full_output,non_blocking=True);return host_full
    write=timing(d2h)
    def kvcopy():return ck.clone(),cv.clone()
    refresh=timing(kvcopy)
    kv_bytes=2*ck.numel()*ck.element_size()*50
    emit(stage='feasibility_timing',baseline=base,cached_path=new,source_output_D2H=write,kv_refresh_copy=refresh,
         rows=[plan.local_rows,plan.width],stable_source_fraction=float(stable.sum()/source.sum()),
         full_50_layer_gpu_cache_gib=kv_bytes/2**30,
         approximate_8_forward_delta_seconds=(5*(base['median_ms']-new['median_ms'])-3*write['median_ms']-2*refresh['median_ms'])*50/1000,
         scope='Projection + actual Ulysses softmax pre/post + KV reconstruction + pinned S read; extra refresh copies included in estimate; omits actual attention, Norm/RoPE, full model and first KV CPU staging cost',
         end_to_end_speed_claim=False)
    emit(stage='completed',live_inference_changed=False)

try:numerical();full()
finally:dist.destroy_process_group()
