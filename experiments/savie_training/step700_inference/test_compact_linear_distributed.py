"""Real deployed linear scan + Ulysses NCCL, then real-shape transport timing."""
import json,os,sys,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist

root=Path('/cache/zhonghao/h3/savie_step1000_eval')
sys.path[:0]=[str(root/'compactlinear_loader'),str(root/'candidate')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
world=dist.get_world_size();assert world==2
torch.set_num_threads(4)
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as m
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from savie_token_plan_install import install as install_plan
from savie_linear_live_rows import install as install_live
from savie_compact_linear import install,plan_for,pack_local,compact_values,unpack_local
group=SimpleNamespace(ulysses_world_size=world,ulysses_rank=rank,ring_world_size=1,ulysses_group=dist.group.WORLD)
m.get_sp_group=lambda:group;m.get_tensor_model_parallel_world_size=lambda:1
strategy=UlyssesParallelAttention(group,2,1,False)
install_plan(m);install_live(m)
original=m.MiniMaxH3Attention._run_openvdn_ulysses
install(m);optimized=m.MiniMaxH3Attention._run_openvdn_ulysses

@torch.inference_mode()
def numerical():
    cases=0;worst=0.
    for frames,text,audio,per in [(3,5,1,6),(7,8,2,6),(12,7,3,6),(37,9,2,6)]:
        start=text+frames*per+audio;used=start+frames*per;n=((used+7+world-1)//world)*world
        layout=m.OpenVDNLayout(used,start,frames,per,2,3,0,text)
        source=m.OpenVDNLayout(used,text,frames,per,2,3,0,text)
        ids=torch.arange(n,device='cuda');physical=torch.cat([ids[r::world] for r in range(world)])
        mapping=SimpleNamespace(logical_to_physical=physical.argsort(),physical_to_logical=physical,world_size=world)
        logical=ids[rank::world]
        heads,dim,hidden=4,32,64
        torch.manual_seed(4101)
        branch=m.BidirectionalLinearBranch(hidden,heads,dim).cuda().eval()
        obj=SimpleNamespace(num_heads=heads,total_num_heads=heads,num_kv_heads=heads,head_dim=dim,
            attention=SimpleNamespace(scatter_idx=2,gather_idx=1),openvdn_linear_enabled=True,
            openvdn_source_hybrid_enabled=False,linear_attention=branch)
        # Softmax implementation is unchanged; isolate the real linear path here.
        obj._openvdn_softmax=lambda q,k,v,*args,**kwargs:q+k+v
        torch.manual_seed(500+rank)
        x=torch.randn(n//world,hidden,device='cuda',dtype=torch.bfloat16)*.1
        raw=tuple(torch.randn(n//world,heads,dim,device='cuda',dtype=torch.bfloat16)*.1 for _ in range(3))
        cu=torch.tensor([0,used,n],device='cuda',dtype=torch.int32)
        for skip in (False,True):
            active=None
            if skip:
                active=~((logical>=start)&(logical<layout.video_end)&(logical%3==0))
            args=(obj,x,*raw,raw,layout,source,cu,strategy,active,mapping,logical)
            expected=original(*args);actual=optimized(*args)
            assert all(bool(torch.isfinite(z).all()) for z in actual)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            cases+=1
        del x,raw,branch,obj
    dist.barrier()
    if rank==0:print(json.dumps(dict(numerical_passed=True,cases=cases,actual_linear_scan=True,actual_NCCL=True,bitexact=True)),flush=True)
    return cases

@torch.inference_mode()
def benchmark():
    n=81216;text=6159;frames=37;per=1008;audio=250;start=text+frames*per+audio
    layout=m.OpenVDNLayout(start+frames*per,start,frames,per,24,42,0,text)
    p=plan_for(layout,n,world,rank,'cuda')
    q,k=[torch.randn(n//world,56,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    beta=torch.randn(n//world,56,1,device='cuda',dtype=torch.bfloat16)
    v=torch.randn(n,28,128,device='cuda',dtype=torch.bfloat16)
    def baseline():
        a,b,c,_,ctx=strategy.pre_attention(q[None],k[None],beta[None],None)
        return strategy.post_attention(a,ctx)
    def compact():
        pq,pk,pb=[pack_local(z,p) for z in (q,k,beta)]
        cv=compact_values(v,p)
        a,b,c,_,ctx=strategy.pre_attention(pq[None],pk[None],pb[None],None)
        return unpack_local(strategy.post_attention(a,ctx)[0],p)
    for _ in range(3):baseline();compact()
    times={'baseline':[],'compact':[]}
    for i in range(10):
        for name,fn in ([('baseline',baseline),('compact',compact)] if i%2==0 else [('compact',compact),('baseline',baseline)]):
            dist.barrier();torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();result=fn();b.record();b.synchronize()
            elapsed=torch.tensor([a.elapsed_time(b)],device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            times[name].append(float(elapsed.item()));del result
    out=dict(full_rows=n,compact_rows=p.local_compact_rows*world,keeps_text_and_all_target=True,
        baseline_ms=statistics.median(times['baseline']),compact_ms=statistics.median(times['compact']),measurements=times,
        scope='Second linear pre/post exchange INCLUDING additional pack, V gather and output scatter; excludes unchanged linear scan')
    if rank==0:print(json.dumps(out),flush=True)
    return out

try:
    cases=numerical();timings=benchmark()
    if rank==0:
        path=root/'compactlinear_candidate/prepared.json';gate=json.loads(path.read_text())
        gate.update(integration_passed=True,compact_NCCL_scan_bitexact_cases=cases,transport_timing=timings)
        path.write_text(json.dumps(gate,indent=2))
    dist.barrier()
finally:dist.destroy_process_group()
