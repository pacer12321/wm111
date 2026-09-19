"""Run with torchrun --nproc_per_node=2; exact real-NCCL mask reuse test."""
import json
import os
import statistics
import time
import torch
import torch.distributed as dist
from savie_step_token_plan import attach_step_plan,gathered_active_mask,token_indices

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
world=dist.get_world_size()
assert world==2
group=dist.group.WORLD
torch.set_num_threads(2)


@torch.inference_mode()
def run():
    timings={'original_50_gathers':[],'cached_1_gather':[]}
    calls=[]
    def counted(parts,local,group):
        calls.append(1)
        dist.all_gather(parts,local,group=group)
    for repeat in range(6):
        logical=torch.arange(81024,device='cuda')
        expected=(logical<43705)|((logical+repeat)%11<5)
        permutation=torch.cat([logical[r::world] for r in range(world)])
        physical=expected[permutation]
        local=physical[rank*40512:(rank+1)*40512].clone()
        for cached in ((False,True) if repeat%2==0 else (True,False)):
            if hasattr(local,'_savie_step_token_plan'):del local._savie_step_token_plan
            dist.barrier();torch.cuda.synchronize()
            start=time.perf_counter();before=len(calls)
            if cached:attach_step_plan(local,physical,rank*40512)
            for block in range(50):
                actual=gathered_active_mask(local,world=world,group=group,all_gather=counted)
                active=token_indices(local);stable=token_indices(local,True)
            torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            assert torch.equal(actual,physical)
            assert torch.equal(active,torch.nonzero(local).flatten())
            assert torch.equal(stable,torch.nonzero(~local).flatten())
            assert len(calls)-before==(1 if cached else 50)
            if repeat>0:timings['cached_1_gather' if cached else 'original_50_gathers'].append(elapsed)
    print(json.dumps(dict(rank=rank,world=world,passed=True,actual_nccl=True,
        epochs=6,blocks_per_epoch=50,gathers_before=50,gathers_after=1,
        median_ms={k:statistics.median(v) for k,v in timings.items()},samples_ms=timings,
        scope='50 mask gathers and active/stable nonzero only, not KV/Ulysses traffic')),flush=True)


try:run()
finally:dist.destroy_process_group()
