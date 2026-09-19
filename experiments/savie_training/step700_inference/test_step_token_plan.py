"""CPU semantics only; does not claim NCCL or speed validation."""
import json
import torch
from savie_step_token_plan import attach_step_plan,token_indices,gathered_active_mask

def run(inference):
    calls=[]
    pg=object()
    for world in (2,4):
        logical=torch.arange(76)%3!=0
        p=torch.cat([torch.arange(r,76,world) for r in range(world)])
        global_mask=logical[p]
        rows=76//world
        def gather(parts,local,group):
            assert group is pg
            calls.append(1)
            for rank,part in enumerate(parts):
                part.copy_(global_mask[rank*rows:(rank+1)*rows])
        for rank in range(world):
            local=global_mask[rank*rows:(rank+1)*rows]
            # Simulate the model's 50 blocks sharing the exact same mask.
            plan=attach_step_plan(local,global_mask,rank*rows)
            before=len(calls)
            for _ in range(50):
                result=gathered_active_mask(local,world=world,group=pg,all_gather=gather)
                assert torch.equal(result,global_mask)
                assert torch.equal(token_indices(local),torch.nonzero(local).flatten())
                assert torch.equal(token_indices(local,True),torch.nonzero(~local).flatten())
            assert len(calls)-before==1
            # A new forward is explicitly re-registered, even for same tensor.
            attach_step_plan(local,global_mask,rank*rows)
            gathered_active_mask(local,world=world,group=pg,all_gather=gather)
            assert len(calls)-before==2
            try:
                gathered_active_mask(local,world=world,group=object(),all_gather=gather)
            except ValueError:
                pass
            else:
                raise AssertionError('Group change must be rejected')
    if not inference:
        g=torch.ones(10,dtype=torch.bool)
        local=g[:5]
        attach_step_plan(local,g,0)
        local[0]=False
        try:token_indices(local)
        except ValueError:pass
        else:raise AssertionError('Mutated mask was accepted')

run(False)
with torch.inference_mode():run(True)
print(json.dumps(dict(passed=True,scope='CPU fake collective only',
    world_sizes=[2,4],blocks_per_step=50,gathers_per_step=1,
    new_forward_invalidates=True,actual_nccl_test_pending=True,
    speed_test_pending=True,deployed=False)),flush=True)
