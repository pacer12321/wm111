"""CPU transport audit for compact TT-linear exchange, NOT deployed.

Keep text K/V/beta: text initializes delta states. Retain text + all target
frames including anchors. No stale KV, new sparsity, or model changes.
"""
import json
import torch


def audit(text,frames,per,audio,pad,world):
    target_start=text+frames*per+audio
    used=target_start+frames*per
    n=((used+pad+world-1)//world)*world
    logical=torch.arange(n)
    full=logical[:,None]*1000+torch.arange(4)[None,:]
    lengths=[];gathered=[];compact_ids=[]
    for rank in range(world):
        local_ids=logical[rank::world]
        choose=(local_ids<text)|((local_ids>=target_start)&(local_ids<target_start+frames*per))
        selected=local_ids[choose]
        lengths.append(selected.numel());gathered.append(full[selected]);compact_ids.append(selected)
    local_rows=max(lengths)
    compact=torch.cat([torch.nn.functional.pad(rows,(0,0,0,local_rows-rows.shape[0]),value=-1) for rows in gathered])
    index=torch.cat([torch.nn.functional.pad(ids,(0,local_rows-ids.numel()),value=-1) for ids in compact_ids])
    original_to_compact=torch.full((n,),-1,dtype=torch.long)
    valid=index>=0
    original_to_compact[index[valid]]=torch.arange(index.numel())[valid]
    consumers=torch.cat((torch.arange(text),torch.arange(target_start+per,target_start+(frames-1)*per)))
    assert bool((original_to_compact[consumers]>=0).all())
    torch.testing.assert_close(compact[original_to_compact[consumers]],full[consumers],rtol=0,atol=0)
    for rank in range(world):
        for_query=compact[rank*local_rows:(rank+1)*local_rows]
        ids=index[rank*local_rows:(rank+1)*local_rows]
        target=(ids>=target_start)&(ids<target_start+frames*per)
        torch.testing.assert_close(for_query[target],full[ids[target]],rtol=0,atol=0)
    return dict(world=world,full_rows=n,compact_rows=compact.shape[0],local_rows=local_rows,
                retained_text=text,retained_target=frames*per,
                exchange_row_reduction=1-compact.shape[0]/n,consumer_values_exact=True,
                scope='Transport mapping only; full linear scan and NCCL speed not yet tested')


if __name__=='__main__':
    cases=0
    for world in (2,4):
        for text in (5,7,8):
            for frames in (3,7,12):
                for audio in (1,2,3):
                    audit(text,frames,6,audio,5,world);cases+=1
    print(json.dumps(dict(cpu_cases=cases,passed=True,keeps_text_state=True)))
    print(json.dumps(audit(6159,37,1008,250,215,2)))
