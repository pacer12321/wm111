"""Exact TT-linear transport compaction: keep text + ALL target K/V.

Only the second raw Q/K/beta exchange and its inverse become shorter.
Softmax communication, odd/even ownership, linear scan, and masks do not change.
"""
from collections import OrderedDict
from dataclasses import dataclass,replace
import torch

_PLANS=OrderedDict()

@dataclass
class Plan:
    layout: object
    local_indices: torch.Tensor
    local_padded_indices: torch.Tensor
    value_indices: torch.Tensor
    logical_to_physical: torch.Tensor
    valid_local_rows: int
    local_full_rows: int
    local_compact_rows: int

def plan_for(layout,packed,world,rank,device):
    key=(layout,packed,world,rank,str(device))
    if key in _PLANS:return _PLANS[key]
    layout.validate(packed)
    assert packed%world==0 and 0<=rank<world
    # CPU metadata construction runs once per geometry, not once per block.
    retained=torch.cat((torch.arange(layout.text_start,layout.text_start+layout.text_len),
                        torch.arange(layout.video_start,layout.video_end)))
    compact_used=retained.numel()
    shards=[retained[retained%world==r] for r in range(world)]
    width=max(x.numel() for x in shards)
    index=torch.cat([torch.nn.functional.pad(x,(0,width-x.numel()),value=-1) for x in shards])
    original_to_compact=torch.full((packed,),-1,dtype=torch.long)
    valid=index>=0
    original_to_compact[index[valid]]=torch.arange(index.numel())[valid]
    logical_to_physical=torch.cat((original_to_compact[retained],torch.nonzero(~valid).flatten()))
    assert torch.equal(logical_to_physical.sort().values,torch.arange(width*world))
    local=(shards[rank]//world).to(device)
    local_padded=torch.nn.functional.pad(local,(0,width-local.numel()),value=0)
    # Dummy padding rows are never consumed by the scan, so any valid row is safe.
    ids=torch.where(valid,index,torch.arange(index.numel())//width)
    value_indices=(ids%world*(packed//world)+ids//world).to(device)
    compact_layout=replace(layout,used_len=compact_used,text_start=0,video_start=layout.text_len)
    compact_layout.validate(width*world)
    result=Plan(compact_layout,local,local_padded,value_indices,logical_to_physical.to(device),
                local.numel(),packed//world,width)
    _PLANS[key]=result
    while len(_PLANS)>8:_PLANS.popitem(last=False)
    return result

def pack_local(tensor,plan):
    assert tensor.shape[0]==plan.local_full_rows
    return tensor.index_select(0,plan.local_padded_indices)

def compact_values(value_head,plan):
    return value_head.index_select(0,plan.value_indices)

def unpack_local(tensor,plan):
    assert tensor.shape[0]==plan.local_compact_rows
    result=tensor.new_zeros((plan.local_full_rows,)+tensor.shape[1:])
    result.index_copy_(0,plan.local_indices,tensor[:plan.valid_local_rows])
    return result

def transformed_source(module):
    import inspect,textwrap
    source=textwrap.dedent(inspect.getsource(module.MiniMaxH3Attention._run_openvdn_ulysses))
    before='''    with _profile_scope("ulysses_linear_pre_attention"):
        raw_q, raw_k, beta_head, _metadata, linear_ctx = strategy.pre_attention(
            qkv_raw[0].unsqueeze(0), qkv_raw[1].unsqueeze(0), beta_local.unsqueeze(0), None,
        )'''
    after='''    if interleaved_map is None or self.openvdn_source_hybrid_enabled:
        raise ValueError("Compact TT linear requires odd/even SP and no source linear")
    with _profile_scope("linear_transport_pack"):
        compact_plan = _savie_compact_plan(layout, packed_len, world, rank, x.device)
        compact_q = _savie_compact_pack(qkv_raw[0], compact_plan)
        compact_k = _savie_compact_pack(qkv_raw[1], compact_plan)
        compact_beta = _savie_compact_pack(beta_local, compact_plan)
        compact_v = _savie_compact_values(v_head[0], compact_plan)
    with _profile_scope("ulysses_linear_pre_attention"):
        raw_q, raw_k, beta_head, _metadata, linear_ctx = strategy.pre_attention(
            compact_q.unsqueeze(0), compact_k.unsqueeze(0), compact_beta.unsqueeze(0), None,
        )
    del compact_q, compact_k, compact_beta'''
    assert source.count(before)==1
    source=source.replace(before,after)
    before='''                (raw_q[0], raw_k[0], v_head[0]),
                beta_head[0, :, :, 0],
                frame_mean,
                layout,
                interleaved_map.logical_to_physical,'''
    after='''                (raw_q[0], raw_k[0], compact_v),
                beta_head[0, :, :, 0],
                frame_mean,
                compact_plan.layout,
                compact_plan.logical_to_physical,'''
    assert source.count(before)==1
    source=source.replace(before,after)
    before='        readout_local = strategy.post_attention(readout_head.unsqueeze(0), linear_ctx).squeeze(0)'
    after=before+'''
    with _profile_scope("linear_transport_unpack"):
        readout_local = _savie_compact_unpack(readout_local, compact_plan)'''
    assert source.count(before)==1
    return source.replace(before,after)

def install(module):
    import linecache
    cls=module.MiniMaxH3Attention
    if getattr(cls,'_savie_compact_linear_installed',False):raise RuntimeError('Compact linear installed twice')
    source='from __future__ import annotations\n'+transformed_source(module)
    module.__dict__.update(_savie_compact_plan=plan_for,_savie_compact_pack=pack_local,
        _savie_compact_values=compact_values,_savie_compact_unpack=unpack_local)
    filename='<savie_compact_linear>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace={};exec(compile(source,filename,'exec'),module.__dict__,namespace)
    cls._run_openvdn_ulysses=namespace['_run_openvdn_ulysses']
    cls._savie_compact_linear_installed=True
    print('SAVIE_COMPACT_LINEAR_INSTALLED keep=text+all_target odd_even_preserved=true',flush=True)
