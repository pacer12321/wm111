"""Post-Ulysses logical packing for partial queries; no communication changes.

The caller still owns odd/even sequence shards. K/V contain every token,
only active queries enter Flex, and outputs scatter to the original physical
rows before Ulysses post_attention. K-order changes may cause BF16 rounding.
"""
from collections import OrderedDict
import torch
from torch.nn.attention.flex_attention import create_block_mask
from savie_mask_arithmetic import make_arithmetic_mask

_PLANS = OrderedDict()
_MAX_PLANS = 8


def build_plan(layout, source, n, permutation, inverse, active_mask, world):
    if active_mask.shape != (n,) or active_mask.dtype != torch.bool:
        raise ValueError('Partial mask must be boolean and cover every physical row')
    if permutation.shape != (n,) or inverse.shape != (n,):
        raise ValueError('SP maps must cover the full post-Ulysses sequence')
    if world < 1 or n % world:
        raise ValueError('Equal-row Ulysses required')
    if any(x.device != permutation.device for x in (inverse, active_mask)):
        raise ValueError('Layout tensors must be on the same device')
    logical = torch.arange(n,device=permutation.device)
    rows = n//world
    expected = (logical%rows)*world+logical//rows
    if not torch.equal(permutation,expected) or not torch.equal(permutation[inverse],logical):
        raise ValueError('Not the expected bijective interleaved map')
    logical_active = torch.nonzero(active_mask.index_select(0,inverse)).flatten()
    physical_queries = inverse.index_select(0,logical_active)
    if not logical_active.numel():
        return dict(logical_queries=logical_active,physical_queries=physical_queries,block_mask=None)
    predicate = make_arithmetic_mask(layout,source,n,logical,logical_active,world_size=1)
    block_mask = create_block_mask(predicate,B=None,H=None,Q_LEN=logical_active.numel(),KV_LEN=n,
                                   device=permutation.device,BLOCK_SIZE=128,_compile=True)
    return dict(logical_queries=logical_active,physical_queries=physical_queries,
                block_mask=block_mask,predicate=predicate)


def partial_softmax(q,k,v,layout,source,active_mask,mapping,flex,scale):
    n = q.shape[0]
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError('MHA Q/K/V shapes must match')
    permutation,inverse = mapping.physical_to_logical,mapping.logical_to_physical
    world = mapping.world_size
    key = (layout,source,n,str(q.device),world)
    cached = _PLANS.get(key)
    if (cached is None or cached[1] is not permutation or cached[2] is not inverse
            or not torch.equal(cached[0],active_mask)):
        plan = build_plan(layout,source,n,permutation,inverse,active_mask,world)
        cached = (active_mask.clone(),permutation,inverse,plan)
        _PLANS[key] = cached
        if len(_PLANS)>_MAX_PLANS:
            _PLANS.popitem(last=False)
    _PLANS.move_to_end(key)
    plan = cached[3]
    if plan['block_mask'] is None:
        return torch.zeros_like(q)
    physical_queries = plan['physical_queries']
    qq = q.index_select(0,physical_queries)
    kk = k.index_select(0,inverse)
    vv = v.index_select(0,inverse)
    out = flex(qq.transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],
               block_mask=plan['block_mask'],scale=scale)[0].transpose(0,1)
    return torch.zeros_like(q).index_copy_(0,physical_queries,out)
