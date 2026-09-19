"""Prototype stable-S KV cache transport. Not installed in serving by default.

Plans describe the SAME odd/even ownership. Only fresh non-stable-S rows cross
Ulysses on reuse forwards. Head-sharded cached K/V are restored by physical ID.
All target rows remain fresh (including T-stable raw Q needed by TT linear).
"""
from dataclasses import dataclass
import torch

@dataclass
class Plan:
    n: int
    world: int
    rank: int
    local_rows: int
    width: int
    local_live: torch.Tensor
    local_pack: torch.Tensor
    global_live: torch.Tensor
    packed_valid: torch.Tensor
    stable_global: torch.Tensor
    stable_local: torch.Tensor

def make_plan(stable_physical,world,rank):
    assert stable_physical.ndim==1 and stable_physical.dtype==torch.bool
    n=stable_physical.numel();assert n%world==0 and 0<=rank<world
    rows=n//world
    # Once per immutable S selector, not every layer.
    masks=stable_physical.detach().cpu().reshape(world,rows)
    live=[torch.nonzero(~mask).flatten() for mask in masks]
    width=max(z.numel() for z in live)
    if width==0:raise ValueError('Non-S global conditions must remain live')
    local=live[rank]
    pack=torch.nn.functional.pad(local,(0,width-local.numel()),value=0)
    valid=torch.cat([torch.arange(x.numel())+r*width for r,x in enumerate(live)])
    global_ids=torch.cat([x+r*rows for r,x in enumerate(live)])
    to=lambda x:x.to(stable_physical.device)
    return Plan(n,world,rank,rows,width,to(local),to(pack),to(global_ids),to(valid),
                torch.nonzero(stable_physical).flatten(),to(torch.nonzero(masks[rank]).flatten()))

def pack_local(x,p):
    assert x.shape[0]==p.local_rows
    return x.index_select(0,p.local_pack)

def expand_head(x,p,cached=None):
    assert x.shape[0]==p.width*p.world
    out=x.new_zeros((p.n,)+x.shape[1:])
    out.index_copy_(0,p.global_live,x.index_select(0,p.packed_valid))
    if cached is not None:
        assert cached.shape==(p.stable_global.numel(),)+x.shape[1:]
        out.index_copy_(0,p.stable_global,cached)
    return out

def compact_head(x,p):
    assert x.shape[0]==p.n
    out=x.new_zeros((p.width*p.world,)+x.shape[1:])
    out.index_copy_(0,p.packed_valid,x.index_select(0,p.global_live))
    return out

def expand_local(x,p):
    assert x.shape[0]==p.width
    out=x.new_zeros((p.local_rows,)+x.shape[1:])
    out.index_copy_(0,p.local_live,x[:p.local_live.numel()])
    return out

def pre(strategy,q,k,v,p,cached_k,cached_v,already_packed=False):
    if not already_packed:q,k,v=[pack_local(z,p) for z in (q,k,v)]
    a,b,c,metadata,ctx=strategy.pre_attention(q[None],k[None],v[None],None)
    return (expand_head(a[0],p)[None],expand_head(b[0],p,cached_k)[None],
            expand_head(c[0],p,cached_v)[None],metadata,ctx)

def post(strategy,head,ctx,p):
    compact=compact_head(head[0],p)
    return expand_local(strategy.post_attention(compact[None],ctx)[0],p)[None]
