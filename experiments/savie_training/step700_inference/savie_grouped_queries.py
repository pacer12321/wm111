"""Candidate only: specialize by QUERY role, never split a query's softmax.

Keeps odd/even ownership, all original legal keys and all real K/V values.
Every query has exactly one attention call over the union of its legal keys.
"""
from collections import OrderedDict
import os
import torch
from torch.nn.attention.flex_attention import create_block_mask

_PLANS=OrderedDict()

def predicate_for(kind,logical_ids,t,s,n):
    nq=logical_ids.numel()
    ss,se=s.video_start,s.video_end
    ts,te=t.video_start,t.video_end
    text0,text1=t.text_start,t.text_start+t.text_len
    used,frames=t.used_len,t.num_frames
    per=s.tokens_per_frame if kind=='source' else t.tokens_per_frame
    start=ss if kind=='source' else ts
    qframes=(logical_ids-start)//per if kind in ('source','target') else None
    regular=(kind in ('source','target') and nq==frames*per and
             torch.equal(logical_ids,torch.arange(start,start+nq,device=logical_ids.device)))
    def mask(b,h,qi,ki):
        valid=(qi>=0)&(qi<nq)&(ki>=0)&(ki<used)
        kt=(ki>=text0)&(ki<text1)
        ks=(ki>=ss)&(ki<se)
        ktarget=(ki>=ts)&(ki<te)
        if kind=='text':
            allowed=kt|ks
        elif kind=='other':
            allowed=ki<used
        else:
            qf=qi//per if regular else qframes[qi.clamp(min=0,max=nq-1)]
            keyf=(ki-(ss if kind=='source' else ts))//per
            local=((keyf>=(qf//5-1)*5)&(keyf<(qf//5+2)*5))
            local=local|(keyf==0)|(keyf==frames-1)|(qf==0)|(qf==frames-1)
            if kind=='source':
                allowed=kt|(ks&local)
            else:
                allowed=((~ks)&(~ktarget))|(ktarget&local)|(ks&((ki-ss)//s.tokens_per_frame==qf))
        return valid&allowed
    return mask

def build_plan(t,s,n,m,keep):
    p,inv=m.physical_to_logical,m.logical_to_physical
    ids=torch.arange(n,device=p.device)
    rows=n//m.world_size
    if n%m.world_size or not torch.equal(p,(ids%rows)*m.world_size+ids//rows) or not torch.equal(p[inv],ids):
        raise ValueError('Incorrect interleaved permutation')
    if keep.shape!=(n,) or keep.dtype!=torch.bool:
        raise ValueError('Invalid active mask')
    logical_keep=keep[inv]
    is_s=(ids>=s.video_start)&(ids<s.video_end)
    is_t=(ids>=t.video_start)&(ids<t.video_end)
    is_text=(ids>=t.text_start)&(ids<t.text_start+t.text_len)
    is_other=(~is_s)&(~is_t)&(~is_text)&(ids<t.used_len)
    groups=[]
    for kind,selected in [('source',is_s),('target',is_t),('text',is_text),('other',is_other)]:
        qids=ids[selected&logical_keep]
        if not qids.numel():continue
        pred=predicate_for(kind,qids,t,s,n)
        bm=create_block_mask(pred,B=None,H=None,Q_LEN=qids.numel(),KV_LEN=n,device=p.device,BLOCK_SIZE=128,_compile=True)
        groups.append(dict(kind=kind,logical_queries=qids,physical_queries=inv[qids],predicate=pred,block_mask=bm))
    return groups,inv[ids[(ids>=t.used_len)&logical_keep]]

def grouped_softmax(self,q,k,v,t,s,active_mask,mapping,flex):
    n=q.shape[0]
    p,inv=mapping.physical_to_logical,mapping.logical_to_physical
    keep=torch.ones(n,device=q.device,dtype=torch.bool) if active_mask is None else active_mask
    source_flash=os.environ.get('SAVIE_SOURCE_QUERY_FLASH')=='1'
    target_flash=os.environ.get('SAVIE_TARGET_QUERY_FLASH')=='1'
    key=(t,s,n,str(q.device),mapping.world_size,active_mask is None,source_flash,target_flash)
    cached=_PLANS.get(key)
    if cached is None or cached[1] is not p or cached[2] is not inv or not torch.equal(cached[0],keep):
        groups,padding=build_plan(t,s,n,mapping,keep)
        source_batches=None
        if source_flash:
            from savie_source_query_flash import prepare_source_batches
            source_mask=((p>=s.video_start)&(p<s.video_end))|((p>=t.text_start)&(p<t.text_start+t.text_len))
            if not bool(keep[source_mask].all()):
                raise ValueError('Static source Flash path requires all source/text query rows')
            source_batches=prepare_source_batches(t,s,n,mapping)
        target_batches=None
        if target_flash:
            from savie_target_query_flash import prepare_target_batches
            target_batches=prepare_target_batches(t,s,n,mapping,active_mask,group_size=8)
        cached=(keep.clone(),p,inv,groups,padding,source_batches,target_batches)
        _PLANS[key]=cached
        if len(_PLANS)>8:_PLANS.popitem(last=False)
    _PLANS.move_to_end(key)
    kk,vv=k.index_select(0,inv),v.index_select(0,inv)
    result=torch.zeros_like(q)
    if source_flash:
        from savie_source_query_flash import fill_source_queries
        fill_source_queries(result,q,kk,vv,cached[5],self.softmax_scale)
    if target_flash:
        from savie_target_query_flash import fill_target_queries
        fill_target_queries(result,q,kk,vv,cached[6],self.softmax_scale)
    for group in cached[3]:
        if source_flash and group['kind'] in ('source','text'):continue
        if target_flash and group['kind']=='target':continue
        ids=group['physical_queries']
        qq=q.index_select(0,ids)
        out=flex(qq.transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],
                 block_mask=group['block_mask'],scale=self.softmax_scale)[0].transpose(0,1)
        result.index_copy_(0,ids,out)
    # Isolated padding rows have one legal key, hence their output equals V.
    if cached[4].numel():
        result.index_copy_(0,cached[4],v.index_select(0,cached[4]))
    return result
