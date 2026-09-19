"""OFFLINE CANDIDATE ONLY: reuse common TT/condition KV across query frames.

Compute disjoint TT+condition and same-frame TS partitions, then merge using
their log-normalizers. This preserves ONE joint softmax, not an unweighted
sum of independently normalized branches. Never installed by a launcher.
"""
import torch

def pairs(t,s,n,m,active=None):
    ids=torch.arange(n,device=m.physical_to_logical.device)
    target=(ids>=t.video_start)&(ids<t.video_end)
    source=(ids>=s.video_start)&(ids<s.video_end)
    global_keys=(ids<t.used_len)&~target&~source
    tf=(ids-t.video_start)//t.tokens_per_frame
    sf=(ids-s.video_start)//s.tokens_per_frame
    keep=torch.ones_like(ids,dtype=torch.bool) if active is None else active[m.logical_to_physical]
    shared=[];same=[]
    for frame in (0,t.num_frames-1):
        qids=ids[target&(tf==frame)&keep]
        if qids.numel():shared.append((qids,ids[global_keys|target]))
    for chunk in range((t.num_frames+4)//5):
        qids=ids[target&(tf>0)&(tf<t.num_frames-1)&(tf//5==chunk)&keep]
        local=((tf>=(chunk-1)*5)&(tf<(chunk+2)*5))|(tf==0)|(tf==t.num_frames-1)
        if qids.numel():shared.append((qids,ids[global_keys|(target&local)]))
    for frame in range(t.num_frames):
        qids=ids[target&(tf==frame)&keep]
        if qids.numel():same.append((qids,ids[source&(sf==frame)]))
    return ids[target&keep],shared,same

def prepare(t,s,n,m,active=None,group_size=4):
    qids,shared,same=pairs(t,s,n,m,active)
    output_index=torch.full((n,),-1,device=qids.device,dtype=torch.long)
    output_index[qids]=torch.arange(qids.numel(),device=qids.device)
    def pack(pairs):
        result=[]
        for begin in range(0,len(pairs),group_size):
            part=pairs[begin:begin+group_size]
            qi=torch.cat([q for q,k in part]);ki=torch.cat([k for q,k in part])
            qlengths=[q.numel() for q,k in part];klengths=[k.numel() for q,k in part]
            cq=torch.tensor([0]+qlengths,device=qids.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
            ck=torch.tensor([0]+klengths,device=qids.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
            result.append((m.logical_to_physical[qi],ki,output_index[qi],cq,ck,max(qlengths),max(klengths)))
        return result
    return m.logical_to_physical[qids],pack(shared),pack(same)

def merge_outputs(a,b,la,lb):
    weight=torch.sigmoid(la-lb).unsqueeze(-1)
    return (a.float()*weight+b.float()*(1-weight)).to(a.dtype)

_MERGE=None
def fill(result,q,k,v,plan,scale):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    global _MERGE
    if _MERGE is None:_MERGE=torch.compile(merge_outputs,fullgraph=True,dynamic=True)
    qids,shared,same=plan
    if not qids.numel():return result
    def run(batches):
        out=q.new_empty((qids.numel(),)+q.shape[1:]);lse=torch.empty(qids.numel(),q.shape[1],device=q.device,dtype=torch.float32)
        for qi,ki,oi,cq,ck,mq,mk in batches:
            partial,normalizer=flash_attn_varlen_func(q=q.index_select(0,qi),k=k.index_select(0,ki),v=v.index_select(0,ki),cu_seqlens_q=cq,cu_seqlens_k=ck,
                max_seqlen_q=mq,max_seqlen_k=mk,softmax_scale=scale,causal=False,return_softmax_lse=True,fa_version=2)
            assert normalizer.shape==(q.shape[1],qi.numel())
            out.index_copy_(0,oi,partial);lse.index_copy_(0,oi,normalizer.transpose(0,1))
        return out,lse
    a,la=run(shared);b,lb=run(same)
    result.index_copy_(0,qids,_MERGE(a,b,la,lb))
    return result

def install(module):
    # Isolated loader only. No cache from any previous attention implementation.
    import savie_target_query_flash as target
    if getattr(target,'_savie_lse_installed',False):raise RuntimeError('Target LSE installed twice')
    target.prepare_target_batches=prepare
    target.fill_target_queries=fill
    target._savie_lse_installed=True
    print('SAVIE_TARGET_LSE_INSTALLED same_connections joint_softmax_preserved=true',flush=True)
