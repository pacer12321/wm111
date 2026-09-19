"""Static, exact SS/text query grouping. No Top-K or removed connections."""
import torch


def prepare_source_batches(t,s,n,m,group_size=4):
    ids=torch.arange(n,device=m.physical_to_logical.device)
    source=(ids>=s.video_start)&(ids<s.video_end)
    text=(ids>=t.text_start)&(ids<t.text_start+t.text_len)
    frame=(ids-s.video_start)//s.tokens_per_frame
    anchor=source&((frame==0)|(frame==s.num_frames-1))
    pairs=[(ids[text|anchor],ids[text|source])]
    for chunk in range((s.num_frames+4)//5):
        qids=ids[source&(~anchor)&(frame//5==chunk)]
        if not qids.numel():continue
        local=(frame>=(chunk-1)*5)&(frame<(chunk+2)*5)
        keys=ids[text|(source&(local|(frame==0)|(frame==s.num_frames-1)))]
        pairs.append((qids,keys))
    batches=[]
    for begin in range(0,len(pairs),group_size):
        current=pairs[begin:begin+group_size]
        qidx=torch.cat([q for q,k in current]);kidx=torch.cat([k for q,k in current])
        qc=[];kc=[];qn=kn=0
        for q,k in current:
            qn+=q.numel();kn+=k.numel();qc.append(qn);kc.append(kn)
        batches.append((m.logical_to_physical[qidx],kidx,qc,kc))
    return batches


def fill_source_queries(result,q,k_logical,v_logical,batches,scale):
    from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import _fusion_attention_tnd
    for qidx,kidx,qc,kc in batches:
        part=_fusion_attention_tnd(q.index_select(0,qidx),k_logical.index_select(0,kidx),
                                   v_logical.index_select(0,kidx),qc,kc,scale)
        result.index_copy_(0,qidx,part)
