"""CPU-only mapping/position/active-mask audit. No model changes or GPU work."""
import dataclasses
import json
from pathlib import Path
import sys
import torch

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
PORT=ROOT/'candidate/vllm_omni/diffusion/models/minimax_h3'
sys.path[:0]=[str(ROOT),str(ROOT/'loader')]
from audit_contract import functions,module
from savie_mask_arithmetic import make_arithmetic_mask

torch.set_num_threads(2)
records=[]
def emit(test,**data):
    row=dict(test=test,**data)
    records.append(row)
    print(json.dumps(row),flush=True)
    Path(__file__).with_suffix('.results.json').write_text(json.dumps(records,indent=2))

def main():
    seq=module('geometry_sequence_map',PORT/'interleaved_sequence_map.py')
    ns=functions(PORT/'minimax_h3_transformer.py',
        {'_InterleavedTensorMap','_build_interleaved_tensor_map','_apply_rope'},
        {'torch':torch,'dataclass':dataclasses.dataclass,'InterleavedSequenceMap':seq.InterleavedSequenceMap})
    build=ns['_build_interleaved_tensor_map']
    cases=0
    for world in (2,4):
        for n in (world,128,81024):
            m=build(n,world,torch.device('cpu'))
            p,inv=m.physical_to_logical,m.logical_to_physical
            ids=torch.arange(n)
            assert torch.equal(p[inv],ids) and torch.equal(inv[p],ids)
            for rank in range(world):
                assert torch.equal(p[rank*m.local_rows:(rank+1)*m.local_rows],ids[rank::world])
            # Every aligned side tensor uses the same permutation and inverse.
            side=torch.stack([ids,ids%3,ids%37],dim=1)
            assert torch.equal(side[p][inv],side)
            cases+=1
    emit('bijection_and_row_aligned_metadata',cases=cases,world_sizes=[2,4],full_request_tokens=81024)
    # Independent element-level connectivity for query/key pairs in all roles.
    op=module('geometry_layout',PORT/'openvdn_npu.py')
    f,per,text,audio=12,12,5,7
    used=text+f*per*2+audio
    n=((used+3+1)//2)*2
    s=op.OpenVDNLayout(used,text,f,per,3,4,0,text)
    t=op.OpenVDNLayout(used,text+f*per+audio,f,per,3,4,0,text)
    m=build(n,2,torch.device('cpu'))
    pred=make_arithmetic_mask(t,s,n,m.physical_to_logical,None,2)
    logical=torch.arange(n)
    actual=pred(0,0,m.logical_to_physical[:,None],m.logical_to_physical[None,:])
    def role(i):
        if i>=used:return 'padding',-1
        if i<text:return 'text',-1
        if s.video_start<=i<s.video_end:return 'source',(i-s.video_start)//per
        if t.video_start<=i<t.video_end:return 'target',(i-t.video_start)//per
        return 'audio',-1
    expected=torch.zeros(n,n,dtype=torch.bool)
    for qi in range(n):
        qr,qf=role(qi)
        for ki in range(n):
            kr,kf=role(ki)
            if qr=='padding' or kr=='padding':
                yes=qr=='padding' and qi==ki
            elif qr in ('source','text') and kr in ('target','audio'):
                yes=False
            elif qr in ('source','target') and kr in ('source','target'):
                if qr==kr:
                    yes=(qf in (0,f-1) or kf in (0,f-1) or (qf//5-1)*5<=kf<(qf//5+2)*5)
                else:
                    yes=qr=='target' and kr=='source' and qf==kf
            else:
                yes=True
            expected[qi,ki]=yes
    assert torch.equal(actual,expected)
    emit('independent_four_branch_predicate',pairs=n*n,mismatches=int((actual!=expected).sum()),
         ss='chunk5 radius1 + first/last query/key anchors',ts='same frame',st='absent',
         text_source_isolated_from_target_audio=True,padding='self only')
    frames,per,text,audio=37,1008,6159,250
    start=text+frames*per+audio
    n=81024
    m=build(n,2,torch.device('cpu'))
    for label,path in [('B',ROOT.parent/'dmd8_b_skip_20260917/selector_dmd8_latent/fixed_selector_payload.pt'),('step700',ROOT/'selector.pt')]:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        active=payload['active_target_mask'].bool()
        logical_keep=torch.ones(n,dtype=torch.bool)
        logical_keep[start:start+active.numel()]=active
        physical=logical_keep[m.physical_to_logical]
        assert torch.equal(physical[m.logical_to_physical],logical_keep)
        ranks=[]
        for rank in range(2):
            ids=m.physical_to_logical[rank*m.local_rows:(rank+1)*m.local_rows]
            target=(ids>=start)&(ids<start+active.numel())
            act=physical[rank*m.local_rows:(rank+1)*m.local_rows]
            ranks.append(dict(rank=rank,target_tokens=int(target.sum()),active_target=int((target&act).sum()),stable_target=int((target&~act).sum()),all_active=int(act.sum())))
        emit('selector_shard_distribution',version=label,ranks=ranks,roundtrip_exact=True)
    emit('complete',scope='CPU geometry and predicates; not an NCCL communication test')

if __name__=='__main__':main()
