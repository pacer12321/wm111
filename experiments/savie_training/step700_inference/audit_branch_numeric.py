"""Small same-input tests with real step700 tensors. No serving/training writes."""
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from audit_contract import module, emit, RUN, ROOT, REPO, PORT

sys.path[:0]=[str(REPO),str(RUN/'loader')]
os.environ['SAVIE_STEP700_CHECKPOINT']='/temp/zhonghao/savie_eval_step700/savie_step000700.pt'
torch.set_num_threads(4)

def relative(x,y):
    return {'relative_l2':float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12)),
            'max_abs':float((x.float()-y.float()).abs().max())}

def main():
    from src.models.linear_attention.branch import BidirectionalLinearBranch as TrainBranch
    from streaming_shards import RangeReader, read_model_interval
    from torch_streaming_shards import read_torch, merge_owned_interval_
    import manifest_builder
    from savie_overlay import overlay_plans, weights
    port=module('audit_infer_branch',PORT/'openvdn_npu.py')
    ckpt=weights()
    manifest=json.loads((RUN/'loader/manifest.json').read_text())
    manifest['model_metadata_validated']=True
    for i,p in enumerate(manifest['plans']):p['model_iteration_index']=i
    readers={}
    def reader(path):
        if path not in readers:readers[path]=RangeReader(path)
        return readers[path]
    def read_interval(plan,start,stop):
        chunks=[]
        for offset, raw in read_model_interval(dataclasses.replace(plan,loras=()),start,stop,1<<18):
            value=torch.from_numpy(raw)
            if plan.source.info['dtype']=='BF16':value=value.view(torch.bfloat16)
            chunks.append(value)
        result=torch.cat(chunks)
        if plan.loras:merge_owned_interval_(result,tensor_start=start,shape=plan.source.info['shape'],loras=plan.loras)
        return result
    torch.manual_seed(4101)
    with safe_open(ROOT/'models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250/linear_branch/model.safetensors',framework='pt',device='cpu') as released:
        for index in (0,24,49):
            prefix=f'transformer_blocks.{index}.attn.linear_attention.'
            state={k[len(prefix):]:released.get_tensor(k) for k in released.keys() if k.startswith(prefix)}
            state.update({k[len(prefix):]:v for k,v in ckpt.items() if k.startswith(prefix)})
            hidden=state['alpha.down.weight'].shape[1]
            heads=state['alpha.A_log'].numel();dim=state['norm.weight'].numel()
            train=TrainBranch(hidden,heads,dim,delta_rule='vdn_solve',bridge='alpha',a_fp32=True,short_conv=('k','v')).to(dtype=torch.bfloat16)
            infer=port.BidirectionalLinearBranch(hidden,heads,dim)
            train.load_state_dict(state,strict=True);infer.load_state_dict(state,strict=True)
            train=train.cuda().eval();infer=infer.cuda().eval()
            frames,per,text=12,4,5
            length=text+frames*per
            x=torch.randn(length,hidden,device='cuda',dtype=torch.bfloat16)*0.2
            qkv=tuple(torch.randn(length,heads,dim,device='cuda',dtype=torch.bfloat16)*0.3 for _ in range(3))
            layout=port.OpenVDNLayout(length,text,frames,per,2,2,0,text)
            bounds=port.window_bounds(frames)
            with torch.no_grad():
                a=train(x[text:],frames,per,bounds,qkv_raw=tuple(z[text:] for z in qkv),frame_size=(2,2),skip_ends=True,text_x=x[:text],text_qkv_raw=tuple(z[:text] for z in qkv),inference=False)
                b=infer(x,qkv,layout)
            emit('linear_branch_real_step700',layer=index,geometry=[frames,2,2],**relative(b,a))
            del train,infer,x,qkv,a,b
            torch.cuda.empty_cache()
            plans=overlay_plans(manifest_builder.tensor_plans_for_main_block(manifest,index,reader),index)
            # Full base+official adapters+SAViE, not just the isolated LoRA delta.
            for plan in plans:
                if not plan.name.endswith(('qkv_proj.weight','out_proj.weight','to_out_linear.weight')):continue
                columns=plan.source.info['shape'][1]
                nrows=plan.source.info['shape'][0]
                for row in (0,nrows//2,nrows-4):
                    start,stop=row*columns,(row+4)*columns
                    merged=read_interval(plan,start,stop)
                    base_plan=dataclasses.replace(plan,loras=())
                    reference=read_interval(base_plan,start,stop)
                    for lo in plan.loras:
                        lower,upper=max(row,lo.start_row),min(row+4,lo.end_row)
                        if lower>=upper:continue
                        rank=lo.a.info['shape'][0]
                        aa=read_torch(lo.a,0,rank*columns).reshape(rank,columns).float()
                        first=lo.b_row_offset+lower-lo.start_row
                        bb=read_torch(lo.b,first*rank,(first+upper-lower)*rank).reshape(-1,rank).float()
                        reference.view(4,columns)[lower-row:upper-row].add_((bb@aa).to(reference.dtype))
                    emit('effective_weight_independent_merge',layer=index,tensor=plan.name,row=row,**relative(merged,reference))
            if index==24:
                wp=next(p for p in plans if p.name=='blocks.24.adaln_proj.linear.weight')
                bp=next(p for p in plans if p.name=='blocks.24.adaln_proj.linear.bias')
                c=wp.source.info['shape'][1]
                activation=F.silu(torch.randn(c))
                outs=[]
                for tag in (0,1):
                    parts=[]
                    for component in range(6):
                        row=tag*6*hidden+component*hidden
                        w=read_interval(wp,row*c,(row+16)*c).reshape(16,c).float()
                        bias=read_interval(bp,row,row+16).float()
                        parts.append(F.linear(activation,w,bias))
                    outs.append(torch.stack(parts))
                emit('adaln_real_weights_tag0_vs_tag1',note='same synthetic time embedding; 16 channels of all 6 components',**relative(outs[1],outs[0]))

if __name__=='__main__':main()
