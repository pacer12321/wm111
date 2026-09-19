"""Reference vs exact-zero-support gate/projection, real shapes on GPU.

Uses real OutputGate class, random finite weights and synthetic readout with
the deployed scan's exact support. This is NOT a generated quality result.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from savie_linear_live_rows import gate_live_rows,add_live_projection,live_rows

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
source=ROOT/'candidate/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py'
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OutputGate')
ns=dict(torch=torch,nn=torch.nn)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns)
OutputGate=ns['OutputGate']
torch.manual_seed(4101)
torch.set_num_threads(4)


def make(n,h,heads,dim,logical,layout,device):
    dtype=torch.bfloat16
    attn=SimpleNamespace(openvdn_source_hybrid_enabled=False,
        linear_attention=SimpleNamespace(output_gate=OutputGate(h,heads,head_dim=dim,bottleneck=dim).to(device)),
        to_out_linear=torch.nn.Linear(heads*dim,h,bias=False,dtype=dtype).to(device))
    x=torch.randn(n,h,device=device,dtype=dtype)
    readout=torch.zeros(n,heads,dim,device=device,dtype=dtype)
    rows=live_rows(logical,layout)
    readout[rows]=torch.randn(rows.numel(),heads,dim,device=device,dtype=dtype)*.1
    base=torch.randn_like(x)*.2
    return attn,x,readout,base


def paths(attn,x,readout,base,logical,layout,active):
    def reference():
        gated=(readout*attn.linear_attention.output_gate(x).to(readout.dtype)).flatten(1)
        if active is None:return base+attn.to_out_linear(gated)
        idx=torch.nonzero(active).flatten()
        out=base.clone()
        out.index_copy_(0,idx,out[idx]+attn.to_out_linear(gated[idx]))
        return out
    def candidate():
        gated,rows=gate_live_rows(attn,x,readout,layout,logical,active)
        return add_live_projection(attn,gated.flatten(1),base.clone(),rows)
    return reference,candidate


@torch.inference_mode()
def run(cpu_only):
    cases=0
    for rank in (0,1):
        logical=torch.arange(rank,254,2)
        layout=SimpleNamespace(video_start=70,video_end=214,tokens_per_frame=12)
        args=make(logical.numel(),32,4,8,logical,layout,'cpu')
        for active in (None,logical%3==0,torch.zeros_like(logical,dtype=torch.bool)):
            before,after=paths(*args,logical,layout,active)
            torch.testing.assert_close(after(),before(),rtol=.02,atol=.004)
            cases+=1
    print(json.dumps(dict(test='cpu_linear_live_rows',cases=cases,passed=True)),flush=True)
    if cpu_only:return
    results=[]
    for rank in (0,1):
        logical=torch.arange(rank,81024,2,device='cuda')
        layout=SimpleNamespace(video_start=43705,video_end=81001,tokens_per_frame=1008)
        args=make(logical.numel(),5376,56,128,logical,layout,'cuda')
        for name,active in [('skip_off',None),('diagnostic_active35pct',(logical<43705)|((logical%20)<7))]:
            before,after=paths(*args,logical,layout,active)
            a,b=before(),after()
            relative=float((a.float()-b.float()).norm()/a.float().norm())
            torch.testing.assert_close(b,a,rtol=.02,atol=.004)
            for _ in range(2):before();after()
            samples={'before':[],'after':[]}
            for repeat in range(7):
                funcs=[('before',before),('after',after)]
                if repeat%2:funcs.reverse()
                for label,fn in funcs:
                    torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
                    samples[label].append((time.perf_counter()-start)*1000)
            row=dict(rank=rank,case=name,full_rows=logical.numel(),live_rows=live_rows(logical,layout,active).numel(),
                     median_ms={k:statistics.median(v) for k,v in samples.items()},relative_l2=relative,
                     samples_ms=samples,scope='OutputGate+linear out projection+gather/scatter; synthetic tensors; NOT total block')
            results.append(row)
            print(json.dumps(row),flush=True)
            (Path(__file__).parent/'linear_live_rows_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true')
    run(p.parse_args().cpu_only)
