"""Execute exact deployed block.forward AST with lightweight rowwise modules.

CPU cases prove transformation correctness, not end-to-end quality. Large
GPU timing omits real attention/MLP GEMMs and reports only this shell's cost.
"""
import argparse,ast,contextlib,copy,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
import torch
from savie_active_post_block import run_active_post_block

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
SOURCE=ROOT/'candidate/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'
tree=ast.parse(SOURCE.read_text())
block=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3DiTBlock')
forward=copy.deepcopy(next(n for n in block.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
forward.name='reference_forward'
selected=[forward]+[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_modulate_gate','_modulate_scale_shift')]
for node in selected:
    node.returns=None
    for arg in ast.walk(node):
        if isinstance(arg,ast.arg):arg.annotation=None
namespace=dict(torch=torch,_BF16_DTYPE=torch.bfloat16,_profile_scope=lambda name:contextlib.nullcontext())
exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(SOURCE),'exec'),namespace)
module=SimpleNamespace(**namespace)
torch.set_num_threads(4)
torch.manual_seed(4101)

class DummyAttention(torch.nn.Module):
    def forward(self,x,**kwargs):
        return x*.17

class TinyMLP(torch.nn.Module):
    def __init__(self,h):
        super().__init__()
        self.up=torch.nn.Linear(h,2*h,bias=False,dtype=torch.bfloat16)
        self.down=torch.nn.Linear(h,h,bias=False,dtype=torch.bfloat16)
    def forward(self,x):
        gate,up=self.up(x).chunk(2,-1)
        return self.down(torch.nn.functional.silu(gate)*up)

class ShellMLP(torch.nn.Module):
    def forward(self,x):return x*.3

def inputs(n,h,device,real_mlp):
    dtype=torch.bfloat16
    scales=[torch.randn(6,h,device=device,dtype=dtype)*.1 for _ in range(6)]
    model=SimpleNamespace(norm1=torch.nn.RMSNorm(h,eps=1e-6,dtype=dtype).to(device),
        norm2=torch.nn.RMSNorm(h,eps=1e-6,dtype=dtype).to(device),
        adaln_proj=lambda t:scales,attn=DummyAttention(),
        mlp=TinyMLP(h).to(device) if real_mlp else ShellMLP())
    x=torch.randn(n,h,device=device,dtype=dtype)
    kwargs=dict(t_emb=torch.zeros(2,8,device=device),combined_indices=torch.arange(n,device=device)%6,
        rope_freqs=None,cu_seqlens=torch.tensor([0,n],device=device),max_seqlen=n)
    return model,x,kwargs

@torch.inference_mode()
def correctness():
    cases=0
    for n,h in ((1,32),(33,32),(259,64)):
        for mode in ('mixed','all_active','all_stable'):
            model,x,kw=inputs(n,h,'cpu',True)
            keep=torch.arange(n)%3!=0
            if mode=='all_active':keep[:]=True
            if mode=='all_stable':keep[:]=False
            kw.update(spotedit_active_mask=keep,spotedit_refresh=True)
            expected=module.reference_forward(model,x,**kw)
            actual=run_active_post_block(model,x,reference_forward=module.reference_forward,module=module,**kw)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            kw['spotedit_refresh']=False
            for _ in range(2):
                x=torch.randn_like(x)
                expected=module.reference_forward(model,x,**kw)
                actual=run_active_post_block(model,x,reference_forward=module.reference_forward,module=module,**kw)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                cases+=1
    print(json.dumps(dict(test='cpu_actual_forward_ast',cases=cases,bit_exact=True,
        refresh_unchanged=True,all_active_and_stable=True,stable_kv_input_not_pruned=True)),flush=True)

def timer(fn):
    torch.cuda.synchronize()
    begin=time.perf_counter()
    out=fn()
    torch.cuda.synchronize()
    return out,(time.perf_counter()-begin)*1000

@torch.inference_mode()
def bench():
    n,h=40512,5376
    model,x,kw=inputs(n,h,'cuda',False)
    target=torch.load(ROOT/'selector.pt',map_location='cpu',weights_only=True)['active_target_mask'].bool()
    whole=torch.ones(n*2,dtype=torch.bool)
    whole[43705:81001]=target
    for rank in (0,1):
        keep=whole[rank::2].to('cuda')
        kw.update(spotedit_active_mask=keep,spotedit_refresh=True)
        module.reference_forward(model,x,**kw)
        kw['spotedit_refresh']=False
        before=lambda:module.reference_forward(model,x,**kw)
        after=lambda:run_active_post_block(model,x,reference_forward=module.reference_forward,module=module,**kw)
        for _ in range(2):a,b=before(),after()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        samples={'before':[],'after':[]}
        for repeat in range(7):
            funcs=[('before',before),('after',after)]
            if repeat%2:funcs.reverse()
            for name,fn in funcs:
                _,ms=timer(fn)
                samples[name].append(ms)
        print(json.dumps(dict(test='GPU_rowwise_shell_only',rank=rank,rows=n,hidden=h,
            active=int(keep.sum()),bit_exact=True,samples=samples,
            medians={k:statistics.median(v) for k,v in samples.items()},
            scope='Norm, modulation, residuals and cache I/O; attention/MLP replaced by cheap rowwise stubs; not total block time')),
            flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    correctness()
    if not args.cpu_only:bench()
