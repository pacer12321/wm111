"""Gate placement correctness; GPU speed is isolated, not a full block."""
import argparse
import ast
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch

ROOT = Path('/cache/zhonghao/h3/savie_step1000_eval')
OLD = ROOT.parent / 'savie_step700_eval'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(ROOT/'speedops_loader'),str(OLD/'candidate'),str(OLD/'repo')]
from savie_active_softmax_projection import active_softmax_projection

# Use the exact deployed class without importing the GPU service package.
gate_source = OLD/'candidate/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py'
tree = ast.parse(gate_source.read_text())
gate_class = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OutputGate')
namespace = dict(torch=torch,nn=torch.nn)
exec(compile(ast.fix_missing_locations(ast.Module(body=[gate_class],type_ignores=[])),str(gate_source),'exec'),namespace)
OutputGate = namespace['OutputGate']


class TupleLinear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


@torch.inference_mode()
def test(full=False):
    n, hidden, heads, dim = (40608,5376,56,128) if full else (67,64,4,16)
    device = 'cuda' if full else 'cpu'
    obj = SimpleNamespace(num_heads=heads,head_dim=dim,
        softmax_gate=OutputGate(hidden,heads).to(device),
        out_proj=TupleLinear(heads*dim,hidden,bias=False,device=device,dtype=torch.bfloat16))
    x=torch.randn(n,hidden,device=device,dtype=torch.bfloat16)
    out=torch.randn(n,heads,dim,device=device,dtype=torch.bfloat16)*.1
    masks=[torch.arange(n,device=device)%3!=0,torch.ones(n,device=device,dtype=torch.bool),torch.zeros(n,device=device,dtype=torch.bool)]
    if full:
        p=torch.load(ROOT/'selector_adaptive_speedops.pt',map_location='cpu',weights_only=True)
        whole=torch.ones(n*2,dtype=torch.bool)
        whole[43705:81001]=p['active_target_mask']
        masks=[whole[r::2].to(device) for r in (0,1)]
    records=[]
    for mask in masks:
        rows=torch.nonzero(mask).flatten()
        def before():
            gated=out*obj.softmax_gate(x).to(out.dtype)
            return obj.out_proj(gated.flatten(1).index_select(0,rows))[0]
        def after():
            return active_softmax_projection(obj,x,out.flatten(1),rows)
        expected,actual=before(),after()
        torch.testing.assert_close(actual,expected,rtol=.025,atol=.006)
        row=dict(rows=n,active=rows.numel(),real_shape=full,passed=True,
                 scope='Gate+output projection only; synthetic x/attention; not full DiT timing')
        if full:
            def timed(fn):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();output=fn();b.record();b.synchronize();return a.elapsed_time(b)
            for _ in range(3):before();after()
            samples={'before':[],'after':[]}
            for i in range(7):
                calls=[('before',before),('after',after)]
                if i%2:calls.reverse()
                for name,fn in calls:samples[name].append(timed(fn))
            row.update(median_ms={k:statistics.median(v) for k,v in samples.items()},samples_ms=samples)
        records.append(row)
        print(json.dumps(row),flush=True)
    if full:(HERE/'active_softmax_projection_results.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    torch.manual_seed(4101);torch.set_num_threads(4)
    test(False)
    if args.gpu:test(True)
