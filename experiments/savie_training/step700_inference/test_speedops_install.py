"""Actual module source installers and packed attention/block composition gate."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval')
OLD=ROOT.parent/'savie_step700_eval'
sys.path[:0]=[str(ROOT/'speedops_loader'),str(OLD/'candidate'),str(OLD/'repo'),str(OLD/'partial_fix_candidate'),str(OLD)]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
module=importlib.import_module('vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer')
original_attention=module.MiniMaxH3Attention.forward
original_block=module.MiniMaxH3DiTBlock.forward
from savie_token_plan_install import install as token_install
from savie_linear_live_rows import install as linear_install,gate_live_rows,live_rows
from savie_active_post_block import install as post_install
token_install(module);linear_install(module);post_install(module)
assert module.MiniMaxH3DiTModel._savie_token_plan_installed
assert module.MiniMaxH3Attention._savie_live_rows_installed
assert module.MiniMaxH3DiTBlock._savie_active_post_installed
torch.manual_seed(4101);torch.set_num_threads(4)


class TupleLinear(torch.nn.Linear):
    def forward(self,x):return super().forward(x),None


@torch.inference_mode()
def attention_gate():
    n,h,heads,d=129,32,4,8
    dtype=torch.bfloat16
    x=torch.randn(n,h,dtype=dtype)
    layout=SimpleNamespace(video_start=70,video_end=214,tokens_per_frame=12)
    cases=0
    for rank in (0,1):
        logical=torch.arange(rank,n*2,2)
        rows=live_rows(logical,layout)
        readout=torch.zeros(n,heads,d,dtype=dtype)
        readout[rows]=torch.randn(rows.numel(),heads,d,dtype=dtype)*.1
        soft=torch.randn_like(readout)*.1
        obj=SimpleNamespace(num_heads=heads,num_kv_heads=heads,head_dim=d,
            qkv_proj=TupleLinear(h,3*heads*d,bias=False,dtype=dtype),
            q_norm=torch.nn.Identity(),k_norm=torch.nn.Identity(),
            out_proj=TupleLinear(heads*d,h,bias=False,dtype=dtype),
            to_out_linear=torch.nn.Linear(heads*d,h,bias=False,dtype=dtype),
            linear_attention=SimpleNamespace(output_gate=module.OutputGate(h,heads,d,bottleneck=d)),
            softmax_gate=module.OutputGate(h,heads),openvdn_enabled=True,
            openvdn_linear_enabled=True,openvdn_source_hybrid_enabled=False,
            attention=SimpleNamespace(_get_active_parallel_strategy=lambda:SimpleNamespace(name='ulysses')))
        for active in (None,torch.ones(n,dtype=torch.bool),logical%3==0,torch.zeros(n,dtype=torch.bool)):
            kw=dict(rope_freqs=None,cu_seqlens=torch.tensor([0,n*2]),max_seqlen=n*2,
                openvdn_layout=layout,spotedit_active_mask=active,spotedit_refresh=active is None,
                local_logical_indices=logical)
            def old_run(*args):return soft, (readout*obj.linear_attention.output_gate(x)).flatten(1)
            obj._run_openvdn_ulysses=old_run
            expected=original_attention(obj,x,**kw)
            def new_run(*args):
                gated,obj._savie_linear_live_rows=gate_live_rows(obj,x,readout,layout,logical,active)
                return soft,gated.flatten(1)
            obj._run_openvdn_ulysses=new_run
            actual=module.MiniMaxH3Attention.forward(obj,x,**kw)
            torch.testing.assert_close(actual,expected,rtol=.02,atol=.006)
            cases+=1
    return cases


@torch.inference_mode()
def grouped_gate():
    from test_partial_layout import layouts,active_cases
    import savie_grouped_queries as candidate
    from torch.nn.attention.flex_attention import flex_attention
    flex=torch.compile(flex_attention,dynamic=True,fullgraph=True)
    torch._dynamo.config.recompile_limit=64
    cases=0
    for world in (2,4):
        t,s,n,m=layouts(frames=12,per=12,text=5,audio=7,pad=3,world=world,device='cuda')
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        obj=SimpleNamespace(softmax_scale=32**-.5)
        for keep in active_cases(t,n,m):
            os.environ['SAVIE_SOURCE_QUERY_FLASH']='0'
            expected=candidate.grouped_softmax(obj,q,k,v,t,s,keep,m,flex)
            os.environ['SAVIE_SOURCE_QUERY_FLASH']='1'
            actual=candidate.grouped_softmax(obj,q,k,v,t,s,keep,m,flex)
            torch.testing.assert_close(actual,expected,rtol=.02,atol=.006)
            # Target perturbation must still be exactly invisible to S/text.
            changed=(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)
            kk,vv=k.clone(),v.clone();kk[changed]*=100;vv[changed]+=100
            other=candidate.grouped_softmax(obj,q,kk,vv,t,s,keep,m,flex)
            context=m.physical_to_logical<s.video_end
            torch.testing.assert_close(other[context],actual[context],rtol=0,atol=0)
            cases+=1
    return cases


cpu_cases=attention_gate()
print(json.dumps(dict(test='actual_install_and_attention_forward',cases=cpu_cases,passed=True)),flush=True)
gpu_cases=grouped_gate()
gate=json.loads((ROOT/'speedops_candidate/prepared.json').read_text())
gate.update(integration_passed=True,attention_forward_cases=cpu_cases,
            grouped_gpu_cases=gpu_cases,source_target_isolation_exact=True,
            full_model_generation_validated=False)
(ROOT/'speedops_candidate/prepared.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
