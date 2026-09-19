"""Exercise actual deployed _embed source and cache installer without model load."""
import ast,json,linecache,types
from pathlib import Path
import torch
from savie_condition_refiner_cache import install

root=Path('/cache/zhonghao/h3/savie_step1000_eval')
path=Path('/cache/zhonghao/h3/savie_step700_eval/candidate/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py')
tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3DiTModel')
embed=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_embed')
source='from __future__ import annotations\n'+ast.unparse(embed)+'\n'
filename='<test_actual_embed>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
module=types.ModuleType('cache_install_test');module.__dict__.update(torch=torch,_FP32_DTYPE=torch.float32,_BF16_DTYPE=torch.bfloat16)
exec(compile(source,filename,'exec'),module.__dict__)
class Model:
    _embed=module._embed
    def _pos_ids(self,value,name):return value
    def forward(self,**kwargs):return self._embed(**self.arguments)
module.MiniMaxH3DiTModel=Model
original=Model._embed
# ast.unparse uses a different format, so install against the ORIGINAL source.
import textwrap
lines=path.read_text().splitlines(True)
source='from __future__ import annotations\n'+textwrap.dedent(''.join(lines[embed.lineno-1:embed.end_lineno]))
linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
exec(compile(source,filename,'exec'),module.__dict__);Model._embed=module._embed;original=Model._embed
class Projection:
    def __init__(self):self.calls=0
    def __call__(self,x):self.calls+=1;return x*.7,None
class Refiner:
    def __init__(self):self.calls=0
    def __call__(self,x,**kwargs):self.calls+=1;return x.sin()+.25
install(module)
@torch.inference_mode()
def test():
    m=Model();m.hidden_size=32;m.video_patch_proj=Projection();m.audio_patch_proj=Projection()
    m.condition_proj=Projection();m.token_refiner=Refiner();m.time_embedder=lambda x:x[:,None].expand(-1,32)*.9
    m.arguments=dict(x=torch.randn(1,19,32),audio_x=torch.randn(1,19,32),text_embeddings_selected=torch.randn(7,32),
        unique_timesteps=torch.tensor([0.]),img_pos=torch.arange(7,17),audio_pos=torch.arange(17,19),text_pos=torch.arange(7),
        refiner_cu_seqlens=torch.tensor([0,7],dtype=torch.int32),refiner_max_seqlen=7,seq_len=19,device=torch.device('cpu'))
    for request in range(2):
        start=m.token_refiner.calls
        for step in range(8):
            m.arguments['x'].add_(.01);m.arguments['unique_timesteps']=torch.tensor([step/8])
            expected=original(m,**m.arguments);count=m.token_refiner.calls
            actual=m.forward(img_pos_info=torch.arange(7,17),update_mask=torch.ones(10,dtype=torch.bool),
                inverse_indices=torch.zeros(19,dtype=torch.long),unique_timesteps=torch.tensor([step/8]))
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert m.token_refiner.calls-count==(1 if step==0 else 0)
        assert m._savie_condition_cache_hits==7 and m._savie_condition_cache_misses==1
    out=dict(passed=True,actual_embed_source_tested=True,requests=2,forwards_per_request=8,cache_misses_per_request=1,
             cache_hits_per_request=7,embeddings_and_timestep_exact=True,scope='Actual embed and installer, mocked deterministic submodules; full-model pending')
    gate_path=root/'conditioncache_candidate/prepared.json';gate=json.loads(gate_path.read_text())
    gate.update(integration_passed=True,condition_cache_install_test=out);gate_path.write_text(json.dumps(gate,indent=2))
    print(json.dumps(out),flush=True)
test()
