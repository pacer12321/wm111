import json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval')
OLD=ROOT.parent/'savie_step700_eval'
sys.path[:0]=[str(ROOT/'targetflash_loader'),str(OLD/'partial_fix_candidate'),str(OLD),str(OLD/'loader'),str(OLD/'candidate'),str(OLD/'repo')]
from test_partial_layout import layouts,active_cases
from savie_grouped_queries import grouped_softmax
from torch.nn.attention.flex_attention import flex_attention
torch.manual_seed(4101);torch.set_num_threads(4)
torch._dynamo.config.recompile_limit=64
torch._dynamo.config.accumulated_recompile_limit=256
os.environ['SAVIE_SOURCE_QUERY_FLASH']='1'
flex=torch.compile(flex_attention,dynamic=True,fullgraph=True)
cases=0
with torch.inference_mode():
    for world in (2,4):
        t,s,n,m=layouts(frames=12,per=12,text=5,audio=7,pad=3,world=world,device='cuda')
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        obj=SimpleNamespace(softmax_scale=32**-.5)
        for keep in active_cases(t,n,m):
            os.environ['SAVIE_TARGET_QUERY_FLASH']='0'
            expected=grouped_softmax(obj,q,k,v,t,s,keep,m,flex)
            os.environ['SAVIE_TARGET_QUERY_FLASH']='1'
            actual=grouped_softmax(obj,q,k,v,t,s,keep,m,flex)
            torch.testing.assert_close(actual,expected,rtol=.025,atol=.006)
            target=(m.physical_to_logical>=t.video_start)&(m.physical_to_logical<t.video_end)
            torch.testing.assert_close(actual[~target],expected[~target],rtol=0,atol=0)
            if keep is not None:assert bool((actual[~keep]==0).all())
            cases+=1
gate_path=ROOT/'targetflash_candidate/prepared.json'
gate=json.loads(gate_path.read_text())
gate.update(integration_passed=True,target_flash_gpu_cases=cases,non_target_output_exact=True)
gate_path.write_text(json.dumps(gate,indent=2))
print(json.dumps(dict(integration_passed=True,cases=cases,non_target_output_exact=True)),flush=True)
