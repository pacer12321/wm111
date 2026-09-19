"""Check the exact new step1000 overlay; no full video generation here."""
import hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
old=root.parent/'savie_step700_eval'
sys.path[:0]=[str(root/'grouped_loader'),str(old/'partial_fix_candidate'),str(old),str(old/'loader'),str(old/'candidate'),str(old/'repo')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
os.environ['SAVIE_GROUPED_QUERY']='1'
import torch
torch.set_num_threads(4)
torch._dynamo.config.recompile_limit=32
from audit_contract import module
from test_partial_layout import layouts,active_cases
from savie_grouped_queries import grouped_softmax
new=module('step1000_grouped_overlay',root/'grouped_loader/savie_overlay.py')
torch.manual_seed(4101)
with torch.inference_mode():
    cases=0
    for world in (2,4):
        t,s,n,m=layouts(frames=12,per=12,world=world,device='cuda')
        q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        self=SimpleNamespace(softmax_scale=32**-.5)
        for keep in active_cases(t,n,m):
            out=new.savie_softmax(self,q,k,v,t,s,keep,m)
            expected=grouped_softmax(self,q,k,v,t,s,keep,m,new._FLEX)
            torch.testing.assert_close(out,expected,rtol=0,atol=0)
            cases+=1
gate=json.loads((root/'grouped_candidate/prepared.json').read_text())
gate.update(integration_passed=True,cases=cases,helper_output_exact=True,
            new_overlay_sha=hashlib.sha256((root/'grouped_loader/savie_overlay.py').read_bytes()).hexdigest())
(root/'grouped_candidate/prepared.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
