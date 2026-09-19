"""Validate the exact overlay that will be deployed, not only its helper."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
HERE=ROOT/'partial_fix_candidate'
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'loader'),str(ROOT/'candidate'),str(ROOT/'repo')]
os.environ.pop('SAVIE_STEP700_CHECKPOINT',None)
import torch
from audit_contract import module
from test_partial_layout import layouts,active_cases
from savie_partial_layout import partial_softmax
old=module('integration_before',ROOT/'loader/savie_overlay.py')
new=module('integration_after',HERE/'savie_overlay.py')
torch.set_num_threads(4)
torch.manual_seed(4101)
with torch.inference_mode():
    t,s,n,m=layouts(frames=12,per=12,device='cuda')
    q,k,v=[torch.randn(n,4,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    self=SimpleNamespace(softmax_scale=32**-.5)
    for keep in active_cases(t,n,m):
        out=new.savie_softmax(self,q,k,v,t,s,keep,m)
        expected=(old.savie_softmax(self,q,k,v,t,s,None,m) if keep is None else
                  partial_softmax(q,k,v,t,s,keep,m,new._FLEX,self.softmax_scale))
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
gate=json.loads((HERE/'partial_layout_passed.json').read_text())
gate.update(integration_passed=True,full_path_bit_exact=True,
    overlay_sha256=hashlib.sha256((HERE/'savie_overlay.py').read_bytes()).hexdigest())
(HERE/'partial_layout_passed.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
