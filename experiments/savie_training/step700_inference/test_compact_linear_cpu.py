import importlib.util,json,sys
from pathlib import Path
import torch
from savie_compact_linear import plan_for,pack_local,compact_values,unpack_local
path=Path('/cache/zhonghao/h3/savie_step700_eval/candidate/vllm_omni/diffusion/models/minimax_h3/openvdn_npu.py')
spec=importlib.util.spec_from_file_location('cpu_openvdn',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cases=0
for world in (2,4):
 for text in (5,7,8):
  for frames in (3,7,12):
   for audio in (1,2,3):
    per=6;start=text+frames*per+audio;used=start+frames*per;n=((used+5+world-1)//world)*world
    layout=m.OpenVDNLayout(used,start,frames,per,2,3,0,text)
    full=torch.arange(n*4).reshape(n,4)
    physical=torch.cat([full[r::world] for r in range(world)])
    plans=[plan_for(layout,n,world,r,'cpu') for r in range(world)]
    packed=torch.cat([pack_local(full[r::world],plans[r]) for r in range(world)])
    consumers=torch.cat((torch.arange(text),torch.arange(start,start+frames*per)))
    compact_logical=torch.arange(consumers.numel())
    for r,p in enumerate(plans):
        torch.testing.assert_close(packed[p.logical_to_physical[compact_logical]],full[consumers],rtol=0,atol=0)
        torch.testing.assert_close(compact_values(physical,p),packed,rtol=0,atol=0)
        # Padding is arbitrary, deliberately poison it to test restoration excludes it.
        local=packed[r*p.local_compact_rows:(r+1)*p.local_compact_rows].clone()
        local[p.valid_local_rows:]=-123
        restored=unpack_local(local,p)
        torch.testing.assert_close(restored[p.local_indices],full[r::world][p.local_indices],rtol=0,atol=0)
    cases+=1
print(json.dumps(dict(passed=True,cases=cases,transport_bitexact=True,scope='CPU mapping; NCCL and linear scan pending')))
