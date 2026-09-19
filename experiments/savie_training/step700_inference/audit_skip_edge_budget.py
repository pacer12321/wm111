"""CPU-only exact legal-softmax-pair budget, not a latency/FLOPs claim."""
import json
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path('/cache/zhonghao/h3')
frames,per,text,audio=37,1008,6159,250
video=frames*per
used=2*video+text+audio
local=[]
for qf in range(frames):
    keys=[kf for kf in range(frames) if
          qf in (0,frames-1) or kf in (0,frames-1) or
          (qf//5-1)*5<=kf<(qf//5+2)*5]
    local.append(len(keys)*per)
local=torch.tensor(local,dtype=torch.int64)
target_keys_B=local+video+text+audio
target_keys_SAViE=local+per+text+audio
total_B=int(((used-video)*used)+(target_keys_B*per).sum())
total_SAViE=int(((local+text)*per).sum()+(target_keys_SAViE*per).sum()+text*(text+video)+audio*used)
rows=[]
for name,path in [('B_selector',root/'dmd8_b_skip_20260917/selector_dmd8_latent/fixed_selector_payload.pt'),
                  ('step700_selector',root/'savie_step700_eval/selector.pt')]:
    mask=torch.load(path,map_location='cpu',weights_only=True)['active_target_mask'].bool()
    assert mask.numel()==video
    stable=(~mask).reshape(frames,per).sum(dim=1)
    for architecture,keys,total in [('B',target_keys_B,total_B),('SAViE',target_keys_SAViE,total_SAViE)]:
        removed=int((stable*keys).sum())
        rows.append(dict(mask=name,architecture=architecture,stable_target_tokens=int(stable.sum()),
                         full_softmax_pairs=total,skipped_softmax_pairs=removed,
                         fraction_of_all_softmax_pairs_skipped=removed/total))
record=dict(scope='Legal attention pairs per head; not measured time; excludes linear/projections/MLP/communication',
            masks_are_diagnostic_only_not_reused_for_step1000=True,results=rows)
print(json.dumps(record,indent=2),flush=True)
Path(__file__).with_suffix('.results.json').write_text(json.dumps(record,indent=2))
