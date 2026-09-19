"""Prepare an isolated skip-OFF step1000 evaluation using validated kernels."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

BASE=Path('/cache/zhonghao/h3')
OLD=BASE/'savie_step700_eval'
RUN=BASE/'savie_step1000_eval'
CKPT=Path('/temp/zhonghao/savie_eval_step1000/savie_step001000.pt')
EXPECTED='af783b204289f051d53b94a25f387fb654bf43ef087f0aa76b8c40c5818b1f02'
assert hashlib.file_digest(CKPT.open('rb'),'sha256').hexdigest()==EXPECTED
RUN.mkdir(exist_ok=True)
loader=RUN/'loader'
loader.mkdir(exist_ok=True)
assert not (RUN/'preflight_passed.json').exists(),'Do not overwrite an existing evaluation'
for path in (OLD/'loader').glob('*.py'):
    shutil.copy2(path,loader/path.name)
shutil.copy2(OLD/'loader/manifest.json',loader/'manifest.json')
shutil.copy2(RUN/'candidate_overlay.py',loader/'savie_overlay.py')
for name in ('candidate','repo','deps','inductor_cache'):
    dest=RUN/name
    if not dest.exists():dest.symlink_to(OLD/name,target_is_directory=True)
os.environ['SAVIE_STEP700_CHECKPOINT']=str(CKPT)  # Existing candidate's hook name, not the chosen step.
os.environ['SAVIE_CHECKPOINT_STEP']='1000'
sys.path[:0]=[str(loader),str(OLD),str(RUN/'repo')]
import torch
torch.set_num_threads(4)
from savie_overlay import weights,overlay_plans
from streaming_shards import RangeReader
from torch_streaming_shards import merge_owned_interval_
import manifest_builder
state=weights()
obj=torch.load(CKPT,map_location='cpu',mmap=True,weights_only=False)
previous=torch.load('/temp/zhonghao/savie_eval_step700/savie_step000700.pt',map_location='cpu',mmap=True,weights_only=False)
assert obj['model_spec']==previous['model_spec'], 'Architecture differs: needs user decision'
assert set(state)==set(previous['weights'])
manifest=json.loads((loader/'manifest.json').read_text())
manifest['model_metadata_validated']=True
for i,p in enumerate(manifest['plans']):p['model_iteration_index']=i
readers={}
def reader(path):
    if path not in readers:readers[path]=RangeReader(path)
    return readers[path]
for index in range(50):
    plans=overlay_plans(manifest_builder.tensor_plans_for_main_block(manifest,index,reader),index)
    if index not in (0,24,49):continue
    for plan in plans:
        for adapter in plan.loras:
            if 'savie' not in adapter.a.key:continue
            columns=plan.source.info['shape'][1]
            start=adapter.start_row*columns+7
            count=columns*2-11
            actual=torch.zeros(count,dtype=torch.bfloat16)
            merge_owned_interval_(actual,tensor_start=start,shape=plan.source.info['shape'],loras=(adapter,))
            a,b=state[adapter.a.key].float(),state[adapter.b.key].float()
            expected=(b[:3]@a).to(torch.bfloat16).flatten()[7:7+count]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
payload=torch.load(OLD/'selector.pt',map_location='cpu',weights_only=True)
payload['active_target_mask']=torch.ones_like(payload['active_target_mask'],dtype=torch.bool)
payload['selector_checkpoint_step']=1000
payload['control_purpose']='skip OFF: all tokens fresh; no cache reads or fusion'
torch.save(payload,RUN/'selector_disabled_control.pt')
(RUN/'selector_disabled_control.json').write_text(json.dumps(dict(step=1000,active_ratio=1.0,stable_ratio=0.0,selector_enabled=False)))
gate=dict(checkpoint_step=1000,checkpoint_sha256=EXPECTED,all_1150_weights_mapped=True,
    sampled_lora_shard_merge_exact=True,architecture_matches_step700=True,skip_enabled=False,
    validated_arithmetic_mask_sha256=hashlib.sha256((loader/'savie_mask_arithmetic.py').read_bytes()).hexdigest(),
    overlay_sha256=hashlib.sha256((loader/'savie_overlay.py').read_bytes()).hexdigest(),
    checkpoint_training_label_bug_not_retroactively_fixed=True)
(RUN/'preflight_passed.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
