"""Isolated step1000 first-x0 calibration; not a full generation benchmark.

Keep grouped-query kernel, weights, source, seed, SP and full-query math.
The first request calibrates the model's OWN latent selector. A second,
identical full-query request gives warmed block-event measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path('/cache/zhonghao/h3')
RUN=ROOT/'savie_step1000_eval'
OLD=ROOT/'savie_step700_eval'
STUDY=RUN/'threshold_study'
STUDY.mkdir(exist_ok=True)
PAYLOAD=STUDY/'selector_original.pt'
MARKER=STUDY/'selector_original.json'
EXPERIMENT=RUN/'result_selector1000_bootstrap'
assert not EXPERIMENT.exists(), 'Do not duplicate an existing bootstrap'
assert not PAYLOAD.exists(), 'Do not replace an existing calibrated selector'
gate=json.loads((RUN/'grouped_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
assert hashlib.sha256((RUN/'grouped_loader/savie_overlay.py').read_bytes()).hexdigest()==gate['new_overlay_sha']
payload=torch.load(RUN/'selector_disabled_control.pt',map_location='cpu',weights_only=True)
payload['active_target_mask']=torch.ones_like(payload['active_target_mask'],dtype=torch.bool)
# Deliberate invalid marker forces calibration from this checkpoint's output.
# This initial control is all-active; no previous model's classification is used.
payload['selector_checkpoint_step']=-1
payload.pop('selector_score',None)
payload.pop('selector_threshold',None)
payload['control_purpose']='All-active bootstrap for OWN step1000 first-x0 scores'
torch.save(payload,PAYLOAD)
os.environ.update(SAVIE_CHECKPOINT_STEP='1000',SAVIE_GROUPED_QUERY='1',
                  ZHONGHAO_H3_BLOCK_PROFILE='1',
                  ZHONGHAO_H3_BLOCK_PROFILE_DIR=str(STUDY/'block_profile'))
source=(OLD/'launch_step700.py').read_text()
changes=[
    ("RUN=ROOT/'savie_step700_eval'","RUN=ROOT/'savie_step1000_eval'"),
    ("LOADER=RUN/'loader'","LOADER=RUN/'grouped_loader'"),
    ("EXPERIMENT=RUN/'result_v1'","EXPERIMENT=RUN/'result_selector1000_bootstrap'"),
    ("['checkpoint_step']==700","['checkpoint_step']==1000"),
    ("ZHONGHAO_H3_REQUEST_STEPS='9'","ZHONGHAO_H3_REQUEST_STEPS='2'"),
    ("str(RUN/'selector.pt')","str(RUN/'threshold_study/selector_original.pt')"),
    ("'selector_ready.json'","'threshold_study/selector_original.json'"),
    ("ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='1'","ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='999'"),
    ("/temp/zhonghao/savie_eval_step700/savie_step000700.pt","/temp/zhonghao/savie_eval_step1000/savie_step001000.pt"),
    ("SAViE step700 DMD8 latent skip SP2 interleaved; seed4101","SAViE1000 OWN first-x0 calibration and full-query diagnostic; NOT full video"),
]
for before,after in changes:
    assert before in source,before
    source=source.replace(before,after)
start=source.index('def run_requests(')
end=source.index('runner.q.request=run_requests',start)
source=source[:start]+'''def run_requests(server,owned,case,steps,directory,manifest):
    runner.update('step1000_own_selector_bootstrap')
    first=request(server,owned,case,2,EXPERIMENT/'first_x0_calibration',manifest)
    selector=json.loads((RUN/'threshold_study/selector_original.json').read_text())
    assert selector['step']==1000,selector
    runner.update('selector_ready_warmed_full_query_profile',selector=selector)
    second=request(server,owned,case,2,directory,manifest)
    second.update(checkpoint_step=1000,actual_dit_forwards=1,
                  token_skip=False,selector=selector,
                  full_video_quality_evaluation=False,formal_speed_comparison=False,
                  scope='First-x0 calibration plus warmed FIRST timestep profile only',
                  calibration_seconds=first['request_seconds'])
    return second
''' + source[end:]
print('Own step1000 latent selector, no old mask; diagnostic first timestep only',flush=True)
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
