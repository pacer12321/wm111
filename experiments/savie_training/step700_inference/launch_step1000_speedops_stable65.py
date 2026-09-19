"""User-selected65% speed probe; wait for the matching no-skip control.

Only the current red-shirt sample is calibrated to65%. This is not a
universal fixed-ratio selector and is not a quality-qualified deployment.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import torch

ROOT=Path('/cache/zhonghao/h3')
RUN=ROOT/'savie_step1000_eval'
OLD=ROOT/'savie_step700_eval'
EXPERIMENT=RUN/'result_skip_stable65_speedops'
assert not EXPERIMENT.exists(),'Do not duplicate the selected threshold test'
gate=json.loads((RUN/'speedops_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(RUN/'speedops_loader/savie_overlay.py')==gate['overlay_sha']
for name,expected in gate['helpers'].items():assert sha(RUN/'speedops_loader'/name)==expected
scan=json.loads((RUN/'threshold_study/threshold_scan.json').read_text())
selected=next(row for row in scan['records'] if row['name']=='stable65')
assert abs(selected['stable_target_fraction']-.65)<.001
payload=torch.load(selected['mask_path'],map_location='cpu',weights_only=True)
assert payload['selector_checkpoint_step']==1000
assert abs(float((~payload['active_target_mask']).float().mean())-.65)<.001
marker=RUN/'threshold_study/stable65_ready.json'
marker.write_text(json.dumps(dict(step=1000,threshold=selected['threshold'],
    active_ratio=selected['active_target_fraction'],stable_ratio=selected['stable_target_fraction'],
    scope='Current sample only; own1000 first-x0; threshold speed probe; quality unvalidated')))
status_path=RUN/'result_skipoff_speedops/status.json'
print('Queued stable65: waiting for same1000 no-skip speedops control; no GPU allocated while waiting',flush=True)
deadline=time.monotonic()+3600
while True:
    if time.monotonic()>deadline:raise TimeoutError('No-skip control did not finish within one hour')
    state=json.loads(status_path.read_text()) if status_path.exists() else {}
    if state.get('status')=='failed':raise RuntimeError('No-skip control failed; do not proceed blindly')
    if state.get('status')=='completed_quality_review_required':break
    time.sleep(20)
# The status is written just before the baseline cleans up its own workers.
time.sleep(20)
os.environ.update(SAVIE_CHECKPOINT_STEP='1000',SAVIE_GROUPED_QUERY='1',
                  SAVIE_SOURCE_QUERY_FLASH='1',SAVIE_SPEEDOPS='1')
os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE',None)
os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE_DIR',None)
source=(OLD/'launch_step700.py').read_text()
changes=[
    ("RUN=ROOT/'savie_step700_eval'","RUN=ROOT/'savie_step1000_eval'"),
    ("LOADER=RUN/'loader'","LOADER=RUN/'speedops_loader'"),
    ("EXPERIMENT=RUN/'result_v1'","EXPERIMENT=RUN/'result_skip_stable65_speedops'"),
    ("['checkpoint_step']==700","['checkpoint_step']==1000"),
    ("str(RUN/'selector.pt')","str(RUN/'threshold_study/stable65_payload.pt')"),
    ("'selector_ready.json'","'threshold_study/stable65_ready.json'"),
    ("/temp/zhonghao/savie_eval_step700/savie_step000700.pt","/temp/zhonghao/savie_eval_step1000/savie_step001000.pt"),
    ("SAViE step700 DMD8 latent skip SP2 interleaved; seed4101","SAViE1000 DMD8 stable65 speed probe SP2 odd/even seed4101"),
    ("checkpoint_step=700,actual_dit_forwards=8","checkpoint_step=1000,actual_dit_forwards=8"),
    ("token_skip='latent_selector_from_step700_first_x0',refresh_forwards=[1,5]",
     "token_skip='own1000_first_x0_threshold_stable65',refresh_forwards=[1,5],quality_validated=False,attention_impl='source_Flash_and_speedops'"),
]
for before,after in changes:
    assert before in source,before
    source=source.replace(before,after)
print('Starting user-selected stable65, same1000/weights/connectivity/chunk5/8 forwards; refresh1and5',flush=True)
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
