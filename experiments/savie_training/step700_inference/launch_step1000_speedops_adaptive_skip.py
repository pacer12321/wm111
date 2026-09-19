"""Latest speedops + original adaptive latent selector, never a ratio target.

Bootstrap from this checkpoint and implementation's own first-x0. Keep
weights, connectivity, odd/even SP, schedule, refresh1/5 and seed unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import torch

ROOT = Path('/cache/zhonghao/h3')
RUN = ROOT / 'savie_step1000_eval'
OLD = ROOT / 'savie_step700_eval'
EXPERIMENT = RUN / 'result_skip_adaptive_speedops'
PAYLOAD = RUN / 'selector_adaptive_speedops.pt'
MARKER = RUN / 'selector_adaptive_speedops.json'
assert not EXPERIMENT.exists(), 'Existing run: inspect instead of duplicate'
assert not PAYLOAD.exists() and not MARKER.exists(), 'Preserve prior calibration'
gate = json.loads((RUN / 'speedops_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step'] == 1000
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(RUN / 'speedops_loader/savie_overlay.py') == gate['overlay_sha']
for name, expected in gate['helpers'].items():
    assert sha(RUN / 'speedops_loader' / name) == expected, name
control = json.loads((RUN / 'result_skipoff_speedops/status.json').read_text())
assert control['status'] == 'completed_quality_review_required'
assert control['result']['checkpoint_step'] == 1000
assert control['result']['token_skip'] is False
payload = torch.load(RUN / 'selector_disabled_control.pt', map_location='cpu', weights_only=True)
payload['active_target_mask'] = torch.ones_like(payload['active_target_mask'], dtype=torch.bool)
payload['selector_checkpoint_step'] = -1  # Force fresh first-x0 calibration.
payload.pop('selector_score', None)
payload.pop('selector_threshold', None)
payload['control_purpose'] = 'All-active bootstrap; original adaptive selector; no target ratio'
torch.save(payload, PAYLOAD)
os.environ.update(SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1',
                  SAVIE_SOURCE_QUERY_FLASH='1', SAVIE_SPEEDOPS='1')
os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE', None)
os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE_DIR', None)
source = (OLD / 'launch_step700.py').read_text()
changes = [
    ("RUN=ROOT/'savie_step700_eval'", "RUN=ROOT/'savie_step1000_eval'"),
    ("LOADER=RUN/'loader'", "LOADER=RUN/'speedops_loader'"),
    ("EXPERIMENT=RUN/'result_v1'", "EXPERIMENT=RUN/'result_skip_adaptive_speedops'"),
    ("['checkpoint_step']==700", "['checkpoint_step']==1000"),
    ("str(RUN/'selector.pt')", "str(RUN/'selector_adaptive_speedops.pt')"),
    ("'selector_ready.json'", "'selector_adaptive_speedops.json'"),
    ("/temp/zhonghao/savie_eval_step700/savie_step000700.pt", "/temp/zhonghao/savie_eval_step1000/savie_step001000.pt"),
    ("SAViE step700 DMD8 latent skip SP2 interleaved; seed4101", "SAViE1000 DMD8 adaptive latent skip latest speedops SP2 interleaved seed4101"),
    ("checkpoint_step=700,actual_dit_forwards=8", "checkpoint_step=1000,actual_dit_forwards=8"),
    ("token_skip='latent_selector_from_step700_first_x0',refresh_forwards=[1,5]",
     "token_skip='adaptive_latent_selector_own1000_speedops_first_x0',refresh_forwards=[1,5],quality_validated=False,attention_impl='grouped_queries_source_Flash_linear_live_rows_active_post_token_plan',stable_target_ratio=None"),
]
for before, after in changes:
    assert before in source, before
    source = source.replace(before, after)
print('Start step1000 latest speedops + ORIGINAL adaptive latent selector; no forced stable ratio', flush=True)
exec(compile(source, str(__file__), 'exec'), {'__name__': '__main__', '__file__': str(__file__)})
