"""Matched step700 control: every target active, all eight forwards fresh.

Uses a separate all-active payload only because the existing interleaved
launcher requires one. No cached state or source fusion is read. Does not
modify the calibrated selector used by the skip run.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/cache/zhonghao/h3/savie_step700_eval')
record = json.loads((ROOT/'flex_fix_deployed.json').read_text())
assert record['deployed'] and record['checkpoint_step'] == 700
for name in ('savie_overlay.py','savie_mask_arithmetic.py'):
    assert hashlib.sha256((ROOT/'loader'/name).read_bytes()).hexdigest() == record['files'][name]
assert not (ROOT/'result_v3_noskip_arithmetic').exists(), 'Do not duplicate an existing run'
import torch
payload = torch.load(ROOT/'selector.pt',map_location='cpu',weights_only=True)
payload['active_target_mask'] = torch.ones_like(payload['active_target_mask'],dtype=torch.bool)
payload['selector_checkpoint_step'] = 700
payload['control_purpose'] = 'skip disabled: all queries active, all steps fresh, no cache reads or fusion'
torch.save(payload,ROOT/'selector_disabled_control.pt')
(ROOT/'selector_disabled_control.json').write_text(json.dumps(dict(step=700,active_ratio=1.0,stable_ratio=0.0,
    selector_enabled=False, purpose=payload['control_purpose'])))
source = (ROOT/'launch_step700.py').read_text()
changes = [
    ("EXPERIMENT=RUN/'result_v1'", "EXPERIMENT=RUN/'result_v3_noskip_arithmetic'"),
    ("str(RUN/'selector.pt')", "str(RUN/'selector_disabled_control.pt')"),
    ("'selector_ready.json'", "'selector_disabled_control.json'"),
    ("ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='1'", "ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='999'"),
    ("SAViE step700 DMD8 latent skip SP2 interleaved; seed4101", "SAViE step700 DMD8 SKIP OFF SP2 interleaved; seed4101"),
    ("partial_query_flex_compile_and_warmup", "full_query_compile_and_warmup"),
    ("request(server,owned,case,3,EXPERIMENT/'warmup_refresh_skip',manifest)", "request(server,owned,case,2,EXPERIMENT/'warmup_refresh_skip',manifest)"),
    ("token_skip='latent_selector_from_step700_first_x0',refresh_forwards=[1,5]", "token_skip=False,refresh_forwards=list(range(1,9))"),
]
for old,new in changes:
    assert old in source, old
    source = source.replace(old,new)
print('MATCHED CONTROL: same step700/seed/DMD8/SP2 arithmetic mask; SKIP OFF; all 8 full fresh forwards; prewarm excluded',flush=True)
exec(compile(source,str(ROOT/'launch_step700.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'launch_step700.py')})
