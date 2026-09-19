"""Create isolated loader only; leave all previous loaders/results untouched."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path('/cache/zhonghao/h3/savie_step1000_eval')
code=root/'speedops_candidate'
loader=root/'speedops_loader'
assert not loader.exists(),'Never overwrite an existing candidate'
gate=json.loads((root/'grouped_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'grouped_loader/savie_overlay.py')==gate['new_overlay_sha']
shutil.copytree(root/'grouped_loader',loader,ignore=shutil.ignore_patterns('__pycache__'))
names=['savie_source_query_flash.py','savie_grouped_queries.py','savie_step_token_plan.py',
       'savie_token_plan_install.py','savie_linear_live_rows.py','savie_active_post_block.py']
for name in names:shutil.copy2(code/name,loader/name)
overlay=(loader/'savie_overlay.py').read_text()
needle='    import manifest_builder\n'
assert overlay.count(needle)==1
insert='''    if os.environ.get("SAVIE_SPEEDOPS") == "1":
        from savie_token_plan_install import install as install_token_plan
        from savie_linear_live_rows import install as install_linear_rows
        from savie_active_post_block import install as install_active_post
        install_token_plan(module)
        install_linear_rows(module)
        install_active_post(module)
'''
(loader/'savie_overlay.py').write_text(overlay.replace(needle,insert+needle))
record=dict(checkpoint_step=1000,integration_passed=False,weights_changed=False,
    connectivity_changed=False,selector_algorithm_changed=False,
    helpers={name:sha(loader/name) for name in names},overlay_sha=sha(loader/'savie_overlay.py'))
(code/'prepared.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
