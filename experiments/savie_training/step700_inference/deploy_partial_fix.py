"""Deploy the tested partial-only implementation; refuse active GPU jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
HERE=ROOT/'partial_fix_candidate'
gate=json.loads((HERE/'partial_layout_passed.json').read_text())
assert gate['passed'] and gate['checkpoint_step']==700 and gate['integration_passed']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HERE/'savie_partial_layout.py')==gate['sha256']
assert sha(HERE/'savie_overlay.py')==gate['overlay_sha256']
old=json.loads((ROOT/'flex_fix_deployed.json').read_text())
assert sha(ROOT/'loader/savie_overlay.py')==old['files']['savie_overlay.py']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU work still active; do not deploy'
files={}
for name in ('savie_partial_layout.py','savie_overlay.py'):
    files[name]=sha(HERE/name)
    for directory in (ROOT,ROOT/'loader'):
        dest=directory/name
        assert dest.resolve().is_relative_to(ROOT.resolve())
        backup=dest.with_name(dest.name+'.before_partial_layout_20260919')
        if dest.exists():
            assert not backup.exists(), 'Never overwrite a deployment backup'
            shutil.copy2(dest,backup)
        temporary=dest.with_name(dest.name+'.deploy_tmp')
        shutil.copy2(HERE/name,temporary)
        os.replace(temporary,dest)
record=dict(deployed=True,checkpoint_step=700,files=files,prior_overlay_sha=old['files']['savie_overlay.py'],
    full_query_path='unchanged arithmetic',partial_query_path='logical pack after Ulysses; scatter before Ulysses',
    connectivity_changed=False,selector_changed=False,weights_changed=False,communication_changed=False,
    bf16_rounding_allowed=True)
(ROOT/'partial_fix_deployed.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
