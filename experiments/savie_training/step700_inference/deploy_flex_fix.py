"""Deploy only a numerically/performance-validated inference overlay; keep backups."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval').resolve()
CANDIDATE=ROOT/'flex_fix_candidate'
passed=json.loads((CANDIDATE/'flex_fix_passed.json').read_text())
assert passed['status']=='passed' and passed['checkpoint_step']==700


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name,expected in passed['files'].items():
    assert sha(CANDIDATE/name)==expected,('candidate changed since tests',name)
old_runtime=passed['baseline_sha256']
expected_old={ROOT/'loader/savie_overlay.py':old_runtime,
              ROOT/'savie_overlay.py':'7996efff4445f601b716db279acb872a27a82334cd8fb4a6fcf3acc66e87c8cb'}
for path,expected in expected_old.items():
    assert path.resolve().is_relative_to(ROOT)
    assert sha(path)==expected,('live file changed',str(path))
    backup=path.with_name(path.name+'.before_flex_arithmetic_20260919')
    if backup.exists():
        assert sha(backup)==expected
    else:
        shutil.copy2(path,backup)

# Dependency first. No deletion; the previous overlays remain recoverable.
for folder in (ROOT/'loader',ROOT):
    for name in ('savie_mask_arithmetic.py','savie_overlay.py'):
        dest=folder/name
        temp=folder/(name+'.deploy_candidate')
        shutil.copy2(CANDIDATE/name,temp)
        os.replace(temp,dest)
        assert sha(dest)==passed['files'][name]
record=dict(passed,deployed=True,old_runtime_sha256=old_runtime,
            changed_scope='inference mask/index implementation only; no weights, selector, training or collectives changed')
(ROOT/'flex_fix_deployed.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
