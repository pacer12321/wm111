"""Same step700 + latent-skip eight-forward test, with only mask implementation fixed."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
record=json.loads((ROOT/'flex_fix_deployed.json').read_text())
assert record['deployed'] and record['checkpoint_step']==700
for name in ('savie_overlay.py','savie_mask_arithmetic.py'):
    assert hashlib.sha256((ROOT/'loader'/name).read_bytes()).hexdigest()==record['files'][name]
path=ROOT/'launch_step700.py'
source=path.read_text()
old="EXPERIMENT=RUN/'result_v1'"
assert source.count(old)==1
assert not (ROOT/'result_v2_arithmetic').exists(),'Do not overwrite or relaunch completed verification'
source=source.replace(old,"EXPERIMENT=RUN/'result_v2_arithmetic'")
print('FLEX_FIX_VERIFY: same step700, seed4101, DMD8, odd/even, latent skip refresh1/5; compile prewarm excluded',flush=True)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
