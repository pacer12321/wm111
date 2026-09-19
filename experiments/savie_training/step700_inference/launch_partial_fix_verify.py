"""Same step700/selector/seed/eight forwards; only partial softmax implementation changes."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/cache/zhonghao/h3/savie_step700_eval')
record=json.loads((ROOT/'partial_fix_deployed.json').read_text())
assert record['deployed'] and record['checkpoint_step']==700
for name,value in record['files'].items():
    assert hashlib.sha256((ROOT/'loader'/name).read_bytes()).hexdigest()==value
source=(ROOT/'launch_step700.py').read_text()
old="EXPERIMENT=RUN/'result_v1'"
assert source.count(old)==1
assert not (ROOT/'result_v4_partial_logical').exists(), 'Do not duplicate a verification run'
source=source.replace(old,"EXPERIMENT=RUN/'result_v4_partial_logical'")
print('VERIFY partial layout fix: SAME step700, selector, odd/even, seed4101, 8 forwards; compile excluded',flush=True)
exec(compile(source,str(ROOT/'launch_step700.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'launch_step700.py')})
