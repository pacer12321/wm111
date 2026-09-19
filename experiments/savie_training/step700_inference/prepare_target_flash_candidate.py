"""Create isolated target-Flash loader; leave 231s/223s controls untouched."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval')
HERE=Path(__file__).resolve().parent
control=json.loads((ROOT/'speedops_candidate/prepared.json').read_text())
assert control['integration_passed'] and control['checkpoint_step']==1000
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in control['helpers'].items():
    assert sha(ROOT/'speedops_loader'/name)==expected,name
loader=ROOT/'targetflash_loader'
assert not loader.exists(),'Do not overwrite an existing tested loader'
shutil.copytree(ROOT/'speedops_loader',loader)
for name in ('savie_grouped_queries.py','savie_target_query_flash.py'):
    shutil.copy2(HERE/name,loader/name)
gate=dict(control,integration_passed=False,parent='speedops_231s_skipoff_223s_skipon',
          change='Exact target frame groups with joint TT+TS+condition softmax',
          full_model_generation_validated=False)
gate['helpers']=dict(control['helpers'])
for name in ('savie_grouped_queries.py','savie_target_query_flash.py'):
    gate['helpers'][name]=sha(loader/name)
out=ROOT/'targetflash_candidate'
out.mkdir(exist_ok=True)
(out/'prepared.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
