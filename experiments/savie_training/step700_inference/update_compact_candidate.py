"""Update only an unrun compact candidate and reset its verification gate."""
import hashlib,json,shutil
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval');here=Path(__file__).resolve().parent
assert not (root/'result_skip_adaptive_compactlinear').exists()
path=root/'compactlinear_candidate/prepared.json';gate=json.loads(path.read_text())
name='savie_compact_linear.py';dest=root/'compactlinear_loader'/name
assert hashlib.sha256(dest.read_bytes()).hexdigest()==gate['helpers'][name]
shutil.copy2(here/name,dest)
gate['helpers'][name]=hashlib.sha256(dest.read_bytes()).hexdigest();gate['integration_passed']=False
path.write_text(json.dumps(gate,indent=2))
print('Updated unrun compact candidate; full validation gate reset',flush=True)
