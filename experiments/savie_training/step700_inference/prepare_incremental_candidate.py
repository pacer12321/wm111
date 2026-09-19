"""Isolate one optimization on the completed step1000 target-Flash baseline."""
import argparse,hashlib,json,shutil
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('variant',choices=['conditioncache','compactlinear','targetlse'])
a=p.parse_args()
root=Path('/cache/zhonghao/h3/savie_step1000_eval');here=Path(__file__).resolve().parent
parent=root/'targetflash_loader';gate=json.loads((root/'targetflash_candidate/prepared.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert gate['integration_passed'] and gate['checkpoint_step']==1000
assert sha(parent/'savie_overlay.py')==gate['overlay_sha']
for name,expected in gate['helpers'].items():assert sha(parent/name)==expected,name
result=json.loads((root/'result_skip_adaptive_targetflash/formal_9step/result.json').read_text())
assert result['http_success'] and result['actual_dit_forwards']==8 and result['checkpoint_step']==1000
loader=root/(a.variant+'_loader');assert not loader.exists(),loader
helper={'conditioncache':'savie_condition_refiner_cache.py','compactlinear':'savie_compact_linear.py',
        'targetlse':'savie_target_lse_candidate.py'}[a.variant]
shutil.copytree(parent,loader);shutil.copy2(here/helper,loader/helper)
overlay=loader/'savie_overlay.py';source=overlay.read_text()
needle='        install_active_post(module)';assert source.count(needle)==1
source=source.replace(needle,needle+'\n        from '+helper[:-3]+' import install as install_incremental\n        install_incremental(module)')
overlay.write_text(source)
gate.update(integration_passed=False,parent='targetflash_216.654535_seconds',change=a.variant,
            full_model_generation_validated=False,overlay_sha=sha(overlay))
gate['helpers'][helper]=sha(loader/helper)
out=root/(a.variant+'_candidate');out.mkdir(exist_ok=True)
(out/'prepared.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate),flush=True)
