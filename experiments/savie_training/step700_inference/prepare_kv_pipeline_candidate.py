"""Stage consolidated exact-connectivity KV implementation in a NEW loader."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval');HERE=Path(__file__).resolve().parent
parent=ROOT/'kvreuse_loader';loader=ROOT/'kvpipeline_loader'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=json.loads((ROOT/'kvreuse_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
assert sha(parent/'savie_overlay.py')==gate['overlay_sha']
for name,digest in gate['helpers'].items():assert sha(parent/name)==digest,name
records=json.loads((HERE/'kv_pipeline_bakeoff_results_20260919.json').read_text())
assert records[-1]['stage']=='completed'
assert any(r.get('stage')=='correctness_gate_passed' and r['cases']==72 for r in records)
options=dict(physical_kv=False,fused_pack=True,source_split=True,source_batch=64,fused_merge=True)
selected=[r for r in records if r.get('stage')=='real_shape_timing' and r['options']==options]
assert {r['mode'] for r in selected}=={'refresh','skip'}
assert all(r['relative_l2']<.006 and r['speed_reduction_pct']>0 for r in selected)
assert not loader.exists(),'Preserve previous loader'
shutil.copytree(parent,loader)
shutil.copy2(HERE/'savie_kv_pipeline.py',loader/'savie_kv_pipeline.py')
path=loader/'savie_overlay.py';source=path.read_text()
needle='        install_kvreuse(module)'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n        from savie_kv_pipeline import install as install_kvpipeline\n        install_kvpipeline(module)')
compile(source,str(path),'exec');path.write_text(source)
gate.update(parent='kvreuse_209.870141s',change='consolidated_KV_pipeline',
    options=options,full_model_generation_validated=False,source_reuse=False,
    attention_output_bit_exact=False,max_measured_relative_l2=max(r['relative_l2'] for r in selected),
    microbenchmark_host='32209',microbenchmark_device='A100',microbenchmark_cases=72,
    full_shape_microbenchmark_passed=True,overlay_sha=sha(path))
gate['helpers']['savie_kv_pipeline.py']=sha(loader/'savie_kv_pipeline.py')
out=ROOT/'kvpipeline_candidate';out.mkdir(exist_ok=False)
(out/'prepared.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate),flush=True)
