"""Independent B + exact-connectivity KV consolidation, no policy changes."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval');HERE=Path(__file__).resolve().parent
parent=ROOT/'source_reuse_loader';loader=ROOT/'source_kvpipeline_loader'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=json.loads((ROOT/'source_reuse_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
assert sha(parent/'savie_overlay.py')==gate['overlay_sha']
for name,digest in gate['helpers'].items():assert sha(parent/name)==digest,name
records=json.loads((HERE/'kv_source_bakeoff_results_20260919.json').read_text())
assert records[-1]['stage']=='completed'
assert any(r.get('stage')=='correctness_gate_passed' and r['cases']==144 for r in records)
options=dict(physical_kv=False,fused_pack=True,source_split=True,source_batch=64,fused_merge=True)
selected=[r for r in records if r.get('stage')=='real_shape_timing' and r['options']==options and r['mode']=='S_selective_proxy_12p5']
assert len(selected)==1 and selected[0]['relative_l2']<.006 and selected[0]['speed_reduction_pct']>0
assert not loader.exists(),'Preserve prior experiment'
shutil.copytree(parent,loader)
shutil.copy2(HERE/'savie_kv_pipeline.py',loader/'savie_kv_pipeline.py')
path=loader/'savie_overlay.py';source=path.read_text()
needle='    install_s_reuse(module)'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n    from savie_kv_pipeline import install as install_source_kvpipeline\n    install_source_kvpipeline(module)')
compile(source,str(path),'exec');path.write_text(source)
gate.update(integration_passed=False,parent='B_selective_S_skip_186.243333s',
    change='partial_S_compatible_consolidated_KV',options=options,
    source_reuse_mode='selective',source_refresh_forwards=[1,2,5],
    full_model_generation_validated=False,attention_output_bit_exact=False,
    non_target_output_exact=False,max_micro_relative_l2=selected[0]['relative_l2'],
    micro_cases=144,micro_source_mask='synthetic 12.5% active; not real selector geometry',
    connectivity_unchanged=True,source_kv_recomputed=True,overlay_sha=sha(path))
gate['helpers']['savie_kv_pipeline.py']=sha(loader/'savie_kv_pipeline.py')
directory=ROOT/'source_kvpipeline_candidate';directory.mkdir(exist_ok=False)
(directory/'prepared.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate),flush=True)
