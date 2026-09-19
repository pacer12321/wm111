import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval')
HERE=Path(__file__).resolve().parent
parent=ROOT/'kvreuse_loader'; loader=ROOT/'source_reuse_loader'
gate=json.loads((ROOT/'kvreuse_candidate/prepared.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert gate['integration_passed'] and gate['checkpoint_step']==1000
assert sha(parent/'savie_overlay.py')==gate['overlay_sha']
for name,value in gate['helpers'].items(): assert sha(parent/name)==value,name
assert not loader.exists(),'Do not overwrite an existing candidate'
shutil.copytree(parent,loader)
shutil.copy2(HERE/'savie_source_reuse.py',loader/'savie_source_reuse.py')
path=loader/'savie_grouped_queries.py'; source=path.read_text()
old="            source_mask=((p>=s.video_start)&(p<s.video_end))|((p>=t.text_start)&(p<t.text_start+t.text_len))"
new="            source_mask=(p>=t.text_start)&(p<t.text_start+t.text_len)"
assert source.count(old)==1;source=source.replace(old,new)
old='source_batches=prepare_source_batches(t,s,n,mapping)'
assert source.count(old)==1;source=source.replace(old,'source_batches=prepare_source_batches(t,s,n,mapping,active_mask=keep)')
path.write_text(source)
overlay=loader/'savie_overlay.py'; source=overlay.read_text()
old='    module.MiniMaxH3DiTModel.forward = forward'
assert source.count(old)==1
source=source.replace(old,old+'\n    from savie_source_reuse import install as install_s_reuse\n    install_s_reuse(module)')
overlay.write_text(source)
gate.update(integration_passed=False,parent='kvreuse_209.870141s',change='approximate_S_query_reuse',
            full_model_generation_validated=False,overlay_sha=sha(overlay),source_selector='hidden-drift; prototype, not quality-calibrated')
for name in ('savie_grouped_queries.py','savie_source_reuse.py'): gate['helpers'][name]=sha(loader/name)
directory=ROOT/'source_reuse_candidate';directory.mkdir(exist_ok=True)
(directory/'prepared.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
