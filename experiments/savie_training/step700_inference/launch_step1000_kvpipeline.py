"""Full eight-forward inference; no S approximation, fresh own T selector."""
import hashlib,json,os
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
gate=json.loads((root/'kvpipeline_candidate/prepared.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert gate['integration_passed'] and gate['full_shape_microbenchmark_passed']
assert sha(root/'kvpipeline_loader/savie_overlay.py')==gate['overlay_sha']
for name,digest in gate['helpers'].items():assert sha(root/'kvpipeline_loader'/name)==digest
os.environ.update(SAVIE_KV_REUSE_GROUP_SIZE='64',SAVIE_KV_PIPELINE_OPTIONS=json.dumps(gate['options']))
source=(root/'speedops_candidate/launch_step1000_speedops_adaptive_skip.py').read_text()
for before,after in [
    ('result_skip_adaptive_speedops','result_skip_adaptive_kvpipeline'),
    ('selector_adaptive_speedops','selector_adaptive_kvpipeline'),
    ('speedops_candidate/prepared.json','kvpipeline_candidate/prepared.json'),
    ('speedops_loader','kvpipeline_loader'),
    ("SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'","SAVIE_TARGET_QUERY_FLASH='1', SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'"),
    ('grouped_queries_source_Flash_linear_live_rows_active_post_token_plan','consolidated_KV_pipeline'),
    ('latest speedops SP2','consolidated_KV_pipeline SP2'),
]:
    assert before in source,before;source=source.replace(before,after)
needle="exec(compile(source, str(__file__), 'exec')"
assert source.count(needle)==1
extra='''source = source.replace("env.update(PYTHONPATH=", "env.update(SAVIE_KV_PIPELINE_OPTIONS=os.environ['SAVIE_KV_PIPELINE_OPTIONS'], PYTHONPATH=")
'''
source=source.replace(needle,extra+needle)
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
