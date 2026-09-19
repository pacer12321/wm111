"""Approved A/B S reuse on frozen 209.87s parent. Runs are sequential on SP2."""
import argparse,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['whole','selective']);p.add_argument('--run-suffix',default='');args=p.parse_args()
assert not args.run_suffix or (args.run_suffix.startswith('_') and args.run_suffix.replace('_','').isalnum())
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
gate=json.loads((root/'source_reuse_candidate/prepared.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert gate['integration_passed'] and gate['checkpoint_step']==1000
assert sha(root/'source_reuse_loader/savie_overlay.py')==gate['overlay_sha']
for name,value in gate['helpers'].items(): assert sha(root/'source_reuse_loader'/name)==value,name
run='result_skip_adaptive_sreuse_'+args.mode+args.run_suffix
selector='selector_adaptive_sreuse_'+args.mode+args.run_suffix
assert not (root/run).exists(), 'Preserve existing experiment'
os.environ.update(SAVIE_KV_REUSE_GROUP_SIZE='64',SAVIE_S_REUSE_MODE=args.mode,
                  SAVIE_S_REUSE_LOG=str(root/(run+'_events.jsonl')))
source=(root/'speedops_candidate/launch_step1000_speedops_adaptive_skip.py').read_text()
for before,after in [
    ('result_skip_adaptive_speedops',run),('selector_adaptive_speedops',selector),
    ('speedops_candidate/prepared.json','source_reuse_candidate/prepared.json'),
    ('speedops_loader','source_reuse_loader'),
    ("SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'","SAVIE_TARGET_QUERY_FLASH='1', SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'"),
    ('grouped_queries_source_Flash_linear_live_rows_active_post_token_plan','kvreuse_S_'+args.mode+'_query_reuse'),
    ('latest speedops SP2','KVreuse S-'+args.mode+' SP2'),
]:
    assert before in source,before;source=source.replace(before,after)
needle="exec(compile(source, str(__file__), 'exec')"
assert source.count(needle)==1
extra='''source = source.replace('warm=request(server,owned,case,3,', 'warm=request(server,owned,case,9,')
source = source.replace("env.update(PYTHONPATH=", "env.update(SAVIE_S_REUSE_MODE=os.environ['SAVIE_S_REUSE_MODE'], SAVIE_S_REUSE_LOG=os.environ['SAVIE_S_REUSE_LOG'], PYTHONPATH=")
source = source.replace('result.update(checkpoint_step=1000,', "result.update(source_reuse_mode=os.environ['SAVIE_S_REUSE_MODE'], source_refresh_forwards=[1,2,5], source_kv_recomputed=True, source_approximation_quality_validated=False, checkpoint_step=1000,")
'''
source=source.replace(needle,extra+needle)
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
