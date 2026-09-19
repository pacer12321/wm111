"""B only: identical S/T policies, independently calibrated, new KV backend."""
import json,os,sys
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
gate=json.loads((root/'source_kvpipeline_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['source_reuse_mode']=='selective'
os.environ['SAVIE_KV_PIPELINE_OPTIONS']=json.dumps(gate['options'])
source=(root/'speedops_candidate/launch_step1000_source_reuse.py').read_text()
source=source.replace('source_reuse_candidate/prepared.json','source_kvpipeline_candidate/prepared.json')
source=source.replace('source_reuse_loader','source_kvpipeline_loader')
needle='env.update(SAVIE_S_REUSE_MODE='
assert source.count(needle)==1
source=source.replace(needle,"env.update(SAVIE_KV_PIPELINE_OPTIONS=os.environ['SAVIE_KV_PIPELINE_OPTIONS'], SAVIE_S_REUSE_MODE=")
source=source.replace("'kvreuse_S_'+args.mode+'_query_reuse'","'kvpipeline_S_'+args.mode+'_query_reuse'")
sys.argv=[str(__file__),'selective','--run-suffix','_kvpipeline']
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
