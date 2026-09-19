"""Reuse verified S-policy/cache tests with the new compiled softmax backend."""
import json,os
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
gate=json.loads((root/'source_kvpipeline_candidate/prepared.json').read_text())
os.environ['SAVIE_KV_PIPELINE_OPTIONS']=json.dumps(gate['options'])
source=(root/'speedops_candidate/test_source_reuse.py').read_text()
source=source.replace("ROOT/'source_reuse_loader'","ROOT/'source_kvpipeline_loader'")
source=source.replace("ROOT/'source_reuse_candidate/prepared.json'","ROOT/'source_kvpipeline_candidate/prepared.json'")
needle='install_target(None)'
assert source.count(needle)==1
source=source.replace(needle,needle+'\nfrom savie_kv_pipeline import install as install_kvpipeline\ninstall_kvpipeline(None)')
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
