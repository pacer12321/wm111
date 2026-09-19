"""Full1000/no-skip speed validation of tested exact-connectivity speedops."""
import hashlib
import json
import os
from pathlib import Path

root=Path('/cache/zhonghao/h3/savie_step1000_eval')
gate=json.loads((root/'speedops_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'speedops_loader/savie_overlay.py')==gate['overlay_sha']
for name,expected in gate['helpers'].items():
    assert sha(root/'speedops_loader'/name)==expected,name
os.environ.update(SAVIE_GROUPED_QUERY='1',SAVIE_SOURCE_QUERY_FLASH='1',SAVIE_SPEEDOPS='1')
os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE',None)
os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE_DIR',None)
script=(root/'launch_step1000_skipoff.py').read_text()
script=script.replace('result_skipoff_indexfix','result_skipoff_speedops')
needle="source=(OLD/'launch_step700.py').read_text()"
assert script.count(needle)==1
script=script.replace(needle,needle+'''.replace("LOADER=RUN/'loader'", "LOADER=RUN/'speedops_loader'")''')
script=script.replace('token_skip=False,refresh_forwards=list(range(1,9))',
                     "token_skip=False,refresh_forwards=list(range(1,9)),attention_impl='grouped_queries_source_Flash_linear_live_rows'")
print('Trained1000 SKIP OFF: source-query static Flash + live TT linear output rows; unchanged connectivity/weights/seed/eight forwards',flush=True)
exec(compile(script,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
