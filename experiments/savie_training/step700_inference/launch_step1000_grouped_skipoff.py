"""Same trained1000/no-skip/8 forward request; change only tested kernel."""
import hashlib,json,os
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
gate=json.loads((root/'grouped_candidate/prepared.json').read_text())
assert gate['integration_passed'] and gate['checkpoint_step']==1000
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'grouped_loader/savie_overlay.py')==gate['new_overlay_sha']
assert sha(root/'grouped_loader/savie_grouped_queries.py')==gate['helper_sha']
os.environ['SAVIE_GROUPED_QUERY']='1'
script=(root/'launch_step1000_skipoff.py').read_text()
script=script.replace('result_skipoff_indexfix','result_skipoff_grouped')
needle="source=(OLD/'launch_step700.py').read_text()"
assert script.count(needle)==1
script=script.replace(needle,needle+'''.replace("LOADER=RUN/'loader'", "LOADER=RUN/'grouped_loader'")''')
script=script.replace('token_skip=False,refresh_forwards=list(range(1,9))',
                     "token_skip=False,refresh_forwards=list(range(1,9)),attention_impl='grouped_queries'")
print('Trained step1000; SKIP OFF; grouped-query kernel only; unchanged 8 forwards/seed/connectivity/SP2',flush=True)
exec(compile(script,str(root/'launch_step1000_grouped_skipoff.py'),'exec'),
     {'__name__':'__main__','__file__':str(root/'launch_step1000_grouped_skipoff.py')})
