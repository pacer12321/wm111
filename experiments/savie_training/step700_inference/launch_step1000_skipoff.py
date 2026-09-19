"""Step1000 DMD8, skip OFF; arithmetic fix and interleaved SP2 retained."""
import os
from pathlib import Path
OLD=Path('/cache/zhonghao/h3/savie_step700_eval')
RUN=Path('/cache/zhonghao/h3/savie_step1000_eval')
assert not (RUN/'result_skipoff_indexfix').exists(),'Do not duplicate inference'
os.environ['SAVIE_CHECKPOINT_STEP']='1000'
source=(OLD/'launch_step700.py').read_text()
changes=[
    ("RUN=ROOT/'savie_step700_eval'","RUN=ROOT/'savie_step1000_eval'"),
    ("EXPERIMENT=RUN/'result_v1'","EXPERIMENT=RUN/'result_skipoff_indexfix'"),
    ("['checkpoint_step']==700","['checkpoint_step']==1000"),
    ("str(RUN/'selector.pt')","str(RUN/'selector_disabled_control.pt')"),
    ("'selector_ready.json'","'selector_disabled_control.json'"),
    ("ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='1'","ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='999'"),
    ("/temp/zhonghao/savie_eval_step700/savie_step000700.pt","/temp/zhonghao/savie_eval_step1000/savie_step001000.pt"),
    ("SAViE step700 DMD8 latent skip SP2 interleaved; seed4101","SAViE step1000 DMD8 SKIP OFF SP2 interleaved; seed4101"),
    ("checkpoint_step=700,actual_dit_forwards=8","checkpoint_step=1000,actual_dit_forwards=8"),
    ("token_skip='latent_selector_from_step700_first_x0',refresh_forwards=[1,5]","token_skip=False,refresh_forwards=list(range(1,9))"),
    ("partial_query_flex_compile_and_warmup","full_query_compile_and_warmup"),
    ("request(server,owned,case,3,EXPERIMENT/'warmup_refresh_skip',manifest)","request(server,owned,case,2,EXPERIMENT/'warmup_refresh_skip',manifest)"),
]
for before,after in changes:
    assert before in source,before
    source=source.replace(before,after)
print('STEP1000: skip OFF, same seed4101/8 forwards/odd-even/DMD8; validated arithmetic fix; compile excluded',flush=True)
exec(compile(source,str(RUN/'launch_step1000_skipoff.py'),'exec'),{'__name__':'__main__','__file__':str(RUN/'launch_step1000_skipoff.py')})
