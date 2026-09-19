"""Same step1000 adaptive skip run, with independently tested target Flash."""
from pathlib import Path
ROOT=Path('/cache/zhonghao/h3/savie_step1000_eval')
source=(ROOT/'speedops_candidate/launch_step1000_speedops_adaptive_skip.py').read_text()
changes=[
    ('result_skip_adaptive_speedops','result_skip_adaptive_targetflash'),
    ('selector_adaptive_speedops','selector_adaptive_targetflash'),
    ('speedops_candidate/prepared.json','targetflash_candidate/prepared.json'),
    ('speedops_loader','targetflash_loader'),
    ("SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'", "SAVIE_TARGET_QUERY_FLASH='1', SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'"),
    ('grouped_queries_source_Flash_linear_live_rows_active_post_token_plan','grouped_queries_source_and_target_Flash_linear_live_rows_active_post_token_plan'),
    ('latest speedops SP2','latest speedops TARGET_FLASH SP2'),
]
for before,after in changes:
    assert before in source,before
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
