"""One-variable full inference ablation on target-Flash + original selector."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('variant',choices=['conditioncache','compactlinear','targetlse'])
a=p.parse_args();root=Path('/cache/zhonghao/h3/savie_step1000_eval')
source=(root/'speedops_candidate/launch_step1000_speedops_adaptive_skip.py').read_text()
for before,after in [
 ('result_skip_adaptive_speedops','result_skip_adaptive_'+a.variant),
 ('selector_adaptive_speedops','selector_adaptive_'+a.variant),
 ('speedops_candidate/prepared.json',a.variant+'_candidate/prepared.json'),
 ('speedops_loader',a.variant+'_loader'),
 ("SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'", "SAVIE_TARGET_QUERY_FLASH='1', SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'"),
 ('grouped_queries_source_Flash_linear_live_rows_active_post_token_plan','grouped_queries_source_and_target_Flash_linear_live_rows_active_post_token_plan_'+a.variant),
 ('latest speedops SP2','latest speedops TARGET_FLASH '+a.variant+' SP2'),
]:
    assert before in source,before
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'),{'__name__':'__main__','__file__':str(__file__)})
