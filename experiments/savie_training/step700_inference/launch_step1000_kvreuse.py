"""Paired skip on/off runs use identical new KV reuse loader and checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['on', 'off'])
args = parser.parse_args()
root = Path('/cache/zhonghao/h3/savie_step1000_eval')
gate = json.loads((root / 'kvreuse_candidate/prepared.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert gate['integration_passed'] and gate['checkpoint_step'] == 1000
assert sha(root / 'kvreuse_loader/savie_overlay.py') == gate['overlay_sha']
for name, expected in gate['helpers'].items():
    assert sha(root / 'kvreuse_loader' / name) == expected, name
os.environ['SAVIE_KV_REUSE_GROUP_SIZE'] = '64'
if args.mode == 'on':
    source = (root / 'speedops_candidate/launch_step1000_speedops_adaptive_skip.py').read_text()
    replacements = [
        ('result_skip_adaptive_speedops', 'result_skip_adaptive_kvreuse'),
        ('selector_adaptive_speedops', 'selector_adaptive_kvreuse'),
        ('speedops_candidate/prepared.json', 'kvreuse_candidate/prepared.json'),
        ('speedops_loader', 'kvreuse_loader'),
        ("SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'", "SAVIE_TARGET_QUERY_FLASH='1', SAVIE_CHECKPOINT_STEP='1000', SAVIE_GROUPED_QUERY='1'"),
        ('grouped_queries_source_Flash_linear_live_rows_active_post_token_plan', 'grouped_queries_target_lse_kvreuse_source_Flash_speedops'),
        ('latest speedops SP2', 'latest speedops KV_REUSE SP2'),
    ]
else:
    os.environ.update(SAVIE_GROUPED_QUERY='1', SAVIE_SOURCE_QUERY_FLASH='1', SAVIE_TARGET_QUERY_FLASH='1', SAVIE_SPEEDOPS='1')
    os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE', None)
    os.environ.pop('ZHONGHAO_H3_BLOCK_PROFILE_DIR', None)
    source = (root / 'launch_step1000_skipoff.py').read_text()
    needle = "source=(OLD/'launch_step700.py').read_text()"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '''.replace("LOADER=RUN/'loader'", "LOADER=RUN/'kvreuse_loader'")''')
    replacements = [
        ('result_skipoff_indexfix', 'result_skipoff_kvreuse'),
        ('token_skip=False,refresh_forwards=list(range(1,9))', "token_skip=False,refresh_forwards=list(range(1,9)),attention_impl='grouped_queries_target_lse_kvreuse_source_Flash_speedops'"),
    ]
for before, after in replacements:
    assert before in source, before
    source = source.replace(before, after)
exec(compile(source, str(__file__), 'exec'), {'__name__': '__main__', '__file__': str(__file__)})
