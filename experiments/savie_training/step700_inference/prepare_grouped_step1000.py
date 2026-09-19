"""Prepare separate step1000 grouped-query loader. Never alter the baseline."""
import hashlib,json,shutil
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
candidate=root/'grouped_candidate'
loader=root/'grouped_loader'
assert not loader.exists(), 'Keep existing candidate immutable'
old_gate=json.loads((root/'preflight_passed.json').read_text())
assert old_gate['checkpoint_step']==1000
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'loader/savie_overlay.py')==old_gate['overlay_sha256']
shutil.copytree(root/'loader',loader,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(candidate/'savie_grouped_queries.py',loader/'savie_grouped_queries.py')
overlay=(loader/'savie_overlay.py').read_text()
needle='    if active_query_mask is not None:\n        # Pack only after all-to-all; preserve ownership, RoPE and connectivity.'
assert overlay.count(needle)==1
insert='''    if os.environ.get("SAVIE_GROUPED_QUERY") == "1":
        from savie_grouped_queries import grouped_softmax
        if _FLEX is None:
            _FLEX = torch.compile(flex_attention, dynamic=True, fullgraph=True)
        return grouped_softmax(self, q, k, v, layout, source_layout,
                               active_query_mask, interleaved_map, _FLEX)
'''
(loader/'savie_overlay.py').write_text(overlay.replace(needle,insert+needle))
record=dict(checkpoint_step=1000,skip_enabled=False,
    old_overlay_sha=old_gate['overlay_sha256'],new_overlay_sha=sha(loader/'savie_overlay.py'),
    helper_sha=sha(loader/'savie_grouped_queries.py'),integration_passed=False,
    connectivity_changed=False,weights_changed=False,selector_changed=False)
(candidate/'prepared.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
