"""Read-only audit of completed isolated eight-forward speed experiments."""
import json,re
from pathlib import Path
root=Path('/cache/zhonghao/h3/savie_step1000_eval')
baseline=json.loads((root/'result_skip_adaptive_targetflash/formal_9step/result.json').read_text())['request_seconds']
for variant in ('targetflash','conditioncache','compactlinear','targetlse'):
    run=root/('result_skip_adaptive_'+variant)
    path=run/'formal_9step/result.json'
    if not path.exists():
        print(json.dumps(dict(variant=variant,status='not_completed')));continue
    result=json.loads(path.read_text());assert result['checkpoint_step']==1000 and result['actual_dit_forwards']==8
    assert result['interleaved_sp'] and result['refresh_forwards']==[1,5]
    log=(run/'server.log').read_text(errors='replace')
    steps=re.findall(r'SPOTEDIT_STEP step=(\d+) mode=(\w+)',log)[-8:]
    assert [int(s) for s,m in steps]==list(range(8)),steps
    assert [int(s)+1 for s,m in steps if m=='FULL_REFRESH']==[1,5],steps
    saved=baseline-result['request_seconds']
    print(json.dumps(dict(variant=variant,seconds=result['request_seconds'],saved_vs_targetflash=saved,
        reduction_percent=100*saved/baseline,stable=result['selector']['stable_ratio'],eight_steps_verified=True,
        selector_bootstrap_excluded=True,compile_warmup_excluded=True,single_run_only=True,
        quality_verified=False)),flush=True)
