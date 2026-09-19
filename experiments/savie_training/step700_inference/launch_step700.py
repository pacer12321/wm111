"""Reuse the proven 30674 DMD8 runner; prewarm first, then time eight forwards."""
import json
import os
from pathlib import Path
import sys

ROOT=Path('/cache/zhonghao/h3')
RUN=ROOT/'savie_step700_eval'
OLD=ROOT/'dmd8_b_skip_20260917'
CANDIDATE=RUN/'candidate'
LOADER=RUN/'loader'
EXPERIMENT=RUN/'result_v1'
CHECKPOINT=ROOT/'models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250'
assert json.loads((RUN/'preflight_passed.json').read_text())['checkpoint_step']==700
os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=str(EXPERIMENT),
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=str(CANDIDATE),
    ZHONGHAO_H3_REQUEST_STEPS='9',
    ZHONGHAO_H3_PRECHECK_STEPS='0',
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=str(RUN/'selector.pt'),
    ZHONGHAO_H3_INTERLEAVED_SP='1',
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS='1',
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS='4',
    ZHONGHAO_H3_EXPERIMENT_LABEL='SAViE step700 DMD8 latent skip SP2 interleaved; seed4101',
    SAVIE_STEP700_CHECKPOINT='/temp/zhonghao/savie_eval_step700/savie_step000700.pt',
    SAVIE_SELECTOR_READY=str(RUN/'selector_ready.json'),
)
sys.path[:0]=[str(LOADER),str(ROOT/'track_b_validation_20260915')]
import run_b_spotedit_step1 as runner
runner.KIT=LOADER
runner.SOURCE_CANDIDATE=CANDIDATE
original_read=runner.q.read
def read(path):
    result=original_read(path)
    if Path(path).name=='cuda_manifest.json' and 'cases' in result:
        result=json.loads(json.dumps(result))
        for name in result['cases']['B']['model_files']:
            result['cases']['B']['model_files'][name]=runner.q.sha(CANDIDATE/runner.q.REL/name)
        g=result['sample']['requested_generation']
        assert g['seed']==4101 and g['flow_shift']==12 and g['audio_flow_shift']==3,g
    return result
runner.q.read=read
original_configure=runner.prepared.configure
def configure(phase):
    original_configure(phase)
    inherited=runner.q.env_for
    def env_for(case):
        env=inherited(case)
        env.update(PYTHONPATH=f'{LOADER}:{CANDIDATE}',
                   ZHONGHAO_H3_OPENVDN='1',ZHONGHAO_H3_OPENVDN_CHECKPOINT=str(CHECKPOINT),
                   ZHONGHAO_H3_PREPARED_OFFLOAD='1',ZHONGHAO_H3_PREPARED_MANIFEST=str(LOADER/'manifest.json'),
                   ZHONGHAO_H3_INTERLEAVED_SP='1',
                   SAVIE_STEP700_CHECKPOINT=os.environ['SAVIE_STEP700_CHECKPOINT'],
                   SAVIE_SELECTOR_READY=os.environ['SAVIE_SELECTOR_READY'],
                   TORCHINDUCTOR_CACHE_DIR=str(RUN/'inductor_cache'),NCCL_DEBUG='WARN')
        return env
    runner.q.env_for=env_for
runner.prepared.configure=configure
request=runner.q.request
def run_requests(server,owned,case,steps,directory,manifest):
    runner.update('selector_bootstrap_and_full_query_compile')
    first=request(server,owned,case,2,EXPERIMENT/'bootstrap_first_x0',manifest)
    if not (RUN/'selector_ready.json').exists():
        raise RuntimeError('step700 first-x0 selector missing')
    runner.update('partial_query_flex_compile_and_warmup',selector=json.loads((RUN/'selector_ready.json').read_text()))
    warm=request(server,owned,case,3,EXPERIMENT/'warmup_refresh_skip',manifest)
    runner.update('formal_eight_forward_inference',compile_excluded=True)
    result=request(server,owned,case,steps,directory,manifest)
    result.update(checkpoint_step=700,actual_dit_forwards=8,interleaved_sp=True,
                  token_skip='latent_selector_from_step700_first_x0',refresh_forwards=[1,5],
                  selector=json.loads((RUN/'selector_ready.json').read_text()),
                  compile_excluded_from_formal=True,bootstrap_seconds=first['request_seconds'],
                  warmup_seconds=warm['request_seconds'])
    return result
runner.q.request=run_requests
runner.main()
