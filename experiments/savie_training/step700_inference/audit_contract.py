"""Read-only audit: production/selector state is never mutated."""
import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zhonghao/h3')
RUN = ROOT / 'savie_step700_eval'
REPO = RUN / 'repo'
PORT = RUN / 'candidate/vllm_omni/diffusion/models/minimax_h3'
sys.path[:0] = [str(REPO), str(RUN/'loader')]
torch.set_num_threads(4)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    picked = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(picked) == len(names), (path, names)
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def emit(name, **values):
    print(json.dumps({'test':name, **values}), flush=True)

def main():
    from src.models.sequence_layout import dual_layout_from_video_spans
    batch = {'torch':torch, 'F':F, 'np':np, 'math':math,
             'dual_layout_from_video_spans':dual_layout_from_video_spans,
             '_INTERP':32, '_T_GROUP':5, '_FRAME_PER_TOKEN':(1,4,4,4,4),
             '_FRAME_RESCALE':5/3, 'VIDEO_TAG':0, 'TEXT_TAG':1, 'AUDIO_TAG':2}
    functions(REPO/'src/training/ref2va_batch.py',
              {'_axis','_time_grid','_time_span','build_ref2va_layout',
               '_optimal_two_means_threshold','unpatchify_video_rows','latent_selector_mask'}, batch)
    ov = functions(RUN/'loader/savie_overlay.py', {'latent_selector_mask','unpatchify_video_rows'},
                   {'torch':torch})
    old = ROOT/'dmd8_b_skip_20260917'
    capture = torch.load(old/'capture_dmd8/first_x0_and_source_latents.pt', map_location='cpu', weights_only=False)
    s, x = capture['source_clean_normalized_latent'], capture['target_first_x0_normalized_latent']
    old_payload = torch.load(old/'selector_dmd8_latent/fixed_selector_payload.pt', map_location='cpu', weights_only=True)
    new_payload = torch.load(RUN/'selector.pt', map_location='cpu', weights_only=True)
    a, score, threshold = ov['latent_selector_mask'](s, x)
    at, st, th = batch['latent_selector_mask'](s, x)
    emit('selector_same_old_input', mask_mismatch_vs_saved=int((a!=old_payload['active_target_mask']).sum()),
         train_vs_port_mask_mismatch=int((a!=at).sum()), train_vs_port_score_max_abs=float((score-st).abs().max()),
         threshold=threshold, training_threshold=th, active_ratio=float(a.float().mean()),
         clean_source_rows_unchanged=torch.equal(old_payload['source_clean_normalized_rows'],new_payload['source_clean_normalized_rows']))
    for label, p, scores, thres in [('old',old_payload,score,threshold),
                                  ('step700',new_payload,new_payload['selector_score'],new_payload['selector_threshold'])]:
        raw=scores>=thres
        emit('selector_distribution',version=label,threshold=thres,
             score_quantiles=torch.quantile(scores.float(),torch.tensor([0.,.1,.25,.5,.75,.9,1.])).tolist(),
             raw_active_ratio=float(raw.float().mean()), active_ratio=float(p['active_target_mask'].float().mean()),
             score_mean=float(scores.float().mean()))
    tokens=module('audit_packed_tokens', PORT/'packed_tokens.py')
    rows=tokens.minimax_h3_patchify_video_latent(s,patch_size=(1,2,2))
    for name, fn in [('training',batch['unpatchify_video_rows']),('port',ov['unpatchify_video_rows'])]:
        restored=fn(rows,s.shape)
        emit('patchify_roundtrip',version=name,max_abs=float((restored-s).abs().max()))
    packing=module('audit_packing',PORT/'packed_sequence.py')
    for dims in [(40,7,6,8,10),(6159,37,48,84,250)]:
        l,t,h,w,audio=dims
        trained=batch['build_ref2va_layout'](l,t,h,w,audio,'cpu')
        inferred=packing.minimax_h3_packed_sequence_ref2va_blocks(text_len=l,latent_t=t,latent_h=h,latent_w=w,
                   audio_t=audio,ref_blocks=[dict(kind='video',ref_audio_t=0,latent_t=t,latent_h=h,latent_w=w)])
        n=trained['layout'].seq_len
        emit('layout',dims=dims,packed_keys=list(inferred),
             position_max_abs=float((trained['position_ids']-inferred['img_position_ids'].reshape(-1,3)[:n]).abs().max()),
             train_text_tag_values=torch.unique(trained['token_tags'][:l]).tolist())
    time=functions(REPO/'src/training/t2va_batch.py',{'few_step_timesteps'},{'torch':torch})
    ti=module('audit_time_request',PORT/'time_request.py')
    tv,ta=time['few_step_timesteps'](8,12,3)
    for label,times,shift in [('video',tv,12),('audio',ta,3)]:
        actual=1-torch.tensor(ti.minimax_h3_time_shift_sigmas(num_steps=9,shift_scale=shift)[:-1])
        emit('schedule',branch=label,max_abs=float((actual-times).abs().max()),times=times.tolist())
    from safetensors import safe_open
    manifest=json.loads((RUN/'loader/manifest.json').read_text())
    emit('manifest',config=manifest['config'])
    branchfile=ROOT/'models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250/linear_branch/model.safetensors'
    with safe_open(branchfile,framework='pt',device='cpu') as f:
        keys=[k for k in f.keys() if k.startswith('transformer_blocks.24.')]
        emit('branch_keys',keys=keys)

if __name__=='__main__':
    main()
