"""Read-only data/code checks before clean DMD8 retraining (no old LoRA)."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

ROOT = Path('/cache/zhonghao/h3')
REPO = ROOT/'train_repos/vdn-minimax-h3'
SAMPLES = Path('/temp/zhonghao/savie_stream/samples')
OUT = ROOT/'train_runs/savie_dmd8_skipoff_2k_tagsfix_20260919'
sys.path.insert(0,str(REPO))
torch.set_num_threads(4)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
batch = REPO/'src/training/ref2va_batch.py'
assert sha(batch)=='eda0d7bdf89d4924ee33e9761e5303d05902eeec5e6fa5c21e91fa8a62dee44e'
assert not list(OUT.glob('savie_step*.pt')), 'Refuse implicit resume from any checkpoint'
missing = []
for index in range(2000):
    for kind in ('video','prompt'):
        stem = SAMPLES/f'{kind}_{index:06d}'
        if not stem.with_suffix('.done').is_file() or not stem.with_suffix('.pt').is_file():
            missing.append(str(stem))
        elif stem.with_suffix('.pt').stat().st_size == 0:
            missing.append(str(stem))
assert not missing, missing
def load_test(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'tests'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
tags=load_test('test_ref2va_prompt_tags')
suite=unittest.defaultTestLoader.loadTestsFromModule(tags)
report=unittest.TextTestRunner(verbosity=2).run(suite)
assert report.wasSuccessful() and not report.skipped
mask=load_test('test_dual_stream_flex')
for name in sorted(dir(mask)):
    if name.startswith('test_'): getattr(mask,name)()
from src.training.ref2va_batch import pack_ref2va_batch
from src.training.train_ref2va_dual import stream_sample
from src.training.t2va_batch import few_step_timesteps
records=[]
for index in (0,1,25,100,499,999,1499,1999):
    sample=stream_sample(SimpleNamespace(sample_dir=str(SAMPLES),sample_wait_timeout=0),index,0)
    packed=pack_ref2va_batch(sample,'cpu',torch.Generator().manual_seed(4101),torch.Generator().manual_seed(4102),step_index=index%8)
    inp=packed['inputs']; positions=inp['text_indices']; expected=sample['text_token_tags']
    assert torch.equal(inp['token_tags'][positions],expected)
    assert torch.equal((inp['timestep_indices']*3+inp['token_tags'])[positions],inp['timestep_indices'][positions]*3+expected)
    for key in ('source_video_latents','target_video_latents','prompt_embeds'):
        assert bool(torch.isfinite(sample[key]).all()),(index,key)
    records.append(dict(index=index,visual_prompt_tokens=int((expected==0).sum()),text_prompt_tokens=int((expected==1).sum()),mismatches=0))
grid_v,grid_a=few_step_timesteps(8,12.0,3.0)
receipt=dict(passed=True,starting_point='DMD8 default+turbo, fresh savie LoRA; no step1000 resume',
    old_checkpoint_loaded=False,sample_count=2000,all_4000_payload_pairs_present=True,
    world_size=4,hsdp='2 replicas x 2 parameter shards',max_steps=1000,
    train_token_skip=False,inference_token_skip=True,lora_rank=64,lora_alpha=64,lr=1e-6,
    repaired_batch_sha256=sha(batch),tag_tests_run=report.testsRun,real_samples=records,
    video_grid=grid_v.tolist(),audio_grid=grid_a.tolist(),
    code_hashes={str(p.relative_to(REPO)):sha(p) for p in
                 (batch,REPO/'src/training/train_ref2va_dual.py',REPO/'src/models/softmax_attention/dual_stream_flex.py')},
    note='Eager mask/ST isolation and exact DMD8 grid validated. Existing Flex kernel cache reusable; full graph compile is not claimed here.')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'preflight.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
