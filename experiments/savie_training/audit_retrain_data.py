"""Read all 2K cache identities/geometry/tags without loading 90GB embeddings."""
from collections import Counter
import json
from pathlib import Path
import time
import torch
ROOT=Path('/temp/zhonghao/savie_stream')
OUT=Path('/cache/zhonghao/h3/train_runs/savie_dmd8_skipoff_2k_tagsfix_20260919')
torch.set_num_threads(4)
manifests={p.name:[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
           for p in (ROOT/'manifests').glob('train_2k*.jsonl')}
shapes=Counter();tags=Counter();ids=[];errors=[];manifest_matches=Counter();examples=[]
begin=time.monotonic()
for i in range(2000):
    try:
        v=torch.load(ROOT/'samples'/f'video_{i:06d}.pt',map_location='cpu',weights_only=True,mmap=True)
        p=torch.load(ROOT/'samples'/f'prompt_{i:06d}.pt',map_location='cpu',weights_only=True,mmap=True)
        if v['sample_id']!=p['sample_id']: raise ValueError(f"video/prompt identity mismatch {v['sample_id']} != {p['sample_id']}")
        a,b=v['source_video_latents'],v['target_video_latents']
        if a.shape!=b.shape: raise ValueError('source/target shape mismatch')
        if a.shape[0]!=24 or len(a.shape)!=4: raise ValueError('unexpected VAE latent layout')
        h,t=p['prompt_embeds'],p['text_token_tags']
        if h.ndim!=2 or h.shape[1]!=5120 or t.shape!=(h.shape[0],): raise ValueError('prompt/tag geometry mismatch')
        if not bool(((t==0)|(t==1)).all()): raise ValueError('unknown modality tag')
        if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()): raise ValueError('nonfinite video cache')
        ids.append(v['sample_id']);shapes[str(tuple(a.shape))]+=1
        tags[str((int((t==0).sum()),int((t==1).sum())))]+=1
        for name,rows in manifests.items():
            if i<len(rows) and rows[i]['sample_id']==v['sample_id']: manifest_matches[name]+=1
        if i in (0,1,499,999,1499,1999):
            examples.append(dict(index=i,id=v['sample_id'],shape=list(a.shape),source_std=float(a.float().std()),target_std=float(b.float().std()),instruction=v.get('instruction')))
    except Exception as exc:
        errors.append(dict(index=i,error=str(exc)))
    if i%250==249: print(json.dumps(dict(scanned=i+1,errors=len(errors),seconds=time.monotonic()-begin)),flush=True)
report=dict(scanned=2000,errors=errors,unique_sample_ids=len(set(ids)),latent_shapes=dict(shapes),
            prompt_tag_counts=dict(tags),matching_manifest_rows=dict(manifest_matches),examples=examples,
            seconds=time.monotonic()-begin,finite_check_scope='all source/target latents; prompt metadata/tags all, full prompt finite sampled by preflight')
(OUT/'data_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
assert not errors and len(ids)==len(set(ids))==2000
assert max(manifest_matches.values(),default=0)==2000, 'No manifest matches all 2K cached pairs'
