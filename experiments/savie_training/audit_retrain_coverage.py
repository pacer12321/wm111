"""Report configuration/data coverage gaps without changing the experiment."""
from collections import Counter
import json,re,sys
from pathlib import Path
import torch
ROOT=Path('/cache/zhonghao/h3');REPO=ROOT/'train_repos/vdn-minimax-h3'
OUT=ROOT/'train_runs/savie_dmd8_skipoff_2k_tagsfix_20260919'
sys.path.insert(0,str(REPO))
from src.models.sequence_layout import DualStreamSequenceLayout
from src.models.softmax_attention.window import window_bounds
from src.models.softmax_attention.dual_stream_flex import make_dual_stream_mask_mod
rows=[]
for frames in (12,37):
    # One token/frame computes frame-edge support exactly; full patches cancel.
    layout=DualStreamSequenceLayout(seq_len=2+2*frames,source_start=2,source_frames=frames,
        source_tokens_per_frame=1,target_start=2+frames,target_frames=frames,
        target_tokens_per_frame=1,text_start=0,text_len=2)
    pred=make_dual_stream_mask_mod(layout,window_bounds(frames,1,5),'cpu',anchor_frames='both')
    si=torch.arange(2,2+frames);ti=si+frames
    ss=pred(None,None,si[:,None],si[None,:]);tt=pred(None,None,ti[:,None],ti[None,:])
    ts=pred(None,None,ti[:,None],si[None,:]);st=pred(None,None,si[:,None],ti[None,:])
    rows.append(dict(frames=frames,ss_frame_edges=int(ss.sum()),tt_frame_edges=int(tt.sum()),
        dense_frame_edges=frames**2,ss_retained_fraction=float(ss.float().mean()),
        ts_frame_edges=int(ts.sum()),st_frame_edges=int(st.sum()),
        ss_per_frame=ss.sum(1).tolist()))
manifest=Path('/temp/zhonghao/savie_stream/manifests/train_2k_reserve_filled_actual.jsonl')
data=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
counts=Counter();label_fields=Counter();examples={}
for item in data:
    instruction=item.get('instruction','').lower()
    for name,pattern in {
        'literal_style':r'\bstyle\b',
        'literal_entire_or_whole':r'\b(entire|whole)\b',
        'color_words':r'\b(red|blue|green|yellow|black|white|color|colour)\b',
        'clothing_words':r'\b(shirt|clothes|clothing|jacket|dress|coat)\b',
    }.items():
        if re.search(pattern,instruction): counts[name]+=1
    for key in ('edit_type','edit_category','category','task_type'):
        if key in item: label_fields[str((key,item[key]))]+=1
report=dict(frame_support=rows,samples=len(data),nonexclusive_text_keyword_counts=dict(counts),
    keyword_counts_are_not_validated_edit_labels=True,available_category_labels=dict(label_fields),
    manifest_sha256=__import__('hashlib').sha256(manifest.read_bytes()).hexdigest(),
    confirmed_audio_rows=dict(training=4,current_eval=250),
    findings_are_distribution_mismatches_not_proven_failure_causes=True)
(OUT/'coverage_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
