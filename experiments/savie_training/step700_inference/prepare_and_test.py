"""Prepare an isolated copy and fail closed on real step700 mapping/mask errors."""
import json
import os
from pathlib import Path
import shutil
import sys
import types

ROOT = Path('/cache/zhonghao/h3')
RUN = ROOT / 'savie_step700_eval'
OLD = ROOT / 'dmd8_b_skip_20260917'
CODE = Path(__file__).parent
os.environ['SAVIE_STEP700_CHECKPOINT'] = '/temp/zhonghao/savie_eval_step700/savie_step000700.pt'
sys.path[:0] = [str(CODE), str(RUN/'repo'), str(RUN/'deps'), str(OLD/'loader')]


def prepare():
    candidate = RUN/'candidate'
    if not candidate.exists():
        shutil.copytree(OLD/'candidate_B_skip', candidate, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    loader = RUN/'loader'
    loader.mkdir(exist_ok=True)
    for path in (OLD/'loader').glob('*.py'):
        shutil.copy2(path, loader/path.name)
    shutil.copy2(OLD/'loader/manifest.json', loader/'manifest.json')
    shutil.copy2(CODE/'streaming_shards.py', loader/'streaming_shards.py')
    shutil.copy2(CODE/'savie_overlay.py', loader/'savie_overlay.py')
    rel = Path('vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py')
    shutil.copy2(CODE/'minimax_h3_transformer.py', candidate/rel)
    sys.path[:0] = [str(loader), str(candidate)]


def test_weights():
    import torch
    import manifest_builder
    from streaming_shards import RangeReader
    from savie_overlay import overlay_plans, weights
    from torch_streaming_shards import merge_owned_interval_
    state = weights()
    obj = torch.load(os.environ['SAVIE_STEP700_CHECKPOINT'], map_location='cpu', mmap=True, weights_only=False)
    config = obj['model_spec']['transforms'][0]['config']
    assert config['anchor_frames'] == 'both', config
    assert config['softmax_attention'] == {'radius': 1, 'chunk': 5}, config
    print(json.dumps({'checkpoint_metadata': obj['metadata'], 'attention_config': config}), flush=True)
    manifest = json.loads((OLD/'loader/manifest.json').read_text())
    # Ordering is irrelevant for isolated plan tests. Production reattaches real meta-model ordering.
    manifest['model_metadata_validated'] = True
    for i, p in enumerate(manifest['plans']):
        p['model_iteration_index'] = i
    readers = {}
    def reader(path):
        if path not in readers:
            readers[path] = RangeReader(path)
        return readers[path]
    total = 0
    for index in range(50):
        plans = overlay_plans(manifest_builder.tensor_plans_for_main_block(manifest,index,reader), index)
        total += sum(1 for k in state if k.startswith(f'transformer_blocks.{index}.'))
        if index not in (0,24,49):
            continue
        for plan in plans:
            if not plan.loras:
                continue
            newest = plan.loras[-(3 if plan.name.endswith('qkv_proj.weight') else 1):]
            if not any('savie' in p.a.key for p in newest):
                continue
            columns = plan.source.info['shape'][1]
            for lora in newest:
                if 'savie' not in lora.a.key:
                    continue
                # Deliberately non-row-aligned shard boundary.
                start = lora.start_row*columns + 7
                count = columns*2 - 11
                target = torch.zeros(count,dtype=torch.bfloat16)
                merge_owned_interval_(target,tensor_start=start,shape=plan.source.info['shape'],loras=(lora,))
                a,b = state[lora.a.key].float(),state[lora.b.key].float()
                expected = (b[:256]@a).to(torch.bfloat16).flatten()[7:7+count]
                torch.testing.assert_close(target,expected,rtol=0,atol=0)
    assert total==1150, total
    print('PASS checkpoint=700 mapped_all_1150 merge_shard_boundaries_exact=true',flush=True)


def test_mask():
    import torch
    from savie_overlay import make_mask
    from vllm_omni.diffusion.models.minimax_h3.openvdn_npu import OpenVDNLayout,window_bounds
    from src.models.sequence_layout import dual_layout_from_video_spans
    from savie_overlay import training_mask_predicate
    make_dual_stream_mask_mod=training_mask_predicate()
    from torch.nn.attention.flex_attention import create_block_mask,flex_attention
    torch.manual_seed(4101)
    device='cuda:0'
    # Both stream boundaries deliberately lie INSIDE 128-token blocks.
    frames,rows,text,audio=7,12,5,6
    source_start=text
    target_start=text+frames*rows+audio
    used=target_start+frames*rows
    length=used+3
    assert length%2==0
    source=OpenVDNLayout(used,source_start,frames,rows,3,4,0,text)
    target=OpenVDNLayout(used,target_start,frames,rows,3,4,0,text)
    perm=torch.cat([torch.arange(i,length,2,device=device) for i in range(2)])
    dual=dual_layout_from_video_spans(used,[dict(role='reference',start=source_start,latent_grid=(frames,3,4)),dict(role='target',start=target_start,latent_grid=(frames,3,4))],torch.arange(text))
    original=make_dual_stream_mask_mod(dual,window_bounds(frames),device,'both')
    logical=torch.arange(length,device=device)
    full_reference=original(0,0,logical[:,None],logical[None,:]) & (logical[:,None]<used)&(logical[None,:]<used)
    full_reference |= (logical[:,None]>=used)&(logical[:,None]==logical[None,:])
    q,k,v=[torch.randn(length,4,32,device=device,dtype=torch.bfloat16) for _ in range(3)]
    flex=torch.compile(flex_attention,dynamic=True)
    for partial in (False,True):
        active=torch.arange(length,device=device)
        if partial:
            logical_active=(perm<target_start)|(perm>=used)|((perm%3)==0)
            active=active[logical_active]
        predicate=make_mask(target,source,length,device,perm,active)
        actual=predicate(0,0,torch.arange(active.numel(),device=device)[:,None],logical[None,:])
        expected=full_reference.index_select(0,perm[active]).index_select(1,perm)
        assert torch.equal(actual,expected)
        bm=create_block_mask(predicate,B=None,H=None,Q_LEN=active.numel(),KV_LEN=length,device=device,_compile=True)
        qp=q[perm[active]].transpose(0,1)[None]
        kp=k[perm].transpose(0,1)[None]
        vp=v[perm].transpose(0,1)[None]
        out=flex(qp,kp,vp,block_mask=bm)
        reference=torch.nn.functional.scaled_dot_product_attention(qp.float(),kp.float(),vp.float(),attn_mask=expected[None,None]).to(out.dtype)
        torch.testing.assert_close(out,reference,atol=.02,rtol=.02)
        changed=(perm>=target_start)&(perm<used)
        kp2=kp.clone();vp2=vp.clone()
        kp2[:,:,changed]=torch.randn_like(kp2[:,:,changed])*100
        vp2[:,:,changed]=torch.randn_like(vp2[:,:,changed])*100
        other=flex(qp,kp2,vp2,block_mask=bm)
        source_q=(perm[active]>=source_start)&(perm[active]<source_start+frames*rows)
        torch.testing.assert_close(out[:,:,source_q],other[:,:,source_q],rtol=0,atol=0)
        print(json.dumps({'test':'compiled_dual_mask','partial':partial,'max_error':float((out-reference).abs().max()),'ST_isolation':'exact','boundary_tokens':'checked'}),flush=True)


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    prepare()
    test_weights()
    test_mask()
    payload=torch.load(OLD/'selector_dmd8_latent/fixed_selector_payload.pt',weights_only=True,map_location='cpu')
    payload['active_target_mask']=torch.ones_like(payload['active_target_mask'],dtype=torch.bool)
    payload['source_shape']=(1,24,37,48,84)
    torch.save(payload,RUN/'selector.pt')
    (RUN/'preflight_passed.json').write_text(json.dumps({'checkpoint_step':700,'all_1150_weights_mapped':True,'compiled_mask_equivalence':True,'source_target_isolation':True}))
