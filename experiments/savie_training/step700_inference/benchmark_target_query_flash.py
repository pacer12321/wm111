"""Independent target-query bakeoff; never import into an active full run."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch

ROOT = Path('/cache/zhonghao/h3/savie_step1000_eval')
OLD = ROOT.parent / 'savie_step700_eval'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'speedops_loader'), str(OLD / 'partial_fix_candidate'),
               str(OLD), str(OLD / 'loader'), str(OLD / 'candidate'), str(OLD / 'repo')]
from savie_target_query_flash import target_pairs, prepare_target_batches, fill_target_queries
from savie_grouped_queries import predicate_for


def layouts(frames, per, text, audio, pad, world=2, device='cpu'):
    ss, se = text, text + frames * per
    ts, te = se + audio, se + audio + frames * per
    n = ((te + pad + world - 1) // world) * world
    s = SimpleNamespace(video_start=ss, video_end=se, num_frames=frames, tokens_per_frame=per)
    t = SimpleNamespace(video_start=ts, video_end=te, num_frames=frames, tokens_per_frame=per,
                        text_start=0, text_len=text, used_len=te)
    ids = torch.arange(n, device=device)
    p = (ids % (n // world)) * world + ids // (n // world)
    m = SimpleNamespace(physical_to_logical=p, logical_to_physical=torch.argsort(p), world_size=world)
    return t, s, n, m


def cpu_contract():
    cases = 0
    for frames in (1, 2, 5, 7, 12, 37):
        for world in (2, 4):
            t, s, n, m = layouts(frames, 6, 7, 3, 5, world)
            logical = m.physical_to_logical
            is_t = (logical >= t.video_start) & (logical < t.video_end)
            for active in (None, ~is_t, ~is_t | (logical % 3 == 0)):
                pairs = target_pairs(t, s, n, m, active)
                expected_q = torch.arange(t.video_start, t.video_end)
                if active is not None:
                    expected_q = expected_q[active[m.logical_to_physical[expected_q]]]
                actual_q = torch.cat([q for q, _ in pairs]) if pairs else torch.empty(0, dtype=torch.long)
                assert torch.equal(actual_q, expected_q)
                if actual_q.numel():
                    pred = predicate_for('target', actual_q, t, s, n)
                    for qids, keys in pairs:
                        qi = torch.searchsorted(actual_q, qids)
                        legal = pred(0, 0, qi[:, None], torch.arange(n)[None, :])
                        wanted = torch.zeros_like(legal)
                        wanted[:, keys] = True
                        assert torch.equal(legal, wanted), (frames, world)
                cases += 1
    print(json.dumps(dict(test='target_frame_group_mask_exact',cases=cases,passed=True)),flush=True)


@torch.inference_mode()
def gpu_benchmark(full, records):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    t, s, n, m = layouts(37, 1008, 6159, 250, 215, device='cuda') if full else layouts(12, 8, 7, 3, 5, device='cuda')
    heads, dim = (28, 128) if full else (4, 32)
    q, k, v = [torch.randn(n, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    kk, vv = k[m.logical_to_physical], v[m.logical_to_physical]
    scale = dim ** -.5
    flex = torch.compile(flex_attention, dynamic=True, fullgraph=True)
    cases = [('skip_off',None)]
    if full:
        payload = torch.load(ROOT / 'selector_adaptive_speedops.pt',map_location='cpu',weights_only=True)
        assert payload['selector_checkpoint_step'] == 1000
        logical = torch.ones(n, dtype=torch.bool, device='cuda')
        logical[t.video_start:t.video_end] = payload['active_target_mask'].to('cuda')
        cases.append(('own1000_adaptive',logical[m.physical_to_logical]))
    else:
        logical = m.physical_to_logical
        target = (logical >= t.video_start) & (logical < t.video_end)
        cases.append(('partial',~target | (logical % 3 == 0)))
    for label, keep in cases:
        pairs = target_pairs(t,s,n,m,keep)
        qids = torch.cat([ids for ids,_ in pairs])
        physical = m.logical_to_physical[qids]
        predicate = predicate_for('target',qids,t,s,n)
        bm = create_block_mask(predicate,None,None,qids.numel(),n,device='cuda',BLOCK_SIZE=128,_compile=True)
        def reference():
            output = torch.zeros_like(q)
            part = flex(q[physical].transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],block_mask=bm,scale=scale)[0].transpose(0,1)
            return output.index_copy_(0,physical,part)
        expected = reference()
        for size in ((1,2,4,8) if full else (4,)):
            batches = prepare_target_batches(t,s,n,m,keep,size)
            def candidate():
                output = torch.zeros_like(q)
                fill_target_queries(output,q,kk,vv,batches,scale)
                return output
            actual = candidate()
            rel = float((actual.float()-expected.float()).norm()/expected.float().norm())
            assert rel < .005, rel
            torch.testing.assert_close(actual,expected,rtol=.025,atol=.006)
            row = dict(real_shape=full,case=label,query_rows=qids.numel(),group_size=size,relative_l2=rel,
                       scope='Target softmax only; fresh synthetic QKV; pack/kernel/scatter included; common KV reorder and one-time plan excluded; no E2E claim')
            if full:
                def timed(fn):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record(); result=fn(); b.record(); b.synchronize(); return a.elapsed_time(b)
                for _ in range(2): reference(); candidate()
                samples={'flex':[],'flash':[]}
                for repeat in range(5):
                    order=[('flex',reference),('flash',candidate)]
                    if repeat%2:order.reverse()
                    for name,fn in order:samples[name].append(timed(fn))
                row.update(median_ms={name:statistics.median(values) for name,values in samples.items()},samples_ms=samples)
            records.append(row)
            (HERE/'target_query_flash_results.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(4101)
    cpu_contract()
    if not args.cpu_only:
        torch._dynamo.config.recompile_limit=64
        torch._dynamo.config.accumulated_recompile_limit=256
        records=[]
        gpu_benchmark(False,records)
        gpu_benchmark(True,records)
