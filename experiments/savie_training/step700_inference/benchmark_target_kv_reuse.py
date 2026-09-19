"""Frozen 210.96s target path versus equal-key grouping/zero-copy KV views."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from benchmark_target_query_flash import layouts, ROOT, HERE
import savie_target_lse_candidate as baseline
import savie_target_kv_reuse as candidate


def cpu_test():
    count = 0
    for world in (2, 4):
        for frames in (3, 5, 7, 12, 37):
            t, s, n, m = layouts(frames, 6, 7, 3, 5, world)
            ids = m.physical_to_logical
            target = (ids >= t.video_start) & (ids < t.video_end)
            keeps = (None, ~target, ~target | (ids % 3 == 0),
                     ~target | (((ids - t.video_start) // t.tokens_per_frame) % 2 == 0))
            for keep in keeps:
                def as_dict(parts):
                    result = {}
                    for qs, ks in parts:
                        for qi in qs.tolist():
                            assert qi not in result
                            result[qi] = ks.tolist()
                    return result
                q0, a0, b0 = baseline.pairs(t, s, n, m, keep)
                q1, a1, b1 = candidate.pairs(t, s, n, m, keep)
                assert torch.equal(q0, q1)
                assert as_dict(a0) == as_dict(a1)
                assert as_dict(b0) == as_dict(b1)
                for size in (8, 16, 64):
                    plan = candidate.prepare(t, s, n, m, keep, size)
                    for batches in plan[1:]:
                        for qi, ki, oi, cq, ck, mq, mk, span, ordered in batches:
                            if span is not None:
                                assert torch.equal(ki, torch.arange(span[0], span[0] + span[1]))
                            if ordered:
                                assert torch.equal(oi, torch.arange(q0.numel()))
                count += 1
    return count


@torch.inference_mode()
def gpu_test(real, records):
    t, s, n, m = layouts(37, 1008, 6159, 250, 215, device='cuda') if real else layouts(12, 8, 7, 3, 5, device='cuda')
    h, d = (28, 128) if real else (4, 32)
    q, k, v = [torch.randn(n, h, d, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    kk, vv = k.index_select(0, m.logical_to_physical), v.index_select(0, m.logical_to_physical)
    logical = torch.ones(n, device='cuda', dtype=torch.bool)
    if real:
        payload = torch.load(ROOT / 'selector_adaptive_targetlse.pt', map_location='cpu', weights_only=True)
        assert payload['selector_checkpoint_step'] == 1000
        logical[t.video_start:t.video_end] = payload['active_target_mask'].cuda()
    else:
        logical[t.video_start:t.video_end] = torch.arange(t.video_end-t.video_start, device='cuda') % 3 == 0
    cases = [('refresh', None), ('skip', logical[m.physical_to_logical])]
    for mode, keep in cases:
        old_plan = baseline.prepare(t, s, n, m, keep, 8)
        def old():
            return baseline.fill(torch.zeros_like(q), q, kk, vv, old_plan, d ** -.5)
        expected = old()
        for size in (8, 16, 64):
            plan = candidate.prepare(t, s, n, m, keep, size)
            def new():
                return candidate.fill(torch.zeros_like(q), q, kk, vv, plan, d ** -.5)
            actual = new()
            error = float((actual.float() - expected.float()).norm() / expected.float().norm())
            torch.testing.assert_close(actual, expected, atol=.008, rtol=.035)
            assert error < .006
            row = dict(real_shape=real, mode=mode, group_size=size, relative_l2=error,
                       baseline_packed_kv_rows=sum(b[1].numel() for part in old_plan[1:] for b in part),
                       candidate_gathered_kv_rows=sum(b[1].numel() for part in plan[1:] for b in part if b[7] is None),
                       candidate_view_kv_rows=sum(b[1].numel() for part in plan[1:] for b in part if b[7] is not None),
                       attention_calls_before=sum(map(len, old_plan[1:])),
                       attention_calls_after=sum(map(len, plan[1:])))
            if real:
                for _ in range(3): old(); new()
                samples = {'baseline_lse': [], 'kv_reuse': []}
                for repeat in range(8):
                    order = [('baseline_lse', old), ('kv_reuse', new)]
                    if repeat % 2: order.reverse()
                    for label, fn in order:
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record(); output = fn(); b.record(); b.synchronize()
                        samples[label].append(a.elapsed_time(b))
                        del output
                row.update(median_ms={key: statistics.median(val) for key, val in samples.items()}, samples=samples)
            records.append(row)
            (HERE / 'target_kv_reuse_results.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(4101)
    print(json.dumps(dict(cpu_cases=cpu_test(), passed=True)), flush=True)
    if not args.cpu_only:
        records = []
        gpu_test(False, records)
        gpu_test(True, records)
