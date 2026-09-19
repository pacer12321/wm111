"""Page sharing bake-off includes real KV pack + attention + output merge."""
import json
import statistics
import torch
from benchmark_target_query_flash import layouts, ROOT, HERE
import savie_target_lse_candidate as reference
import savie_target_kv_reuse as reuse
import savie_target_paged_kv as paged


def cpu_test():
    count = 0
    for world in (2, 4):
        for frames in (3, 7, 12, 37):
            t, s, n, m = layouts(frames, 6, 7, 3, 5, world)
            ids = m.physical_to_logical
            target = (ids >= t.video_start) & (ids < t.video_end)
            for keep in (None, ~target, ~target | (ids % 3 == 0)):
                for size in (16, 64, 256):
                    plan = paged.prepare(t, s, n, m, keep, size)
                    qids, shared, same = reuse.pairs(t, s, n, m, keep)
                    pool = plan[1].view(-1, size)
                    for groups, branch in zip((shared, same), plan[3:]):
                        if not groups:
                            assert branch is None
                            continue
                        table, lengths = branch[4], branch[3]
                        for i, (_, expected) in enumerate(groups):
                            actual = pool[table[i].long()].flatten()[:int(lengths[i])]
                            assert torch.equal(actual, expected)
                    count += 1
    print(json.dumps(dict(cpu_reconstruct_cases=count, passed=True)), flush=True)


@torch.inference_mode()
def main():
    torch.manual_seed(4101); torch.set_num_threads(4)
    cpu_test()
    t, s, n, m = layouts(37, 1008, 6159, 250, 215, device='cuda')
    q, k, v = [torch.randn(n, 28, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    kk, vv = k.index_select(0, m.logical_to_physical), v.index_select(0, m.logical_to_physical)
    logical = torch.ones(n, device='cuda', dtype=torch.bool)
    payload = torch.load(ROOT / 'selector_adaptive_targetlse.pt', map_location='cpu', weights_only=True)
    logical[t.video_start:t.video_end] = payload['active_target_mask'].cuda()
    records = []
    for mode, keep in [('refresh', None), ('skip', logical[m.physical_to_logical])]:
        old = reference.prepare(t, s, n, m, keep, 8)
        simple = reuse.prepare(t, s, n, m, keep, 64)
        def baseline(): return reference.fill(torch.zeros_like(q), q, kk, vv, old, 128 ** -.5)
        def simple_reuse(): return reuse.fill(torch.zeros_like(q), q, kk, vv, simple, 128 ** -.5)
        expected = baseline()
        for size in (16, 64, 256):
            plan = paged.prepare(t, s, n, m, keep, size)
            def candidate(): return paged.fill(torch.zeros_like(q), q, kk, vv, plan, 128 ** -.5)
            actual = candidate()
            error = float((actual.float() - expected.float()).norm() / expected.float().norm())
            torch.testing.assert_close(actual, expected, atol=.008, rtol=.035)
            assert error < .006
            funcs = [('frozen_lse', baseline), ('simple_reuse', simple_reuse), ('paged', candidate)]
            for _ in range(3):
                for _, fn in funcs: fn()
            samples = {name: [] for name, _ in funcs}
            for repeat in range(8):
                ordered = funcs[repeat % 3:] + funcs[:repeat % 3]
                for name, fn in ordered:
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); output = fn(); b.record(); b.synchronize()
                    samples[name].append(a.elapsed_time(b)); del output
            row = dict(mode=mode, page_size=size, unique_packed_rows=plan[1].numel(),
                       relative_l2=error, medians_ms={key: statistics.median(val) for key, val in samples.items()}, samples=samples)
            records.append(row)
            (HERE / 'target_paged_kv_results.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__': main()
