"""Time only the original text-state block; never cache or change its output.

One synchronization at each model-forward boundary, none inside text blocks.
NPU event intervals include stream idle/host dispatch gaps; CPU enqueue wall
is recorded separately. Neither is a promise of equivalent end-to-end savings.
"""
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import time

_BRANCH = None
_ROWS = []
_STEP = 0


@contextmanager
def branch_scope(side, layer):
    global _BRANCH
    assert side in ('T', 'S') and 0 <= layer < 50
    previous = _BRANCH
    _BRANCH = (side, layer)
    try:
        yield
    finally:
        _BRANCH = previous


@contextmanager
def text_scope():
    import torch
    assert _BRANCH is not None and _STEP > 0, 'Unbound text timing call'
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    side, layer = _BRANCH
    start.record()
    cpu_start = time.perf_counter()
    try:
        yield
    finally:
        cpu_seconds = time.perf_counter() - cpu_start
        end.record()
        _ROWS.append({'side': side, 'layer': layer, 'start': start, 'end': end,
                      'cpu_enqueue_seconds': cpu_seconds})


def install_forward(cls):
    assert not getattr(cls, '_text_timer_installed', False)
    cls._text_timer_installed = True
    original = cls.forward
    @functools.wraps(original)
    def timed(self, **kwargs):
        global _ROWS, _STEP
        import torch
        import torch.distributed as dist
        _ROWS = []
        _STEP += 1
        rank = dist.get_rank()
        assert dist.get_world_size() == 4
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        cpu_start = time.perf_counter()
        result = original(self, **kwargs)
        cpu_wall = time.perf_counter() - cpu_start
        end.record()
        flush_start = time.perf_counter()
        torch.npu.synchronize()
        rows = [{'side': r['side'], 'layer': r['layer'],
                 'npu_interval_ms': r['start'].elapsed_time(r['end']),
                 'cpu_enqueue_seconds': r['cpu_enqueue_seconds']} for r in _ROWS]
        assert [(r['side'], r['layer']) for r in rows] == [(s, i) for i in range(50) for s in ('T', 'S')]
        record = {'rank': rank, 'world': 4, 'forward': _STEP, 'text_calls': rows,
                  'forward_npu_interval_ms': start.elapsed_time(end),
                  'forward_cpu_wall_seconds': cpu_wall,
                  'sync_and_event_readback_seconds': time.perf_counter() - flush_start,
                  'metric': 'current_stream_elapsed_includes_host_gaps_not_pure_kernel_busy_time'}
        root = Path(os.environ['H3_PROFILE_OUTPUT'])
        root.mkdir(parents=True, exist_ok=True)
        with (root / f'text_state.rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if rank == 0:
            totals = {s: sum(r['npu_interval_ms'] for r in rows if r['side'] == s) for s in ('T','S')}
            print('TEXT_STATE_TIMING ' + json.dumps({'forward': _STEP, 'rank0_ms': totals}), flush=True)
        return result
    cls.forward = timed
