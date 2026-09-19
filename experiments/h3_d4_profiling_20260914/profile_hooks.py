"""Diagnostic-only observers. No attention, weights, scheduling or tensor edits.

Forward 3: CPU/NPU operator trace on rank 0. Forward 4: nested NPU event
intervals on all ranks. Other forwards: CPU wall only. Inclusive intervals
include stream waits/host starvation and are NOT kernel busy-time measurements.
"""
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import time

_ACTIVE = False
_EVENTS = False
_STACK = []
_ROWS = []
_LAYOUT = None
_COLLECTIVE = 0
_CALL = 0
_INSTALLED = False


def output(name, value):
    root = Path(os.environ['H3_PROFILE_OUTPUT'])
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value, indent=2) + '\n')


@contextmanager
def span(name):
    if not _ACTIVE:
        yield
        return
    import torch
    row = None
    if _EVENTS:
        row = {'id': len(_ROWS), 'parent': _STACK[-1] if _STACK else None, 'name': name,
               'start': torch.npu.Event(enable_timing=True),
               'end': torch.npu.Event(enable_timing=True)}
        _ROWS.append(row)
        _STACK.append(row['id'])
        row['start'].record()
    try:
        with torch.profiler.record_function('H3PROFILE/' + name):
            yield
    finally:
        if row is not None:
            row['end'].record()
            assert _STACK.pop() == row['id']


def observer(fn, name):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with span(name):
            return fn(*args, **kwargs)
    return wrapped


def install(namespace):
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Profiling observers installed twice')
    _INSTALLED = True
    import sys
    import torch
    import torch.distributed as dist
    import torch_npu
    prefix = namespace['__name__'].rsplit('.', 1)[0]
    op = sys.modules[prefix + '.openvdn_npu']
    modules = [op, sys.modules[namespace['__name__']]]
    dual = sys.modules.get(prefix + '.dual_stream_attention')
    if dual is not None:
        modules.append(dual)

    # Replace all already-imported aliases of a function with the SAME observer.
    # This preserves the number and order of all original function executions.
    for attr, label in {
        '_fusion_attention_tnd': 'softmax/fusion_kernel_call',
        '_frame_statistics': 'linear/frame_statistics',
        '_vdn_factor_apply': 'linear/cholesky_solve',
        '_scan_states': 'linear/bidirectional_scan',
        '_gather_linear_state': 'linear/gather_state',
        'local_frame_sums_counts': 'linear/frame_means_local',
    }.items():
        original = getattr(op, attr)
        wrapped = observer(original, label)
        for module in modules:
            for key, value in list(vars(module).items()):
                if value is original:
                    setattr(module, key, wrapped)

    op.LinearAttentionSepConv.apply = observer(op.LinearAttentionSepConv.apply, 'linear/short_conv')
    op.FrameKDAAlpha.forward_head_shard = observer(op.FrameKDAAlpha.forward_head_shard, 'linear/alpha')
    original_delta = op._delta_factor_apply
    @functools.wraps(original_delta)
    def delta(*args, **kwargs):
        # Inside the video scan this is video state; the separate call in
        # forward_head_shard initializes text state. Do not conflate the two.
        inside_scan = any(_ROWS[i]['name'] == 'linear/bidirectional_scan' for i in _STACK)
        # Trace-only scopes have no event stack; tensor frame count identifies
        # the text state for this fixed 37-latent-frame diagnostic profile.
        if not _EVENTS:
            tensor_a = args[2] if len(args) > 2 else kwargs['A']
            inside_scan = tensor_a.shape[0] != 1
        with span('linear/video_delta' if inside_scan else 'linear/text_delta'):
            return original_delta(*args, **kwargs)
    op._delta_factor_apply = delta
    original_linear = op.BidirectionalLinearBranch.forward_head_shard

    @functools.wraps(original_linear)
    def linear(self, *args, **kwargs):
        layout = args[3] if len(args) > 3 else kwargs['layout']
        stream = 'S' if _LAYOUT is not None and layout.video_start == getattr(_LAYOUT, 'source_start', -1) else 'T'
        with span('linear/' + stream):
            return original_linear(self, *args, **kwargs)
    op.BidirectionalLinearBranch.forward_head_shard = linear

    attention = namespace['MiniMaxH3Attention']
    original_ulysses = attention._run_openvdn_ulysses

    @functools.wraps(original_ulysses)
    def ulysses(self, *args, **kwargs):
        global _LAYOUT, _COLLECTIVE
        previous = _LAYOUT
        _LAYOUT = args[5] if len(args) > 5 else kwargs['layout']
        _COLLECTIVE = 0
        strategy = args[7] if len(args) > 7 else kwargs['strategy']
        if not getattr(strategy, '_h3_profile_observed', False):
            strategy.pre_attention = observer(strategy.pre_attention, 'communication/ulysses_pre')
            strategy.post_attention = observer(strategy.post_attention, 'communication/ulysses_post')
            strategy._h3_profile_observed = True
        try:
            with span('attention/hybrid_core'):
                return original_ulysses(self, *args, **kwargs)
        finally:
            _LAYOUT = previous
    attention._run_openvdn_ulysses = ulysses
    attention._openvdn_softmax = observer(attention._openvdn_softmax, 'softmax/local_global_total')
    attention.forward = observer(attention.forward, 'attention/total')
    namespace['MiniMaxH3MLP'].forward = observer(namespace['MiniMaxH3MLP'].forward, 'mlp')
    namespace['MiniMaxH3DiTBlock'].forward = observer(namespace['MiniMaxH3DiTBlock'].forward, 'block')

    original_reduce = dist.all_reduce
    @functools.wraps(original_reduce)
    def all_reduce(*args, **kwargs):
        global _COLLECTIVE
        if not _ACTIVE or _LAYOUT is None:
            return original_reduce(*args, **kwargs)
        _COLLECTIVE += 1
        stream = 'T' if _COLLECTIVE <= 2 else 'S'
        with span('communication/frame_allreduce_' + stream):
            return original_reduce(*args, **kwargs)
    dist.all_reduce = all_reduce

    model = namespace['MiniMaxH3DiTModel']
    model._embed = observer(model._embed, 'embedding_and_text_refiner')
    original_forward = model.forward
    @functools.wraps(original_forward)
    def forward(self, **kwargs):
        global _CALL, _ACTIVE, _EVENTS, _ROWS
        _CALL += 1
        index = _CALL
        rank = dist.get_rank()
        prof = None
        if index in (3, 4):
            dist.barrier()
            torch.npu.synchronize()
        if index == 3 and rank == 0:
            prof = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(Path(os.environ['H3_PROFILE_OUTPUT']) / 'trace'), worker_name='rank0',
                    analyse_flag=True, async_mode=False))
            prof.start()
        _ACTIVE = (index == 3 and rank == 0) or index == 4
        _EVENTS = index == 4
        _ROWS = []
        started = time.perf_counter()
        try:
            with span('dit_forward'):
                result = original_forward(self, **kwargs)
        finally:
            wall = time.perf_counter() - started
            _ACTIVE = False
            _EVENTS = False
            if prof is not None:
                prof.stop()
            if index == 4:
                torch.npu.synchronize()
                output(f'event_intervals.rank{rank}.json', {
                    'case': os.environ['H3_PROFILE_CASE'], 'rank': rank, 'forward': index,
                    'metric': 'inclusive_current_stream_elapsed_ms_NOT_kernel_busy_time',
                    'rows': [{k: v for k, v in row.items() if k not in ('start', 'end')} |
                             {'elapsed_ms': row['start'].elapsed_time(row['end'])} for row in _ROWS]})
            if index in (3, 4):
                dist.barrier()
            with (Path(os.environ['H3_PROFILE_OUTPUT']) / f'forwards.rank{rank}.jsonl').open('a') as f:
                f.write(json.dumps({'forward': index, 'cpu_wall_seconds': wall,
                                    'instrumented': index in (3, 4), 'rank': rank}) + '\n')
        return result
    model.forward = forward
    print('H3_PROFILE_OBSERVERS_INSTALLED ' + namespace['__file__'], flush=True)
