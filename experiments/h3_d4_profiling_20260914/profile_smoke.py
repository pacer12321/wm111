"""Small profiler API/export check, launched only under the supervisor leases."""
import json
import os
from pathlib import Path
import torch
import torch_npu

root = Path(os.environ['H3_PROFILE_OUTPUT']) / 'smoke'
torch.npu.set_device(0)
a = torch.ones((256, 256), device='npu', dtype=torch.bfloat16)
for _ in range(3):
    b = a @ a
torch.npu.synchronize()
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    record_shapes=False, profile_memory=False, with_stack=False,
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root), worker_name='smoke',
                                                               analyse_flag=True, async_mode=False)):
    with torch.profiler.record_function('H3PROFILE/smoke'):
        for _ in range(3):
            b = a @ a
        torch.npu.synchronize()
traces = list(root.rglob('trace_view.json'))
assert len(traces) == 1, traces
trace = json.loads(traces[0].read_text())
events = trace['traceEvents'] if isinstance(trace, dict) else trace
assert any(e.get('name') == 'H3PROFILE/smoke' for e in events)
assert any('MatMul' in e.get('name', '') or 'Matmul' in e.get('name', '') for e in events)
print(json.dumps({'profiler_smoke_passed': True, 'trace': str(traces[0])}), flush=True)
