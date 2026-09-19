"""Check real NPU event timing before loading the model; no global profiler."""
import json
import math
import torch
import torch_npu
import text_state_timer as timer

torch.npu.set_device(0)
a = torch.ones((256,256), dtype=torch.bfloat16, device='npu')
for _ in range(3):
    b = a @ a
torch.npu.synchronize()
timer._STEP = 1
for side in ('T','S'):
    with timer.branch_scope(side,0):
        with timer.text_scope():
            for _ in range(3):
                b = a @ a
torch.npu.synchronize()
values = [r['start'].elapsed_time(r['end']) for r in timer._ROWS]
assert len(values) == 2 and all(math.isfinite(t) and t > 0 for t in values)
print(json.dumps({'event_timer_smoke_passed': True, 'T_ms': values[0], 'S_ms': values[1]}),flush=True)
