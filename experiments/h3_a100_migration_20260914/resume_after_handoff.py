"""One pre-inference queue resume after verified own reservation release."""
import fcntl
import json
import subprocess
from pathlib import Path
import run_abcd_cuda as queue

history = queue.WORK / 'history/holder_teardown_wait_20260914'
lock = (queue.WORK / 'queue.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not history.exists() and not (queue.WORK / 'results').exists()
state = queue.read(queue.WORK / 'queue_status.json')
assert state['status'] == 'failed' and not state.get('formal_completed')
assert 'GPU2/3 are not idle' in state['error']
assert queue.pid_identity(state['pid']) is None
assert queue.read(queue.WORK / 'cuda_kernel_test.json')['passed']
assert queue.read(queue.WORK / 'environment_status.json')['status'] == 'installed_not_gpu_validated'
assert queue.read(queue.HOLDER / 'reservation.json')['status'] == 'released'
queue.assert_idle()
history.mkdir(parents=True)
for name in ('queue_status.json', 'queue.log', 'cuda_kernel_test.json', 'finalize_status.json'):
    path = queue.WORK / name
    path.rename(history / name)
fcntl.flock(lock, fcntl.LOCK_UN)
result = subprocess.check_output([str(queue.ENV / 'bin/python'), str(queue.WORK / 'run_abcd_cuda.py'), '--launch'], text=True)
print(result, flush=True)
