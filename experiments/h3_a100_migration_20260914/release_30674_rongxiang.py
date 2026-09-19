"""Release only the positively identified, user-authorized stopped training."""
import json
import os
from pathlib import Path
import signal
import socket
import time

assert socket.gethostname() == 'os-node-created-mgf6h'
TASK = Path('/home/ma-user/workspace/rongxiang/ViBT-Wan')


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return fields[19], fields[1]
    except FileNotFoundError:
        return None


targets = {}
for pid, suffix in [(1233274, 'train_sr_wan22_ti2v_edit.py'),
                    (1233271, 'train_sr_wan22_ti2v_edit.sh')]:
    if identity(pid) is None:
        continue
    proc = Path(f'/proc/{pid}')
    assert proc.stat().st_uid == os.getuid()
    assert proc.joinpath('cwd').resolve() == TASK
    cmd = proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert any(arg.endswith(suffix.encode()) for arg in cmd), (pid, 'identity mismatch')
    targets[pid] = identity(pid)
if 1233274 in targets:
    assert targets[1233274][1] == '1233271'
    assert not Path('/proc/1233274/task/1233274/children').read_text().strip()

for sig in (signal.SIGTERM, signal.SIGCONT):
    for pid, saved in targets.items():
        if identity(pid) == saved:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
deadline = time.monotonic() + 10
while time.monotonic() < deadline and any(identity(p) == i for p, i in targets.items()):
    time.sleep(0.2)
for pid, saved in targets.items():
    if identity(pid) == saved:
        os.kill(pid, signal.SIGKILL)
time.sleep(1)
print(json.dumps({'terminated_authorized_pids': list(targets),
                  'remaining_same_identity': [p for p, i in targets.items() if identity(p) == i],
                  'other_processes_untouched': True}))
