import os
import signal
import socket
import time
import shutil
from pathlib import Path

assert socket.gethostname() == 'os-node-created-mgf6h'
root = Path('/cache/zhonghao/h3').resolve()
targets = [root / 'reservation_30674_gpu0_20260914', root / 'reservation_30674_gpu1_20260914']
for target in targets:
    assert not target.is_symlink() and target.resolve().parent == root
    if target.exists():
        import json
        meta = json.loads((target / 'reservation.json').read_text())
        assert meta['owner'] == 'zhonghao' and meta['status'] == 'released'
        assert not Path('/proc/' + str(meta['pid'])).exists(), 'reservation PID still exists'

def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[19], int(stat[1]), (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (FileNotFoundError, ProcessLookupError):
        return None

job = '/home/ma-user/workspace/xuchubo/neighborhood_readout_30674_20260915_retry01_01/'
launcher = identity(1486143)
captured = {}
if launcher:
    assert job + 'ae_launch.py --worker' in launcher[2], launcher
    captured[1486143] = launcher
    while True:
        before = len(captured)
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            pid = int(p.name)
            value = identity(pid)
            if value and value[1] in captured:
                captured[pid] = value
        if len(captured) == before:
            break
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, original in captured.items():
            now = identity(pid)
            if now and now[0] == original[0]:
                try:
                    os.kill(pid, sig)
                    print('signal', sig.name, pid, flush=True)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(8)
for target in targets:
    if target.exists():
        shutil.rmtree(target)
        print('deleted', target, flush=True)
print('No experiment restart. Model, code and result files preserved.', flush=True)
