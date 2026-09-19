"""Stop only two freshly verified xiacong GPU job trees, preserving all files."""
import json
import os
from pathlib import Path
import signal
import socket
import time
from clear_30674_new_jobs_2341 import identity


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    specs = [(1470068, 1470072, 1470074, 'projector_in_loop.py', b'1'),
             (1470153, 1470157, 1470159, 'train_projector.py', b'0')]
    owned = {}
    ordered_parents = []
    for wrapper, shell, worker, name, gpu in specs:
        wargv = Path(f'/proc/{wrapper}/cmdline').read_bytes().split(b'\0')
        assert wargv[:3] == [b'bash', b'/home/ma-user/workspace/xiacong/ReMoGen/bin/xcrun', b'bash']
        sargv = Path(f'/proc/{shell}/cmdline').read_bytes().split(b'\0')
        assert sargv[:2] == [b'bash', b'-c'] and name.encode() in sargv[2]
        argv = Path(f'/proc/{worker}/cmdline').read_bytes().split(b'\0')
        assert argv[:2] == [b'/cache/xiacong/envs/wanres/bin/python', name.encode()]
        assert b'CUDA_VISIBLE_DEVICES=' + gpu in Path(f'/proc/{worker}/environ').read_bytes().split(b'\0')
        assert identity(worker)[1] == shell and identity(shell)[1] == wrapper
        assert Path(f'/proc/{worker}/cwd').resolve() == Path('/home/ma-user/workspace/xiacong/encoder-seam')
        for pid in (wrapper, shell, worker):
            owned[pid] = identity(pid)[0]
        ordered_parents.extend((wrapper, shell))
    rows = {int(p.name): identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    while True:
        children = {p: v[0] for p, v in rows.items() if v and v[1] in owned and p not in owned}
        if not children:
            break
        owned.update(children)
    record = {'targets': owned, 'signals': []}
    with Path('/cache/zhonghao/h3/clear_xiacong_30674_2354.json').open('x') as log:
        log.write(json.dumps(record))
        log.flush()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in ordered_parents + [p for p in owned if p not in ordered_parents]:
                now = identity(pid)
                if now and now[0] == owned[pid] and now[2] != 'Z':
                    try:
                        os.kill(pid, sig)
                        record['signals'].append([pid, sig.name])
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + (8 if sig == signal.SIGTERM else 3)
            while time.monotonic() < deadline:
                if not any((v := identity(p)) and v[0] == start and v[2] != 'Z' for p, start in owned.items()):
                    break
                time.sleep(0.25)
        record['remaining'] = {p: v for p, start in owned.items()
                               if (v := identity(p)) and v[0] == start and v[2] != 'Z'}
        log.seek(0)
        log.truncate()
        log.write(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    assert not record['remaining']


if __name__ == '__main__':
    main()
