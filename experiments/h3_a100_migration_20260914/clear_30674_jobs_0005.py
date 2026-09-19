"""Explicitly authorized cleanup of current GPU jobs; no deletion or GPU reset."""
import json
import os
from pathlib import Path
import signal
import socket
import time
from clear_30674_new_jobs_2341 import identity


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    owned, parents = {}, []
    specs = [(1477252,1477256,1477258,'harvest.py',b'1'),
             (1477345,1477349,1477351,'train_projector.py',b'0'),
             (1477449,1477453,1477455,'trajectory_distill.py',b'0')]
    for wrapper, shell, worker, name, gpu in specs:
        assert Path(f'/proc/{wrapper}/cmdline').read_bytes().split(b'\0')[:3] == [
            b'bash',b'/home/ma-user/workspace/xiacong/ReMoGen/bin/xcrun',b'bash']
        assert Path(f'/proc/{worker}/cmdline').read_bytes().split(b'\0')[:2] == [
            b'/cache/xiacong/envs/wanres/bin/python',name.encode()]
        assert identity(worker)[1] == shell and identity(shell)[1] == wrapper
        assert b'CUDA_VISIBLE_DEVICES='+gpu in Path(f'/proc/{worker}/environ').read_bytes().split(b'\0')
        for pid in (wrapper,shell,worker):
            owned[pid] = identity(pid)[0]
        parents.extend((wrapper,shell))
    # Independently verified new Occlu4D GPU finetune, not its unrelated older jobs.
    occlu = identity(1478771)
    if occlu is not None:
        assert occlu[0] == 3598559000
        assert occlu[1] in (1,1478770)
        assert Path('/proc/1478771/cmdline').read_bytes().split(b'\0')[:3] == [b'python',b'-m',b'examples.finetune']
        assert Path('/proc/1478771/cwd').resolve() == Path('/home/ma-user/workspace/hanmo/Occlu4D')
        owned[1478771] = occlu[0]
        if identity(1478770) is not None:
            assert occlu[1] == 1478770
            assert b'python -m examples.finetune' in Path('/proc/1478770/cmdline').read_bytes()
            owned[1478770] = identity(1478770)[0]
            parents.append(1478770)
    else:
        assert identity(1478770) is None  # Both exited independently; do not target another PID.
    rows = {int(p.name):identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    while True:
        more = {p:v[0] for p,v in rows.items() if v and v[1] in owned and p not in owned}
        if not more:
            break
        owned.update(more)
    audit={'targets':owned,'signals':[]}
    with Path('/cache/zhonghao/h3/clear_30674_jobs_0005.json').open('x') as log:
        log.write(json.dumps(audit)); log.flush()
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid in parents+[p for p in owned if p not in parents]:
                current=identity(pid)
                if current and current[0]==owned[pid] and current[2]!='Z':
                    try:
                        os.kill(pid,sig); audit['signals'].append([pid,sig.name])
                    except ProcessLookupError:
                        pass
            deadline=time.monotonic()+(8 if sig==signal.SIGTERM else 3)
            while time.monotonic()<deadline:
                if not any((v:=identity(p)) and v[0]==start and v[2]!='Z' for p,start in owned.items()):
                    break
                time.sleep(.25)
        audit['remaining']={p:v for p,start in owned.items() if (v:=identity(p)) and v[0]==start and v[2]!='Z'}
        log.seek(0); log.truncate(); log.write(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)
    assert not audit['remaining']


if __name__=='__main__':
    main()
