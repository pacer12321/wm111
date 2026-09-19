"""User-authorized termination of three verified new 30674 jobs, not files."""
import json
import os
from pathlib import Path
import signal
import socket
import time

TARGETS = {
    1461870: (3598350150, 1461868, 'projector_in_loop.py'),
    1462377: (3598352783, 1462375, 'train_projector.py'),
    1466211: (3598409361, 1466209, 'trajectory_distill.py'),
}


def identity(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text()
        parts = data[data.rfind(')') + 2:].split()
        return int(parts[19]), int(parts[1]), parts[0]
    except FileNotFoundError:
        return None


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    selected = {}
    parents = []
    for pid, (start, parent, script) in TARGETS.items():
        actual = identity(pid)
        assert actual and actual[:2] == (start, parent), (pid, actual)
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert argv[0] == b'/cache/xiacong/envs/wanres/bin/python'
        assert script.encode() in argv
        assert Path(f'/proc/{pid}/cwd').resolve() == Path('/home/ma-user/workspace/xiacong/encoder-seam')
        parent_id = identity(parent)
        parent_argv = Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
        assert parent_id and parent_argv[:2] == [b'bash', b'-c']
        assert script.encode() in parent_argv[2] and b'workspace/xiacong/encoder-seam' in parent_argv[2]
        selected[pid] = start
        selected[parent] = parent_id[0]
        parents.append(parent)
    rows = {int(p.name): identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    while True:
        children = {pid: value[0] for pid, value in rows.items()
                    if value and value[1] in selected and pid not in selected}
        if not children:
            break
        selected.update(children)
    evidence = {'verified_roots': list(TARGETS), 'selected_pid_starttime': selected, 'signals': []}
    output = Path('/cache/zhonghao/h3/clear_new_jobs_20260914_2341.json')
    # Exclusive evidence file doubles as a one-shot guard against accidental replay.
    with output.open('x') as handle:
        handle.write(json.dumps(evidence, indent=2))
        handle.flush()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            # Parents first: stop their queued harvest/training commands as well.
            for pid in parents + [p for p in selected if p not in parents]:
                actual = identity(pid)
                if actual and actual[0] == selected[pid] and actual[2] != 'Z':
                    try:
                        os.kill(pid, sig)
                        evidence['signals'].append([pid, sig.name])
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + (8 if sig == signal.SIGTERM else 3)
            while time.monotonic() < deadline:
                if not any((v := identity(p)) and v[0] == start and v[2] != 'Z'
                           for p, start in selected.items()):
                    break
                time.sleep(0.25)
        evidence['remaining'] = {p: v for p, start in selected.items()
                                 if (v := identity(p)) and v[0] == start and v[2] != 'Z'}
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)
    assert not evidence['remaining'], 'Some verified processes remain'


if __name__ == '__main__':
    main()
