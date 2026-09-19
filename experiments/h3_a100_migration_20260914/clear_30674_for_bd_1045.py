"""One-shot, explicitly authorized cleanup of verified GPU job trees only."""
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

TARGETS = {
    1550121: (3599553099, '/home/ma-user/workspace/hanmo/Occlu4D', 'examples.train'),
    1668052: (3601677411, '/home/ma-user/workspace/xiacong/ReMoGen', 'train_projector.py'),
    1692739: (3602063228, '/home/ma-user/workspace/xiacong/ReMoGen', 'harvest.py'),
}


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        parts = raw[raw.rfind(')') + 2:].split()
        return {'start': int(parts[19]), 'parent': int(parts[1]), 'state': parts[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    targets = TARGETS
    suffix = '1045'
    if sys.argv[1:] == ['--new-trajectory']:
        targets = {1712196: (3602389678, '/home/ma-user/workspace/xiacong/ReMoGen', 'trajectory_distill.py')}
        suffix = 'new_trajectory'
    elif sys.argv[1:] == ['--restarted-pair']:
        targets = {
            1713045: (3602400932, '/home/ma-user/workspace/xiacong/ReMoGen', 'harvest.py'),
            1713146: (3602400990, '/home/ma-user/workspace/xiacong/ReMoGen', 'train_projector.py'),
        }
        suffix = 'restarted_pair'
    else:
        assert not sys.argv[1:]
    selected = {}
    for pid, (start, cwd, token) in targets.items():
        got = identity(pid)
        if got is None:
            continue
        assert got['start'] == start, ('PID reused', pid, got)
        assert str(Path(f'/proc/{pid}/cwd').resolve()) == cwd
        assert token in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        selected[pid] = start
    rows = {int(p.name): identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    while True:
        children = {pid: row['start'] for pid, row in rows.items()
                    if row and row['parent'] in selected and pid not in selected}
        if not children:
            break
        selected.update(children)
    evidence = {'selected_pid_start': selected, 'signals': [], 'scope': 'verified GPU job trees; no file deletion'}
    output = Path(f'/cache/zhonghao/h3/clear_30674_for_bd_20260915_{suffix}.json')
    with output.open('x') as log:
        log.write(json.dumps(evidence)); log.flush()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid, start in selected.items():
                got = identity(pid)
                if got and got['start'] == start and got['state'] != 'Z':
                    try:
                        os.kill(pid, sig)
                        evidence['signals'].append([pid, sig.name])
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + (8 if sig == signal.SIGTERM else 3)
            while time.monotonic() < deadline:
                if not any((got := identity(pid)) and got['start'] == start and got['state'] != 'Z'
                           for pid, start in selected.items()):
                    break
                time.sleep(0.25)
        evidence['remaining'] = {pid: got for pid, start in selected.items()
                                 if (got := identity(pid)) and got['start'] == start and got['state'] != 'Z'}
        log.seek(0); log.truncate(); log.write(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)
    assert not evidence['remaining']


if __name__ == '__main__':
    main()
