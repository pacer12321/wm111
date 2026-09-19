"""User-authorized, bounded cleanup of verified reinserted GPU job trees.

No persistent service, no data deletion, no GPU reservation allocation.
New same-account GPU-device clients are verified by /proc identity and open
device handles, not by assuming host GPU PIDs equal container PIDs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/cache/zhonghao/h3')
CONTROL = ROOT / 'bd_gpu_guard_20260915_v3'
RUN = ROOT / 'bd_prepared_20260915_v2'
UUIDS = ['GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670', 'GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3']
FOREIGN = '/home/ma-user/workspace/xiacong/encoder-seam'
PROGRAMS = {'harvest.py', 'train_projector.py', 'trajectory_distill.py', 'projector_in_loop.py'}


def record(pid):
    p = Path(f'/proc/{pid}')
    try:
        data = (p / 'stat').read_text()
        fields = data[data.rfind(')') + 2:].split()
        return dict(pid=pid, start=int(fields[19]), parent=int(fields[1]), state=fields[0], uid=p.stat().st_uid,
                    argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
                    cwd=str((p/'cwd').resolve()))
    except (OSError, ProcessLookupError):
        return None


def has_gpu_fd(pid):
    try:
        for fd in Path(f'/proc/{pid}/fd').iterdir():
            try:
                name = os.readlink(fd)
                if name.startswith('/dev/nvidia') and name[len('/dev/nvidia'):].isdigit():
                    return True
            except OSError:
                pass
    except OSError:
        pass
    return False


def sweep(log):
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    assert uuids == UUIDS, 'Wrong device mapping: cleanup disabled'
    rows = {int(p.name): record(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    first_guard_start = json.loads((ROOT/'bd_gpu_guard_20260915'/'pid.json').read_text())['start']
    protected = {pid for pid,r in rows.items() if r and (r['cwd'].startswith(str(ROOT)+'/')
                 or (r['argv'] and r['argv'][0].startswith(str(ROOT)+'/')))}
    # Preserve the verified pre-existing diagnostic and its descendants;
    # it is not a newly inserted job. Never protect its PID without identity.
    if rows.get(105960) and rows[105960]['start'] == 3352476376:
        protected.add(105960)
    while True:
        children = {pid for pid,r in rows.items() if r and r['parent'] in protected}
        if children <= protected:
            break
        protected |= children
    selected, evidence = {}, []
    for pid, row in rows.items():
        if not row or row['state'] == 'Z' or not row['argv'] or pid in protected or row['uid'] != os.getuid():
            continue
        if Path(row['argv'][0]).name in ('nvidia-smi', 'nvtop', 'nvitop'):
            continue
        program = next((arg for arg in row['argv'][1:] if arg in PROGRAMS), None)
        # Old unrelated CPU/background services are out of scope. Known
        # active jobs were separately verified; all new GPU clients are in
        # scope under the user's explicit automatic-cleanup authorization.
        if program is None and row['start'] < first_guard_start:
            continue
        if not has_gpu_fd(pid):
            continue
        root = row
        # Bound ancestor selection to this exact single-job bash/xcrun chain;
        # never stop a shell, remote editor or another user's agent process.
        while True:
            parent = rows.get(root['parent'])
            if not parent or not parent['argv'] or Path(parent['argv'][0]).name != 'bash':
                break
            if parent['cwd'] not in (FOREIGN, '/home/ma-user/workspace/xiacong/ReMoGen'):
                break
            command = ' '.join(parent['argv'])
            if program is None or program not in command or 'encoder-seam' not in command:
                break
            root = parent
        selected[root['pid']] = root['start']
        evidence.append(dict(gpu_worker=row, verified_launcher=root))
    if not selected:
        return
    while True:
        children = {pid:r['start'] for pid,r in rows.items()
                    if r and r['parent'] in selected and pid not in selected}
        if not children:
            break
        selected.update(children)
    event = dict(time=time.time(), evidence=evidence, selected=selected, signals=[])
    log.write(json.dumps(dict(event, phase='before_signal'))+'\n'); log.flush()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid,start in selected.items():
            now=record(pid)
            if now and now['start']==start and now['state']!='Z':
                try:
                    os.kill(pid,sig)
                    event['signals'].append([pid,sig.name])
                except ProcessLookupError:
                    pass
        time.sleep(3 if sig==signal.SIGTERM else 1)
    event['remaining'] = {pid:r for pid,start in selected.items()
                          if (r:=record(pid)) and r['start']==start and r['state']!='Z'}
    log.write(json.dumps(dict(event,phase='after_signal'))+'\n'); log.flush()
    print(json.dumps({'stopped_roots':[r['verified_launcher']['pid'] for r in evidence],
                      'remaining':list(event['remaining'])}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--launch',action='store_true'); args=parser.parse_args()
    assert socket.gethostname()=='os-node-created-mgf6h'
    if args.launch:
        CONTROL.mkdir(exist_ok=False)
        old_info=json.loads((ROOT/'bd_gpu_guard_20260915'/'pid.json').read_text())
        old=record(old_info['pid'])
        if old and old['start']==old_info['start']:
            assert str(Path(__file__).resolve()) in old['argv'], 'Old guard identity mismatch'
            os.kill(old['pid'],signal.SIGTERM)
        with (CONTROL/'dispatch.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (CONTROL/'pid.json').write_text(json.dumps({'pid':proc.pid,'start':record(proc.pid)['start']}))
        print(json.dumps({'guard_pid':proc.pid,'max_duration_hours':12,'stop_file':str(CONTROL/'STOP')})); return
    deadline=time.monotonic()+12*3600
    with (CONTROL/'events.jsonl').open('x') as log:
        while time.monotonic()<deadline and not (CONTROL/'STOP').exists():
            status=RUN/'queue_status.json'
            if status.exists():
                try:
                    if json.loads(status.read_text()).get('status') == 'completed_timing_first_quality_pending':
                        break
                except json.JSONDecodeError:
                    pass
            sweep(log)
            time.sleep(5)
    (CONTROL/'ended.json').write_text(json.dumps({'time':time.time(),'status':'ended'}))


if __name__=='__main__': main()
