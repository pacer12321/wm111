"""Read-only process/cgroup evidence. No model imports or process termination.

Use --once for baseline; otherwise watch one explicitly identified D controller.
Output is exclusive-create so evidence from previous attempts is never replaced.
"""
import argparse
import datetime
import json
from pathlib import Path
import time

CGROUP = Path('/sys/fs/cgroup/memory')


def pairs(text):
    return {key: int(value) for key, value in
            (line.split() for line in text.splitlines() if line.strip())}


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return int(fields[19]), int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def snapshot(root=None):
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for name in ('usage_in_bytes', 'limit_in_bytes', 'max_usage_in_bytes',
                 'failcnt', 'memsw.limit_in_bytes'):
        try:
            result[name] = int((CGROUP / ('memory.' + name)).read_text())
        except OSError as exc:
            result[name] = {'error': str(exc)}
    for name in ('oom_control', 'stat'):
        try:
            result[name] = pairs((CGROUP / ('memory.' + name)).read_text())
        except OSError as exc:
            result[name] = {'error': str(exc)}
    # RSS records only our explicitly selected process tree, not other users' argv.
    if root is not None:
        rows = {int(p.name): identity(int(p.name)) for p in Path('/proc').iterdir()
                if p.name.isdigit()}
        selected = {root}
        while True:
            children = {pid for pid, value in rows.items() if value and value[1] in selected}
            if children <= selected:
                break
            selected |= children
        result['processes'] = []
        for pid in sorted(selected):
            try:
                fields = {}
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    key, _, value = line.partition(':')
                    if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmLck', 'State'):
                        fields[key] = value.strip()
                result['processes'].append({'pid': pid, 'identity': rows.get(pid), **fields})
            except OSError:
                pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--once', action='store_true')
    group.add_argument('--watch-pid', type=int)
    args = parser.parse_args()
    original = identity(args.watch_pid) if args.watch_pid else None
    if args.watch_pid:
        cmd = Path(f'/proc/{args.watch_pid}/cmdline').read_bytes()
        if original is None or b'/cache/zhonghao/h3/' not in cmd:
            raise RuntimeError('Not a live experiment controller; refusing unrelated PID')
    with args.output.open('x') as output:
        while True:
            sample = snapshot(args.watch_pid)
            if args.watch_pid:
                sample['controller_identity_matches'] = identity(args.watch_pid) == original
            output.write(json.dumps(sample) + '\n')
            output.flush()
            if args.once or not sample['controller_identity_matches']:
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
