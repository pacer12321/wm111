"""Read-only container GPU-client identity; never use Ngid as a host PID."""
import os
from pathlib import Path


def scan_clients(proc=Path('/proc'), devices=('/dev/nvidia3', '/dev/nvidia7'),
                 unreadable=None, capture_unreadable=False):
    found = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        start = None
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            raw = (entry/'stat').read_text()
            fields = raw[raw.rfind(')')+2:].split()
            start = int(fields[19])
            if fields[0] == 'Z':
                continue
            argv = (entry/'cmdline').read_bytes().split(b'\0')
            if not argv or Path(os.fsdecode(argv[0])).name in ('nvidia-smi','nvtop','nvitop'):
                continue
            for fd in (entry/'fd').iterdir():
                try:
                    if os.readlink(fd) in devices:
                        found[int(entry.name)] = int(fields[19])
                        break
                except FileNotFoundError:
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A validated idle baseline may include the container init process
            # whose fd links are hidden even from the same UID. Record that
            # limitation explicitly; never grant this exemption to a new PID.
            if unreadable is not None and start is not None:
                if capture_unreadable:
                    unreadable[int(entry.name)] = start
                    continue
                if unreadable.get(int(entry.name)) == start:
                    continue
            raise RuntimeError(f'Cannot inspect GPU client namespace at {entry}')
    return found


if __name__=='__main__':
    import json
    excluded={}
    print(json.dumps({'clients':scan_clients(unreadable=excluded,capture_unreadable=True),
                      'unreadable_baseline':excluded}))


def check_clients(clients, baseline, owned):
    unexpected = {pid:start for pid,start in clients.items()
                  if owned.get(pid) != start and baseline.get(pid) != start}
    if unexpected:
        raise RuntimeError(f'Unowned GPU device clients in container namespace: {unexpected}')
    return {pid:start for pid,start in clients.items() if owned.get(pid) == start}
