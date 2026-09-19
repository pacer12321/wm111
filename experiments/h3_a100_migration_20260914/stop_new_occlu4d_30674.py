"""Stop exactly the newly-started Occlu4D torchrun tree on assigned host 30674."""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

ROOT_PID = 1925292
ROOT_START_TICKS = "3604506929"
EXPECTED = "torchrun --nproc_per_node=2 -m examples.train"


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, PermissionError):
        return []


def descendants(pid: int) -> list[int]:
    found: list[int] = []
    stack = [pid]
    while stack:
        parent = stack.pop()
        for child in children(parent):
            found.append(child)
            stack.append(child)
    return found


if start_ticks(ROOT_PID) != ROOT_START_TICKS:
    raise SystemExit("Refusing: root PID was reused or already exited")
cmdline = Path(f"/proc/{ROOT_PID}/cmdline").read_bytes().replace(b"\0", b" ").decode()
if EXPECTED not in cmdline:
    raise SystemExit(f"Refusing unexpected command: {cmdline}")

targets = descendants(ROOT_PID) + [ROOT_PID]
for pid in reversed(targets):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

time.sleep(4)
remaining = [pid for pid in targets if start_ticks(pid) is not None]
for pid in reversed(remaining):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

time.sleep(1)
remaining = [pid for pid in targets if start_ticks(pid) is not None]
print({"root": ROOT_PID, "targets": len(targets), "remaining": remaining})
if remaining:
    raise SystemExit("Some targeted processes survived")
