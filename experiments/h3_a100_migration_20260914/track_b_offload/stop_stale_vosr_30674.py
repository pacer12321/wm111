"""Stop the user-authorized stale VOSR tree while preserving the live H3 run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import time


ROOT_PID = 105960
EXPECTED = b"/cache/zihaohe/vosr_bench.py"
AUDIT = Path("/cache/zhonghao/h3/stale_vosr_cleanup_20260916.json")


def identity(pid: int) -> tuple[str, str] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        return stat[21], stat[2]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def children(pid: int) -> list[int]:
    try:
        return [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except (FileNotFoundError, PermissionError):
        return []


def descendants(pid: int) -> list[int]:
    result, stack = [], [pid]
    while stack:
        parent = stack.pop()
        for child in children(parent):
            result.append(child)
            stack.append(child)
    return result


def alive(pid: int, start: str) -> bool:
    value = identity(pid)
    return value is not None and value[0] == start and value[1] != "Z"


def main() -> None:
    assert socket.gethostname() == "os-node-created-mgf6h"
    root_identity = identity(ROOT_PID)
    if root_identity is None:
        raise SystemExit("stale VOSR root already exited")
    command = Path(f"/proc/{ROOT_PID}/cmdline").read_bytes()
    if EXPECTED not in command:
        raise SystemExit(f"refusing unexpected process: {command!r}")
    targets = descendants(ROOT_PID) + [ROOT_PID]
    identities = {pid: identity(pid)[0] for pid in targets if identity(pid) is not None}
    evidence = {"root": ROOT_PID, "targets": identities, "signals": []}
    with AUDIT.open("x") as handle:
        for sig, timeout in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
            for pid in reversed(targets):
                start = identities.get(pid)
                if start is not None and alive(pid, start):
                    try:
                        os.kill(pid, sig)
                        evidence["signals"].append([pid, sig.name])
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not any(alive(pid, start) for pid, start in identities.items()):
                    break
                time.sleep(0.25)
        evidence["remaining_live"] = [
            pid for pid, start in identities.items() if alive(pid, start)
        ]
        handle.write(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)
    if evidence["remaining_live"]:
        raise SystemExit("some verified VOSR processes remain live")


if __name__ == "__main__":
    main()
