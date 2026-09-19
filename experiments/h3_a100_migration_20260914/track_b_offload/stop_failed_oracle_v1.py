"""Stop only the failed v1 Oracle capture server tree authorized by the user."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import time


ROOT_PID = 2474637
EXPECTED_PORT = b"--port\x0019127\x00"
EXPECTED_CANDIDATE = b"attention_oracle_multilayer_code_20260916_v1"
AUDIT = Path("/cache/zhonghao/h3/attention_oracle_multilayer_redshirt_20260916_v1/cleanup.json")


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def children(pid: int) -> list[int]:
    try:
        return [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except (FileNotFoundError, PermissionError):
        return []


def descendants(pid: int) -> list[int]:
    found, stack = [], [pid]
    while stack:
        parent = stack.pop()
        for child in children(parent):
            found.append(child)
            stack.append(child)
    return found


def main() -> None:
    assert socket.gethostname() == "os-node-created-mgf6h"
    root_start = start_ticks(ROOT_PID)
    if root_start is None:
        raise SystemExit("v1 server already exited")
    command = Path(f"/proc/{ROOT_PID}/cmdline").read_bytes()
    environment = Path(f"/proc/{ROOT_PID}/environ").read_bytes()
    if b"vllm_omni.entrypoints.cli.main" not in command or EXPECTED_PORT not in command:
        raise SystemExit(f"refusing unexpected root command: {command!r}")
    if EXPECTED_CANDIDATE not in environment:
        raise SystemExit("refusing: root environment is not the failed v1 candidate")
    targets = descendants(ROOT_PID) + [ROOT_PID]
    identities = {pid: start_ticks(pid) for pid in targets}
    evidence = {
        "root": ROOT_PID,
        "root_start_ticks": root_start,
        "targets": identities,
        "signals": [],
    }
    with AUDIT.open("x") as handle:
        for sig, timeout in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
            for pid in reversed(targets):
                if start_ticks(pid) == identities[pid]:
                    try:
                        os.kill(pid, sig)
                        evidence["signals"].append([pid, sig.name])
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not any(start_ticks(pid) == identities[pid] for pid in targets):
                    break
                time.sleep(0.25)
        evidence["remaining"] = [
            pid for pid in targets if start_ticks(pid) == identities[pid]
        ]
        handle.write(json.dumps(evidence, indent=2))
    print(json.dumps(evidence), flush=True)
    if evidence["remaining"]:
        raise SystemExit("some verified v1 processes remain")


if __name__ == "__main__":
    main()
