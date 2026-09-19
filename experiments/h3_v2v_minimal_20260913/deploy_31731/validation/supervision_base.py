"""Independent copy of A's lease/process/cleanup semantics, with explicit cards.

No import of the live 30213 supervisor. Personal per-card locks coordinate our
own jobs; they do not grant authority over another user's jobs. Fresh idle
inspection under these locks is mandatory. Never stop an existing foreign PID.
"""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
import uuid


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def proc_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "pgrp": int(fields[2]),
                "session": int(fields[3]), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def selected_idle(output, cards):
    headers = list(re.finditer(r"^\|\s*NPU\s+Chip\s*\|\s*Process id\s*\|\s*Process name\s*\|"
                               r"\s*Process memory\(MB\)\s*\|\s*$", output, re.MULTILINE))
    if len(headers) != 1:
        raise RuntimeError("Unrecognized or ambiguous npu-smi process table")
    empty, busy = set(), set()
    for line in output[headers[0].end():].splitlines():
        line = line.strip()
        if not line or re.fullmatch(r"[+\-=]+", line):
            continue
        idle = re.fullmatch(r"\|\s*No running processes found in NPU (\d+)\s*\|", line)
        if idle:
            card = int(idle.group(1))
            if card in empty:
                raise RuntimeError("Duplicate idle-card row")
            empty.add(card)
            continue
        row = re.fullmatch(r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*[^|]+\|\s*\d+\s*\|", line)
        if row:
            busy.add(int(row.group(1)))
            continue
        raise RuntimeError(f"Unrecognized process row: {line!r}")
    if not set(cards).issubset(empty) or set(cards) & busy:
        raise RuntimeError(f"Selected cards not explicitly idle: selected={cards}, idle={empty}, busy={busy}")
    return {"selected_physical_cards": list(cards), "explicitly_idle_cards": sorted(empty), "other_busy_cards": sorted(busy)}


def selected_health_memory(output, cards, minimum_free_mib):
    health, memory = {}, {}
    current_card = None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            continue
        card_model = re.fullmatch(r"(\d+)\s+([A-Za-z0-9][A-Za-z0-9 _-]*)", parts[1])
        if card_model and re.search(r"[A-Za-z]", card_model.group(2)):
            current_card = int(card_model.group(1))
            if current_card in cards:
                if current_card in health:
                    raise RuntimeError("Duplicate selected-card health row")
                health[current_card] = {"model": card_model.group(2), "health": parts[2]}
            continue
        # Standard npu-smi card block's third row ends in HBM used/total MiB.
        values = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if current_card in cards and values:
            if current_card in memory:
                raise RuntimeError("Ambiguous selected-card HBM row")
            used, total = map(int, values[-1])
            if used < 0 or total <= 0 or used > total:
                raise RuntimeError("Invalid HBM counters")
            memory[current_card] = {"used_mib": used, "total_mib": total, "free_mib": total-used}
    if set(health) != set(cards) or any(row["health"] != "OK" for row in health.values()):
        raise RuntimeError("Every selected physical NPU must report health OK")
    if set(memory) != set(cards) or any(row["free_mib"] < minimum_free_mib for row in memory.values()):
        raise RuntimeError("Selected-card free HBM was not verified or is insufficient")
    return {"health": health, "hbm": memory}


def host_memory_snapshot(minimum_available, *, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    text = (proc_root / "meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("Cannot verify host MemAvailable")
    available = int(match.group(1)) * 1024
    candidates = [(cgroup_root / "memory.max", cgroup_root / "memory.current"),
                  (cgroup_root / "memory/memory.limit_in_bytes", cgroup_root / "memory/memory.usage_in_bytes")]
    chosen = next(((limit, current) for limit, current in candidates if limit.is_file() and current.is_file()), None)
    if chosen is None:
        raise RuntimeError("Cannot verify container memory limit/current usage")
    limit_text = chosen[0].read_text().strip()
    used = int(chosen[1].read_text().strip())
    limit = None if limit_text == "max" else int(limit_text)
    if used < 0 or (limit is not None and limit <= 0):
        raise RuntimeError("Invalid cgroup memory counters")
    remaining = None if limit is None or limit >= 1 << 60 else max(0, limit-used)
    effective = min(available, remaining) if remaining is not None else available
    if effective < minimum_available:
        raise RuntimeError(f"Insufficient host/container memory for tiny test: {effective} bytes available")
    return {"host_mem_available_bytes": available, "cgroup_limit_bytes": limit, "cgroup_used_bytes": used,
            "effective_available_bytes": effective, "required_available_bytes": minimum_available,
            "cgroup_files": [str(path) for path in chosen]}


class ProcessSupervisor:
    def __init__(self, selected, run_id):
        self.selected, self.run_id = selected, run_id
        self.run_dir = None
        self.locks, self.children, self.handles = [], [], []
        self.server = None
        self.status = {}

    def take_lock(self, path):
        # Only create our private mutex. No truncate/unlink/LOCK_UN operation.
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_uid != os.getuid():
            os.close(fd)
            raise RuntimeError(f"Personal mutex must be a private, single-link regular file: {path}")
        handle = os.fdopen(fd, "r+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise RuntimeError(f"Personal resource mutex is already held: {path}")
        self.locks.append(handle)

    def spawn(self, command, **kwargs):
        child = subprocess.Popen(command, start_new_session=True, env=self.env(), cwd=self.selected.code_root,
                                 pass_fds=tuple(handle.fileno() for handle in self.locks), **kwargs)
        self.children.append((child, proc_identity(child.pid)))
        return child

    def owned_group_members(self, pgid):
        members = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                identity = proc_identity(int(entry.name))
                if (not identity or identity["pgrp"] != pgid or identity["session"] != pgid or identity["state"] == "Z"):
                    continue
                if f"H3_MINIMAL_RUN_ID={self.run_id}".encode() in (entry / "environ").read_bytes().split(b"\0"):
                    members.append(identity["pid"])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return members

    def cleanup(self):
        if self.run_dir is not None:
            self.update("cleaning_up_own_processes")
        pending = []
        for child, identity in reversed(self.children):
            current = proc_identity(child.pid)
            leader_matches = (identity is not None and current is not None
                              and current["start_ticks"] == identity["start_ticks"]
                              and current["pgrp"] == child.pid and current["session"] == child.pid)
            if leader_matches or self.owned_group_members(child.pid):
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    pending.append(child)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        while pending and time.monotonic() < deadline:
            pending = [child for child in pending if self.owned_group_members(child.pid)]
            if pending:
                time.sleep(1)
        for child in pending:
            if self.owned_group_members(child.pid):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for child, _ in self.children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        remaining = {child.pid: self.owned_group_members(child.pid) for child, _ in self.children}
        remaining = {group: pids for group, pids in remaining.items() if pids}
        self.status.update(remaining_owned_process_groups=remaining, cleanup_completed=not remaining,
                           needs_attention=bool(remaining))
        if self.server is not None:
            try:
                result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=False)
                (self.run_dir / "npu_after.txt").write_text(result.stdout + "\n" + result.stderr)
                if result.returncode:
                    raise RuntimeError("Post-cleanup npu-smi failed")
                self.status["resource_release_check"] = selected_idle(result.stdout, self.selected.cards)
                self.status["selected_cards_verified_idle_after_cleanup"] = True
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.status.update(selected_cards_verified_idle_after_cleanup=False,
                                   resource_release_check_error=str(exc), needs_attention=True)
        for stream in self.handles:
            stream.close()
        return not remaining

    def release_locks(self):
        for handle in reversed(self.locks):
            handle.close()
        self.locks.clear()


def interrupted(signum, _frame):
    raise InterruptedError(f"Received signal {signum}; stopping only this run's own processes")
