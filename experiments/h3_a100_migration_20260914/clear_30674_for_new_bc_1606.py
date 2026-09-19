"""Stop only the verified xiacong GPU job trees on user-reserved port 30674."""

import json
import os
from pathlib import Path
import signal
import time


EXPECTED = {
    1831828: (3603533554, b"/home/ma-user/workspace/xiacong/encoder-seam/supervise.sh"),
    1881811: (3603905790, b"/home/ma-user/workspace/xiacong/ReMoGen/bin/xcrun"),
    1881849: (3603905837, b"/home/ma-user/workspace/xiacong/ReMoGen/bin/xcrun"),
}
AUDIT = Path("/cache/zhonghao/h3/clear_30674_for_new_bc_20260915_1606.json")


def identity(pid):
    proc = Path(f"/proc/{pid}")
    try:
        raw = (proc / "stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "cmdline": (proc / "cmdline").read_bytes(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def snapshot():
    rows = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            row = identity(int(entry.name))
            if row is not None:
                rows[row["pid"]] = row
    return rows


def descendants(rows, roots):
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if row["ppid"] in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def main():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    initial = snapshot()
    for pid, (start, needle) in EXPECTED.items():
        row = initial.get(pid)
        if row is None or row["start_ticks"] != start or needle not in row["cmdline"]:
            raise RuntimeError(f"verified root changed: {pid}")
    selected = descendants(initial, EXPECTED)
    record = {
        "requested_scope": "verified xiacong GPU job trees only",
        "roots": sorted(EXPECTED),
        "selected": sorted(selected),
        "signals": [],
        "files_deleted": False,
    }
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current = snapshot()
        live = []
        for pid in sorted(selected, reverse=True):
            original = initial.get(pid)
            row = current.get(pid)
            if row is not None and original is not None and row["start_ticks"] == original["start_ticks"]:
                os.kill(pid, sig)
                live.append(pid)
        record["signals"].append({"signal": sig.name, "pids": live})
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                now = snapshot()
                if not any(pid in now and now[pid]["start_ticks"] == initial[pid]["start_ticks"] for pid in selected):
                    break
                time.sleep(0.5)
    final = snapshot()
    record["remaining"] = sorted(
        pid for pid in selected
        if pid in final and final[pid]["start_ticks"] == initial[pid]["start_ticks"]
    )
    record["status"] = "stopped" if not record["remaining"] else "failed"
    AUDIT.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    if record["remaining"]:
        raise RuntimeError(record["remaining"])


if __name__ == "__main__":
    main()
