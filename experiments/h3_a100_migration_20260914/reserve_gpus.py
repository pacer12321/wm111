"""Explicit user-authorized, low-compute memory hold for port 32209 GPUs 2/3.

Run --launch once to detach; touch STOP in this script's directory to release.
This is not a scheduler reservation. Automatic expiry: 24 hours.
"""
import ctypes as c
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TARGETS = {
    "2": "GPU-856faa17-eeba-f3e1-36b7-720cf8424a84",
    "3": "GPU-5d12ae6b-1304-87d8-6b26-835e6a754c92",
}
HOLD_BYTES = 70 * 1024**3
TTL = 24 * 3600


def query(kind, fields):
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [[part.strip() for part in line.split(",")] for line in output.splitlines() if line.strip()]


def check_idle():
    rows = query("gpu", "index,uuid,pci.bus_id,memory.used,utilization.gpu")
    selected = {}
    for index, uuid, bus, memory, utilization in rows:
        if index in TARGETS:
            if uuid != TARGETS[index] or int(memory) > 10 or int(utilization) != 0:
                raise RuntimeError(f"Target no longer idle or identity changed: {index} {uuid} {memory} {utilization}")
            selected[index] = bus
    if set(selected) != set(TARGETS):
        raise RuntimeError("Target GPUs missing")
    for uuid, pid in query("compute-apps", "gpu_uuid,pid"):
        if uuid in TARGETS.values():
            raise RuntimeError(f"Target has existing compute process {uuid} {pid}")
    return selected


def main():
    if "--launch" in sys.argv:
        check_idle()
        if (ROOT / "reservation.json").exists() or (ROOT / "STOP").exists():
            raise RuntimeError("Use a fresh run directory; refusing duplicate launch")
        with (ROOT / "reservation.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"launched_pid": child.pid, "directory": str(ROOT)}))
        return

    selected = check_idle()
    driver = c.CDLL("libcuda.so.1")

    def call(name, types, *args):
        fn = getattr(driver, name)
        fn.argtypes = types
        fn.restype = c.c_int
        rc = fn(*args)
        if rc:
            raise RuntimeError(f"{name} failed with CUDA error {rc}")

    call("cuInit", [c.c_uint], 0)
    contexts = []
    stop = False
    state = {"pid": os.getpid(), "owner": "zhonghao", "targets": TARGETS,
             "bytes_per_gpu": HOLD_BYTES, "purpose": "H3 acceleration preparation; no benchmark running",
             "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "expires_at": dt.datetime.fromtimestamp(time.time() + TTL, dt.timezone.utc).isoformat()}

    def on_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        for index, bus in selected.items():
            dev = c.c_int()
            call("cuDeviceGetByPCIBusId", [c.POINTER(c.c_int), c.c_char_p], c.byref(dev), bus.encode())
            ctx = c.c_void_p()
            call("cuCtxCreate_v2", [c.POINTER(c.c_void_p), c.c_uint, c.c_int], c.byref(ctx), 0, dev)
            contexts.append(ctx)
            pointer = c.c_uint64()
            call("cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t], c.byref(pointer), HOLD_BYTES)
            print(f"Holding {HOLD_BYTES // 1024**3} GiB on GPU{index} ({TARGETS[index]})", flush=True)
        state["status"] = "holding"
        (ROOT / "reservation.json").write_text(json.dumps(state, indent=2))
        deadline = time.monotonic() + TTL
        while not stop and time.monotonic() < deadline and not (ROOT / "STOP").exists():
            time.sleep(5)
        state["status"] = "released"
        state["reason"] = "signal" if stop else "stop_file" if (ROOT / "STOP").exists() else "24h_expiry"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        for ctx in reversed(contexts):
            try:
                call("cuCtxDestroy_v2", [c.c_void_p], ctx)
            except Exception as exc:
                print(f"Context cleanup warning: {exc}", flush=True)
        state["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (ROOT / "reservation.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
