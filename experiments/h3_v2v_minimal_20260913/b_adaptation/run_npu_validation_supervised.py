#!/usr/bin/env python3
"""One leased four-card B wrapper correctness test, never a video/model run.

Linux-only standard-library supervisor. Reuses the reviewed A supervisor's
process identity, pass_fds, idle-table parser and lease-preserving cleanup.
All B validation state is separate; A's a_status.json is never written.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SUPERVISOR_PATH = SCRIPT_DIR.parent / "run_a_supervised.py"
spec = importlib.util.spec_from_file_location("h3_reviewed_supervisor_base", BASE_SUPERVISOR_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import reviewed supervisor {BASE_SUPERVISOR_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

CARDS = (2, 3, 6, 7)
MASTER_PORT = 29673
COLLECTIVE_TIMEOUT = 180
RUN_TIMEOUT = 900
OUTPUT_ROOT = Path("/cache/zhonghao/h3_v2v_minimal_20260913/b_validation")
EXPECTED_SCRIPT_DIR = Path("/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/b_adaptation")
VENDOR_ROOT = EXPECTED_SCRIPT_DIR / "vendor/vllm-omni"
LEASE_ROOT = Path("/cache/zihaohe/generation-supervision/runs")


class ValidationSupervisor(base.Supervisor):
    def __init__(self):
        if tuple(base.CARDS) != CARDS:
            raise RuntimeError("Reviewed idle/cleanup helper card allocation changed")
        super().__init__({}, SCRIPT_DIR)
        self.status = {
            "case": "B_NPU_wrapper_validation",
            "run_id": self.run_id,
            "supervisor_pid": os.getpid(),
            "supervisor_proc_identity": base.proc_identity(os.getpid()),
            "started_at": base.utc_now(),
            "allocated_physical_npu_ids": list(CARDS),
            "master_port": MASTER_PORT,
            "collective_timeout_seconds": COLLECTIVE_TIMEOUT,
            "outer_timeout_seconds": RUN_TIMEOUT,
            "scope": "Tiny wrapper correctness only; no full H3 weights, video, or speed claim.",
        }

    def update(self, phase, **values):
        self.status.update(values)
        self.status.update(phase=phase, updated_at=base.utc_now())
        if self.run_dir is not None:
            base.atomic_json(self.run_dir / "validation_status.json", self.status)
            base.atomic_json(OUTPUT_ROOT / "validation_status.json", self.status)
        print(json.dumps({"phase": phase, "updated_at": self.status["updated_at"], **values},
                         ensure_ascii=False), flush=True)

    def acquire(self):
        # Separate own lock/state from A; cooperative per-card locks serialize
        # this validation with A and all other cooperating users.
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        own_lock = (OUTPUT_ROOT / "run.lock").open("a+b")
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            own_lock.close()
            raise RuntimeError("Another B NPU validation holds its run.lock")
        self.locks.append(own_lock)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = OUTPUT_ROOT / "runs" / f"{stamp}_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status["run_directory"] = str(self.run_dir)
        self.update("acquiring_device_leases")
        for card in CARDS:
            path = LEASE_ROOT / f"new_layout_training_device{card}.lock"
            # Existing lease only, read-only descriptor: no create/truncate or
            # deletion. Failure unwinds already-held leases in main().
            handle = path.open("rb")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError(f"Physical NPU {card} lease is held; validation was not launched")
            self.locks.append(handle)
        self.update("device_leases_acquired")

    def preflight(self):
        if SCRIPT_DIR != EXPECTED_SCRIPT_DIR:
            raise RuntimeError(f"Deploy this supervisor only at {EXPECTED_SCRIPT_DIR}")
        required = [
            BASE_SUPERVISOR_PATH,
            SCRIPT_DIR / "launch_npu_validation.sh",
            SCRIPT_DIR / "tests/npu_wrapper_regression.py",
            SCRIPT_DIR / "upstream/openvdn_npu.py",
            SCRIPT_DIR / "patched/openvdn_npu.py",
            SCRIPT_DIR / "patched/minimax_h3_transformer.py",
            VENDOR_ROOT / "vllm_omni/__init__.py",
            Path("/cache/yunfeng/minimax_h3_npu/scripts/env.sh"),
            Path("/cache/yunfeng/envs/minimax-h3-npu/bin/python"),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing required validation files: {missing}")
        for executable in ("npu-smi", "bash"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"Required executable missing from PATH: {executable}")
        if len(self.locks) != len(CARDS) + 1:
            raise RuntimeError("Fresh resource inspection requires all four leases")
        check = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                               timeout=30, check=False)
        (self.run_dir / "npu_before.txt").write_text(check.stdout + "\n" + check.stderr)
        if check.returncode:
            raise RuntimeError(f"npu-smi failed with code {check.returncode}")
        idle = base.require_selected_cards_idle(check.stdout)
        health = {}
        for match in re.finditer(r"^\|\s*(\d+)\s+\S+\s*\|\s*([^|]+)\|", check.stdout, re.MULTILINE):
            card = int(match.group(1))
            if card in CARDS:
                if card in health:
                    raise RuntimeError(f"Ambiguous health rows for NPU {card}")
                health[card] = match.group(2).strip()
        if health != {card: "OK" for card in CARDS}:
            raise RuntimeError(f"Selected card health is not explicitly OK: {health}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", MASTER_PORT))
        source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in required[:6]}
        self.update("preflight_passed", resource_snapshot=idle, selected_card_health=health,
                    source_sha256=source_hashes)

    def env(self):
        env = os.environ.copy()
        env.update(
            H3_ROOT=str(self.run_dir),
            H3_OUTPUT=str(self.run_dir),
            ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, CARDS)),
            H3_MINIMAL_RUN_ID=self.run_id,
            H3_B_VALIDATION_SUPERVISOR_PID=str(os.getpid()),
            H3_B_VALIDATION_MASTER_PORT=str(MASTER_PORT),
            H3_B_VALIDATION_COLLECTIVE_TIMEOUT=str(COLLECTIVE_TIMEOUT),
            H3_B_VALIDATION_RESULT=str(self.run_dir / "npu_wrapper.json"),
            PYTHONPATH=str(VENDOR_ROOT),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
        )
        return env

    def launch(self):
        log = (self.run_dir / "validation.log").open("ab", buffering=0)
        self.handles.append(log)
        # The inherited spawn supplies pass_fds for every lease and creates a
        # fresh session. 'server' is only the inherited cleanup's child handle;
        # this command executes the tiny test, never a video service.
        started = time.monotonic()
        self.server = self.spawn(["bash", str(SCRIPT_DIR / "launch_npu_validation.sh")],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.update("validation_running", worker_pid=self.server.pid,
                    worker_started_at=base.utc_now(),
                    worker_proc_identity=base.proc_identity(self.server.pid))
        deadline = started + RUN_TIMEOUT
        while self.server.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"NPU wrapper validation exceeded {RUN_TIMEOUT} seconds; cleaning up this run only")
            time.sleep(1)
        elapsed = time.monotonic() - started
        self.status["worker_returncode"] = self.server.returncode
        self.status["validation_wall_seconds"] = elapsed
        if self.server.returncode:
            raise RuntimeError(f"NPU validation exited {self.server.returncode}; inspect validation.log")
        result_path = self.run_dir / "npu_wrapper.json"
        if not result_path.is_file():
            raise RuntimeError("Validation exited zero without npu_wrapper.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result.get("status") != "passed" or result.get("physical_cards") != list(CARDS)
                or result.get("world_size") != len(CARDS)):
            raise RuntimeError("NPU wrapper summary did not verify the requested four-card pass")
        rank_results = result.get("rank_results", [])
        if len(rank_results) != len(CARDS) or any(item.get("status") != "passed" for item in rank_results):
            raise RuntimeError("NPU wrapper summary lacks four passing rank results")
        self.update("validation_passed_before_cleanup", result_path=str(result_path),
                    validation_wall_seconds=elapsed, worker_returncode=0,
                    validation_passed=True)


def main():
    supervisor = ValidationSupervisor()
    signal.signal(signal.SIGTERM, base.interrupted)
    signal.signal(signal.SIGINT, base.interrupted)
    error = None
    try:
        supervisor.acquire()
        supervisor.preflight()
        supervisor.launch()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if supervisor.run_dir is not None:
            supervisor.update("failed_before_cleanup", error=error)
        else:
            print(error, file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean = supervisor.cleanup()
            if supervisor.run_dir is not None:
                if not clean and error is None:
                    error = "Owned validation processes remain; do not reuse cards until checked"
                if supervisor.status.get("needs_attention") and error is None:
                    error = "Post-validation resource release could not be verified"
                phase = "needs_attention" if supervisor.status.get("needs_attention") else ("failed" if error else "completed")
                supervisor.update(phase, finished_at=base.utc_now(), error=error,
                                  note="Only tiny B wrapper correctness validation; no full-model B/C or acceleration result.")
        finally:
            # Close only, no LOCK_UN/deletion: inherited descriptors retain the
            # lease if an owned child cannot be terminated.
            supervisor.release_locks()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
