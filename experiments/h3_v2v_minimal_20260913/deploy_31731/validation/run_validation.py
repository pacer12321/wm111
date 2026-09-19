#!/usr/bin/env python3
"""One opt-in B tiny-wrapper test on all eight 31731 cards; no full model."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import profiles
import supervision_base as base

RUN_TIMEOUT = 900
COLLECTIVE_TIMEOUT = 180
EXPECTED_CASES = {"source_prefix", "source_suffix"}
EXPECTED_CHECKS = {"wrapper-softmax", "wrapper-linear-gated", "actual-attention-forward-projected",
                   "source-text-anchor-padding-linear-zero", "padding-final-output-zero"}


def verify_results(path, selected, host, run_id, manifest):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Tiny result is missing/nonregular")
    result = json.loads(path.read_text())
    expected = {"status": "passed", "group": selected.group, "host": host, "run_id": run_id,
                "physical_cards": list(selected.cards), "world_size": profiles.WORLD, "dtype": "bfloat16",
                "source_sha256": manifest, "test_script_sha256": manifest["validation/npu_wrapper_regression.py"]["sha256"]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Tiny summary differs from this host/group/run/source proof")
    ranks = result.get("rank_results", [])
    if len(ranks) != profiles.WORLD:
        raise RuntimeError("Tiny summary must contain exactly eight rank records")
    seen = set()
    strategy = manifest["strategy/ulysses.py"]
    for item in ranks:
        rank = item.get("rank")
        if (type(rank) is not int or rank not in range(profiles.WORLD) or rank in seen
                or item.get("physical_card") != selected.cards[rank]
                or item.get("status") != "passed" or item.get("host") != host
                or item.get("group") != selected.group or item.get("run_id") != run_id
                or item.get("source_sha256") != manifest
                or item.get("real_strategy_source") != strategy["path"]
                or item.get("real_strategy_sha256") != strategy["sha256"]
                or len(item.get("tests", [])) != 2):
            raise RuntimeError("Invalid/duplicate/mismatched rank or actual strategy proof")
        if {test.get("case") for test in item["tests"]} != EXPECTED_CASES:
            raise RuntimeError("Rank did not test both source-prefix and source-suffix layouts")
        if item.get("parallelism_proof") != {
                "ulysses_world_size": profiles.WORLD, "ulysses_rank": rank,
                "ring_world_size": 1, "tensor_parallel_world_size": 1,
                "collective_global_ranks": list(range(profiles.WORLD)), "heads_per_ulysses_rank": 7}:
            raise RuntimeError("Rank lacks a complete single-group USP8 collective proof")
        for test in item["tests"]:
            checks = test.get("checks", [])
            if len(checks) != 5 or {check.get("check") for check in checks} != EXPECTED_CHECKS:
                raise RuntimeError("Rank is missing an expected wrapper check")
            for check in checks:
                if any(type(check.get(key)) not in (int, float) or not math.isfinite(check[key]) or check[key] < 0
                       for key in ("max_abs", "rms_error", "golden_rms", "atol", "rtol")):
                    raise RuntimeError("Check metrics are invalid/nonfinite")
                exact = check["check"].endswith("-zero")
                if (check["atol"], check["rtol"]) != ((0.0, 0.0) if exact else (1/512, 1/128)):
                    raise RuntimeError("Check tolerances differ from the reviewed BF16/exact gates")
                if exact and (check["max_abs"] != 0 or check["rms_error"] != 0):
                    raise RuntimeError("An exact-zero check has nonzero error")
        rank_path = path.with_name(f"{path.stem}.rank{rank}.json")
        if rank_path.is_symlink() or json.loads(rank_path.read_text()) != item:
            raise RuntimeError("Embedded rank proof differs from its standalone file")
        seen.add(rank)
    return result


class ValidationSupervisor(base.ProcessSupervisor):
    def __init__(self, group):
        selected = profiles.profile(group)
        self.host = profiles.require_host()  # REAL hostname and boot, no override argument.
        super().__init__(selected, uuid.uuid4().hex)
        self.manifest = None
        self.status = {"case": "B_tiny_wrapper_31731", "group": group, "host": self.host,
                       "run_id": self.run_id, "allocated_physical_npu_ids": list(selected.cards),
                       "supervisor_pid": os.getpid(), "supervisor_proc_identity": base.proc_identity(os.getpid()),
                       "started_at": base.utc_now(), "master_port": selected.master_port,
                       "outer_timeout_seconds": RUN_TIMEOUT, "collective_timeout_seconds": COLLECTIVE_TIMEOUT,
                       "scope": "Tiny correctness/interface only; no H3 weights, video, training or acceleration claim."}

    def update(self, phase, **values):
        self.status.update(values, phase=phase, updated_at=base.utc_now())
        if self.run_dir is not None:
            base.atomic_json(self.run_dir / "validation_status.json", self.status)
            base.atomic_json(self.selected.output_root / "validation_status.json", self.status)
        print(json.dumps({"phase": phase, "group": self.selected.group, "run_id": self.run_id, **values}), flush=True)

    def acquire(self):
        for path in (self.selected.output_root, self.selected.lease_root):
            profiles.canonical_private(path)
            path.mkdir(parents=True, exist_ok=True)
            profiles.canonical_private(path)
        self.take_lock(self.selected.output_root / "run.lock")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        new_run = self.selected.output_root / "runs" / f"{stamp}_{self.run_id}"
        profiles.canonical_private(new_run)
        new_run.mkdir(parents=True, exist_ok=False)
        # Never attach status-writing state to an existing/colliding run.
        self.run_dir = new_run
        self.status["run_directory"] = str(self.run_dir)
        base.atomic_json(self.run_dir / "host_identity.json", self.host)
        temporary = profiles.INSTALL / "tmp" / f"v_{self.run_id}"
        profiles.canonical_private(temporary)
        temporary.mkdir(parents=True, exist_ok=False)
        profiles.canonical_private(temporary)
        self.update("acquiring_personal_device_leases")
        for card in self.selected.cards:
            self.take_lock(self.selected.lease_root / f"device{card}.lock")
        self.update("personal_device_leases_acquired")

    def preflight(self):
        profiles.require_host(self.host)
        if Path(__file__).resolve().parent != self.selected.code_root:
            raise RuntimeError(f"Deploy this supervisor only at {self.selected.code_root}")
        if len(self.locks) != profiles.WORLD + 1 or any(handle.closed for handle in self.locks):
            raise RuntimeError("Fresh inspection requires group mutex plus all eight per-card leases")
        self.manifest = profiles.source_manifest(self.selected)
        python = profiles.INSTALL / "env/bin/python"
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RuntimeError("Private H3 Python is not ready")
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=False)
        (self.run_dir / "npu_before.txt").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            raise RuntimeError("npu-smi failed")
        idle = base.selected_idle(result.stdout, self.selected.cards)
        cards = base.selected_health_memory(result.stdout, self.selected.cards, profiles.MIN_HBM_AVAILABLE_MIB)
        memory = base.host_memory_snapshot(profiles.MIN_HOST_AVAILABLE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", self.selected.master_port))
        base.atomic_json(self.run_dir / "source_manifest.json", self.manifest)
        self.update("preflight_passed", resource_snapshot=idle, card_health_memory=cards,
                    host_memory=memory, source_sha256=self.manifest,
                    lease_scope="Personal locks coordinate zhonghao jobs; no foreign process is stopped or displaced.")

    def env(self):
        env = os.environ.copy()
        env.update(H3_ROOT=str(self.run_dir), H3_OUTPUT=str(self.run_dir), H3_MINIMAL_RUN_ID=self.run_id,
                   H3_VALIDATION_GROUP=self.selected.group, H3_VALIDATION_SUPERVISOR_PID=str(os.getpid()),
                   H3_VALIDATION_MASTER_PORT=str(self.selected.master_port),
                   H3_VALIDATION_RESULT=str(self.run_dir / "npu_wrapper.json"),
                   H3_GROUP_ID=f"zhonghao_31731_{self.selected.group}_{self.run_id}",
                   ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, self.selected.cards)),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        return env

    def launch(self):
        profiles.require_host(self.host)
        if profiles.source_manifest(self.selected) != self.manifest:
            raise RuntimeError("Candidate/runtime sources changed after preflight")
        log = (self.run_dir / "validation.log").open("ab", buffering=0)
        self.handles.append(log)
        started = time.monotonic()
        self.server = self.spawn(["bash", str(self.selected.code_root / "launch_validation.sh")],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.update("validation_running", worker_pid=self.server.pid,
                    worker_proc_identity=base.proc_identity(self.server.pid), worker_started_at=base.utc_now())
        while self.server.poll() is None:
            if time.monotonic() - started > RUN_TIMEOUT:
                raise TimeoutError("Tiny wrapper exceeded 900 seconds; cleanup only this run")
            time.sleep(1)
        if self.server.returncode:
            raise RuntimeError(f"Tiny wrapper exited {self.server.returncode}; inspect validation.log")
        profiles.require_host(self.host)
        if profiles.source_manifest(self.selected) != self.manifest:
            raise RuntimeError("Candidate/runtime sources changed during the tiny test")
        path = self.run_dir / "npu_wrapper.json"
        verify_results(path, self.selected, self.host, self.run_id, self.manifest)
        self.update("validation_passed_before_cleanup", validation_passed=True, result_path=str(path),
                    result_sha256=profiles.digest(path), validation_wall_seconds=time.monotonic()-started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(profiles.GROUPS), required=True)
    parser.add_argument("--allow-npu", action="store_true", help="Explicit authorization for this tiny NPU test")
    args = parser.parse_args(argv)
    if not args.allow_npu:
        parser.error("No NPU test is started without --allow-npu")
    supervisor = ValidationSupervisor(args.group)
    signal.signal(signal.SIGTERM, base.interrupted)
    signal.signal(signal.SIGINT, base.interrupted)
    error = None
    try:
        supervisor.acquire()
        supervisor.preflight()
        supervisor.launch()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        supervisor.update("failed_before_cleanup", error=error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean = supervisor.cleanup()
            if not clean and error is None:
                error = "Owned validation processes remain; do not reuse selected cards"
            if supervisor.status.get("needs_attention") and error is None:
                error = "Selected-card release could not be verified"
            supervisor.update("needs_attention" if supervisor.status.get("needs_attention") else ("failed" if error else "completed"),
                              error=error, finished_at=base.utc_now(),
                              note="Tiny B wrapper correctness only. No full-model B/C video or performance result.")
        finally:
            supervisor.release_locks()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
