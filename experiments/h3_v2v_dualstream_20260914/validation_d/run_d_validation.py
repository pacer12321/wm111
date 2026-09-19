#!/usr/bin/env python3
"""One explicitly authorized D tiny run; never schedules or starts full H3."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid

import d_profiles as p

TIMEOUT = 900


def supervisor_class(base):
    class DValidation(base.ProcessSupervisor):
        def __init__(self):
            self.host = p.require_host()
            super().__init__(p.Profile(), uuid.uuid4().hex)
            self.manifest = None
            self.status = dict(case=p.CASE, mode=p.MODE, host=self.host, group=self.selected.group, run_id=self.run_id,
                               supervisor_pid=os.getpid(), supervisor_proc_identity=base.proc_identity(os.getpid()),
                               allocated_physical_npu_ids=list(p.CARDS), started_at=base.utc_now(),
                               outer_timeout_seconds=TIMEOUT, collective_timeout_seconds=180,
                               scope="Synthetic D/SP8 correctness only; no weights/video/speed claim.")

        def update(self, phase, **values):
            self.status.update(values, phase=phase, updated_at=base.utc_now())
            if self.run_dir is not None:
                base.atomic_json(self.run_dir / "d_validation_status.json", self.status)
                base.atomic_json(self.selected.output_root / "d_validation_status.json", self.status)
            print(json.dumps(dict(phase=phase, run_id=self.run_id, **values)), flush=True)

        def acquire(self):
            for path in (self.selected.output_root, self.selected.lease_root):
                p.canonical(path).mkdir(parents=True, exist_ok=True)
                p.canonical(path)
            self.take_lock(self.selected.output_root / "run.lock")
            new = self.selected.output_root / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + self.run_id)
            p.canonical(new).mkdir(parents=True, exist_ok=False)
            self.run_dir = new
            self.status["run_directory"] = str(new)
            base.atomic_json(new / "host_identity.json", self.host)
            for card in p.CARDS:
                self.take_lock(self.selected.lease_root / f"device{card}.lock")
            p.canonical(p.INSTALL / "tmp" / f"d_{self.run_id}").mkdir(parents=True, exist_ok=False)
            self.update("personal_device_leases_acquired")

        def preflight(self):
            p.require_host(self.host)
            if Path(__file__).resolve().parent != p.CODE_ROOT or len(self.locks) != 9:
                raise RuntimeError("Wrong deployment or incomplete nine-lock ownership")
            self.manifest = p.source_manifest()
            python = p.INSTALL / "env/bin/python"
            if not python.is_file() or not os.access(python, os.X_OK):
                raise RuntimeError("Private Python unavailable")
            result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
            (self.run_dir / "npu_before.txt").write_text(result.stdout + "\n" + result.stderr)
            if result.returncode:
                raise RuntimeError("npu-smi failed")
            idle = base.selected_idle(result.stdout, p.CARDS)
            hbm = base.selected_health_memory(result.stdout, p.CARDS, p.MIN_HBM_MIB)
            memory = base.host_memory_snapshot(p.MIN_HOST_AVAILABLE)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", p.PORT))
            base.atomic_json(self.run_dir / "source_manifest.json", self.manifest)
            validation_policy = dict(
                schema_version=1, mode=p.MODE, review_status="approved_for_inference",
                scope="synthetic_validation_only",
                attention=dict(endpoint_policy="vdn_anchors", auxiliary_policy="preserve_existing",
                               source_linear_text="text", target_linear_text="text", chunk_frames=5,
                               chunk_radius=1, share_parameters=True, independent_stream_states=True),
            )
            base.atomic_json(self.run_dir / "validation_policy.json", validation_policy)
            base.atomic_json(self.run_dir / "owner_proof.json", dict(
                supervisor_pid=os.getpid(), proc_identity=base.proc_identity(os.getpid()),
                run_id=self.run_id, host=self.host, case=p.CASE,
                lease_fds=[handle.fileno() for handle in self.locks],
                lease_paths=[str(self.selected.output_root / "run.lock")] +
                            [str(self.selected.lease_root / f"device{card}.lock") for card in p.CARDS],
                cards=list(p.CARDS), resources_checked=True))
            self.update("preflight_passed", validation_policy_sha256=p.digest(self.run_dir / "validation_policy.json"), resource_snapshot=idle, card_health_memory=hbm,
                        host_memory=memory, source_sha256=self.manifest)

        def env(self):
            env = os.environ.copy()
            env.update(H3_ROOT=str(self.run_dir), H3_OUTPUT=str(self.run_dir), H3_MINIMAL_RUN_ID=self.run_id,
                       H3_D_SUPERVISOR_PID=str(os.getpid()), H3_GROUP_ID=f"zhonghao_31731_D8_{self.run_id}",
                       ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, p.CARDS)),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
            policy = self.run_dir / "validation_policy.json"
            env.update(D_REVIEWED_POLICY_PATH=str(policy), D_REVIEWED_POLICY_SHA256=p.digest(policy))
            return env

        def launch(self):
            p.require_host(self.host)
            if p.source_manifest() != self.manifest:
                raise RuntimeError("Sources changed after preflight")
            log = (self.run_dir / "d_validation.log").open("ab", buffering=0)
            self.handles.append(log)
            began = time.monotonic()
            self.server = self.spawn(["bash", str(p.CODE_ROOT / "launch_d_validation.sh")],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            self.update("validation_running", worker_pid=self.server.pid,
                        worker_proc_identity=base.proc_identity(self.server.pid))
            while self.server.poll() is None:
                if time.monotonic() - began > TIMEOUT:
                    raise TimeoutError("D tiny exceeded fixed 900-second limit")
                time.sleep(1)
            if self.server.returncode:
                raise RuntimeError(f"D tiny exited {self.server.returncode}")
            p.require_host(self.host)
            if p.source_manifest() != self.manifest:
                raise RuntimeError("Sources changed during D tiny")
            from d_npu_regression import verify_results
            result = self.run_dir / "d_tiny.json"
            verify_results(result, self.host, self.run_id, self.manifest)
            self.update("validation_passed_before_cleanup", validation_passed=True,
                        result_path=str(result), result_sha256=p.digest(result),
                        validation_wall_seconds=time.monotonic() - began)
    return DValidation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_npu:
        parser.error("C NPU execution requires explicit --allow-npu")
    p.require_host()
    base = p.load_helper("supervision_base.py")
    supervisor = supervisor_class(base)()
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
            if (not clean or supervisor.status.get("needs_attention")) and error is None:
                error = "Owned process/card release requires attention"
            supervisor.update("needs_attention" if supervisor.status.get("needs_attention") else
                              ("failed" if error else "completed"), error=error, finished_at=base.utc_now())
        finally:
            supervisor.release_locks()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
