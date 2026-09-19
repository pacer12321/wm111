"""Run the read-only Oracle analysis after the owned capture fully cleans up."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/cache/zhonghao/h3")
SUPPORT = ROOT / "attention_oracle_multilayer_support_20260916_v2"
EXPERIMENT = ROOT / "attention_oracle_multilayer_redshirt_20260916_v2"
OUTPUT = EXPERIMENT / "oracle_analysis_v1"
STATE_PATH = EXPERIMENT / "oracle_supervisor_status.json"
PYTHON = ROOT / "env_cuda_v1/bin/python"
GPU_UUID = "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670"


def update(status: str, **values) -> None:
    state = {
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **values,
    }
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(STATE_PATH)
    print(json.dumps(state), flush=True)


def gpu_memory_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_UUID}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(completed.stdout.strip())


def main() -> None:
    if STATE_PATH.exists() or OUTPUT.exists():
        raise RuntimeError("supervisor output already exists")
    update("waiting_for_capture_cleanup")
    while not (EXPERIMENT / "final_memory.json").exists():
        status_path = EXPERIMENT / "status.json"
        if status_path.exists():
            capture = json.loads(status_path.read_text())
            if capture.get("status") == "failed":
                update("blocked_capture_failed", capture_status=capture)
                return
        time.sleep(20)

    capture = json.loads((EXPERIMENT / "status.json").read_text())
    if capture.get("status") != "completed_analysis_pending":
        update("blocked_capture_not_complete", capture_status=capture)
        return

    # Never contend with a new external GPU task.  The capture itself has
    # already released both workers when final_memory.json appears.
    while True:
        used = gpu_memory_mib()
        if used < 4096:
            break
        update("waiting_for_gpu0", gpu0_memory_used_mib=used)
        time.sleep(30)

    command = [
        str(PYTHON),
        str(SUPPORT / "analyze_attention_oracle_multilayer.py"),
        "--capture-root", str(EXPERIMENT / "qk_capture"),
        "--output-dir", str(OUTPUT),
        "--layers", "8,24,41",
        "--steps", "0,12,24,36,48",
        "--ks", "5,8,12,16",
        "--device", "cuda:0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    compat = (
        f"{ROOT}/cuda_compat13/usr/local/cuda-13.0/compat:"
        f"{ROOT}/env_cuda_v1/lib:{ROOT}/python312_runtime/lib"
    )
    environment["LD_LIBRARY_PATH"] = compat + ":" + environment.get("LD_LIBRARY_PATH", "")
    update("running_oracle_analysis", command=command)
    with (EXPERIMENT / "oracle_analysis.log").open("x") as log:
        completed = subprocess.run(
            command,
            cwd=SUPPORT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        update("failed_oracle_analysis", returncode=completed.returncode)
        raise SystemExit(completed.returncode)
    update("completed", output=str(OUTPUT))


if __name__ == "__main__":
    main()
