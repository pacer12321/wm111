from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/cache/zhonghao/h3")
EXPERIMENT = ROOT / "svoo_isa_all_layers_redshirt_20260917_v1"
TOOLS = ROOT / "ss_spatial_local_code_20260917_v1/tools"
PYTHON = ROOT / "env_cuda_v1/bin/python"
OUTPUT = ROOT / "svoo_isa_all_layers_oracle_20260917_v1"
CONTROLLER_PID = ROOT / "svoo_isa_all_layers_redshirt_20260917_v1.controller.pid"
STATE_PATH = OUTPUT / "status.json"


def update(status: str, **values) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **values,
    }
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATE_PATH)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_used_mib() -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir()
    pid = int(CONTROLLER_PID.read_text().strip())
    update("waiting_for_capture", controller_pid=pid)
    while alive(pid):
        time.sleep(30)

    capture_status = json.loads((EXPERIMENT / "status.json").read_text())
    if capture_status.get("status") != "completed_quality_review_required":
        update("blocked_capture_failed", capture_status=capture_status)
        raise RuntimeError("all-layer capture did not complete")

    update("waiting_for_two_idle_gpus")
    consecutive_idle = 0
    while consecutive_idle < 2:
        used = gpu_used_mib()
        consecutive_idle = consecutive_idle + 1 if used == [0, 0] else 0
        if consecutive_idle < 2:
            time.sleep(15)

    layers = ",".join(str(index) for index in range(48))
    capture_dir = EXPERIMENT / "source_qkv_capture"
    environment = os.environ.copy()
    compat = ROOT / "cuda_compat13/usr/local/cuda-13.0/compat"
    environment["LD_LIBRARY_PATH"] = ":".join(
        item for item in (str(compat), environment.get("LD_LIBRARY_PATH", "")) if item
    )
    processes = []
    for gpu, branch in enumerate(("qskt", "qsks")):
        command = [
            str(PYTHON),
            str(TOOLS / "analyze_svoo_isa_branch_oracle.py"),
            str(capture_dir),
            str(OUTPUT / f"{branch}_ideal_all_layers.json"),
            "--branch",
            branch,
            "--layers",
            layers,
            "--exact-ratios",
            "0.25,0.375,0.5",
            "--full-query-fractions",
            "0.5",
            "--device",
            "cuda",
        ]
        branch_environment = environment.copy()
        branch_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = (OUTPUT / f"{branch}.log").open("x")
        process = subprocess.Popen(
            command,
            cwd=TOOLS,
            env=branch_environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((branch, process, log))
    update(
        "running_all_layer_oracles",
        pids={branch: process.pid for branch, process, _ in processes},
    )
    return_codes = {}
    for branch, process, log in processes:
        return_codes[branch] = process.wait()
        log.close()
    if any(return_codes.values()):
        update("failed", return_codes=return_codes)
        raise RuntimeError(f"oracle failures: {return_codes}")
    update("completed", return_codes=return_codes)


if __name__ == "__main__":
    main()
