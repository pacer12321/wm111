"""Wait for both assigned A100s, then run B capture and B-aware Oracle analysis."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/cache/zhonghao/h3")
SUPPORT = ROOT / "attention_oracle_b_multilayer_support_20260916_v1"
EXPERIMENT = ROOT / "attention_oracle_b_multilayer_redshirt_20260916_v1"
CONTROL = ROOT / "attention_oracle_b_multilayer_pipeline_20260916_v1"
PYTHON = ROOT / "env_cuda_v1/bin/python"
UUIDS = [
    "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
    "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
]


def update(status: str, **values) -> None:
    CONTROL.mkdir(exist_ok=True)
    state = {
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **values,
    }
    temporary = CONTROL / "status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(CONTROL / "status.json")
    print(json.dumps(state), flush=True)


def gpu_rows() -> list[dict[str, int | str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        uuid, memory, utilization = [part.strip() for part in line.split(",")]
        if uuid in UUIDS:
            rows.append({"uuid": uuid, "memory_mib": int(memory), "utilization": int(utilization)})
    if {row["uuid"] for row in rows} != set(UUIDS):
        raise RuntimeError(f"assigned GPU UUID mismatch: {rows}")
    return rows


def main() -> None:
    if CONTROL.exists() or EXPERIMENT.exists():
        raise RuntimeError("B v1 pipeline output already exists")
    CONTROL.mkdir()
    stable = 0
    while stable < 3:
        rows = gpu_rows()
        idle = all(row["memory_mib"] < 2048 and row["utilization"] <= 5 for row in rows)
        stable = stable + 1 if idle else 0
        update("waiting_for_two_idle_a100s", gpu_rows=rows, consecutive_idle_checks=stable)
        if stable < 3:
            time.sleep(20)

    update("running_b_dense_ts_teacher_capture", gpu_rows=gpu_rows())
    environment = dict(os.environ)
    hashes = json.loads((SUPPORT / "candidate_hashes.json").read_text())
    environment["ZHONGHAO_H3_B_CAPTURE_TRANSFORMER_SHA256"] = hashes["transformer_sha256"]
    environment["ZHONGHAO_H3_B_CAPTURE_PIPELINE_SHA256"] = hashes["pipeline_sha256"]
    with (CONTROL / "capture_launcher.log").open("x") as log:
        capture = subprocess.run(
            [str(PYTHON), str(SUPPORT / "launch_b_attention_capture_timesteps.py")],
            cwd=SUPPORT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if capture.returncode:
        update("failed_b_teacher_capture", returncode=capture.returncode)
        raise SystemExit(capture.returncode)
    capture_status = json.loads((EXPERIMENT / "status.json").read_text())
    if capture_status.get("status") != "completed_analysis_pending":
        update("blocked_capture_status", capture_status=capture_status)
        return

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = UUIDS[0]
    compat = (
        f"{ROOT}/cuda_compat13/usr/local/cuda-13.0/compat:"
        f"{ROOT}/env_cuda_v1/lib:{ROOT}/python312_runtime/lib"
    )
    environment["LD_LIBRARY_PATH"] = compat + ":" + environment.get("LD_LIBRARY_PATH", "")
    output = EXPERIMENT / "oracle_analysis_b_v1"
    command = [
        str(PYTHON), str(SUPPORT / "analyze_attention_oracle_multilayer.py"),
        "--capture-root", str(EXPERIMENT / "qk_capture"),
        "--output-dir", str(output),
        "--layers", "8,24,41",
        "--steps", "0,12,24,36,48",
        "--ks", "8,12,16",
        "--teacher-mode", "b_vdn",
        "--device", "cuda:0",
    ]
    update("running_b_oracle_analysis", command=command)
    with (CONTROL / "oracle_analysis.log").open("x") as log:
        analysis = subprocess.run(
            command, cwd=SUPPORT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    if analysis.returncode:
        update("failed_b_oracle_analysis", returncode=analysis.returncode)
        raise SystemExit(analysis.returncode)
    update("completed", output=str(output))


if __name__ == "__main__":
    main()
