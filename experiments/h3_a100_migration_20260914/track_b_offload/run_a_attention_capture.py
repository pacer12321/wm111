"""Run one isolated Ref2VA-base request and capture dense Q/K at steps 0 and 48."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path("/cache/zhonghao/h3")
ORIGINAL = ROOT / "a100_v1"
PREPARED = ROOT / "bd_prepared_20260915_v2"
KIT = ROOT / "track_b_validation_20260915"
SUPPORT = ROOT / "attention_oracle_multilayer_support_20260916_v2"
EXPERIMENT = Path(
    os.environ.get(
        "ZHONGHAO_H3_BASE_CAPTURE_EXPERIMENT",
        str(ROOT / "attention_map_base_dense_redshirt_20260916_v6"),
    )
)
CANDIDATE = ROOT / "attention_oracle_multilayer_code_20260916_v1/candidate_A_capture"
CAPTURE_DIR = EXPERIMENT / "qk_capture"
CAPTURE_STEPS = os.environ.get("ZHONGHAO_H3_BASE_CAPTURE_STEPS", "0,48")
CAPTURE_STEP_VALUES = [
    int(value) for value in CAPTURE_STEPS.split(",") if value.strip()
]
CAPTURE_LAYERS = os.environ.get("ZHONGHAO_H3_ATTENTION_CAPTURE_LAYERS", "24")
CAPTURE_LAYER_VALUES = [
    int(value) for value in CAPTURE_LAYERS.split(",") if value.strip()
]
TRANSFORMER_SHA256 = os.environ["ZHONGHAO_H3_BASE_CAPTURE_TRANSFORMER_SHA256"]
PIPELINE_SHA256 = os.environ["ZHONGHAO_H3_BASE_CAPTURE_PIPELINE_SHA256"]

sys.path.insert(0, str(ORIGINAL))
import run_abcd_cuda as q
sys.path.insert(0, str(KIT))
import run_prepared_bd as prepared
from d_memory_guard_track_a import MemoryGuard
from d_memory_probe import snapshot


STATE = {
    "experiment": (
        "A Ref2VA base dense attention steps "
        + ", ".join(str(value) for value in CAPTURE_STEP_VALUES)
    ),
    "case": "A",
    "requested_schedule_points": 50,
    "actual_dit_forwards": 49,
    "status": "starting",
}


def update(status: str, **values) -> None:
    STATE.update(status=status, updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **values)
    temporary = EXPERIMENT / "status.json.tmp"
    temporary.write_text(json.dumps(STATE, indent=2))
    temporary.replace(EXPERIMENT / "status.json")
    print(json.dumps({"status": status, **values}), flush=True)


def main() -> None:
    assert socket.gethostname() == "os-node-created-mgf6h"
    EXPERIMENT.mkdir(exist_ok=False)
    lock = (ORIGINAL / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepared.idle()
    baseline = snapshot()
    file_cache = max(0, baseline["stat"]["cache"] - baseline["stat"]["shmem"])
    if baseline["usage_in_bytes"] - file_cache >= 48 * 1024**3:
        raise RuntimeError("CPU non-file baseline too high")

    manifest = q.read(ORIGINAL / "cuda_manifest.json")
    source_candidate = ORIGINAL / "candidates/A"
    for filename, expected in manifest["cases"]["A"]["model_files"].items():
        if q.sha(source_candidate / q.REL / filename) != expected:
            raise RuntimeError(f"Original A candidate changed: {filename}")
        candidate_expected = expected
        if filename == "pipeline_minimax_h3.py":
            candidate_expected = PIPELINE_SHA256
        elif filename == "minimax_h3_transformer.py":
            candidate_expected = TRANSFORMER_SHA256
        if q.sha(CANDIDATE / q.REL / filename) != candidate_expected:
            raise RuntimeError(f"Isolated A capture candidate changed: {filename}")

    q.WORK = PREPARED
    q.PORT = 19127
    q.UUIDS = prepared.UUIDS
    q.gpu_rows = prepared.gpu_rows
    q.assert_idle = prepared.idle
    q.update = update
    q.STATE = STATE
    environment = q.env_for("A")
    environment["PYTHONPATH"] = f"{SUPPORT}:{KIT}:{CANDIDATE}"
    environment["LD_LIBRARY_PATH"] += f":{ROOT}/python312_runtime/lib"
    environment["ZHONGHAO_H3_PREPARED_OFFLOAD"] = "1"
    environment["ZHONGHAO_H3_PREPARED_BASE_ONLY"] = "1"
    environment["ZHONGHAO_H3_PREPARED_MANIFEST"] = str(KIT / "validated_manifest_v1.json")
    environment["ZHONGHAO_H3_ATTENTION_CAPTURE_DIR"] = str(CAPTURE_DIR)
    environment["ZHONGHAO_H3_ATTENTION_CAPTURE_STEPS"] = CAPTURE_STEPS
    environment["ZHONGHAO_H3_ATTENTION_CAPTURE_LAYERS"] = CAPTURE_LAYERS
    environment["TMPDIR"] = str(ROOT / "tmp/base_dense_attention_capture")
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    command = [
        str(q.ENV / "bin/python"), "-m", "vllm_omni.entrypoints.cli.main", "serve",
        str(ROOT / "models/MiniMax-H3/Ref2VA"), "--omni", "--host", "127.0.0.1",
        "--port", str(q.PORT), "--trust-remote-code", "--init-timeout", "3600",
        "--stage-init-timeout", "3600", "--num-gpus", "2", "--usp", "2",
        "--ring", "1", "--text-encoder-tp-size", "2", "--enable-layerwise-offload",
        "--vae-parallel-mode", "tile", "--vae-use-tiling", "--vae-patch-parallel-size", "2",
        "--diffusion-attention-backend", "FLASH_ATTN", "--disable-multithread-weight-load",
    ]
    (EXPERIMENT / "launch.json").write_text(json.dumps({
        "command": command,
        "candidate": str(CANDIDATE),
        "source": manifest["source_path"],
        "sample": manifest["sample"],
        "capture_steps": CAPTURE_STEP_VALUES,
        "capture_layers": CAPTURE_LAYER_VALUES,
        "attention": "original Ref2VA dense; no OpenVDN; no selector; no token skip",
    }, indent=2))

    guard = MemoryGuard(EXPERIMENT / "memory_trace.jsonl", lambda: STATE.get("status"))
    original_capture = q.capture_tree
    owned: dict[int, int] = {}
    server = None

    def capture(pid: int, owned_rows: dict[int, int]) -> None:
        original_capture(pid, owned_rows)
        guard.check()

    q.capture_tree = capture
    guard.start()
    try:
        update("loading_model")
        with (EXPERIMENT / "server.log").open("x") as log:
            server = subprocess.Popen(
                command,
                cwd=CANDIDATE,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = q.pid_identity(server.pid)
            if identity:
                owned[server.pid] = identity[0]
            deadline = time.monotonic() + 4000
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                capture(server.pid, owned)
                if server.poll() is not None:
                    raise RuntimeError("A/base model load failed; see server.log")
                try:
                    with opener.open(f"http://127.0.0.1:{q.PORT}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("A/base model readiness timeout")
            update("model_ready_running_request")
            result = q.request(
                server,
                owned,
                "A",
                50,
                EXPERIMENT / "formal_50step",
                manifest,
            )
            result.update(
                attention="original Ref2VA dense",
                capture_steps=CAPTURE_STEP_VALUES,
                capture_layers=CAPTURE_LAYER_VALUES,
                formal_speed_comparison=False,
            )
            (EXPERIMENT / "formal_50step/result.json").write_text(json.dumps(result, indent=2))
            update("completed_analysis_pending", result=result)
    except BaseException as exc:
        update("failed", error=repr(exc), memory_guard_reason=guard.reason)
        raise
    finally:
        if server is not None:
            q.cleanup(server, owned)
        guard.close()
        (EXPERIMENT / "final_memory.json").write_text(json.dumps(snapshot(), indent=2))


if __name__ == "__main__":
    main()
