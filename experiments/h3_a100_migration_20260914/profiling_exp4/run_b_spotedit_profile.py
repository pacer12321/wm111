"""Run one isolated B denoising step and decode its first-step x0 estimate."""
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
EXPERIMENT = Path(
    os.environ.get(
        "ZHONGHAO_H3_SPOTEDIT_EXPERIMENT",
        str(ROOT / "spotedit_b_first_x0_20260915_v3"),
    )
)
SOURCE_CANDIDATE = PREPARED / "candidates" / "B"
CANDIDATE = Path(os.environ.get("ZHONGHAO_H3_SPOTEDIT_CANDIDATE", str(SOURCE_CANDIDATE)))
CAPTURE_DIR = os.environ.get("ZHONGHAO_H3_SELECTOR_CAPTURE_DIR")
CAPTURE_PIPELINE_SHA256 = os.environ.get("ZHONGHAO_H3_CAPTURE_PIPELINE_SHA256")
TRANSFORMER_SHA256 = os.environ.get("ZHONGHAO_H3_TRANSFORMER_SHA256")
SELECTOR_PAYLOAD = os.environ.get("ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD")
REQUEST_STEPS = int(os.environ.get("ZHONGHAO_H3_REQUEST_STEPS", "2"))
PRECHECK_STEPS = int(os.environ.get("ZHONGHAO_H3_PRECHECK_STEPS", "0"))
EXPERIMENT_LABEL = os.environ.get("ZHONGHAO_H3_EXPERIMENT_LABEL", "B first denoising prediction decoded as x0")
SAMPLE_MANIFEST_PATH = os.environ.get("ZHONGHAO_H3_SAMPLE_MANIFEST")

sys.path.insert(0, str(ORIGINAL))
import run_abcd_cuda as q
sys.path.insert(0, str(KIT))
import run_prepared_bd as prepared
from d_memory_guard_track_a import MemoryGuard
from d_memory_probe import snapshot


STATE = {
    "experiment": EXPERIMENT_LABEL,
    "case": "B",
    "requested_schedule_points": REQUEST_STEPS,
    "actual_dit_forwards": REQUEST_STEPS - 1,
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

    prepared.configure("spotedit_b_first_x0")
    q.PORT = 19127
    q.update = update
    q.STATE = STATE
    manifest = q.read(PREPARED / "cuda_manifest.json")
    if SAMPLE_MANIFEST_PATH:
        sample = q.read(Path(SAMPLE_MANIFEST_PATH))
        manifest = {**manifest, "source_path": sample["source_video"], "sample": sample}
    candidate = CANDIDATE
    for filename, expected in manifest["cases"]["B"]["model_files"].items():
        if q.sha(SOURCE_CANDIDATE / q.REL / filename) != expected:
            raise RuntimeError(f"Prepared B source candidate changed: {filename}")
        override = None
        if filename == "pipeline_minimax_h3.py" and CAPTURE_PIPELINE_SHA256:
            override = CAPTURE_PIPELINE_SHA256
        if filename == "minimax_h3_transformer.py" and TRANSFORMER_SHA256:
            override = TRANSFORMER_SHA256
        candidate_expected = override or expected
        if q.sha(candidate / q.REL / filename) != candidate_expected:
            raise RuntimeError(f"Isolated B candidate changed: {filename}")

    environment = q.env_for("B")
    environment["PYTHONPATH"] = str(KIT) + ":" + str(candidate)
    if CAPTURE_DIR:
        environment["ZHONGHAO_H3_SELECTOR_CAPTURE_DIR"] = CAPTURE_DIR
    if SELECTOR_PAYLOAD:
        environment["ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD"] = SELECTOR_PAYLOAD
        environment["ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS"] = os.environ.get(
            "ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS", "4"
        )
        environment["ZHONGHAO_H3_SPOTEDIT_RESET_STEPS"] = os.environ.get(
            "ZHONGHAO_H3_SPOTEDIT_RESET_STEPS", "13,22,31"
        )
    environment["TMPDIR"] = str(ROOT / "tmp" / "spotedit_b_first_x0")
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    command = [
        str(q.ENV / "bin/python"), "-m", "vllm_omni.entrypoints.cli.main", "serve",
        str(ROOT / "models/MiniMax-H3/Ref2VA"), "--omni", "--host", "127.0.0.1",
        "--port", str(q.PORT), "--trust-remote-code", "--init-timeout", "3600",
        "--stage-init-timeout", "3600", "--num-gpus", "2", "--usp", "2", "--ring", "1",
        "--text-encoder-tp-size", "2", "--enable-layerwise-offload", "--vae-parallel-mode", "tile",
        "--vae-use-tiling", "--vae-patch-parallel-size", "2",
        "--diffusion-attention-backend", "FLASH_ATTN", "--disable-multithread-weight-load",
    ]
    (EXPERIMENT / "launch.json").write_text(json.dumps({
        "command": command,
        "candidate": str(candidate),
        "source": manifest["source_path"],
        "sample": manifest["sample"],
        "selector_capture_dir": CAPTURE_DIR,
        "selector_payload": SELECTOR_PAYLOAD,
        "requested_schedule_points": REQUEST_STEPS,
        "interpretation": (
            "Two schedule points execute one DiT forward and return its first x0; "
            "the normal formal setting is 50 schedule points."
        ),
    }, indent=2))

    guard = MemoryGuard(EXPERIMENT / "memory_trace.jsonl", lambda: STATE.get("status"))
    original_capture = q.capture_tree

    def capture(pid, owned):
        original_capture(pid, owned)
        guard.check()

    q.capture_tree = capture
    owned = {}
    server = None
    guard.start()
    try:
        update("loading_model")
        with (EXPERIMENT / "server.log").open("x") as log:
            server = subprocess.Popen(command, cwd=candidate, env=environment, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = q.pid_identity(server.pid)
            if identity:
                owned[server.pid] = identity[0]
            deadline = time.monotonic() + 4000
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                capture(server.pid, owned)
                if server.poll() is not None:
                    raise RuntimeError("B model load failed; see server.log")
                try:
                    with opener.open(f"http://127.0.0.1:{q.PORT}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("B model readiness timeout")
            update("model_ready_running_request", requested_steps=REQUEST_STEPS)
            if PRECHECK_STEPS:
                precheck = q.request(
                    server,
                    owned,
                    "B",
                    PRECHECK_STEPS,
                    EXPERIMENT / f"precheck_{PRECHECK_STEPS}step",
                    manifest,
                )
                precheck.update(
                    schedule_points=PRECHECK_STEPS,
                    actual_dit_forwards=PRECHECK_STEPS - 1,
                    purpose="trigger at least one cache/skip step before the full run",
                    formal_speed_comparison=False,
                )
                (EXPERIMENT / f"precheck_{PRECHECK_STEPS}step" / "result.json").write_text(
                    json.dumps(precheck, indent=2)
                )
                update("precheck_completed_running_request", precheck=precheck)
            output_name = "first_x0" if REQUEST_STEPS == 2 else f"formal_{REQUEST_STEPS}step"
            result = q.request(server, owned, "B", REQUEST_STEPS, EXPERIMENT / output_name, manifest)
            result.update(
                schedule_points=REQUEST_STEPS,
                actual_dit_forwards=REQUEST_STEPS - 1,
                decoded_semantics=(
                    "first denoising model prediction x0 ([sigma=1, sigma=0])"
                    if REQUEST_STEPS == 2
                    else "full denoising output"
                ),
                purpose=(
                    "test whether early prediction already exposes edit-localization signal"
                    if REQUEST_STEPS == 2
                    else EXPERIMENT_LABEL
                ),
                formal_speed_comparison=False,
            )
            (EXPERIMENT / output_name / "result.json").write_text(json.dumps(result, indent=2))
            update("completed_quality_review_required", result=result)
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
