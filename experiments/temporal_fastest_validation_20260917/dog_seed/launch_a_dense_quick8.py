"""Eight-schedule-point original Ref2VA dense control for strict temporal reordering."""

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
EXPERIMENT = ROOT / "temporal_dog_A_dense_quick8_20260917_v3_single_gpu"
CANDIDATE = ORIGINAL / "candidates/A"
SAMPLE_MANIFEST = (
    ROOT
    / "data/generalization_candidates_20260917/temporal_reorder_dog/manifest_strict.json"
)

sys.path.insert(0, str(ORIGINAL))
import run_abcd_cuda as q
sys.path.insert(0, str(KIT))
import run_prepared_bd as prepared
from d_memory_guard_track_a import MemoryGuard
from d_memory_probe import snapshot


STATE = {
    "experiment": "A original Ref2VA dense strict temporal-reordering quick8",
    "case": "A",
    "requested_schedule_points": 8,
    "actual_dit_forwards": 7,
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

    base_manifest = q.read(ORIGINAL / "cuda_manifest.json")
    sample_manifest = json.loads(SAMPLE_MANIFEST.read_text())
    manifest = {
        **base_manifest,
        "source_path": sample_manifest["source_video"],
        "sample": sample_manifest,
    }
    for filename, expected in base_manifest["cases"]["A"]["model_files"].items():
        if q.sha(CANDIDATE / q.REL / filename) != expected:
            raise RuntimeError(f"Original A candidate changed: {filename}")

    q.WORK = PREPARED
    q.PORT = 19127
    q.UUIDS = [prepared.UUIDS[0]]
    all_gpu_rows = prepared.gpu_rows
    q.gpu_rows = lambda: [row for row in all_gpu_rows() if row[1] == q.UUIDS[0]]
    q.assert_idle = prepared.idle
    q.update = update
    q.STATE = STATE
    environment = q.env_for("A")
    environment["PYTHONPATH"] = str(CANDIDATE)
    compat = ROOT / "cuda_compat13/usr/local/cuda-13.0/compat"
    environment["LD_LIBRARY_PATH"] = ":".join(
        filter(None, (str(compat), environment.get("LD_LIBRARY_PATH", ""),
                      str(ROOT / "python312_runtime/lib")))
    )
    environment.pop("ZHONGHAO_H3_PREPARED_OFFLOAD", None)
    environment.pop("ZHONGHAO_H3_PREPARED_BASE_ONLY", None)
    environment.pop("ZHONGHAO_H3_PREPARED_MANIFEST", None)
    environment["TMPDIR"] = str(ROOT / "tmp/temporal_dog_A_dense_quick8_single_gpu")
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    command = [
        str(q.ENV / "bin/python"), "-m", "vllm_omni.entrypoints.cli.main", "serve",
        str(ROOT / "models/MiniMax-H3/Ref2VA"), "--omni", "--host", "127.0.0.1",
        "--port", str(q.PORT), "--trust-remote-code", "--init-timeout", "3600",
        "--stage-init-timeout", "3600", "--num-gpus", "1", "--usp", "1",
        "--ring", "1", "--text-encoder-tp-size", "1", "--enable-layerwise-offload",
        "--vae-parallel-mode", "tile", "--vae-use-tiling", "--vae-patch-parallel-size", "1",
        "--diffusion-attention-backend", "FLASH_ATTN", "--disable-multithread-weight-load",
    ]
    (EXPERIMENT / "launch.json").write_text(json.dumps({
        "command": command,
        "candidate": str(CANDIDATE),
        "source": manifest["source_path"],
        "sample": manifest["sample"],
        "attention": "original Ref2VA dense; single GPU; no OpenVDN; no selector; no token skip",
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
            result = q.request(server, owned, "A", 8, EXPERIMENT / "formal_8step", manifest)
            result.update(
                attention="original Ref2VA dense",
                formal_speed_comparison=False,
                purpose="capability attribution only: A dense versus B VDN-TT",
            )
            (EXPERIMENT / "formal_8step/result.json").write_text(json.dumps(result, indent=2))
            update("completed", result=result)
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
