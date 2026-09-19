"""Run an isolated B or B-K Oracle-proxy Ref2VA generation on two A100s."""

from __future__ import annotations

import argparse
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
SUPPORT = ROOT / "bk_oracle_support_20260916_v1"
CANDIDATE = ROOT / "bk_oracle_code_20260916_v1/candidate_BK_oracle"
ROUTES = ROOT / "bk_oracle_routes_redshirt_20260916_v1"

sys.path.insert(0, str(ORIGINAL))
import run_abcd_cuda as q
sys.path.insert(0, str(KIT))
import run_prepared_bd as prepared
from d_memory_guard_track_a import MemoryGuard
from d_memory_probe import snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("B", "K0", "K12", "K16"), required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19131)
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1:
        raise ValueError("steps must be >=2 and repeats >=1")
    if socket.gethostname() != "os-node-created-mgf6h":
        raise RuntimeError("This runner is pinned to the validated 30674 host")
    args.experiment.mkdir(exist_ok=False, parents=True)
    lock = (ORIGINAL / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepared.idle()

    state = {
        "mode": args.mode,
        "steps": args.steps,
        "repeats": args.repeats,
        "status": "starting",
        "candidate": str(CANDIDATE),
        "sample": "shirt_red_couple_124",
        "oracle_scope": "nearest sampled layer/timestep B-dense-teacher route proxy",
    }

    def update(status: str, **values) -> None:
        state.update(status=status, updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **values)
        tmp = args.experiment / "status.json.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.experiment / "status.json")
        print(json.dumps({"status": status, **values}), flush=True)

    baseline = snapshot()
    file_cache = max(0, baseline["stat"]["cache"] - baseline["stat"]["shmem"])
    if baseline["usage_in_bytes"] - file_cache >= 48 * 1024**3:
        raise RuntimeError("CPU non-file baseline too high")

    manifest = q.read(ORIGINAL / "cuda_manifest.json")
    q.WORK = PREPARED
    q.PORT = args.port
    q.UUIDS = prepared.UUIDS
    q.gpu_rows = prepared.gpu_rows
    q.assert_idle = prepared.idle
    q.update = update
    q.STATE = state
    environment = q.env_for("B")
    environment["PYTHONPATH"] = f"{SUPPORT}:{KIT}:{CANDIDATE}"
    environment["LD_LIBRARY_PATH"] += f":{ROOT}/python312_runtime/lib"
    environment["ZHONGHAO_H3_PREPARED_OFFLOAD"] = "1"
    environment.pop("ZHONGHAO_H3_PREPARED_BASE_ONLY", None)
    environment["ZHONGHAO_H3_PREPARED_MANIFEST"] = str(KIT / "validated_manifest_v1.json")
    environment["TMPDIR"] = str(ROOT / f"tmp/bk_oracle_{args.mode.lower()}")
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    route_file = None
    if args.mode != "B":
        route_file = ROUTES / f"routes_k{args.mode[1:]}.json"
        if not route_file.is_file():
            raise FileNotFoundError(route_file)
        environment["ZHONGHAO_H3_TS_ROUTE_FILE"] = str(route_file)
    else:
        environment.pop("ZHONGHAO_H3_TS_ROUTE_FILE", None)

    command = [
        str(q.ENV / "bin/python"), "-m", "vllm_omni.entrypoints.cli.main", "serve",
        str(ROOT / "models/MiniMax-H3/Ref2VA"), "--omni", "--host", "127.0.0.1",
        "--port", str(q.PORT), "--trust-remote-code", "--init-timeout", "3600",
        "--stage-init-timeout", "3600", "--num-gpus", "2", "--usp", "2",
        "--ring", "1", "--text-encoder-tp-size", "2", "--enable-layerwise-offload",
        "--vae-parallel-mode", "tile", "--vae-use-tiling", "--vae-patch-parallel-size", "2",
        "--diffusion-attention-backend", "FLASH_ATTN", "--disable-multithread-weight-load",
    ]
    (args.experiment / "launch.json").write_text(json.dumps({
        "command": command,
        "route_file": str(route_file) if route_file else None,
        "comparison_base": "B: identical candidate with route disabled",
        "true_sparse_kernel": True,
        "dense_mask": False,
        "visible_rule": "A_struct union S_i union Oracle Top-K",
    }, indent=2))

    guard = MemoryGuard(args.experiment / "memory_trace.jsonl", lambda: state.get("status"))
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
        with (args.experiment / "server.log").open("x") as log:
            server = subprocess.Popen(
                command, cwd=CANDIDATE, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            identity = q.pid_identity(server.pid)
            if identity:
                owned[server.pid] = identity[0]
            deadline = time.monotonic() + 4000
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                capture(server.pid, owned)
                if server.poll() is not None:
                    raise RuntimeError("model load failed; see server.log")
                try:
                    with opener.open(f"http://127.0.0.1:{q.PORT}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("model readiness timeout")
            update("model_ready")
            results = []
            for repeat in range(args.repeats):
                label = "runtime_preheat" if args.repeats > 1 and repeat == 0 else f"timing_{repeat}"
                output = args.experiment / label
                update("running_request", repeat=repeat, label=label)
                result = q.request(server, owned, "B", args.steps, output, manifest)
                result.update(
                    mode=args.mode,
                    route_file=str(route_file) if route_file else None,
                    formal_speed_comparison=args.steps == 50,
                    runtime_preheat=args.repeats > 1 and repeat == 0,
                    attention=(
                        "B VDN T-T local+linear; dense T-S"
                        if args.mode == "B"
                        else f"B VDN T-T local+linear; true sparse T-S {args.mode}"
                    ),
                )
                (output / "result.json").write_text(json.dumps(result, indent=2))
                results.append(result)
            update("completed", results=results)
    except BaseException as exc:
        update("failed", error=repr(exc), memory_guard_reason=guard.reason)
        raise
    finally:
        if server is not None:
            q.cleanup(server, owned)
        guard.close()
        (args.experiment / "final_memory.json").write_text(json.dumps(snapshot(), indent=2))


if __name__ == "__main__":
    main()
