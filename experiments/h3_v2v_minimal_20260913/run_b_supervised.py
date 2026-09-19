#!/usr/bin/env python3
"""Lease-protected B: default smoke-only, explicit same-service smoke + formal.

This Linux-only entry point requires a passing, version-matched tiny NPU test.
It reuses A's request/probe, process identity, FD inheritance and own-session
cleanup, but owns separate B state/output paths. Only --mode smoke-then-formal
permits ONE 50-step request, after a NEW same-service smoke passes strict gates.
Historical smoke evidence (including another loader version) is never imported.
No training, checkpoint modification, or operation on another user's processes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

import run_a_supervised as base


CARDS = (2, 3, 6, 7)
PORT = 19098
B_HEALTH_TIMEOUT = 2700
SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_SCRIPT_DIR = Path("/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913")
PROJECT_OUTPUT = Path("/cache/zhonghao/h3_v2v_minimal_20260913")
OUTPUT_ROOT = PROJECT_OUTPUT / "b_model"
ADAPTATION_ROOT = SCRIPT_DIR / "b_adaptation"
VENDOR_ROOT = ADAPTATION_ROOT / "vendor/vllm-omni"
VENDOR_MODEL_ROOT = VENDOR_ROOT / "vllm_omni/diffusion/models/minimax_h3"
CHECKPOINT = Path("/cache/yunfeng/models/OpenVDN-vdn-minimax-h3/stage-b-step-2000")
REF2VA_ROOT = Path("/cache/yunfeng/models/MiniMax-H3/Ref2VA")
MODES = ("smoke-only", "smoke-then-formal")
OFFICIAL_COMMIT = "2f740c9291431d89d4f2330743b093fac4390d09"
PATCHED_FILES = ("minimax_h3_transformer.py", "openvdn_npu.py",
                 "openvdn_checkpoint.py", "pipeline_minimax_h3.py")
GENERATION_FIELDS = ("width", "height", "fps", "duration_seconds", "seed",
                     "num_inference_steps", "flow_shift", "audio_flow_shift")
EXPECTED_PARALLELISM = {
    "num_gpus": 4, "usp": 4, "ring": 1, "dit_tensor_parallel_size": 1,
    "text_encoder_tp_size": 4, "layerwise_offload": True,
    "vae_parallel_mode": "tile", "vae_patch_parallel_size": 4,
    "listen_host": "127.0.0.1", "listen_port": PORT,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_identity():
    return {"hostname": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "machine": os.uname().machine}


def safetensors_identity(path):
    """Header hash plus file identity; NOT a full tensor-payload checksum."""
    path = path.resolve(strict=True)
    info = path.stat()
    with path.open("rb") as stream:
        length = stream.read(8)
        if len(length) != 8:
            raise RuntimeError(f"Truncated safetensors length: {path}")
        size = struct.unpack("<Q", length)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise RuntimeError(f"Invalid safetensors header size: {path}")
        raw = stream.read(size)
    if len(raw) != size:
        raise RuntimeError(f"Truncated safetensors header: {path}")
    header = json.loads(raw)
    count = len([key for key in header if key != "__metadata__"])
    return {"path": str(path), "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "device": info.st_dev, "inode": info.st_ino,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "tensor_count": count}


def runtime_weight_manifest():
    """Snapshot all model asset identities; stream no 135GB payload hash."""
    files = {}
    for label, root in (("ref2va", REF2VA_ROOT), ("stage_b", CHECKPOINT)):
        if not root.is_dir():
            raise RuntimeError(f"Missing model root: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in {".git", ".cache"} for part in path.relative_to(root).parts):
                continue
            info = path.stat()
            entry = {"path": str(path.resolve(strict=True)), "size_bytes": info.st_size,
                     "mtime_ns": info.st_mtime_ns, "device": info.st_dev, "inode": info.st_ino}
            if path.suffix == ".safetensors":
                entry = safetensors_identity(path)
            elif info.st_size <= 1024 * 1024:
                entry["sha256"] = digest(path)
            files[f"{label}/{path.relative_to(root).as_posix()}"] = entry
    branch = files.get("stage_b/linear_branch/model.safetensors")
    lora = files.get("stage_b/adapters/default/adapter_model.safetensors")
    index = REF2VA_ROOT / "transformer/model.safetensors.index.json"
    if branch is None or branch["tensor_count"] != 800 or lora is None or lora["tensor_count"] != 416:
        raise RuntimeError("Incomplete original Stage-B checkpoint headers")
    weight_map = json.loads(index.read_text())["weight_map"]
    if len(weight_map) != 535 or len(set(weight_map.values())) != 13:
        raise RuntimeError("Expected original 535-tensor/13-shard Ref2VA transformer")
    for name in set(weight_map.values()):
        if f"ref2va/transformer/{name}" not in files:
            raise RuntimeError(f"Missing original Ref2VA shard: {name}")
    return {"files": files, "branch": branch, "lora": lora,
            "payload_hash_scope": "File identity/size/mtime and safetensors header hashes; not full tensor-payload hashes."}


def logging_configuration():
    # Standard logging formatter provides genuine process IDs, unlike counting
    # four unlabelled messages and assuming they came from four distinct ranks.
    return {"version": 1, "disable_existing_loggers": False,
            "formatters": {"h3b": {"format": "H3B pid=%(process)d %(asctime)s %(levelname)s %(name)s %(message)s"}},
            "handlers": {"h3b": {"class": "logging.StreamHandler", "formatter": "h3b", "stream": "ext://sys.stderr"}},
            "loggers": {name: {"handlers": ["h3b"], "level": "INFO", "propagate": False}
                        for name in ("vllm", "vllm_omni")}}


def parse_full_load_evidence(log_text, weights):
    records = {}
    for marker, field in (("OPENVDN_B_LOAD_RECORD", "load"), ("OPENVDN_B_CPU_LOADING_END", "cpu_loading")):
        pattern = re.compile(r"H3B pid=(\d+) [^\n]*?" + marker + r" (\{[^\n]*\})")
        for match in pattern.finditer(log_text):
            pid = int(match.group(1))
            row = records.setdefault(pid, {})
            if field in row:
                raise RuntimeError(f"Duplicate {marker} for worker PID {pid}")
            row[field] = json.loads(match.group(2))
    if len(records) != 4 or any(set(row) != {"load", "cpu_loading"} for row in records.values()):
        raise RuntimeError("Need full-load and restored-CPU-loader records from four distinct worker PIDs")
    expected = {"checkpoint": str(CHECKPOINT), "base_partition": "ref2va", "base_tensor_count": 535,
                "branch_tensor_count": 800, "lora_pairs_merged": 208, "lora_rank": 64, "lora_alpha": 64,
                "lora_scale": 1.0, "official_scale_source_commit": OFFICIAL_COMMIT,
                "merge_dtype": "FP32 delta, cast to parameter dtype, then add",
                "qkv_merge_layout": "post-base-loader contiguous Q/K/V thirds"}
    for pid, row in records.items():
        load, cpu = row["load"], row["cpu_loading"]
        if any(load.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Incomplete/mismatched original weight coverage for PID {pid}")
        for kind in ("branch", "lora"):
            if any(load.get(kind, {}).get(key) != weights[kind][key]
                   for key in ("path", "size_bytes", "header_sha256", "tensor_count")):
                raise RuntimeError(f"Loaded {kind} differs from this run's checkpoint for PID {pid}")
        if (cpu.get("status") != "completed" or cpu.get("intraop_loading") != 4
                or type(cpu.get("intraop_before")) is not int or cpu["intraop_before"] <= 0
                or cpu.get("intraop_after") != cpu["intraop_before"]
                or type(cpu.get("interop_before")) is not int or cpu["interop_before"] <= 0
                or cpu.get("interop_after") != cpu["interop_before"]
                or cpu.get("interop_loading") != cpu["interop_before"]):
            raise RuntimeError(f"New four-thread CPU loader did not restore inference threads for PID {pid}")
        timings = cpu.get("phase_seconds", {})
        if any(type(timings.get(key)) not in (int, float) or not math.isfinite(timings[key]) or timings[key] < 0
               for key in ("base", "branch", "lora")):
            raise RuntimeError(f"Missing/invalid base/branch/LoRA load timing for PID {pid}")
    return records


def selected_worker_map(output):
    selected = {}
    pattern = re.compile(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*[^|]+\|\s*\d+\s*\|$", re.MULTILINE)
    for match in pattern.finditer(output):
        card, _chip, pid = map(int, match.groups())
        if card in CARDS:
            if card in selected or pid in selected.values():
                raise RuntimeError("Expected exactly one distinct B worker per selected physical card")
            selected[card] = pid
    if set(selected) != set(CARDS):
        raise RuntimeError("Four selected-card worker contexts must remain alive after smoke")
    return selected


def verify_smoke_artifacts(run_dir, smoke, config):
    directory = run_dir / "smoke_2step"
    result_path, request_path = directory / "result.json", directory / "request.json"
    if json.loads(result_path.read_text()) != smoke:
        raise RuntimeError("In-memory smoke differs from this run's immutable result")
    if (smoke.get("success") is not True or smoke.get("http_code") != "200"
            or smoke.get("curl_returncode") != 0 or smoke.get("requested_steps") != 2
            or smoke.get("content_type", "").split(";")[0] != "video/mp4"
            or smoke.get("ffprobe", {}).get("verified") is not True):
        raise RuntimeError("Fresh same-service smoke must succeed and have verified video metadata")
    expected_output = run_dir / "output/smoke_2step.mp4"
    output = Path(smoke.get("output_video", ""))
    if (output != expected_output or output.resolve(strict=True) != expected_output.resolve(strict=True)
            or output.is_symlink() or output.stat().st_size <= 0):
        raise RuntimeError("Smoke output must be this run's nonempty regular video")
    request = json.loads(request_path.read_text())
    generation, fields = config["requested_generation"], request.get("fields", {})
    expected_fields = {key: generation[key] for key in ("width", "height", "fps", "flow_shift", "seed")}
    expected_fields.update(prompt=config["edit_prompt"], num_inference_steps=2)
    if (request.get("requested_steps") != 2 or request.get("source_video") != config["source_video"]
            or any(fields.get(key) != value for key, value in expected_fields.items())):
        raise RuntimeError("Smoke used a different source/prompt/sampler configuration")
    extra = json.loads(fields.get("extra_params", "{}"))
    if extra != {"task": "ref2va", "duration": generation["duration_seconds"],
                 "audio_flow_shift": generation["audio_flow_shift"]}:
        raise RuntimeError("Smoke extra parameters differ from the formal Ref2VA configuration")
    checks = smoke["ffprobe"].get("checks", {})
    frames = checks.get("frame_count")
    if (checks.get("dimensions") != [generation["width"], generation["height"]]
            or checks.get("fps") != generation["fps"] or type(frames) is not int or frames <= 0):
        raise RuntimeError("Smoke dimensions/FPS/frame count are not all verified")
    return {"request_path": str(request_path), "request_sha256": digest(request_path),
            "result_path": str(result_path), "result_sha256": digest(result_path),
            "output_video": str(output), "output_sha256": digest(output), "output_bytes": output.stat().st_size,
            "quality_evidence": False}


def require_selected_cards_healthy(output):
    """Read the card/model/Health rows, failing closed on missing/duplicate rows."""
    health = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4:
            continue
        card_model = re.fullmatch(r"(\d+)\s+([A-Za-z0-9][A-Za-z0-9 _-]*)", parts[1])
        if card_model is None or re.search(r"[A-Za-z]", card_model.group(2)) is None:
            continue
        card = int(card_model.group(1))
        if card not in CARDS:
            continue
        if card in health:
            raise RuntimeError(f"Duplicate npu-smi health row for physical NPU {card}")
        health[card] = {"model": card_model.group(2), "health": parts[2]}
    if set(health) != set(CARDS) or any(row["health"] != "OK" for row in health.values()):
        raise RuntimeError(f"Selected NPUs are not all explicitly healthy: {health}")
    return health


def check_validation_result(path):
    if not path.is_absolute():
        raise ValueError("--validation-result must be an absolute path")
    path = path.resolve(strict=True)
    allowed = (PROJECT_OUTPUT / "b_validation/runs").resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError(f"Validation result must be a file under {allowed}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("status") != "passed" or result.get("physical_cards") != list(CARDS)
            or result.get("world_size") != len(CARDS) or result.get("dtype") != "bfloat16"):
        raise RuntimeError("Tiny NPU summary must report passed, BF16, and physical cards 2/3/6/7")
    source_hashes = result.get("source_sha256", {})
    required_test_sources = ("patched/minimax_h3_transformer.py", "patched/openvdn_npu.py")
    for relative in required_test_sources:
        current = ADAPTATION_ROOT / relative
        if not current.is_file() or source_hashes.get(relative) != digest(current):
            raise RuntimeError(f"NPU validation is missing or stale for {relative}; rerun the tiny test")
    rank_results = result.get("rank_results", [])
    if len(rank_results) != len(CARDS):
        raise RuntimeError("Tiny NPU summary must contain four passing rank results")
    rank_records = {}
    for item in rank_results:
        rank = item.get("rank")
        if (type(rank) is not int or rank not in range(len(CARDS)) or rank in rank_records
                or item.get("status") != "passed" or item.get("physical_card") != CARDS[rank]
                or item.get("source_sha256") != source_hashes or not item.get("tests")):
            raise RuntimeError("Invalid, duplicate, empty, or mismatched passing rank result")
        rank_path = path.with_name(f"{path.stem}.rank{rank}.json")
        on_disk = json.loads(rank_path.read_text(encoding="utf-8"))
        if on_disk != item:
            raise RuntimeError(f"Embedded rank {rank} result differs from its standalone result file")
        rank_records[rank] = {"path": str(rank_path), "sha256": digest(rank_path)}
    vendor_hashes = {}
    for name in PATCHED_FILES:
        candidate = ADAPTATION_ROOT / "patched" / name
        vendor = VENDOR_MODEL_ROOT / name
        if not candidate.is_file() or not vendor.is_file():
            raise RuntimeError(f"Missing candidate/vendor file for {name}")
        candidate_hash, vendor_hash = digest(candidate), digest(vendor)
        if candidate_hash != vendor_hash:
            raise RuntimeError(f"Vendor {name} does not match the reviewed candidate")
        vendor_hashes[name] = candidate_hash
    return {
        "result_path": str(path), "result_sha256": digest(path),
        "source_sha256": source_hashes, "vendor_candidate_sha256": vendor_hashes,
        "rank_results": rank_records,
        "note": "Tiny NPU correctness passed; full weights/offload/layout/quality remain to be checked.",
    }


class BSupervisor(base.Supervisor):
    def __init__(self, config, validation_result, mode="smoke-only"):
        if mode not in MODES:
            raise ValueError(f"Unsupported B mode: {mode}")
        if tuple(base.CARDS) != CARDS or base.PORT != PORT or base.HEALTH_TIMEOUT != 1800:
            raise RuntimeError("Reviewed A supervisor card/port/startup defaults changed")
        super().__init__(config, SCRIPT_DIR)
        self.validation_result = validation_result
        self.validated_gate = None
        self.mode = mode
        self.formal_manifest = None
        self._requests_started = False
        self.status.update(case="B", mode=mode,
                           stage_b_checkpoint=str(CHECKPOINT),
                           validation_result=str(validation_result),
                           formal_50step_started=False, formal_50step_completed=False,
                           formal_request_attempts=0, quality_evidence=False,
                           startup_timeouts={"orchestrator_seconds": 2400,
                                             "stage_init_seconds": 2400,
                                             "supervisor_health_seconds": B_HEALTH_TIMEOUT})

    def update(self, phase, **values):
        self.status.update(values)
        self.status.update(phase=phase, updated_at=base.utc_now())
        if self.run_dir is not None:
            base.atomic_json(self.run_dir / "b_status.json", self.status)
            base.atomic_json(OUTPUT_ROOT / "b_status.json", self.status)
            if self.mode == "smoke-then-formal":
                base.atomic_json(OUTPUT_ROOT / "b_formal_status.json", self.status)
        print(json.dumps({"phase": phase, "updated_at": self.status["updated_at"], **values},
                         ensure_ascii=False), flush=True)

    def acquire(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        own_lock = (OUTPUT_ROOT / "run.lock").open("a+b")
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            own_lock.close()
            raise RuntimeError("Another supervised full B experiment holds its run.lock")
        self.locks.append(own_lock)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        prefix = "b_smoke_formal_" if self.mode == "smoke-then-formal" else "b_"
        self.run_dir = OUTPUT_ROOT / "runs" / f"{prefix}{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "output").mkdir()
        self.status["run_directory"] = str(self.run_dir)
        base.atomic_json(self.run_dir / "experiment.json", self.config)
        self.update("acquiring_device_leases")
        for card in CARDS:
            path = base.LEASE_ROOT / f"new_layout_training_device{card}.lock"
            handle = path.open("rb")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError(f"Physical NPU {card} lease is held; no B model was launched")
            self.locks.append(handle)
        self.update("device_leases_acquired")

    def preflight(self):
        if SCRIPT_DIR != EXPECTED_SCRIPT_DIR:
            raise RuntimeError(f"Deploy this supervisor only at {EXPECTED_SCRIPT_DIR}")
        if len(self.locks) != len(CARDS) + 1:
            raise RuntimeError("All four existing device leases must be held before preflight")
        if self.config.get("allocated_physical_npu_ids") != list(CARDS):
            raise RuntimeError("experiment.json conflicts with the fixed physical 2/3/6/7 allocation")
        configured_parallelism = self.config.get("A_launch_configuration", {})
        for key, expected in EXPECTED_PARALLELISM.items():
            if configured_parallelism.get(key) != expected:
                raise RuntimeError(f"A/B parallelism must match: {key} != {expected!r}")
        # Same shared experiment.json, plus a comparison to the successful A's
        # immutable per-run snapshot for all source/prompt/sampler fields.
        a_run = Path(self.config["latest_A_run"]["run_directory"])
        if not a_run.is_absolute() or not a_run.resolve().is_relative_to((PROJECT_OUTPUT / "runs").resolve()):
            raise RuntimeError("A run reference must stay within the authorized original run directory")
        a_config_path = a_run / "experiment.json"
        a_config = json.loads(a_config_path.read_text(encoding="utf-8-sig"))
        for key in ("source_video", "edit_prompt"):
            if self.config.get(key) != a_config.get(key):
                raise RuntimeError(f"B {key} differs from the completed A run")
        generation = self.config["requested_generation"]
        for key in GENERATION_FIELDS:
            if generation.get(key) != a_config["requested_generation"].get(key):
                raise RuntimeError(f"B generation field {key} differs from A")
        if generation.get("num_inference_steps") != 50:
            raise RuntimeError("Keep the matched formal configuration at 50; the separate smoke request always uses 2")
        source = Path(self.config["source_video"])
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty source video: {source}")
        required = [SCRIPT_DIR / "launch_b_server.sh", VENDOR_ROOT / "vllm_omni/__init__.py",
                    CHECKPOINT / "linear_branch/model.safetensors",
                    CHECKPOINT / "adapters/default/adapter_model.safetensors"]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("A B launcher, vendor package, or full Stage-B weight file is missing/empty")
        for executable in ("npu-smi", "curl", "bash"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"Required executable is missing: {executable}")
        self.validated_gate = check_validation_result(self.validation_result)
        base.atomic_json(self.run_dir / "validated_npu_gate.json", self.validated_gate)
        check = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                               timeout=30, check=False)
        (self.run_dir / "npu_before.txt").write_text(check.stdout + "\n" + check.stderr)
        if check.returncode:
            raise RuntimeError(f"npu-smi returned {check.returncode}")
        idle = base.require_selected_cards_idle(check.stdout)
        health = require_selected_cards_healthy(check.stdout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", PORT))
        self.update("preflight_passed", resource_snapshot=idle, resource_health=health,
                    validation_gate=self.validated_gate,
                    matched_a_config={"path": str(a_config_path), "sha256": digest(a_config_path)})
        if self.mode == "smoke-then-formal":
            base.atomic_json(self.run_dir / "formal_logging.json", logging_configuration())
            self.formal_manifest = self.runtime_manifest()
            base.atomic_json(self.run_dir / "formal_runtime_manifest.json", self.formal_manifest)
            self.update("formal_mode_preflight_passed", formal_runtime_manifest=self.formal_manifest,
                        note="A new two-step smoke under these exact files must pass before the one formal request.")

    def runtime_manifest(self):
        source_files = [SCRIPT_DIR / "run_a_supervised.py", SCRIPT_DIR / "run_b_supervised.py",
                        SCRIPT_DIR / "launch_b_server.sh", self.run_dir / "formal_logging.json"]
        return {"host": host_identity(), "run_id": self.run_id, "cards": list(CARDS),
                "parallelism": EXPECTED_PARALLELISM.copy(),
                "generation": {key: self.config["requested_generation"][key] for key in GENERATION_FIELDS},
                "source_video": self.config["source_video"], "source_sha256": digest(Path(self.config["source_video"])),
                "edit_prompt": self.config["edit_prompt"], "tiny_and_candidate_gate": check_validation_result(self.validation_result),
                "supervision_files": {str(path): digest(path) for path in source_files},
                "weights": runtime_weight_manifest()}

    def require_fresh_smoke_for_formal(self):
        if self.mode != "smoke-then-formal" or self.formal_manifest is None:
            raise RuntimeError("Formal requests require explicit mode and this run's immutable manifest")
        if self.status.get("formal_request_attempts") != 0 or self.status.get("formal_50step_started"):
            raise RuntimeError("This supervisor may attempt exactly one formal request, never retry it")
        if len(self.locks) != 5 or any(handle.closed for handle in self.locks):
            raise RuntimeError("All four card leases and the B run lock must remain held")
        idle = self.status.get("resource_snapshot", {})
        if (idle.get("selected_physical_cards") != list(CARDS)
                or not set(CARDS).issubset(idle.get("explicitly_idle_cards", []))):
            raise RuntimeError("Missing this run's fresh preflight idle-card proof under all leases")
        if self.runtime_manifest() != self.formal_manifest:
            raise RuntimeError("Host/cards/input/source/weights changed since this service was launched")
        expected = self.status.get("server_proc_identity", {})
        current = base.proc_identity(self.server.pid)
        if (self.server.poll() is not None or current is None
                or any(current.get(key) != expected.get(key) for key in ("pid", "start_ticks", "pgrp", "session"))):
            raise RuntimeError("The exact same smoke service must still be alive for the formal request")
        smoke = verify_smoke_artifacts(self.run_dir, self.status.get("smoke_2step", {}), self.config)
        log_path = self.run_dir / "server.log"
        log_raw = log_path.read_bytes()
        load = parse_full_load_evidence(log_raw.decode("utf-8", errors="replace"), self.formal_manifest["weights"])
        check = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=False)
        (self.run_dir / "npu_before_formal.txt").write_text(check.stdout + "\n" + check.stderr)
        if check.returncode:
            raise RuntimeError("npu-smi failed before formal request")
        health = require_selected_cards_healthy(check.stdout)
        workers = selected_worker_map(check.stdout)
        owned = set(self.owned_group_members(self.server.pid))
        if set(workers.values()) != set(load) or not set(load).issubset(owned):
            raise RuntimeError("Full-load records do not match the four selected-card workers owned by this service")
        identities = {str(card): base.proc_identity(pid) for card, pid in workers.items()}
        if any(identity is None or identity.get("pid") != workers[int(card)]
               or identity.get("pgrp") != self.server.pid or identity.get("session") != self.server.pid
               for card, identity in identities.items()):
            raise RuntimeError("A selected worker exited/changed session while formal evidence was collected")
        evidence = {"status": "passed", "mode": "same_service_new_smoke", "run_id": self.run_id,
                    "host": self.formal_manifest["host"], "cards": list(CARDS),
                    "candidate_gate": self.validated_gate, "smoke": smoke,
                    "server_identity": current, "worker_identities_by_card": identities,
                    "loaded_records_by_pid": load, "health": health,
                    "server_log_prefix_bytes": len(log_raw), "server_log_prefix_sha256": hashlib.sha256(log_raw).hexdigest(),
                    "preflight_cards_verified_idle_under_leases": True,
                    "cleanup_policy": "Same live service: retain all leases through both requests, then verify cleanup. No previous smoke is reused.",
                    "quality_evidence": False, "actual_model_forward_calls": None}
        base.atomic_json(self.run_dir / "formal_smoke_gate.json", evidence)
        self.update("formal_smoke_gate_passed", formal_smoke_gate=evidence)
        return evidence

    def run_requests(self):
        if self._requests_started:
            raise RuntimeError("B request workflow is single-use; no automatic retry or second formal request")
        self._requests_started = True
        self.request("smoke_2step", 2)
        if self.mode == "smoke-only":
            self.update("smoke_passed_waiting_for_review", review_required=True,
                        review_items=["Full branch/LoRA load coverage", "Actual packed layout",
                                      "CPU fallback and memory/runtime warnings", "Smoke video inspection"],
                        note="Smoke-only mode: stop and release cards; no formal request was authorized.")
            return
        self.require_fresh_smoke_for_formal()
        self.update("formal_50step_authorized", formal_50step_started=True, formal_request_attempts=1,
                    formal_request_started_at=base.utc_now(),
                    note="Exactly one matched 50-step request; startup and the smoke are outside its timing.")
        self.request("b_50step", 50)
        formal = self.status.get("b_50step", {})
        if formal.get("success") is not True or formal.get("ffprobe", {}).get("verified") is not True:
            raise RuntimeError("Formal response is not a verified successful video")
        self.update("formal_50step_passed_waiting_for_review", formal_50step_completed=True,
                    review_required=True, quality_evidence=False,
                    note="Formal B video exists; quality, NFE and algorithmic acceleration interpretation still need review.")

    def env(self):
        env = super().env()
        prior_pythonpath = env.get("PYTHONPATH", "")
        env.update(
            PYTHONPATH=str(VENDOR_ROOT) + (":" + prior_pythonpath if prior_pythonpath else ""),
            ZHONGHAO_H3_OPENVDN="1",
            ZHONGHAO_H3_OPENVDN_CHECKPOINT=str(CHECKPOINT),
            H3_B_SUPERVISOR_PID=str(os.getpid()),
            H3_B_VALIDATION_RESULT=str(self.validation_result.resolve()),
        )
        if self.mode == "smoke-then-formal":
            env.update(VLLM_CONFIGURE_LOGGING="1", VLLM_LOGGING_CONFIG_PATH=str(self.run_dir / "formal_logging.json"))
        return env

    def launch(self):
        # Check once more immediately before exec, not only before resource
        # inspection. No untested candidate/vendor replacement is allowed.
        if check_validation_result(self.validation_result) != self.validated_gate:
            raise RuntimeError("Validation proof or candidate/vendor files changed after preflight")
        if self.mode == "smoke-then-formal" and self.runtime_manifest() != self.formal_manifest:
            raise RuntimeError("Formal-mode runtime files changed immediately before launch")
        log = (self.run_dir / "server.log").open("ab", buffering=0)
        self.handles.append(log)
        self.server = self.spawn(["bash", str(SCRIPT_DIR / "launch_b_server.sh")],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.update("server_starting", server_pid=self.server.pid,
                    server_started_at=base.utc_now(), server_proc_identity=base.proc_identity(self.server.pid))
        # Full CPU finite validation + LoRA merge is startup work, not measured
        # generation time. The CLI's inner timeout must not kill a healthy load.
        deadline = time.monotonic() + B_HEALTH_TIMEOUT
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"B server exited during startup: {self.server.returncode}; see server.log")
            try:
                with opener.open(base.BASE_URL + "/health", timeout=5) as response:
                    if response.status == 200:
                        self.update("server_healthy", server_ready_at=base.utc_now())
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(f"B server did not become healthy within {B_HEALTH_TIMEOUT} seconds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-result", type=Path, required=True,
                        help="Absolute path of the passing, version-matched tiny NPU summary")
    parser.add_argument("--mode", choices=MODES, default="smoke-only",
                        help="Explicit smoke-then-formal runs a NEW smoke and at most one 50-step request on the same service")
    args = parser.parse_args(argv)
    if not args.validation_result.is_absolute():
        parser.error("--validation-result must be an absolute path")
    config = json.loads((SCRIPT_DIR / "experiment.json").read_text(encoding="utf-8-sig"))
    supervisor = BSupervisor(config, args.validation_result, mode=args.mode)
    signal.signal(signal.SIGTERM, base.interrupted)
    signal.signal(signal.SIGINT, base.interrupted)
    error = None
    try:
        supervisor.acquire()
        supervisor.preflight()
        supervisor.launch()
        supervisor.run_requests()
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
                    error = "Owned B processes remain after cleanup; do not reuse the cards until checked"
                if supervisor.status.get("needs_attention") and error is None:
                    error = "Post-B resource release could not be verified"
                completed = ("formal_completed_review_required" if supervisor.status.get("formal_50step_completed")
                             else "smoke_completed_review_required")
                phase = "needs_attention" if supervisor.status.get("needs_attention") else ("failed" if error else completed)
                supervisor.update(phase, finished_at=base.utc_now(), error=error,
                                  note=("One formal B request completed; own processes/cards cleaned. Video quality, NFE, and acceleration claim remain unreviewed."
                                        if phase == "formal_completed_review_required" else
                                        "Inspect mode/request counters and error: never infer formal completion from smoke or startup. No automatic retry."))
        finally:
            supervisor.release_locks()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
