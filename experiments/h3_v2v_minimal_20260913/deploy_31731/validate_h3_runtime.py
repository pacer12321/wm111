#!/usr/bin/env python3
"""Offline CPU import/path/ELF checks for the private 31731 H3 installation.

Source metadata was read on 30213; no target runtime pass is implied by this
script existing. No model, tensor, NPU availability query, or device operation
is requested. Later NPU initialization attempts are explicitly blocked.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import struct
import subprocess
import sys

ROOT = Path("/cache/zhonghao/h3")
ENV = ROOT / "env"
SOURCE_ROOTS = tuple(ROOT / "src" / name for name in ("vllm", "vllm-ascend", "vllm-omni"))
TARGET_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
OLD_PREFIXES = ("/cache/yunfeng/envs/minimax-h3-npu", "/cache/yunfeng/minimax_h3_npu/src")
EXPECTED_PYTHON = "3.12.14"
EXPECTED_VERSIONS = {
    "torch": "2.10.0", "torch-npu": "2.10.0.post2", "vllm": "0.26.0",
    "vllm-ascend": "0.26.0rc1", "vllm-omni": "0.26.0+npu",
    "transformers": "5.14.1", "safetensors": "0.8.0", "numpy": "1.26.4",
    "einops": "0.8.2", "diffusers": "0.38.0", "accelerate": "1.12.0",
    "tokenizers": "0.22.2", "huggingface-hub": "1.29.0", "pillow": "12.3.0",
    "scipy": "1.13.1", "setuptools": "80.10.2", "wheel": "0.48.0", "pip": "26.2.1",
}
SYSTEM_LIBRARY_ROOTS = tuple(Path(p) for p in ("/lib", "/lib64", "/usr/lib", "/usr/lib64"))
DRIVER_ROOT = Path("/usr/local/Ascend/driver")


def inside(path, roots):
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(Path(root).resolve()) for root in roots)


def has_old_prefix(text):
    return any(re.search(re.escape(prefix) + r"(?=$|[/\\\s\"':])", text) for prefix in OLD_PREFIXES)


def allowed_library_path(path):
    return inside(path, (ENV, *SOURCE_ROOTS, *SYSTEM_LIBRARY_ROOTS, DRIVER_ROOT))


def check_binding():
    if socket.gethostname() != TARGET_HOST:
        raise RuntimeError("Run only on the approved 31731 node")
    if Path(sys.prefix).resolve() != ENV or not inside(sys.executable, (ENV,)):
        raise RuntimeError("Use the PRIVATE_ENV/bin/python interpreter")
    if platform.python_version() != EXPECTED_PYTHON or platform.machine() != "aarch64":
        raise RuntimeError("Python version or CPU architecture differs from source baseline")
    for path in (ROOT, ENV, *SOURCE_ROOTS):
        if not path.is_dir() or path.resolve() != path:
            raise RuntimeError(f"Missing/noncanonical private installation root: {path}")


def check_environment(values):
    records, warnings = {}, []
    exact = {"H3_INSTALL": ROOT, "H3_ENV": ENV,
             "ASCEND_HOME_PATH": ENV / "Ascend/cann-9.0.1",
             "ASCEND_TOOLKIT_HOME": ENV / "Ascend/cann-9.0.1"}
    for key, expected in exact.items():
        value = values.get(key, "")
        if not value or Path(value).resolve() != expected.resolve():
            raise RuntimeError(f"Wrong {key}; source env_h3_31731.sh first")
        records[key] = value
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PATH"):
        value = values.get(key, "")
        if has_old_prefix(value):
            raise RuntimeError(f"Old source prefix remains in active {key}")
        entries = value.split(":") if value else []
        if key != "PATH" and not entries:
            raise RuntimeError(f"Required active path is empty: {key}")
        if "" in entries:
            warnings.append(f"{key} contains an empty segment (current directory)")
        for entry in filter(None, entries):
            if not Path(entry).is_absolute():
                raise RuntimeError(f"Relative path in {key}: {entry}")
            if key == "LD_LIBRARY_PATH" and not allowed_library_path(entry):
                raise RuntimeError(f"Foreign runtime library directory: {entry}")
            if key == "PYTHONPATH" and not inside(entry, (ENV, *SOURCE_ROOTS)):
                raise RuntimeError(f"Foreign Python path: {entry}")
        records[key] = entries
    for key, value in values.items():
        if key.startswith(("ASCEND_", "ATB_", "CANN_")) and key.endswith(("PATH", "HOME", "DIR")):
            if not value:
                continue
            parts = value.split(":")
            allowed = (ENV,)
            if key == "ASCEND_DRIVER_PATH":
                allowed = (DRIVER_ROOT,)
            elif "CACHE" in key:
                allowed = (ROOT,)
            if any(not part or not Path(part).is_absolute() or not inside(part, allowed) for part in parts):
                raise RuntimeError(f"CANN/ATB runtime path is not private: {key}")
            records[key] = value
    if not any(key.startswith("ATB_") and key.endswith(("PATH", "HOME", "DIR")) for key in records):
        raise RuntimeError("ATB path variables were not established by its set_env.sh")
    return records, warnings


def check_metadata():
    result = {}
    for name, expected in EXPECTED_VERSIONS.items():
        dist = importlib.metadata.distribution(name)
        if dist.version != expected:
            raise RuntimeError(f"Metadata version mismatch: {name}: {dist.version} != {expected}")
        metadata_path = Path(dist._path)
        if not inside(metadata_path, (ENV,)):
            raise RuntimeError(f"Distribution metadata is not private: {name}: {metadata_path}")
        result[name] = {"version": dist.version, "metadata": str(metadata_path)}
    return result


def runtime_text_has_old_prefix(text):
    # Comments and Conda package provenance are not executable path bindings.
    active = "\n".join(line for line in text.splitlines()
                       if not line.lstrip().startswith("#") or line.startswith("#!"))
    return has_old_prefix(active)


def check_runtime_prefix_files():
    site = ENV / "lib/python3.12/site-packages"
    candidates = set()
    for pattern in ("*.pth", "*.egg-link", "__editable__*.py"):
        candidates.update(site.glob(pattern))
    for name in ("pip", "pip3", "pip3.12", "vllm", "vllm-omni"):
        path = ENV / "bin" / name
        if path.exists():
            candidates.add(path)
    candidates.update((ENV / "Ascend").rglob("set_env.sh"))
    scanned = []
    for path in sorted(candidates):
        if not inside(path, (ENV,)) or not path.is_file():
            raise RuntimeError(f"Runtime text binding escapes private environment: {path}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise RuntimeError(f"Unexpectedly large runtime binding: {path}")
        text = path.read_text(encoding="utf-8")
        if runtime_text_has_old_prefix(text):
            raise RuntimeError(f"Old prefix in active runtime file: {path}")
        scanned.append(str(path))
    return {"scanned": scanned, "provenance_exemptions": [
        "conda-meta/history", "conda-meta/*.json link/source records",
        "*.dist-info/direct_url.json provenance (actual .pth/import paths checked separately)"]}


def module_origin(module):
    value = vars(module).get("__file__")
    if not value or not inside(value, (ENV, *SOURCE_ROOTS)):
        raise RuntimeError(f"Module was imported outside private installation: {module.__name__}: {value}")
    return str(Path(value).resolve())


def forbid_device_init(*args, **kwargs):
    raise RuntimeError("CPU-only validation blocked an attempted NPU/device initialization")


def forbid_network(*args, **kwargs):
    raise RuntimeError("Offline validation blocked a network connection")


def prepare_offline():
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
        os.environ[name] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    sys.dont_write_bytecode = True
    socket.socket.connect = forbid_network
    socket.socket.connect_ex = forbid_network
    socket.create_connection = forbid_network


def import_cpu_modules():
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError("Use a fresh interpreter; torch/NPU was imported before CPU guards")
    result = {}
    torch = importlib.import_module("torch")
    result["torch"] = {"file": module_origin(torch), "runtime_version": str(torch.__version__)}
    if str(torch.__version__).split("+")[0] != EXPECTED_VERSIONS["torch"]:
        raise RuntimeError("Imported torch version differs from source metadata base version")
    npu = importlib.import_module("torch_npu")
    result["torch_npu"] = {"file": module_origin(npu)}
    if vars(npu.npu).get("_initialized") is not False:
        raise RuntimeError("torch_npu import unexpectedly initialized its device runtime")
    # The reviewed source lazy initializer invokes _C._npu_init. Block both
    # before importing vLLM and any platform plugins, without querying a device.
    npu.npu._lazy_init = forbid_device_init
    npu._C._npu_init = forbid_device_init
    if hasattr(torch, "cuda"):
        torch.cuda._lazy_init = forbid_device_init
    for name in ("vllm", "vllm_ascend", "vllm_omni", "safetensors", "numpy", "einops", "transformers"):
        module = importlib.import_module(name)
        result[name] = {"file": module_origin(module)}
        if vars(npu.npu).get("_initialized") is not False:
            raise RuntimeError(f"Device runtime initialized while importing {name}")
    return result, {"python_npu_initialized": False, "device_operations_requested": 0,
                    "npu_lazy_init_and_c_init_guarded": True,
                    "not_a_device_health_or_inference_test": True}


def elf_machine(path):
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:4] != b"\x7fELF" or header[4] != 2 or header[5] not in (1, 2):
        raise RuntimeError(f"Not a valid 64-bit ELF: {path}")
    machine = struct.unpack("<H" if header[5] == 1 else ">H", header[18:20])[0]
    if machine != 183:
        raise RuntimeError(f"Wrong ELF architecture (expected AArch64=183): {path}: {machine}")
    return machine


def parse_ldd(text):
    if "not found" in text:
        raise RuntimeError("Missing ELF dependency: " + "; ".join(line.strip() for line in text.splitlines() if "not found" in line))
    if has_old_prefix(text):
        raise RuntimeError("ELF dependency resolved into old source installation")
    result = []
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)", line)
        if match:
            library, path = match.groups()
            if not allowed_library_path(path):
                raise RuntimeError(f"ELF dependency resolved outside private/system/driver roots: {library}: {path}")
            if library.startswith(("libascendcl.so", "libopapi.so", "libatb.so", "libhccl.so")) and not inside(path, (ENV,)):
                raise RuntimeError(f"CANN/ATB library did not resolve into private environment: {library}: {path}")
            result.append({"library": library, "path": path})
    return result


def check_abi():
    site = ENV / "lib/python3.12/site-packages"
    patterns = [site / "torch/_C*.so", site / "torch_npu/_C*.so",
                site / "safetensors/_safetensors_rust*.so",
                site / "torch/lib/libtorch_cpu.so", site / "torch_npu/lib/libtorch_npu.so",
                ROOT / "src/vllm-ascend/vllm_ascend/vllm_ascend_C*.so",
                ROOT / "src/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so"]
    result = []
    for pattern in patterns:
        matches = list(pattern.parent.glob(pattern.name))
        if len(matches) != 1 or not matches[0].is_file() or not inside(matches[0], (ENV, *SOURCE_ROOTS)):
            raise RuntimeError(f"Expected exactly one private ABI library: {pattern}")
        path = matches[0]
        machine = elf_machine(path)
        process = subprocess.run(["/usr/bin/ldd", str(path)], capture_output=True, text=True, timeout=30)
        output = process.stdout + process.stderr
        if process.returncode:
            raise RuntimeError(f"ldd failed for {path}: {output.strip()}")
        result.append({"file": str(path), "elf_machine": machine, "dependencies": parse_ldd(output)})
    return result


def main():
    report = {"status": "running", "scope": "offline CPU imports, metadata, path and ELF dependencies",
              "source_baseline": "30213 read-only metadata audit", "target_runtime_previously_verified": False}
    try:
        check_binding()
        prepare_offline()
        report["environment"], report["warnings"] = check_environment(os.environ)
        report["metadata"] = check_metadata()
        report["runtime_prefix_files"] = check_runtime_prefix_files()
        report["modules"], report["device_guard"] = import_cpu_modules()
        report["abi"] = check_abi()
        report["status"] = "passed_cpu_runtime_checks_only"
        report["npu_inference_verified"] = False
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed_cpu_runtime_checks_only" else 1


if __name__ == "__main__":
    sys.exit(main())
