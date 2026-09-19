"""Fixed, isolated C tiny deployment; reuse reviewed B helpers without editing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import socket
import sys

INSTALL = Path("/cache/zhonghao/h3")
CODE_ROOT = INSTALL / "c_validation_code"
B_CODE = INSTALL / "validation_code"
VENDOR = INSTALL / "candidates/c_v1/vllm-omni"
EXPECTED_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
WORLD, CARDS, PORT = 8, tuple(range(8)), 29675
MIN_HOST_AVAILABLE, MIN_HBM_MIB = 64 * 1024**3, 8192
CASE = "C_strict_tiny_SP8_31731_v1"
PINNED_HELPERS = {
    "profiles.py": "f7ff6a1bfe7d4a91f628e861c435619f731338bf6b36ff81ae780ee0d5511269",
    "supervision_base.py": "5889609771d9f80565e52f3d6c0bbb239d4a555c28cc8ffec0b65c8abb4cb344",
    "npu_wrapper_regression.py": "7679909766ca09168e29ad7a9a5943571ab2acec7838c9c5096f158105a0c698",
}
PINNED_C = {
    "openvdn_npu.py": "4808ef3017343904d8b52cda9cfebc94798f4d609f8ec19e63c59b92abf3381f",
    "openvdn_checkpoint.py": "fbe011bae524bea16f54a14032e61e82f2c68aa4da6d426b98a13e097fb8f19f",
    "minimax_h3_transformer.py": "750056c7166238dc6275e9163f751034ecf2f8e3b1c03af400ea6fe9c83043df",
    "pipeline_minimax_h3.py": "9a124e4bef1421c2f0a3e7c45afa4260f975eee09a4565e7343e07daa8648414",
    "strict_source_layout.py": "0986978dbd005ec97af88613073465b5eada55434b0293b8098a5f40bfe81028",
    "strict_source_attention.py": "b96d20631f4cf9b5a06ea562e63c127c3e1e7c22fef0eb234aac9b1929941158",
}
GOLDEN_SHA = "bb06d4183e79dc5837d19a68ebaf1abdb520098dc50b05b6e22d34e9259cb43c"
OWN_FILES = ("c_profiles.py", "c_cases.py", "c_npu_regression.py", "run_c_validation.py", "launch_c_validation.sh")


@dataclass(frozen=True)
class Profile:
    group: str = "01234567"
    cards: tuple = CARDS
    master_port: int = PORT
    code_root: Path = CODE_ROOT
    vendor_root: Path = VENDOR
    output_root: Path = INSTALL / "c_validation/01234567"
    lease_root: Path = INSTALL / "validation/card_locks"  # same locks as A/B
    env_script: Path = INSTALL / "env_h3_31731.sh"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(INSTALL) or path.resolve() != path:
        raise RuntimeError(f"Noncanonical/nonprivate C path: {path}")
    return path


def require_host(expected=None):
    actual = dict(hostname=socket.gethostname(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  machine=os.uname().machine)
    if (actual["hostname"] != EXPECTED_HOST or actual["machine"] != "aarch64"
            or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", actual["boot_id"])
            or (expected is not None and expected != actual)):
        raise RuntimeError("C validation host/boot identity mismatch")
    return actual


def verify_file(path, expected=None):
    canonical(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing/nonregular source: {path}")
    sha = digest(path)
    if expected is not None and sha != expected:
        raise RuntimeError(f"Reviewed source hash mismatch: {path}")
    return dict(path=str(path), sha256=sha)


def source_manifest():
    model = VENDOR / "vllm_omni/diffusion/models/minimax_h3"
    result = {f"candidate/{name}": verify_file(model / name, sha) for name, sha in PINNED_C.items()}
    result.update({f"helper/{name}": verify_file(B_CODE / name, sha) for name, sha in PINNED_HELPERS.items()})
    result.update({f"validation/{name}": verify_file(CODE_ROOT / name) for name in OWN_FILES})
    result["upstream/openvdn_npu.py"] = verify_file(B_CODE / "golden/openvdn_npu.py", GOLDEN_SHA)
    for name in ("ulysses.py", "base.py", "factory.py"):
        result[f"strategy/{name}"] = verify_file(VENDOR / "vllm_omni/diffusion/attention/parallel" / name)
    result["runtime/env"] = verify_file(INSTALL / "env_h3_31731.sh")
    return result


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helper(name):
    path = B_CODE / name
    verify_file(path, PINNED_HELPERS[name])
    if name == "npu_wrapper_regression.py":
        # Its pure holder/QKV/compare functions depend on original B profiles.
        # The helper's worker/main are never called or monkey-patched.
        load_helper("profiles.py")
    return load_file("profiles" if name == "profiles.py" else "_c_reused_" + path.stem, path)
