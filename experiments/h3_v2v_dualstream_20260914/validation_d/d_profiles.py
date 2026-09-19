"""Fixed D synthetic validation, with explicit pinned new sources and old safety helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys

INSTALL = Path("/cache/zhonghao/h3")
CODE_ROOT = INSTALL / "dualstream_v1/validation_d"
B_CODE = INSTALL / "validation_code"
VENDOR = INSTALL / "dualstream_v1/candidate/vllm-omni"
EXPECTED_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
WORLD, CARDS, PORT = 8, tuple(range(8)), 29685
MIN_HOST_AVAILABLE, MIN_HBM_MIB = 64 * 1024**3, 8192
CASE = "D_tiny_SP8"
MODE = "D_source_hybrid_same_frame_v1"
PINNED_HELPERS = {
    "profiles.py": "f7ff6a1bfe7d4a91f628e861c435619f731338bf6b36ff81ae780ee0d5511269",
    "supervision_base.py": "5889609771d9f80565e52f3d6c0bbb239d4a555c28cc8ffec0b65c8abb4cb344",
    "npu_wrapper_regression.py": "7679909766ca09168e29ad7a9a5943571ab2acec7838c9c5096f158105a0c698",
}
CANDIDATE_FILES = ("openvdn_npu.py", "openvdn_checkpoint.py", "minimax_h3_transformer.py",
                   "pipeline_minimax_h3.py", "strict_source_layout.py", "strict_source_attention.py",
                   "dual_stream_attention.py", "dual_stream_adapter.py")
GOLDEN_SHA = "bb06d4183e79dc5837d19a68ebaf1abdb520098dc50b05b6e22d34e9259cb43c"
OWN_FILES = ("d_profiles.py", "d_cases.py", "d_cpu_regression.py", "d_npu_regression.py",
             "run_d_validation.py", "launch_d_validation.sh", "d_validation_verifier.py")


@dataclass(frozen=True)
class Profile:
    group: str = "01234567"
    cards: tuple = CARDS
    master_port: int = PORT
    code_root: Path = CODE_ROOT
    vendor_root: Path = VENDOR
    output_root: Path = INSTALL / "dualstream_v1/validation_results/01234567"
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


def validation_pins():
    path = CODE_ROOT / "pins.json"
    expected = os.environ.get("D_VALIDATION_PINS_SHA256", "")
    if len(expected) != 64:
        raise RuntimeError("D validation requires an explicitly pinned pins.json digest")
    verify_file(path, expected)
    pins = json.loads(path.read_text())
    if pins.get("schema_version") != 1 or pins.get("case") != CASE or pins.get("mode") != MODE:
        raise RuntimeError("D validation pin identity mismatch")
    if set(pins.get("candidate", {})) != set(CANDIDATE_FILES) or set(pins.get("validation", {})) != set(OWN_FILES):
        raise RuntimeError("D validation pin coverage incomplete")
    for section in ("candidate", "validation"):
        if any(not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None for sha in pins[section].values()):
            raise RuntimeError("Malformed D validation source digest")
    return pins


def source_manifest():
    model = VENDOR / "vllm_omni/diffusion/models/minimax_h3"
    result = {f"candidate/{name}": verify_file(model / name, sha) for name, sha in validation_pins()["candidate"].items()}
    result.update({f"helper/{name}": verify_file(B_CODE / name, sha) for name, sha in PINNED_HELPERS.items()})
    result.update({f"validation/{name}": verify_file(CODE_ROOT / name, validation_pins()["validation"][name]) for name in OWN_FILES})
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
    return load_file("profiles" if name == "profiles.py" else "_d_reused_" + path.stem, path)
