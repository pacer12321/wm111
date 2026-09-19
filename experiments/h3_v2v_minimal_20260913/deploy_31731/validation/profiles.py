"""Fixed 31731 tiny-validation profiles; no arbitrary host/path/card CLI."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import socket

INSTALL = Path("/cache/zhonghao/h3")
CODE_ROOT = INSTALL / "validation_code"
VENDOR_ROOT = INSTALL / "candidates/b_v1/vllm-omni"
MODEL_MODULE = VENDOR_ROOT / "vllm_omni/diffusion/models/minimax_h3"
ENV_SCRIPT = INSTALL / "env_h3_31731.sh"
EXPECTED_HOST = "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0"
GROUPS = {"01234567": tuple(range(8))}
PORTS = {"01234567": 29673}
WORLD = 8
GOLDEN_SHA256 = "bb06d4183e79dc5837d19a68ebaf1abdb520098dc50b05b6e22d34e9259cb43c"
CANDIDATE_SHA256 = {
    "minimax_h3_transformer.py": "c1b6b89a1be2a0cac3a2bf3add15ed73d19340086ede91e4a085a218581a0c83",
    "openvdn_npu.py": "4808ef3017343904d8b52cda9cfebc94798f4d609f8ec19e63c59b92abf3381f",
    "openvdn_checkpoint.py": "fbe011bae524bea16f54a14032e61e82f2c68aa4da6d426b98a13e097fb8f19f",
    "pipeline_minimax_h3.py": "de86c6d435254db31e4c094f13d06f71aa87c5fe26d5c344b9c97c610febd690",
}
MIN_HOST_AVAILABLE = 64 * 1024**3
MIN_HBM_AVAILABLE_MIB = 8192


@dataclass(frozen=True)
class Profile:
    group: str
    cards: tuple[int, ...]
    master_port: int
    output_root: Path
    lease_root: Path
    code_root: Path
    vendor_root: Path
    env_script: Path


def profile(group):
    if group not in GROUPS:
        raise ValueError("Only the explicit four-card groups 0123 and 4567 are allowed")
    return Profile(group, GROUPS[group], PORTS[group], INSTALL / "validation" / group,
                   INSTALL / "validation/card_locks", CODE_ROOT, VENDOR_ROOT, ENV_SCRIPT)


def read_host_identity():
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id):
        raise RuntimeError("Host boot ID is unavailable/invalid")
    return {"hostname": socket.gethostname(), "boot_id": boot_id, "machine": os.uname().machine}


def require_host(expected=None):
    actual = read_host_identity()
    if actual["hostname"] != EXPECTED_HOST or actual["machine"] != "aarch64":
        raise RuntimeError(f"This validation is restricted to 31731's expected host: {actual}")
    if expected is not None and actual != expected:
        raise RuntimeError("Host/boot changed after validation was authorized")
    return actual


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_private(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(INSTALL):
        raise RuntimeError(f"Path is outside the private H3 installation: {path}")
    if path.resolve(strict=False) != path:
        raise RuntimeError(f"Symlink/noncanonical private path: {path}")
    return path


def source_manifest(selected):
    model = selected.vendor_root / "vllm_omni/diffusion/models/minimax_h3"
    paths = {"upstream/openvdn_npu.py": selected.code_root / "golden/openvdn_npu.py",
             **{f"candidate/{name}": model / name for name in CANDIDATE_SHA256},
             "strategy/ulysses.py": selected.vendor_root / "vllm_omni/diffusion/attention/parallel/ulysses.py",
             "strategy/base.py": selected.vendor_root / "vllm_omni/diffusion/attention/parallel/base.py",
             "strategy/factory.py": selected.vendor_root / "vllm_omni/diffusion/attention/parallel/factory.py",
             **{f"validation/{name}": selected.code_root / name for name in
                ("profiles.py", "supervision_base.py", "run_validation.py", "launch_validation.sh", "npu_wrapper_regression.py")},
             "runtime/env_h3_31731.sh": selected.env_script}
    manifest = {}
    for label, path in paths.items():
        canonical_private(path)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing/nonregular validation source: {path}")
        manifest[label] = {"path": str(path), "sha256": digest(path)}
    if manifest["upstream/openvdn_npu.py"]["sha256"] != GOLDEN_SHA256:
        raise RuntimeError("Golden OpenVDN reference differs from the reviewed original")
    for name, expected in CANDIDATE_SHA256.items():
        if manifest[f"candidate/{name}"]["sha256"] != expected:
            raise RuntimeError(f"Candidate b_v1 {name} differs from the reviewed source")
    return manifest


def require_cards(group, physical_cards):
    selected = profile(group)
    if tuple(physical_cards) != selected.cards or len(physical_cards) != WORLD:
        raise RuntimeError("Spawned physical-card allocation differs from the selected profile")
    expected = ",".join(map(str, selected.cards))
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != expected:
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES differs from the leased physical group")
    return selected
