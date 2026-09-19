"""Create one isolated prepared-offload A/base candidate for attention capture."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys


ROOT = Path("/cache/zhonghao/h3")
SOURCE = ROOT / "a100_v1/candidates/A"
TARGET = ROOT / "attention_map_base_dense_code_20260916_v1/candidate_A_capture"
REL = Path("vllm_omni/diffusion/models/minimax_h3")
sys.path.insert(0, str(ROOT / "track_b_validation_20260915"))
from generate_pipeline_patch import BOOTSTRAP


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if TARGET.parent.exists():
    raise RuntimeError(f"isolated destination already exists: {TARGET.parent}")
TARGET.parent.mkdir(parents=True)
shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns("__pycache__", ".git"))
pipeline = TARGET / REL / "pipeline_minimax_h3.py"
source = pipeline.read_text()
if "_h3_prepared_install" in source:
    raise RuntimeError("source A pipeline unexpectedly already contains prepared bootstrap")
pipeline.write_text(source.rstrip() + "\n" + BOOTSTRAP)
print(f"TARGET={TARGET}")
print(f"PIPELINE_SHA256={sha(pipeline)}")
