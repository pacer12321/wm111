"""Launch a 12-schedule-point Experiment-4 block-level CUDA profile."""

from __future__ import annotations

import os
import sys


ROOT = "/cache/zhonghao/h3"
EXPERIMENT = f"{ROOT}/exp4_block_profile_20260916_v3"
os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=EXPERIMENT,
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=f"{ROOT}/exp4_block_profile_code_20260916_v1/candidate",
    ZHONGHAO_H3_TRANSFORMER_SHA256=(
        "3327b0dfb686addb5bd8665a20b1c92bb2fa684e50ca70737ea55413ce62b438"
    ),
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=(
        f"{ROOT}/spotedit_selector_capture_20260915_v1/vae_perceptual_v1/"
        "fixed_selector_payload.pt"
    ),
    ZHONGHAO_H3_REQUEST_STEPS="12",
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="4",
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="13,22,31",
    ZHONGHAO_H3_ENABLE_SOURCE_HYBRID="0",
    ZHONGHAO_H3_BLOCK_PROFILE="1",
    ZHONGHAO_H3_BLOCK_PROFILE_DIR=f"{EXPERIMENT}/block_profile",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "Experiment 4 block-level CUDA profile; 4 refresh forwards followed by "
        "7 partial-query skip forwards"
    ),
)
sys.path.insert(0, f"{ROOT}/track_b_validation_20260915")

import run_b_spotedit_step1 as runner

runner.q.UUIDS = [
    "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
    "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
]
runner.main()
