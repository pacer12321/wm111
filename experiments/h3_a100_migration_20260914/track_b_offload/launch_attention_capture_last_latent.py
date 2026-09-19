"""Capture the final denoising-step Q/K map for experiment 4 latent selector."""

from __future__ import annotations

import os
import sys


ROOT = "/cache/zhonghao/h3"
os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=(
        f"{ROOT}/attention_map_last_latent_redshirt_20260915_v1"
    ),
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=(
        f"{ROOT}/attention_map_last_latent_code_20260915_v1/"
        "candidate_B_attn_capture"
    ),
    ZHONGHAO_H3_TRANSFORMER_SHA256=(
        "a3b3e23ba9fca3036a91682fd972b9eb93ed225a2cf495a35ff36e15c27a0881"
    ),
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=(
        f"{ROOT}/spotedit_selector_capture_20260915_v1/latent_analysis_v2/"
        "fixed_selector_payload.pt"
    ),
    ZHONGHAO_H3_REQUEST_STEPS="50",
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="4",
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="13,22,31",
    ZHONGHAO_H3_ENABLE_SOURCE_HYBRID="0",
    ZHONGHAO_H3_ATTENTION_CAPTURE_DIR=(
        f"{ROOT}/attention_map_last_latent_redshirt_20260915_v1/qk_capture"
    ),
    ZHONGHAO_H3_ATTENTION_CAPTURE_LAYERS="24",
    # A 50-point schedule performs 49 DiT forwards indexed 0..48.
    ZHONGHAO_H3_ATTENTION_CAPTURE_STEP="48",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "Experiment 4 latent selector final DiT forward attention capture; "
        "B attention; dense S-S; partial-query; fusion; warmup; reset"
    ),
)
sys.path.insert(0, f"{ROOT}/track_b_validation_20260915")

import run_b_spotedit_step1 as runner

runner.q.UUIDS = [
    "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
    "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
]
runner.main()
