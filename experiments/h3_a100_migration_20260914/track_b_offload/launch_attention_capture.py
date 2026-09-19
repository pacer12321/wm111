"""Capture one real post-RoPE OpenVDN Q/K diagnostic without changing output."""

from __future__ import annotations

import os
import sys


ROOT = "/cache/zhonghao/h3"
os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=f"{ROOT}/attention_map_midlayer_redshirt_20260915_v3",
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=(
        f"{ROOT}/attention_map_code_20260915_v1/candidate_B_attn_capture"
    ),
    ZHONGHAO_H3_TRANSFORMER_SHA256=(
        "00f91011ddafea9bd9932733f5caa2be321d004903215f150074f9aa3151bce4"
    ),
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=(
        f"{ROOT}/spotedit_selector_capture_20260915_v1/vae_perceptual_v1/"
        "fixed_selector_payload.pt"
    ),
    ZHONGHAO_H3_REQUEST_STEPS="2",
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="4",
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="13,22,31",
    ZHONGHAO_H3_ENABLE_SOURCE_HYBRID="0",
    ZHONGHAO_H3_ATTENTION_CAPTURE_DIR=(
        f"{ROOT}/attention_map_midlayer_redshirt_20260915_v3/qk_capture"
    ),
    ZHONGHAO_H3_ATTENTION_CAPTURE_LAYERS="24",
    ZHONGHAO_H3_ATTENTION_CAPTURE_STEP="0",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "One-forward read-only post-RoPE attention Q/K capture at DiT layer 24"
    ),
)
sys.path.insert(0, f"{ROOT}/track_b_validation_20260915")

import run_b_spotedit_step1 as runner

runner.q.UUIDS = [
    "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
    "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
]
runner.main()
