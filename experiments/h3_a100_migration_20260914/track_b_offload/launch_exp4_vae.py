"""One-shot launcher for experiment 4 with the VAE-perceptual selector."""

from __future__ import annotations

import os
import sys


ROOT = "/cache/zhonghao/h3"
os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=f"{ROOT}/ablation4_vae_perceptual_redshirt_20260915_v1",
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=(
        f"{ROOT}/spotedit_partial_query_code_20260915_v1/candidate_B_spot_partial"
    ),
    ZHONGHAO_H3_TRANSFORMER_SHA256=(
        "4f60d5f881cfe7814ff56aa5939a52e8948bf5c06971ca97d756eb35e1258d26"
    ),
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=(
        f"{ROOT}/spotedit_selector_capture_20260915_v1/vae_perceptual_v1/"
        "fixed_selector_payload.pt"
    ),
    ZHONGHAO_H3_REQUEST_STEPS="50",
    ZHONGHAO_H3_PRECHECK_STEPS="6",
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="4",
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="13,22,31",
    ZHONGHAO_H3_ENABLE_SOURCE_HYBRID="0",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "Experiment 4 VAE-perceptual selector; B attention; dense S-S; "
        "partial-query; fusion; warmup; reset"
    ),
)
sys.path.insert(0, f"{ROOT}/track_b_validation_20260915")

import run_b_spotedit_step1 as runner

runner.q.UUIDS = [
    "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
    "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
]
runner.main()
