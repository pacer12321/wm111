"""Capture three Ref2VA depths at five denoising timesteps for Oracle routing."""

from __future__ import annotations

import os


ROOT = "/cache/zhonghao/h3"
os.environ.update(
    ZHONGHAO_H3_BASE_CAPTURE_EXPERIMENT=(
        f"{ROOT}/attention_oracle_multilayer_redshirt_20260916_v2"
    ),
    ZHONGHAO_H3_BASE_CAPTURE_STEPS="0,12,24,36,48",
    ZHONGHAO_H3_ATTENTION_CAPTURE_LAYERS="8,24,41",
    ZHONGHAO_H3_BASE_CAPTURE_TRANSFORMER_SHA256=(
        "a6ff0b8f70ebda9469fb00b201da27b748ac14bcb88506c09222ec2b351b294b"
    ),
    ZHONGHAO_H3_BASE_CAPTURE_PIPELINE_SHA256=(
        "64f49496f11531453cc71a16ff8e4589d4ebe462d872c7c9a23d13bb5eddc404"
    ),
)

import run_a_attention_capture


run_a_attention_capture.main()
