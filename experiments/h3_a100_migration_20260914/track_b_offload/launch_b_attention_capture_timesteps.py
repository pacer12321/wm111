"""Launch the audited B/VDN multilayer dense-T-S teacher capture."""

from __future__ import annotations

import os


os.environ.update(
    ZHONGHAO_H3_B_CAPTURE_TRANSFORMER_SHA256=os.environ[
        "ZHONGHAO_H3_B_CAPTURE_TRANSFORMER_SHA256"
    ],
    ZHONGHAO_H3_B_CAPTURE_PIPELINE_SHA256=os.environ[
        "ZHONGHAO_H3_B_CAPTURE_PIPELINE_SHA256"
    ],
)

import run_b_attention_capture


run_b_attention_capture.main()
