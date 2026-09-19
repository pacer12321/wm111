"""Run the strict dog temporal-reordering control with B and no token skip."""

from __future__ import annotations

import os
import sys


ROOT = "/cache/zhonghao/h3"
os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=f"{ROOT}/temporal_dog_B_noskip_20260917_v1",
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=f"{ROOT}/bd_prepared_20260915_v2/candidates/B",
    ZHONGHAO_H3_SAMPLE_MANIFEST=(
        f"{ROOT}/data/generalization_candidates_20260917/temporal_reorder_dog/"
        "manifest_strict.json"
    ),
    ZHONGHAO_H3_REQUEST_STEPS="50",
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "Strict temporal-reordering control: B attention, no token skip, dense S-S"
    ),
)
sys.path.insert(0, f"{ROOT}/track_b_validation_20260915")

import run_b_spotedit_step1 as runner


runner.q.UUIDS = [
    "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
    "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
]
runner.main()
