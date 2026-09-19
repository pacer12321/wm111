from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


ROOT = Path("/cache/zhonghao/h3")
CODE = ROOT / "interleaved_sp_code_20260916_v1"
CANDIDATE = CODE / "candidate"
TRANSFORMER = (
    CANDIDATE
    / "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("contiguous", "contiguous_repeat", "interleaved"),
    )
    args = parser.parse_args()

    experiment = ROOT / f"interleaved_sp_equivalence_20260916_v1_{args.mode}"
    os.environ.update(
        ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=str(experiment),
        ZHONGHAO_H3_SPOTEDIT_CANDIDATE=str(CANDIDATE),
        ZHONGHAO_H3_TRANSFORMER_SHA256=sha256(TRANSFORMER),
        ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=str(
            ROOT
            / "spotedit_selector_capture_20260915_v1/vae_perceptual_v1/"
            "fixed_selector_payload.pt"
        ),
        ZHONGHAO_H3_REQUEST_STEPS="2",
        ZHONGHAO_H3_PRECHECK_STEPS="0",
        ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="4",
        ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="13,22,31",
        ZHONGHAO_H3_ENABLE_SOURCE_HYBRID="0",
        ZHONGHAO_H3_INTERLEAVED_SP="1" if args.mode == "interleaved" else "0",
        ZHONGHAO_H3_EQ_CAPTURE_DIR=str(experiment / "attention_capture"),
        ZHONGHAO_H3_EQ_CAPTURE_BLOCK="0",
        ZHONGHAO_H3_EXPERIMENT_LABEL=(
            f"Block-0 attention numerical equivalence capture ({args.mode})"
        ),
    )
    sys.path.insert(0, str(ROOT / "track_b_validation_20260915"))
    import run_b_spotedit_step1 as runner

    runner.q.UUIDS = [
        "GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670",
        "GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3",
    ]
    runner.main()


if __name__ == "__main__":
    main()
