from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/cache/zhonghao/h3")
RUN = ROOT / "dmd8_b_skip_20260917"
CANDIDATE = RUN / "candidate_B_skip"
LOADER = RUN / "loader"
EXPERIMENT = RUN / "results/B_DMD8_interleaved_full_compute"
CHECKPOINT = ROOT / "models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
SELECTOR = RUN / "selector_dmd8_all_active/fixed_selector_payload.pt"

os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=str(EXPERIMENT),
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=str(CANDIDATE),
    ZHONGHAO_H3_REQUEST_STEPS="9",
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=str(SELECTOR),
    ZHONGHAO_H3_INTERLEAVED_SP="1",
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="999",
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "B-DMD8 interleaved full-compute control; all active; seed 4101"
    ),
)

sys.path.insert(0, str(ROOT / "track_b_validation_20260915"))
sys.path.insert(0, str(LOADER))
import run_b_spotedit_step1 as runner

runner.KIT = LOADER
runner.SOURCE_CANDIDATE = CANDIDATE
original_read = runner.q.read


def read_with_isolated_hashes(path):
    value = original_read(path)
    if Path(path).name == "cuda_manifest.json" and "cases" in value:
        value = json.loads(json.dumps(value))
        for filename in value["cases"]["B"]["model_files"]:
            candidate_file = CANDIDATE / runner.q.REL / filename
            value["cases"]["B"]["model_files"][filename] = runner.q.sha(candidate_file)
    return value


runner.q.read = read_with_isolated_hashes
original_configure = runner.prepared.configure


def configure(phase):
    original_configure(phase)
    inherited_env_for = runner.q.env_for

    def env_for(case):
        value = inherited_env_for(case)
        value["PYTHONPATH"] = f"{LOADER}:{CANDIDATE}"
        value["ZHONGHAO_H3_OPENVDN"] = "1"
        value["ZHONGHAO_H3_OPENVDN_CHECKPOINT"] = str(CHECKPOINT)
        value["ZHONGHAO_H3_PREPARED_OFFLOAD"] = "1"
        value["ZHONGHAO_H3_PREPARED_MANIFEST"] = str(LOADER / "manifest.json")
        value["ZHONGHAO_H3_INTERLEAVED_SP"] = "1"
        return value

    runner.q.env_for = env_for


runner.prepared.configure = configure
runner.main()

result_path = EXPERIMENT / "formal_9step/result.json"
result = json.loads(result_path.read_text())
if result.get("actual_dit_forwards") != 8:
    raise RuntimeError(f"DMD8 interleaved control forward-count gate failed: {result.get('actual_dit_forwards')}")
result.update(
    formal_speed_comparison=True,
    checkpoint=str(CHECKPOINT),
    checkpoint_stage="dmd",
    checkpoint_step=250,
    adapters=["default", "turbo"],
    turbo_num_steps=8,
    token_skip=False,
    interleaved_sp=True,
    selector=str(SELECTOR),
    active_ratio=1.0,
    refresh_forward_numbers=list(range(1, 9)),
    comparison_contract="same interleaved token layout as skip variants; all target tokens active",
)
result_path.write_text(json.dumps(result, indent=2))
print(json.dumps({"status": "B_DMD8_INTERLEAVED_CONTROL_COMPLETE", "seconds": result.get("request_seconds")}), flush=True)
