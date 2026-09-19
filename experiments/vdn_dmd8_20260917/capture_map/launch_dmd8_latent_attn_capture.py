from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/cache/zhonghao/h3")
RUN = ROOT / "dmd8_b_skip_20260917"
CANDIDATE = RUN / "candidate_B_skip_attn_capture"
LOADER = RUN / "loader"
EXPERIMENT = RUN / "results/B_DMD8_latent_attn_capture_all8"
CHECKPOINT = ROOT / "models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"
SELECTOR = RUN / "selector_dmd8_latent/fixed_selector_payload.pt"
CAPTURE = EXPERIMENT / "qk_capture"

os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=str(EXPERIMENT),
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=str(CANDIDATE),
    ZHONGHAO_H3_REQUEST_STEPS="9",
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD=str(SELECTOR),
    ZHONGHAO_H3_INTERLEAVED_SP="1",
    ZHONGHAO_H3_SPOTEDIT_INITIAL_STEPS="1",
    ZHONGHAO_H3_SPOTEDIT_RESET_STEPS="4",
    ZHONGHAO_H3_SS_ORACLE_CAPTURE_DIR=str(CAPTURE),
    ZHONGHAO_H3_SS_ORACLE_CAPTURE_STEPS="0,1,2,3,4,5,6,7",
    ZHONGHAO_H3_SS_ORACLE_CAPTURE_LAYERS="8,24,41",
    ZHONGHAO_H3_EXPERIMENT_LABEL=(
        "DMD8 latent-skip attention capture: source queries, layers 8/24/41, all 8 forwards"
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
        value["ZHONGHAO_H3_SS_ORACLE_CAPTURE_DIR"] = str(CAPTURE)
        value["ZHONGHAO_H3_SS_ORACLE_CAPTURE_STEPS"] = "0,1,2,3,4,5,6,7"
        value["ZHONGHAO_H3_SS_ORACLE_CAPTURE_LAYERS"] = "8,24,41"
        return value

    runner.q.env_for = env_for


runner.prepared.configure = configure
runner.main()

result_path = EXPERIMENT / "formal_9step/result.json"
result = json.loads(result_path.read_text())
if result.get("actual_dit_forwards") != 8:
    raise RuntimeError(f"DMD8 capture forward-count gate failed: {result.get('actual_dit_forwards')}")
result.update(
    formal_speed_comparison=False,
    checkpoint=str(CHECKPOINT),
    checkpoint_stage="dmd",
    checkpoint_step=250,
    adapters=["default", "turbo"],
    turbo_num_steps=8,
    token_skip=True,
    interleaved_sp=True,
    selector=str(SELECTOR),
    capture_layers=[8, 24, 41],
    capture_steps=list(range(8)),
    capture_root=str(CAPTURE),
)
result_path.write_text(json.dumps(result, indent=2))
print(json.dumps({"status": "DMD8_ATTN_CAPTURE_COMPLETE", "result": str(result_path)}), flush=True)
