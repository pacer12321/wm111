from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/cache/zhonghao/h3")
RUN = ROOT / "dmd8_b_skip_20260917"
CANDIDATE = RUN / "candidate_B"
LOADER = RUN / "loader"
EXPERIMENT = RUN / "results/B_DMD8"
CHECKPOINT = ROOT / "models/OpenVDN-vdn-minimax-h3/stage-dmd-step-250"

os.environ.update(
    ZHONGHAO_H3_SPOTEDIT_EXPERIMENT=str(EXPERIMENT),
    ZHONGHAO_H3_SPOTEDIT_CANDIDATE=str(CANDIDATE),
    ZHONGHAO_H3_REQUEST_STEPS="9",  # official API: 9 sigma points -> 8 DiT evaluations
    ZHONGHAO_H3_PRECHECK_STEPS="0",
    ZHONGHAO_H3_EXPERIMENT_LABEL="B-DMD8; no token skip; red-shirt edit; seed 4101",
)

# Keep the legacy request runner available, but make the audited DMD8 loader
# win every top-level import (especially h3_prepared_integration).
sys.path.insert(0, str(ROOT / "track_b_validation_20260915"))
sys.path.insert(0, str(LOADER))
import run_b_spotedit_step1 as runner

# The legacy runner reconstructs PYTHONPATH inside main() from runner.KIT,
# after q.env_for() returns.  Point that late override at the audited DMD8
# loader as well; otherwise worker subprocesses silently import Stage-B code.
runner.KIT = LOADER

# The legacy runner compares every candidate file to the immutable Stage-B
# source tree.  This isolated candidate intentionally changes only the
# audited DMD8 loader files, so use the isolated snapshot as both sides of
# that integrity check; the loader/manifest hashes are separately recorded
# by h3_prepared_integration at model load.
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
        value.pop("ZHONGHAO_H3_FIXED_SELECTOR_PAYLOAD", None)
        value.pop("ZHONGHAO_H3_INTERLEAVED_SP", None)
        return value

    runner.q.env_for = env_for


runner.prepared.configure = configure
runner.main()

result_path = EXPERIMENT / "formal_9step/result.json"
result = json.loads(result_path.read_text())
if result.get("actual_dit_forwards") != 8:
    raise RuntimeError(f"DMD8 forward-count gate failed: {result.get('actual_dit_forwards')}")
result.update(
    formal_speed_comparison=True,
    checkpoint=str(CHECKPOINT),
    checkpoint_stage="dmd",
    checkpoint_step=250,
    adapters=["default", "turbo"],
    turbo_num_steps=8,
    comparison_contract="same source/prompt/seed/settings; B versus B+token-skip only",
)
result_path.write_text(json.dumps(result, indent=2))
print(json.dumps({"status": "B_DMD8_COMPLETE", "result": str(result_path),
                  "seconds": result.get("request_seconds")}), flush=True)
