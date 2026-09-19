#!/usr/bin/env bash
set -euo pipefail
: "${H3_ROOT:?}" "${H3_MINIMAL_RUN_ID:?}" "${H3_VLM_SUPERVISOR_PID:?}" "${H3_VLM_LEASE_FDS:?}" "${H3_GROUP_ID:?}"
[[ "$(hostname)" == 'ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0' &&
   "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ && "${PPID}" == "${H3_VLM_SUPERVISOR_PID}" &&
   "${ASCEND_RT_VISIBLE_DEVICES:-}" == '0,1' &&
   "${H3_ROOT}" == /cache/zhonghao/h3/color_trial_v1/vlm_results/shirt_red_couple_124/runs/*"_${H3_MINIMAL_RUN_ID}" &&
   "$(realpath -e -- "${H3_ROOT}")" == "${H3_ROOT}" &&
   -f "${H3_ROOT}/frozen_evidence.json" && ! -L "${H3_ROOT}/frozen_evidence.json" ]] || exit 2
export TMPDIR="/cache/zhonghao/h3/tmp/vlm_${H3_MINIMAL_RUN_ID}"
[[ -d "${TMPDIR}" && "$(realpath -e -- "${TMPDIR}")" == "${TMPDIR}" ]] || exit 2
unset PYTHONPATH VLLM_LOGGING_CONFIG_PATH
source /cache/zhonghao/h3/env_h3_31731.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec /cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/vlm/vlm_worker.py --allow-npu
