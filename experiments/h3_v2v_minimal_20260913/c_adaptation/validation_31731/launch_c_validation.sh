#!/usr/bin/env bash
set -euo pipefail
: "${H3_ROOT:?}" "${H3_OUTPUT:?}" "${H3_MINIMAL_RUN_ID:?}" "${H3_C_SUPERVISOR_PID:?}" "${H3_GROUP_ID:?}"
if [[ "$(hostname)" != "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0" ||
      ! "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ || "${PPID}" != "${H3_C_SUPERVISOR_PID}" ||
      "${ASCEND_RT_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" || "${H3_ROOT}" != "${H3_OUTPUT}" ||
      "${H3_ROOT}" != /cache/zhonghao/h3/c_validation/01234567/runs/*"_${H3_MINIMAL_RUN_ID}" ||
      "$(realpath -e -- "${H3_ROOT}")" != "${H3_ROOT}" ]]; then
  echo "Wrong host/allocation or no matching C lease-holding supervisor" >&2; exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="/cache/zhonghao/h3/tmp/c_${H3_MINIMAL_RUN_ID}"
source /cache/zhonghao/h3/env_h3_31731.sh
export PYTHONPATH="/cache/zhonghao/h3/candidates/c_v1/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec /cache/zhonghao/h3/env/bin/python /cache/zhonghao/h3/c_validation_code/c_npu_regression.py \
  --allow-npu --run-id "${H3_MINIMAL_RUN_ID}" --output "${H3_ROOT}/c_tiny.json"
