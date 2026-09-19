#!/usr/bin/env bash
# Private 31731 tiny test only. Requires its own lease-holding supervisor.
set -euo pipefail
: "${H3_ROOT:?}" "${H3_OUTPUT:?}" "${H3_MINIMAL_RUN_ID:?}"
: "${H3_VALIDATION_GROUP:?}" "${H3_VALIDATION_SUPERVISOR_PID:?}"
: "${H3_VALIDATION_MASTER_PORT:?}" "${H3_VALIDATION_RESULT:?}" "${H3_GROUP_ID:?}"
if [[ "$(hostname)" != "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0" ||
      ! "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ || "${PPID}" != "${H3_VALIDATION_SUPERVISOR_PID}" ]]; then
  echo "Wrong host or missing lease-holding supervisor identity" >&2; exit 2
fi
case "${H3_VALIDATION_GROUP}" in
  01234567) task_cards=0,1,2,3,4,5,6,7; task_port=29673 ;;
  *) echo "Unknown fixed four-card group" >&2; exit 2 ;;
esac
task_prefix="/cache/zhonghao/h3/validation/${H3_VALIDATION_GROUP}/runs/"
if [[ "${ASCEND_RT_VISIBLE_DEVICES:-}" != "${task_cards}" || "${H3_VALIDATION_MASTER_PORT}" != "${task_port}" ||
      "${H3_ROOT}" != "${task_prefix}"* || "$(realpath -e -- "${H3_ROOT}")" != "${H3_ROOT}" ||
      "${H3_ROOT##*/}" != *"_${H3_MINIMAL_RUN_ID}" || "${H3_OUTPUT}" != "${H3_ROOT}" ||
      "${H3_VALIDATION_RESULT}" != "${H3_ROOT}/npu_wrapper.json" || -e "${H3_VALIDATION_RESULT}" ||
      ! -f "${H3_ROOT}/host_identity.json" || ! -f "${H3_ROOT}/source_manifest.json" ]]; then
  echo "Noncanonical/mismatched card, port, run output or proof" >&2; exit 2
fi
export TMPDIR="/cache/zhonghao/h3/tmp/v_${H3_MINIMAL_RUN_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
source /cache/zhonghao/h3/env_h3_31731.sh
# Source the runtime FIRST; prepend the candidate AFTER its original-src paths.
export PYTHONPATH="/cache/zhonghao/h3/candidates/b_v1/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${task_cards}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
printf '31731 B tiny group=%s cards=%s port=%s run=%s\n' "${H3_VALIDATION_GROUP}" "${task_cards}" "${task_port}" "${H3_MINIMAL_RUN_ID}"
exec /cache/zhonghao/h3/env/bin/python /cache/zhonghao/h3/validation_code/npu_wrapper_regression.py \
  --allow-npu --group "${H3_VALIDATION_GROUP}" --physical-cards "${task_cards}" \
  --run-id "${H3_MINIMAL_RUN_ID}" --host-proof "${H3_ROOT}/host_identity.json" \
  --manifest-proof "${H3_ROOT}/source_manifest.json" \
  --master-port "${task_port}" --collective-timeout 180 --output "${H3_VALIDATION_RESULT}"
