#!/usr/bin/env bash
# Invoked only by run_npu_validation_supervised.py while all 2367 leases are held.
# No video service, checkpoint download/loading, training, or shared-file writes.
set -euo pipefail

: "${H3_ROOT:?The supervisor must set an isolated validation run directory}"
: "${H3_OUTPUT:?The supervisor must set this run's output directory}"
: "${H3_MINIMAL_RUN_ID:?The supervisor must provide its unique run identifier}"
: "${H3_B_VALIDATION_SUPERVISOR_PID:?This launcher requires its lease-holding supervisor}"
: "${H3_B_VALIDATION_RESULT:?The supervisor must provide this run's result path}"
: "${H3_B_VALIDATION_MASTER_PORT:?The supervisor must set the independent test port}"
: "${H3_B_VALIDATION_COLLECTIVE_TIMEOUT:?The supervisor must set the collective timeout}"

if [[ ! "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ ]]; then
  echo "Invalid per-run identifier" >&2
  exit 2
fi
case "${H3_ROOT}" in
  /cache/zhonghao/h3_v2v_minimal_20260913/b_validation/runs/*) ;;
  *) echo "Refusing H3_ROOT outside the isolated B validation runs" >&2; exit 2 ;;
esac
if [[ "$(realpath -e -- "${H3_ROOT}")" != "${H3_ROOT}" || \
      "${H3_ROOT##*/}" != *"_${H3_MINIMAL_RUN_ID}" || \
      "${H3_OUTPUT}" != "${H3_ROOT}" || \
      "${H3_B_VALIDATION_RESULT}" != "${H3_ROOT}/npu_wrapper.json" ]]; then
  echo "Refusing noncanonical or mismatched validation output paths" >&2
  exit 2
fi
if [[ "${PPID}" != "${H3_B_VALIDATION_SUPERVISOR_PID}" || \
      "${H3_B_VALIDATION_MASTER_PORT}" != "29673" || \
      "${H3_B_VALIDATION_COLLECTIVE_TIMEOUT}" != "180" ]]; then
  echo "Refusing launch without the expected supervisor or fixed test parameters" >&2
  exit 2
fi
if [[ -e "${H3_B_VALIDATION_RESULT}" ]]; then
  echo "Refusing to overwrite an existing validation result" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=2,3,6,7
export H3_ENV=/cache/yunfeng/envs/minimax-h3-npu
export H3_MODEL=/cache/yunfeng/models/MiniMax-H3
# Short, private scratch directory avoids Unix socket path-length failures.
export TMPDIR="/cache/zhonghao/h3tmp/bv_${H3_MINIMAL_RUN_ID}"
export XDG_CACHE_HOME="${H3_ROOT}/cache"
export TORCHINDUCTOR_CACHE_DIR="${H3_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${H3_ROOT}/cache/triton"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${H3_OUTPUT}"

# H3_ROOT/H3_OUTPUT are already isolated before sourcing this shared file.
source /cache/yunfeng/minimax_h3_npu/scripts/env.sh
export PYTHONPATH=/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/b_adaptation/vendor/vllm-omni
export PATH="/cache/yunfeng/minimax_h3_npu/bin:${PATH}"

printf 'B tiny NPU validation: cards=%s, port=%s, result=%s\n' \
  "${ASCEND_RT_VISIBLE_DEVICES}" "${H3_B_VALIDATION_MASTER_PORT}" "${H3_B_VALIDATION_RESULT}"
exec "${H3_ENV}/bin/python" \
  /home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/b_adaptation/tests/npu_wrapper_regression.py \
  --allow-npu \
  --base /home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/b_adaptation \
  --master-port 29673 --collective-timeout 180 \
  --output "${H3_B_VALIDATION_RESULT}"
