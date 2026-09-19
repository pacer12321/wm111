#!/usr/bin/env bash
# Full original Ref2VA plus Stage-B OpenVDN, only under run_b_supervised.py leases.
# The supervisor defaults to smoke-only; its explicit smoke-then-formal mode
# permits one matched formal request after the same-service smoke gates pass.
set -euo pipefail

: "${H3_ROOT:?The supervisor must set the isolated B run directory}"
: "${H3_OUTPUT:?The supervisor must set the isolated B output directory}"
: "${H3_MINIMAL_RUN_ID:?The supervisor must set the unique process marker}"
: "${H3_B_SUPERVISOR_PID:?A lease-holding B supervisor is required}"
: "${H3_B_VALIDATION_RESULT:?The version-matched NPU result must be validated}"
if [[ ! "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ || "${PPID}" != "${H3_B_SUPERVISOR_PID}" ]]; then
  echo "Refusing launch without the expected B supervisor/run identifier" >&2
  exit 2
fi
case "${H3_ROOT}" in
  /cache/zhonghao/h3_v2v_minimal_20260913/b_model/runs/b_*) ;;
  *) echo "Refusing H3_ROOT outside the isolated B runs" >&2; exit 2 ;;
esac
if [[ "$(realpath -e -- "${H3_ROOT}")" != "${H3_ROOT}" || \
      "${H3_OUTPUT}" != "${H3_ROOT}/output" || ! -f "${H3_B_VALIDATION_RESULT}" ]]; then
  echo "Refusing noncanonical B paths or missing validation evidence" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=2,3,6,7
export H3_ENV=/cache/yunfeng/envs/minimax-h3-npu
export H3_MODEL=/cache/yunfeng/models/MiniMax-H3
export H3_PORT=19098
export TMPDIR="/cache/zhonghao/h3tmp/${H3_MINIMAL_RUN_ID}"
export XDG_CACHE_HOME="${H3_ROOT}/cache"
export TORCHINDUCTOR_CACHE_DIR="${H3_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${H3_ROOT}/cache/triton"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${H3_OUTPUT}"

# All directories written by the shared env script are overridden beforehand.
source /cache/yunfeng/minimax_h3_npu/scripts/env.sh
export PATH="/cache/yunfeng/minimax_h3_npu/bin:${PATH}"
export PYTHONPATH="/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/b_adaptation/vendor/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
export ZHONGHAO_H3_OPENVDN=1
export ZHONGHAO_H3_OPENVDN_CHECKPOINT=/cache/yunfeng/models/OpenVDN-vdn-minimax-h3/stage-b-step-2000
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

printf 'B Ref2VA supervised service: cards=%s, model=%s, Stage-B=%s, output=%s, port=%s\n' \
  "${ASCEND_RT_VISIBLE_DEVICES}" "${H3_MODEL}/Ref2VA" \
  "${ZHONGHAO_H3_OPENVDN_CHECKPOINT}" "${H3_OUTPUT}" "${H3_PORT}"
# Same A parallelism/offload settings. DiT TP remains the original default 1;
# text encoder TP=4 is separate. No distillation or shortened formal sampler.
exec "${H3_ENV}/bin/vllm" serve "${H3_MODEL}/Ref2VA" \
  --omni --host 127.0.0.1 --port "${H3_PORT}" --trust-remote-code \
  --init-timeout 2400 --stage-init-timeout 2400 \
  --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
  --diffusion-attention-backend FLASH_ATTN
