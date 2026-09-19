#!/usr/bin/env bash
# Original H3 Ref2VA A only; invoked under run_a_supervised.py's card leases.
set -euo pipefail

: "${H3_ROOT:?The supervisor must set an isolated per-run H3_ROOT}"
: "${H3_OUTPUT:?The supervisor must set an isolated per-run H3_OUTPUT}"
case "${H3_ROOT}" in
  /cache/zhonghao/h3_v2v_minimal_20260913/runs/a_*) ;;
  *) echo "Refusing H3_ROOT outside the isolated A run directory" >&2; exit 2 ;;
esac
case "${H3_OUTPUT}" in
  "${H3_ROOT}"/*) ;;
  *) echo "Refusing output outside this run" >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES=2,3,6,7
export H3_ENV=/cache/yunfeng/envs/minimax-h3-npu
export H3_MODEL=/cache/yunfeng/models/MiniMax-H3
export H3_PORT=19098
# ZMQ Unix socket filenames append a UUID; sockaddr_un allows only 107 bytes.
# Keep this per-run scratch path short, still entirely under zhonghao's cache.
: "${H3_MINIMAL_RUN_ID:?The supervisor must supply a unique run identifier}"
export TMPDIR="/cache/zhonghao/h3tmp/${H3_MINIMAL_RUN_ID}"
export XDG_CACHE_HOME="${H3_ROOT}/cache"
export TORCHINDUCTOR_CACHE_DIR="${H3_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${H3_ROOT}/cache/triton"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${H3_OUTPUT}"

# This script's external overrides prevent env.sh writing under shared H3_ROOT.
source /cache/yunfeng/minimax_h3_npu/scripts/env.sh
export PATH="/cache/yunfeng/minimax_h3_npu/bin:${PATH}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

printf 'A baseline: cards=%s, model=%s, output=%s, port=%s\n' \
  "${ASCEND_RT_VISIBLE_DEVICES}" "${H3_MODEL}/Ref2VA" "${H3_OUTPUT}" "${H3_PORT}"
exec "${H3_ENV}/bin/vllm" serve "${H3_MODEL}/Ref2VA" \
  --omni --host 127.0.0.1 --port "${H3_PORT}" --trust-remote-code \
  --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
  --diffusion-attention-backend FLASH_ATTN
