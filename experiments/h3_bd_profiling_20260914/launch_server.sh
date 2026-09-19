#!/usr/bin/env bash
set -euo pipefail
: "${H3_MINIMAL_RUN_ID:?}" "${H3_PROFILE_CASE:?}" "${H3_PROFILE_OUTPUT:?}"
: "${H3_PROFILE_SUPERVISOR:?}" "${H3_PROFILE_VENDOR:?}"
[[ "${PPID}" == "${H3_PROFILE_SUPERVISOR}" && "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ ]] || exit 2
[[ "$(hostname)" == ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0 ]] || exit 2
case "${H3_PROFILE_CASE}" in B|D) ;; *) exit 2 ;; esac
[[ "${H3_PROFILE_VENDOR}" == "/cache/zhonghao/h3/profiling_bd_v1/candidates/${H3_PROFILE_CASE}/vllm-omni" ]] || exit 2
unset PYTHONPATH
source /cache/zhonghao/h3/env_h3_31731.sh
export PYTHONPATH="/cache/zhonghao/h3/profiling_bd_v1:${H3_PROFILE_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ZHONGHAO_H3_OPENVDN=1
export ZHONGHAO_H3_OPENVDN_CHECKPOINT=/cache/zhonghao/h3/models/OpenVDN-vdn-minimax-h3/stage-b-step-2000
export D_REVIEWED_POLICY_PATH=/cache/zhonghao/h3/dualstream_v1/deploy_d/reviewed_policy.json
export D_REVIEWED_POLICY_SHA256=a412150f34a5f3c564e908e5ba85dcebe32ca641b2d139ce6963cbfc5d3949cd
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400
# ZMQ appends a UUID to TMPDIR; UNIX socket paths must stay under 107 bytes.
export TMPDIR="/cache/zhonghao/h3/tmp/p_${H3_MINIMAL_RUN_ID}_${H3_PROFILE_CASE}"
[[ ! -e "${TMPDIR}" ]] || exit 2
mkdir "${TMPDIR}"
exec "${H3_ENV}/bin/vllm" serve "${H3_MODEL}/Ref2VA" \
  --omni --host 127.0.0.1 --port 19111 --trust-remote-code \
  --init-timeout 2400 --stage-init-timeout 2400 \
  --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
