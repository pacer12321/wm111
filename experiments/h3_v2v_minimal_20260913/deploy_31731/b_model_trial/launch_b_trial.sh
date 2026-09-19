#!/usr/bin/env bash
# B only: complete released Stage-B branch/LoRA, original linear + global source.
# Direct invocation is refused; supervisor holds all eight shared card leases.
set -euo pipefail
: "${H3_ROOT:?}" "${H3_OUTPUT:?}" "${H3_MINIMAL_RUN_ID:?}" "${H3_GROUP_ID:?}"
: "${H3_TRIAL_GROUP:?}" "${H3_TRIAL_SAMPLE:?}" "${H3_TRIAL_SUPERVISOR_PID:?}" "${H3_PORT:?}"
if [[ "$(hostname)" != "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0" ||
      ! "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ || "${PPID}" != "${H3_TRIAL_SUPERVISOR_PID}" ]]; then
  echo 'Wrong host or missing lease-holding B supervisor' >&2; exit 2
fi
case "${H3_TRIAL_GROUP}" in
  01234567) task_cards=0,1,2,3,4,5,6,7; task_port=19099 ;;
  *) echo 'Only the fixed eight-card group 01234567 is allowed' >&2; exit 2 ;;
esac
case "${H3_TRIAL_SAMPLE}" in
  lake_snow|explicit_reverse_couple_124) ;;
  *) echo 'Only the frozen lake_snow/reverse profiles are allowed' >&2; exit 2 ;;
esac
task_prefix="/cache/zhonghao/h3/b_model_trial/${H3_TRIAL_GROUP}/${H3_TRIAL_SAMPLE}/B/runs/b_"
if [[ "${ASCEND_RT_VISIBLE_DEVICES:-}" != "${task_cards}" || "${H3_PORT}" != "${task_port}" ||
      "${H3_ROOT}" != "${task_prefix}"* || "${H3_ROOT##*/}" != *"_${H3_MINIMAL_RUN_ID}" ||
      "$(realpath -e -- "${H3_ROOT}")" != "${H3_ROOT}" || "${H3_OUTPUT}" != "${H3_ROOT}/output" ||
      "${H3_GROUP_ID}" != "zhonghao_31731_B_${H3_TRIAL_GROUP}_${H3_MINIMAL_RUN_ID}" ||
      ! -f "${H3_ROOT}/frozen_evidence.json" || ! -f "${H3_ROOT}/host_identity.json" ||
      ! -f "${H3_ROOT}/formal_logging.json" || -L "${H3_ROOT}/formal_logging.json" ||
      "${VLLM_LOGGING_CONFIG_PATH:-}" != "${H3_ROOT}/formal_logging.json" ]]; then
  echo 'Mismatched/noncanonical B group, card, port, output, run or proof' >&2; exit 2
fi
export TMPDIR="/cache/zhonghao/h3/tmp/b_${H3_MINIMAL_RUN_ID}"
[[ -d "${TMPDIR}" && "$(realpath -e -- "${TMPDIR}")" == "${TMPDIR}" ]] || exit 2
unset PYTHONPATH ZHONGHAO_H3_OPENVDN_CHECKPOINT
source /cache/zhonghao/h3/env_h3_31731.sh
# The bootstrap first isolates CANN/ATB and retains its private Python tail.
# Only after that do we put the validated B vendor before the original source.
export PYTHONPATH="/cache/zhonghao/h3/candidates/b_v1/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
export ZHONGHAO_H3_OPENVDN=1
export ZHONGHAO_H3_OPENVDN_CHECKPOINT=/cache/zhonghao/h3/models/OpenVDN-vdn-minimax-h3/stage-b-step-2000
export ASCEND_RT_VISIBLE_DEVICES="${task_cards}"
export H3_PORT="${task_port}"
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOGGING_COLOR=0
export VLLM_LOGGING_CONFIG_PATH="${H3_ROOT}/formal_logging.json"
# Waiting ceiling only; supervisor times the entire individual HTTP request.
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600
mkdir -p "${XDG_CACHE_HOME}" "${H3_OUTPUT}"
printf 'B sample=%s group=%s cards=%s port=%s run=%s source_policy=global\n' "${H3_TRIAL_SAMPLE}" "${H3_TRIAL_GROUP}" "${task_cards}" "${task_port}" "${H3_MINIMAL_RUN_ID}"
# 8 DiT workers require one all-rank text group and VAE patch group in this
# implementation. Old textTP4/VAE4 were invalid with world8, not equivalent.
exec "${H3_ENV}/bin/vllm" serve "${H3_MODEL}/Ref2VA" \
  --omni --host 127.0.0.1 --port "${H3_PORT}" --trust-remote-code \
  --init-timeout 2400 --stage-init-timeout 2400 \
  --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
