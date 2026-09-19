#!/usr/bin/env bash
# Original dense Ref2VA A only, under this group's lease-holding supervisor.
set -euo pipefail
: "${H3_ROOT:?}" "${H3_OUTPUT:?}" "${H3_MINIMAL_RUN_ID:?}" "${H3_GROUP_ID:?}"
: "${H3_TRIAL_GROUP:?}" "${H3_TRIAL_SAMPLE:?}" "${H3_TRIAL_SUPERVISOR_PID:?}" "${H3_PORT:?}"
if [[ "$(hostname)" != "ma-job-042f8bbb-1a1b-41f0-bd5e-6ee86199e9e3-worker-0" ||
      ! "${H3_MINIMAL_RUN_ID}" =~ ^[0-9a-f]{32}$ || "${PPID}" != "${H3_TRIAL_SUPERVISOR_PID}" ]]; then
  echo 'Wrong host or missing lease-holding supervisor' >&2; exit 2
fi
case "${H3_TRIAL_GROUP}" in
  01234567) task_cards=0,1,2,3,4,5,6,7; task_port=19098 ;;
  *) echo 'Only the fixed eight-card group 01234567 is allowed' >&2; exit 2 ;;
esac
case "${H3_TRIAL_SAMPLE}" in
  shirt_red_couple_124) ;;
  *) echo 'Only the frozen shirt_red_couple_124 profile is allowed' >&2; exit 2 ;;
esac
task_prefix="/cache/zhonghao/h3/color_trial_v1/results/${H3_TRIAL_GROUP}/${H3_TRIAL_SAMPLE}/A/runs/a_"
if [[ "${ASCEND_RT_VISIBLE_DEVICES:-}" != "${task_cards}" || "${H3_PORT}" != "${task_port}" ||
      "${H3_ROOT}" != "${task_prefix}"* || "${H3_ROOT##*/}" != *"_${H3_MINIMAL_RUN_ID}" ||
      "$(realpath -e -- "${H3_ROOT}")" != "${H3_ROOT}" || "${H3_OUTPUT}" != "${H3_ROOT}/output" ||
      ! -f "${H3_ROOT}/frozen_evidence.json" || ! -f "${H3_ROOT}/host_identity.json" ]]; then
  echo 'Mismatched/noncanonical group, card, port, output or proof' >&2; exit 2
fi
export TMPDIR="/cache/zhonghao/h3/tmp/a_${H3_MINIMAL_RUN_ID}"
[[ -d "${TMPDIR}" && "$(realpath -e -- "${TMPDIR}")" == "${TMPDIR}" ]] || exit 2
unset PYTHONPATH ZHONGHAO_H3_OPENVDN_CHECKPOINT VLLM_LOGGING_CONFIG_PATH
source /cache/zhonghao/h3/env_h3_31731.sh
# The reviewed bootstrap has scrubbed inherited/candidate paths and retained
# private CANN Python packages. Keep those tails; original source stays first.
export PYTHONPATH="/cache/zhonghao/h3/src/vllm-omni:/cache/zhonghao/h3/src/vllm-ascend:/cache/zhonghao/h3/src/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export ZHONGHAO_H3_OPENVDN=0
export ASCEND_RT_VISIBLE_DEVICES="${task_cards}"
export H3_PORT="${task_port}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
mkdir -p "${XDG_CACHE_HOME}" "${H3_OUTPUT}"
printf 'A sample=%s group=%s cards=%s port=%s run=%s\n' "${H3_TRIAL_SAMPLE}" "${H3_TRIAL_GROUP}" "${task_cards}" "${task_port}" "${H3_MINIMAL_RUN_ID}"
exec "${H3_ENV}/bin/vllm" serve "${H3_MODEL}/Ref2VA" \
  --omni --host 127.0.0.1 --port "${H3_PORT}" --trust-remote-code \
  --init-timeout 2400 --stage-init-timeout 2400 \
  --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
