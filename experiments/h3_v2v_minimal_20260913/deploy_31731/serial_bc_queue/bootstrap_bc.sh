#!/usr/bin/env bash
# Fixed environment boundary for coordinator and both owned C supervisors.
set -euo pipefail
case "${1:-}" in queue|C_tiny|C_full) task_kind="$1" ;; *) exit 2 ;; esac
[[ "$#" == 1 ]] || exit 2
if [[ "${task_kind}" != queue ]]; then
  [[ "${H3_SERIAL_BC_QUEUE_ID:-}" == b_6362ce1811da4db69d335b3b3ca5d1a4_c_tiny_c_lake ]] || exit 2
fi
unset PYTHONPATH PYTHONHOME VLLM_LOGGING_CONFIG_PATH
source /cache/zhonghao/h3/env_h3_31731.sh
[[ "$(command -v npu-smi)" == /usr/local/sbin/npu-smi && -x /usr/local/sbin/npu-smi ]]
[[ "$(command -v python)" == /cache/zhonghao/h3/env/bin/python ]]
[[ "$(command -v ffmpeg)" == /cache/zhonghao/h3/bin/ffmpeg && -x /cache/zhonghao/h3/bin/ffmpeg ]]
case "${task_kind}" in
  queue) exec /cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/serial_bc_queue_code/serial_bc_queue.py --allow-c-chain ;;
  C_tiny) exec /cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/c_validation_code/run_c_validation.py --allow-npu ;;
  C_full) exec /cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/c_model_trial_code/run_c_trial.py --group 01234567 --sample lake_snow --allow-npu ;;
esac
