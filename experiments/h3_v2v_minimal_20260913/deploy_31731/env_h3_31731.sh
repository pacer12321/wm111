#!/usr/bin/env bash
# Source this ONLY for zhonghao's private H3 installation on port 31731.
# This does not start a service or select/reserve any NPU devices.
set -euo pipefail
export H3_INSTALL=/cache/zhonghao/h3
export H3_ENV="${H3_INSTALL}/env"
export H3_MODEL="${H3_INSTALL}/models/MiniMax-H3"
export H3_VDN_CHECKPOINT="${H3_INSTALL}/models/OpenVDN-vdn-minimax-h3/stage-b-step-2000"
export H3_ROOT="${H3_ROOT:-${H3_INSTALL}/runtime}"
export H3_OUTPUT="${H3_OUTPUT:-${H3_ROOT}/output}"

# CANN's generated remove_env only removes versions under its OWN installation
# prefix. It cannot remove inherited /usr/local/Ascend/cann-8.5.2, toolkit/latest
# or system NNAL. Build this process's search paths from trusted roots before
# sourcing 9.0.1; never edit the system installation/default shell environment.
_h3_path_allowed() {
    local task_kind="$1" task_entry="$2"
    # Empty/current-directory/relative and parent-traversing paths are unsafe.
    [[ "${task_entry}" == /* && "${task_entry}" != *'/../'* && "${task_entry}" != */.. &&
       "${task_entry}" != *'/./'* && "${task_entry}" != */. ]] || return 1
    case "${task_kind}:${task_entry}" in
        bin:"${H3_ENV}"/*|bin:"${H3_INSTALL}/bin"|bin:/usr/local/ffmpeg/bin|bin:/usr/local/Ascend/driver/tools|\
        bin:/usr/local/sbin|bin:/usr/local/bin|bin:/usr/sbin|bin:/usr/bin|bin:/sbin|bin:/bin) return 0 ;;
        lib:"${H3_ENV}"|lib:"${H3_ENV}"/*|lib:/usr/local/Ascend/driver|lib:/usr/local/Ascend/driver/*|\
        lib:/lib|lib:/lib/*|lib:/lib64|lib:/lib64/*|lib:/usr/lib|lib:/usr/lib/*|lib:/usr/lib64|lib:/usr/lib64/*) return 0 ;;
        python:"${H3_ENV}"/*|python:"${H3_INSTALL}/src/vllm"|python:"${H3_INSTALL}/src/vllm"/*|\
        python:"${H3_INSTALL}/src/vllm-ascend"|python:"${H3_INSTALL}/src/vllm-ascend"/*|\
        python:"${H3_INSTALL}/src/vllm-omni"|python:"${H3_INSTALL}/src/vllm-omni"/*) return 0 ;;
        build:"${H3_ENV}"|build:"${H3_ENV}"/*|build:/usr/local/Ascend/driver|build:/usr/local/Ascend/driver/*|\
        build:/usr|build:/usr/include|build:/usr/include/*|build:/usr/lib|build:/usr/lib/*|\
        build:/usr/lib64|build:/usr/lib64/*|build:/lib|build:/lib/*|build:/lib64|build:/lib64/*) return 0 ;;
    esac
    return 1
}

_h3_clean_list() {
    local task_name="$1" task_kind="$2" task_mode="$3" task_entry task_result=""
    local -a task_entries=()
    IFS=: read -r -a task_entries <<< "${!task_name-}"
    for task_entry in "${task_entries[@]}"; do
        [[ -n "${task_entry}" ]] || continue
        if ! _h3_path_allowed "${task_kind}" "${task_entry}"; then
            if [[ "${task_mode}" == strict ]]; then
                printf 'Foreign/noncanonical %s entry after private toolkit setup; refusing runtime\n' "${task_name}" >&2
                return 1
            fi
            continue
        fi
        case ":${task_result}:" in
            *":${task_entry}:"*) ;;
            *) task_result="${task_result:+${task_result}:}${task_entry}" ;;
        esac
    done
    printf -v "${task_name}" '%s' "${task_result}"
    export "${task_name}"
}

# No foreign preload or Python interpreter-home binding is allowed to override
# the selected private interpreter. Device allocation/HCCL controls are untouched.
unset LD_PRELOAD LD_AUDIT PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
_h3_clean_list PATH bin inherited
_h3_clean_list LD_LIBRARY_PATH lib inherited
_h3_clean_list PYTHONPATH python inherited
for _h3_task_variable in CMAKE_PREFIX_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH PKG_CONFIG_PATH; do
    _h3_clean_list "${_h3_task_variable}" build inherited
done
# Remove stale accelerator PATH/HOME/DIR bindings, not controls such as the
# physical-card mask, HCCL settings or numeric ATB tuning flags. The private
# vendor scripts establish their own canonical bindings below.
for _h3_task_variable in ${!ASCEND_@} ${!ATB_@} ${!CANN_@} ${!NNAE_@} ${!DDK_@}; do
    case "${_h3_task_variable}" in
        *PATH|*HOME|*DIR) unset "${_h3_task_variable}" ;;
    esac
done
unset TOOLCHAIN_HOME TBE_IMPL_PATH
export ASCEND_DRIVER_PATH=/usr/local/Ascend/driver
export PATH="${H3_ENV}/bin:/usr/local/ffmpeg/bin:/usr/local/Ascend/driver/tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${H3_ENV}/lib:${H3_ENV}/lib/python3.12/site-packages/torch/lib:${H3_ENV}/lib/python3.12/site-packages/torch_npu/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${H3_INSTALL}/src/vllm-omni:${H3_INSTALL}/src/vllm-ascend:${H3_INSTALL}/src/vllm${PYTHONPATH:+:${PYTHONPATH}}"
[[ "$(realpath -e -- "${H3_ENV}")" == "${H3_ENV}" ]]
export ASCEND_HOME_PATH="${H3_ENV}/Ascend/cann-9.0.1"
export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"
for _h3_task_script in "${ASCEND_HOME_PATH}/set_env.sh" "${H3_ENV}/Ascend/nnal/nnal/atb/set_env.sh"; do
    [[ -f "${_h3_task_script}" && "$(realpath -e -- "${_h3_task_script}")" == "${_h3_task_script}" ]]
done
set +u
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${H3_ENV}/Ascend/nnal/nnal/atb/set_env.sh"
set -u
# NNAL may clear this variable while loading. Set it after both vendor scripts.
export ATB_SHARE_MEMORY_NAME_SUFFIX="${H3_GROUP_ID:-zhonghao_h3_31731_validation}"
export PATH="${H3_INSTALL}/bin:${H3_ENV}/bin:/usr/local/ffmpeg/bin:${PATH}"
export PYTHONPATH="${H3_INSTALL}/src/vllm-omni:${H3_INSTALL}/src/vllm-ascend:${H3_INSTALL}/src/vllm${PYTHONPATH:+:${PYTHONPATH}}"
# Deduplicate after vendors prepend, and reject any foreign path they introduce.
_h3_clean_list PATH bin strict
_h3_clean_list LD_LIBRARY_PATH lib strict
_h3_clean_list PYTHONPATH python strict
for _h3_task_variable in CMAKE_PREFIX_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH PKG_CONFIG_PATH; do
    _h3_clean_list "${_h3_task_variable}" build strict
done
[[ "${ASCEND_HOME_PATH}" == "${H3_ENV}/Ascend/cann-9.0.1" && "${ASCEND_TOOLKIT_HOME}" == "${ASCEND_HOME_PATH}" &&
   "${ASCEND_OPP_PATH}" == "${ASCEND_HOME_PATH}/opp" && "${ASCEND_AICPU_PATH}" == "${ASCEND_HOME_PATH}" &&
   "${ATB_HOME_PATH}" == "${H3_ENV}/Ascend/nnal/nnal/atb/"* &&
   "$(realpath -m -- "${ATB_HOME_PATH}")" == "${H3_ENV}/"* ]]
for _h3_task_variable in ${!ASCEND_@} ${!ATB_@} ${!CANN_@} ${!NNAE_@} ${!DDK_@}; do
    case "${_h3_task_variable}" in
        ASCEND_DRIVER_PATH) [[ "${!_h3_task_variable}" == /usr/local/Ascend/driver ]] ;;
        *PATH|*HOME|*DIR) [[ "${!_h3_task_variable}" == "${H3_ENV}" || "${!_h3_task_variable}" == "${H3_ENV}/"* ]] ;;
    esac
done
unset _h3_task_variable _h3_task_script
unset -f _h3_path_allowed _h3_clean_list
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-1800}"
export XDG_CACHE_HOME="${H3_ROOT}/cache"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
