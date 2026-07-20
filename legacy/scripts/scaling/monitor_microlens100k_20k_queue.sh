#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/xqp_data/RecSys26"
LOG_ROOT="${ROOT}/logs/scaling_microlens100k_20k"
LAUNCHER="${ROOT}/scripts/scaling/launch_microlens100k_20k_cell.sh"
LAUNCHER_LOG_DIR="${LOG_ROOT}/launchers"
GPU_LIST="${GPU_LIST:-2 3}"
MEM_FREE_THRESHOLD_MB="${MEM_FREE_THRESHOLD_MB:-2000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

mkdir -p "${LAUNCHER_LOG_DIR}"

JOBS=(
  "I04-semid tiger_compat M01-t5s I04_M01_20k"
  "I07-rkmeans tiger_compat M01-t5s I07_M01_20k"
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

compat_ready() {
  local id_dir="$1"
  local compat_dir="$2"
  local compat_root="${LOG_ROOT}/${id_dir}/${compat_dir}"
  [[ -s "${compat_root}/cached_ids.npy" && -s "${compat_root}/tiger_config.json" ]]
}

job_started() {
  local tag="$1"
  [[ -f "${LAUNCHER_LOG_DIR}/queue_${tag}.started" ]]
}

refresh_gpu_state() {
  local idx
  local uuid
  local mem
  local pid

  GPU_MEM_USED=()
  GPU_UUID_TO_INDEX=()
  GPU_APP_COUNT=()

  while IFS=, read -r idx uuid mem; do
    idx="$(echo "${idx}" | xargs)"
    uuid="$(echo "${uuid}" | xargs)"
    mem="$(echo "${mem}" | xargs)"
    [[ -z "${idx}" || -z "${uuid}" ]] && continue
    GPU_MEM_USED["${idx}"]="${mem}"
    GPU_UUID_TO_INDEX["${uuid}"]="${idx}"
    GPU_APP_COUNT["${idx}"]=0
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)

  while IFS=, read -r uuid pid; do
    uuid="$(echo "${uuid}" | xargs)"
    pid="$(echo "${pid}" | xargs)"
    [[ -z "${uuid}" || -z "${pid}" ]] && continue
    idx="${GPU_UUID_TO_INDEX[${uuid}]:-}"
    [[ -z "${idx}" ]] && continue
    GPU_APP_COUNT["${idx}"]=$(( ${GPU_APP_COUNT[${idx}]:-0} + 1 ))
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
}

gpu_free() {
  local gpu="$1"
  local mem="${GPU_MEM_USED[${gpu}]:-999999}"
  local app_count="${GPU_APP_COUNT[${gpu}]:-999999}"
  [[ "${app_count}" -eq 0 && "${mem}" -lt "${MEM_FREE_THRESHOLD_MB}" ]]
}

all_started() {
  local entry
  for entry in "${JOBS[@]}"; do
    read -r _ _ _ tag <<<"${entry}"
    if ! job_started "${tag}"; then
      return 1
    fi
  done
  return 0
}

while true; do
  if all_started; then
    echo "[$(timestamp)] all queued MicroLens-100K 20k cells have been started"
    exit 0
  fi

  declare -A GPU_MEM_USED
  declare -A GPU_UUID_TO_INDEX
  declare -A GPU_APP_COUNT
  refresh_gpu_state

  used_this_round=" "
  launched=0
  for entry in "${JOBS[@]}"; do
    read -r id_dir compat_dir model_size tag <<<"${entry}"
    if job_started "${tag}"; then
      continue
    fi
    if ! compat_ready "${id_dir}" "${compat_dir}"; then
      echo "[$(timestamp)] waiting compat: ${id_dir}/${compat_dir}"
      continue
    fi

    for gpu in ${GPU_LIST}; do
      if [[ "${used_this_round}" == *" ${gpu} "* ]]; then
        continue
      fi
      if gpu_free "${gpu}"; then
        log_path="${LAUNCHER_LOG_DIR}/${tag}_gpu${gpu}.nohup"
        echo "[$(timestamp)] launching ${tag}: ${id_dir} ${compat_dir} ${model_size} on gpu${gpu}"
        echo "[$(timestamp)] gpu=${gpu}" > "${LAUNCHER_LOG_DIR}/queue_${tag}.started"
        nohup bash "${LAUNCHER}" "${id_dir}" "${compat_dir}" "${model_size}" "${gpu}" > "${log_path}" 2>&1 < /dev/null &
        echo "[$(timestamp)] ${tag} pid=$! log=${log_path}"
        used_this_round="${used_this_round}${gpu} "
        launched=1
        sleep 20
        break
      fi
    done
  done

  if [[ "${launched}" -eq 0 ]]; then
    echo "[$(timestamp)] no launch this round; sleeping ${SLEEP_SECONDS}s"
  fi
  sleep "${SLEEP_SECONDS}"
done
