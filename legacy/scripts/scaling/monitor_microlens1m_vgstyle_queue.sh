#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/xqp_data/RecSys26"
LOG_ROOT="${ROOT}/logs/scaling_microlens1m_vgstyle"
LAUNCHER="${ROOT}/scripts/scaling/launch_microlens1m_vgstyle_cell.sh"
LAUNCHER_LOG_DIR="${LOG_ROOT}/launchers"
GPU_LIST="${GPU_LIST:-0 1 3 4 5 6}"
MEM_FREE_THRESHOLD_MB="${MEM_FREE_THRESHOLD_MB:-2000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

mkdir -p "${LAUNCHER_LOG_DIR}"

JOBS=(
  "I04-semid tiger_compat_title_hierarchy M01-t5s I04_M01"
  "I04-semid tiger_compat_title_hierarchy M02-t5b I04_M02"
  "I04-semid tiger_compat_title_hierarchy M03-t5l I04_M03"
  "I05-letter tiger_compat M01-t5s I05_M01"
  "I05-letter tiger_compat M02-t5b I05_M02"
  "I05-letter tiger_compat M03-t5l I05_M03"
  "I07-rkmeans tiger_compat_balanced M01-t5s I07_M01"
  "I07-rkmeans tiger_compat_balanced M02-t5b I07_M02"
  "I07-rkmeans tiger_compat_balanced M03-t5l I07_M03"
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

job_started() {
  local id_dir="$1"
  local model_size="$2"
  local tag="$3"
  [[ -f "${LAUNCHER_LOG_DIR}/queue_${tag}.started" ]] && return 0
  find "${LOG_ROOT}/${id_dir}/${model_size}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q .
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
    read -r id_dir _ model_size tag <<<"${entry}"
    if ! job_started "${id_dir}" "${model_size}" "${tag}"; then
      return 1
    fi
  done
  return 0
}

while true; do
  if all_started; then
    echo "[$(timestamp)] all queued MicroLens-1M VG-style cells have been started"
    exit 0
  fi

  declare -A GPU_MEM_USED
  declare -A GPU_UUID_TO_INDEX
  declare -A GPU_APP_COUNT
  refresh_gpu_state

  used_this_round=" "
  for entry in "${JOBS[@]}"; do
    read -r id_dir compat_dir model_size tag <<<"${entry}"
    if job_started "${id_dir}" "${model_size}" "${tag}"; then
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
        sleep 20
        break
      fi
    done
  done

  echo "[$(timestamp)] no additional free GPU found; sleeping ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done
