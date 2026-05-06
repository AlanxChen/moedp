#!/bin/bash

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

GPU_CSV="0"
SEED_CSV="0"
RUN_TAG=$(date -u +"%m%d%H%M")
ADDITION_START="0000"
TASK_REGEX=""
LOG_ROOT="${REPO_ROOT}/logs/taskmd_train"
DRY_RUN="false"
LAUNCH_GAP_SEC=120
JOB_TIMEOUT_HOURS=20
JOB_TIMEOUT_SEC=$((JOB_TIMEOUT_HOURS * 3600))
JOB_TIMEOUT_KILL_AFTER_SEC=300

# Fixed train set for the current mainline.
TASK_NAMES=(
    "Hammer_Cleanup_T0"
    "Kitchen_T0"
    "Coffee_Preparation_T0"
    "Mug_Cleanup_T0"
    "Kitchen_Cleanup_T0"
    "Table_Cleanup_T0"
)

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_taskmd_multi_gpu.sh [options]

Options:
  --gpus 0,1,2,3        Comma-separated GPU ids. Default: 0
  --seeds 0,1           Comma-separated seeds. Default: 0
  --run-tag 0412exp     Prefix used for log directory names. Default: UTC MMDDHHMM
  --addition-start 0100 First numeric addition_info value. Default: 0000
  --task-regex REGEX    Only schedule tasks whose names match the regex.
  --log-root PATH       Root directory for training logs.
  --dry-run             Print the generated commands without launching training.
  --help                Show this message.

Behavior:
  - Uses the fixed current task list embedded in this script.
  - Schedules two trainings per task:
      1. dp_unet
      2. dp_unet_mlp_moe, with task-specific MoE hyperparameters loaded from moe_dp/config/task/<task_name>.yaml
  - Runs at most one training per GPU at a time.
  - Enforces a 120-second gap between any two training launches across all GPUs.
  - Caps each training job at 20 hours, then sends TERM and escalates to KILL after 300 seconds if needed.
EOF
}

trim_csv_array() {
    local -n array_ref=$1
    local i
    for i in "${!array_ref[@]}"; do
        array_ref[$i]="${array_ref[$i]//[[:space:]]/}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_CSV="$2"
            shift 2
            ;;
        --seeds)
            SEED_CSV="$2"
            shift 2
            ;;
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --addition-start)
            ADDITION_START="$2"
            shift 2
            ;;
        --task-regex)
            TASK_REGEX="$2"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'."
            print_usage
            exit 1
            ;;
    esac
done

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEED_CSV}"
trim_csv_array GPUS
trim_csv_array SEEDS

if [[ ${#GPUS[@]} -eq 0 || -z "${GPUS[0]}" ]]; then
    echo "Error: no GPU ids were provided."
    exit 1
fi

if [[ ${#SEEDS[@]} -eq 0 || -z "${SEEDS[0]}" ]]; then
    echo "Error: no seeds were provided."
    exit 1
fi

if [[ ! "${ADDITION_START}" =~ ^[0-9]+$ ]]; then
    echo "Error: --addition-start must be a non-negative integer, for example 0000 or 0100."
    exit 1
fi

ADDITION_WIDTH=${#ADDITION_START}
if (( ADDITION_WIDTH < 4 )); then
    ADDITION_WIDTH=4
fi
addition_counter=$((10#${ADDITION_START}))

RUN_LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
mkdir -p "${RUN_LOG_DIR}"

declare -a JOBS=()
declare -a PLAN_LINES=()

for task_name in "${TASK_NAMES[@]}"; do
    if [[ -n "${TASK_REGEX}" ]] && [[ ! "${task_name}" =~ ${TASK_REGEX} ]]; then
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        dp_addition_info=$(printf "%0${ADDITION_WIDTH}d" "${addition_counter}")
        JOBS+=("dp_unet|${task_name}|${dp_addition_info}|${seed}")
        PLAN_LINES+=("dp_unet|${task_name}|${dp_addition_info}|${seed}")
        addition_counter=$((addition_counter + 1))

        moe_addition_info=$(printf "%0${ADDITION_WIDTH}d" "${addition_counter}")
        JOBS+=("dp_unet_mlp_moe|${task_name}|${moe_addition_info}|${seed}")
        PLAN_LINES+=("dp_unet_mlp_moe|${task_name}|${moe_addition_info}|${seed}")
        addition_counter=$((addition_counter + 1))
    done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "Error: no jobs were generated. Check --task-regex or TASK_SPECS."
    exit 1
fi

PLAN_PATH="${RUN_LOG_DIR}/job_plan.tsv"
{
    echo -e "job_idx\tgpu_slot\talg_name\ttask_name\taddition_info\tseed"
    for job_idx in "${!PLAN_LINES[@]}"; do
        IFS='|' read -r alg_name task_name addition_info seed <<< "${PLAN_LINES[$job_idx]}"
        gpu_slot=${GPUS[$((job_idx % ${#GPUS[@]}))]}
        echo -e "${job_idx}\t${gpu_slot}\t${alg_name}\t${task_name}\t${addition_info}\t${seed}"
    done
} > "${PLAN_PATH}"

echo "Generated ${#JOBS[@]} training jobs from the fixed task list in scripts/train_taskmd_multi_gpu.sh"
echo "Run tag: ${RUN_TAG}"
echo "Addition start: ${ADDITION_START}"
echo "GPU list: ${GPU_CSV}"
echo "Seeds: ${SEED_CSV}"
echo "Launch gap: ${LAUNCH_GAP_SEC}s"
echo "Per-job timeout: ${JOB_TIMEOUT_HOURS}h"
echo "Logs: ${RUN_LOG_DIR}"
echo "Plan: ${PLAN_PATH}"

if [[ "${DRY_RUN}" == "true" ]]; then
    for job_idx in "${!PLAN_LINES[@]}"; do
        IFS='|' read -r alg_name task_name addition_info seed <<< "${PLAN_LINES[$job_idx]}"
        gpu_slot=${GPUS[$((job_idx % ${#GPUS[@]}))]}
        echo "[DRY-RUN][gpu ${gpu_slot}] setsid timeout --signal=TERM --kill-after=${JOB_TIMEOUT_KILL_AFTER_SEC}s ${JOB_TIMEOUT_SEC}s bash scripts/train_policy.sh ${alg_name} ${task_name} ${addition_info} ${seed} ${gpu_slot}"
    done
    exit 0
fi

if ! command -v setsid >/dev/null 2>&1; then
    echo "Error: setsid is required so Ctrl-C can stop each training process group."
    exit 1
fi

worker_pids=()
LAUNCH_LOCK_FILE="${RUN_LOG_DIR}/.launch_gap.lock"
LAUNCH_STATE_FILE="${RUN_LOG_DIR}/.last_launch_time"
rm -f "${RUN_LOG_DIR}"/.active_job_pgid_worker_* 2>/dev/null || true

signal_active_job_groups() {
    local signal="$1"
    local pgid_file
    local pgid

    for pgid_file in "${RUN_LOG_DIR}"/.active_job_pgid_worker_*; do
        [[ -e "${pgid_file}" ]] || continue
        read -r pgid < "${pgid_file}" || continue
        if [[ "${pgid}" =~ ^[0-9]+$ ]]; then
            echo "[STOP] Sending ${signal} to active training process group ${pgid}"
            kill "-${signal}" -- "-${pgid}" 2>/dev/null || true
        fi
    done
}

cleanup_workers() {
    local pid
    trap - INT TERM
    signal_active_job_groups TERM
    for pid in "${worker_pids[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    sleep 5
    signal_active_job_groups KILL
    for pid in "${worker_pids[@]}"; do
        kill -KILL "${pid}" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    rm -f "${RUN_LOG_DIR}"/.active_job_pgid_worker_* 2>/dev/null || true
}

wait_for_launch_slot() {
    local gpu_id="$1"
    local now
    local last_launch=0
    local wait_sec=0

    exec 9>"${LAUNCH_LOCK_FILE}"
    flock 9

    now=$(date +%s)
    if [[ -f "${LAUNCH_STATE_FILE}" ]]; then
        read -r last_launch < "${LAUNCH_STATE_FILE}" || last_launch=0
    fi

    if [[ "${last_launch}" =~ ^[0-9]+$ ]]; then
        wait_sec=$((last_launch + LAUNCH_GAP_SEC - now))
    fi

    if (( wait_sec > 0 )); then
        echo "[WAIT][gpu ${gpu_id}] Sleeping ${wait_sec}s to keep ${LAUNCH_GAP_SEC}s between launches"
        sleep "${wait_sec}"
    fi

    date +%s > "${LAUNCH_STATE_FILE}"
    flock -u 9
    exec 9>&-
}

trap 'echo "Stopping all training workers"; cleanup_workers; exit 130' INT
trap 'echo "Stopping all training workers"; cleanup_workers; exit 143' TERM

for gpu_index in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$gpu_index]}"
    (
        active_job_pgid=""
        active_job_pgid_file="${RUN_LOG_DIR}/.active_job_pgid_worker_${gpu_index}"

        stop_active_job() {
            trap - INT TERM
            if [[ -n "${active_job_pgid}" ]]; then
                echo "[STOP][gpu ${gpu_id}] Terminating active training process group ${active_job_pgid}"
                kill -TERM -- "-${active_job_pgid}" 2>/dev/null || true
                sleep 5
                kill -KILL -- "-${active_job_pgid}" 2>/dev/null || true
            fi
            rm -f "${active_job_pgid_file}" 2>/dev/null || true
            exit 130
        }

        trap 'stop_active_job' INT TERM

        worker_status=0
        for ((job_idx=gpu_index; job_idx<${#JOBS[@]}; job_idx+=${#GPUS[@]})); do
            IFS='|' read -r alg_name task_name addition_info seed <<< "${JOBS[$job_idx]}"
            log_file="${RUN_LOG_DIR}/$(printf "%02d" "${job_idx}")_${task_name}_${alg_name}_seed${seed}.log"
            job_status=0

            wait_for_launch_slot "${gpu_id}"
            echo "[START][gpu ${gpu_id}] ${alg_name} ${task_name} seed=${seed} addition_info=${addition_info} timeout=${JOB_TIMEOUT_HOURS}h"
            setsid timeout --signal=TERM --kill-after="${JOB_TIMEOUT_KILL_AFTER_SEC}s" "${JOB_TIMEOUT_SEC}s" \
                bash "${SCRIPT_DIR}/train_policy.sh" \
                "${alg_name}" \
                "${task_name}" \
                "${addition_info}" \
                "${seed}" \
                "${gpu_id}" > "${log_file}" 2>&1 &
            active_job_pgid=$!
            echo "${active_job_pgid}" > "${active_job_pgid_file}"
            wait "${active_job_pgid}"
            job_status=$?
            active_job_pgid=""
            rm -f "${active_job_pgid_file}" 2>/dev/null || true

            if [[ ${job_status} -eq 0 ]]; then
                echo "[DONE][gpu ${gpu_id}] ${alg_name} ${task_name} seed=${seed}. Log: ${log_file}"
                continue
            fi

            worker_status=1
            if [[ ${job_status} -eq 124 || ${job_status} -eq 137 ]]; then
                echo "[TIMEOUT][gpu ${gpu_id}] ${alg_name} ${task_name} seed=${seed} exceeded ${JOB_TIMEOUT_HOURS}h. See ${log_file}"
                continue
            fi

            echo "[FAIL][gpu ${gpu_id}] ${alg_name} ${task_name} seed=${seed} exit_code=${job_status}. See ${log_file}"
        done
        exit "${worker_status}"
    ) &
    worker_pids+=("$!")
done

overall_status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        overall_status=1
    fi
done

if [[ ${overall_status} -ne 0 ]]; then
    echo "Completed with failures. Check logs under ${RUN_LOG_DIR}."
    exit 1
fi

echo "All scheduled training jobs finished successfully."
