#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_runtime_env.sh
source "${SCRIPT_DIR}/_runtime_env.sh"

DEBUG=False
DEFAULT_TAG="eval"
DEFAULT_SEED=42
DEFAULT_GPU_ID=0
DEFAULT_N_ENVS=25
DEFAULT_N_TEST=50
DEFAULT_N_TEST_VIS=20

print_usage() {
    cat <<'EOF'
Usage:
  Recommended named-argument form:
    bash scripts/eval_policy_mimicgen_async.sh \
      --alg dp_unet_mlp_moe \
      --task Kitchen_Cleanup_T0 \
      --checkpoint data/outputs/your_train_run/checkpoints/latest.ckpt \
      --tag eval1 \
      --seed 3 \
      --gpu 0 \
      --n-envs 25 \
      --n-test 50 \
      --n-test-vis 20

  Backward-compatible positional form:
    bash scripts/eval_policy_mimicgen_async.sh \
      dp_unet_mlp_moe \
      Kitchen_Cleanup_T0 \
      eval1 \
      3 \
      0 \
      data/outputs/your_train_run/checkpoints/latest.ckpt \
      25 \
      50 \
      20

Required arguments:
  --alg           Policy config name, for example: dp_unet or dp_unet_mlp_moe
  --task          MimicGen task name, for example: Kitchen_Cleanup_T0
  --checkpoint    Checkpoint to evaluate

Optional arguments:
  --tag           Extra label added to the rollout run name (default: eval)
  --seed          Evaluation seed written into the run name and Hydra config (default: 42)
  --gpu           CUDA device index exposed through CUDA_VISIBLE_DEVICES (default: 0)
  --n-envs        Number of parallel environments used during evaluation (default: 25)
  --n-test        Number of test episodes per setting (default: 50)
  --n-test-vis    Number of recorded videos per setting (default: 20)
  --debug         Run rollout with training.debug=True and offline logging
  -h, --help      Show this help message
EOF
}

require_integer() {
    local value="$1"
    local name="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: ${name} must be a non-negative integer, got '${value}'." >&2
        exit 1
    fi
}

alg_name=""
task_name=""
run_tag="${DEFAULT_TAG}"
seed="${DEFAULT_SEED}"
gpu_id="${DEFAULT_GPU_ID}"
checkpoint_path=""
n_envs="${DEFAULT_N_ENVS}"
n_test="${DEFAULT_N_TEST}"
n_test_vis="${DEFAULT_N_TEST_VIS}"

parse_named_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --alg)
                alg_name="${2:-}"
                shift 2
                ;;
            --task)
                task_name="${2:-}"
                shift 2
                ;;
            --tag)
                run_tag="${2:-}"
                shift 2
                ;;
            --seed)
                seed="${2:-}"
                shift 2
                ;;
            --gpu)
                gpu_id="${2:-}"
                shift 2
                ;;
            --checkpoint)
                checkpoint_path="${2:-}"
                shift 2
                ;;
            --n-envs)
                n_envs="${2:-}"
                shift 2
                ;;
            --n-test)
                n_test="${2:-}"
                shift 2
                ;;
            --n-test-vis)
                n_test_vis="${2:-}"
                shift 2
                ;;
            --debug)
                DEBUG=True
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option '$1'." >&2
                print_usage >&2
                exit 1
                ;;
        esac
    done
}

parse_legacy_args() {
    if [[ $# -lt 6 ]]; then
        echo "Error: legacy positional mode requires at least 6 arguments." >&2
        print_usage >&2
        exit 1
    fi

    alg_name="$1"
    task_name="$2"
    run_tag="$3"
    seed="$4"
    gpu_id="$5"
    checkpoint_path="$6"
    n_envs="${7:-$DEFAULT_N_ENVS}"
    n_test="${8:-$DEFAULT_N_TEST}"
    n_test_vis="${9:-$DEFAULT_N_TEST_VIS}"
}

if [[ $# -eq 0 ]]; then
    print_usage
    exit 1
fi

if [[ "${1}" == -* ]]; then
    parse_named_args "$@"
else
    parse_legacy_args "$@"
fi

if [[ -z "$alg_name" || -z "$task_name" || -z "$checkpoint_path" ]]; then
    echo "Error: --alg, --task, and --checkpoint are required." >&2
    print_usage >&2
    exit 1
fi

if [[ $alg_name == dp_* ]]; then
    observation_type="image"
else
    echo "Error: Unsupported alg_name. This main branch only keeps image-based dp models such as 'dp_unet' and 'dp_unet_mlp_moe'."
    exit 1
fi

task_env=${task_name%%_*}
echo "Observation type: $observation_type"
echo "Task environment: $task_env"
task=${task_name}
echo "Task config: $task"

config_name=${alg_name}

if [[ "$checkpoint_path" != /* ]]; then
    app_relative_checkpoint="${MOEDP_REPO_ROOT}/moe-dp/${checkpoint_path}"
    repo_relative_checkpoint="${MOEDP_REPO_ROOT}/${checkpoint_path}"
    if [[ -f "$app_relative_checkpoint" ]]; then
        checkpoint_path="$app_relative_checkpoint"
    else
        checkpoint_path="$repo_relative_checkpoint"
    fi
fi

if [[ ! -f "$checkpoint_path" ]]; then
    echo "Error: checkpoint file does not exist: $checkpoint_path"
    exit 1
fi

require_integer "$seed" "seed"
require_integer "$gpu_id" "gpu"
require_integer "$n_envs" "n-envs"
require_integer "$n_test" "n-test"
require_integer "$n_test_vis" "n-test-vis"

if [[ "$n_envs" -le 0 ]]; then
    echo "Error: n_envs must be a positive integer."
    exit 1
fi

if [[ "$n_test" -le 0 ]]; then
    echo "Error: n_test must be a positive integer."
    exit 1
fi

if [[ "$n_test_vis" -lt 0 ]]; then
    echo "Error: n_test_vis must be zero or a positive integer."
    exit 1
fi

exp_name=${task_name}-${alg_name}-${run_tag}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "Run tag: $run_tag"
echo "Checkpoint: $checkpoint_path"
echo "Concurrent environments: $n_envs"
echo "Test episodes per setting: $n_test"
echo "Recorded videos per setting: $n_test_vis"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd "${MOEDP_REPO_ROOT}/moe-dp"

export CUDA_VISIBLE_DEVICES=${gpu_id}
"${MOEDP_PYTHON_BIN}" rollout_mimicgen_async.py --config-name=${config_name}.yaml \
    task=${task} \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.resume=False \
    exp_name=${exp_name} \
    task.env_runner._target_="moe_dp.env_runner.mimicgen_runner_eval_reset_obj_async.MimicgenRunner" \
    task.env_runner.n_envs=${n_envs} \
    task.env_runner.n_test=${n_test} \
    task.env_runner.n_test_vis=${n_test_vis} \
    logging.mode=${wandb_mode} \
    multi_run.run_dir=${run_dir} \
    +rollout.checkpoint_path="${checkpoint_path}" \
    +rollout.output_subdir="rollout"
