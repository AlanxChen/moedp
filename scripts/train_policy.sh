#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_runtime_env.sh
source "${SCRIPT_DIR}/_runtime_env.sh"

# Usage:
# bash scripts/train_policy.sh dp_unet_mlp_moe Kitchen_T0 9002 0 0
# args: alg_name task_name addition_info seed gpu_id

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}

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
addition_info=${3}
seed=${4}
gpu_id=${5}
n_demo=100

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

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
echo "Number of demonstrations: $n_demo"

export CUDA_VISIBLE_DEVICES=${gpu_id}
"${MOEDP_PYTHON_BIN}" train.py --config-name=${config_name}.yaml \
    task=${task} \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    task.env_runner._target_="moe_dp.env_runner.mimicgen_runner_eval_reset_obj_async.MimicgenRunner" \
    checkpoint.save_ckpt=${save_ckpt} \
    multi_run.run_dir=${run_dir} \
    task.n_demo=${n_demo}
