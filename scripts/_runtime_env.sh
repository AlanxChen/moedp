#!/bin/bash

if [[ -n "${MOEDP_RUNTIME_READY:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MOEDP_REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

python_name=${PYTHON_BIN:-python}
if ! command -v "${python_name}" >/dev/null 2>&1; then
    echo "Error: python executable '${python_name}' was not found in the current environment." >&2
    return 1 2>/dev/null || exit 1
fi

export MOEDP_PYTHON_BIN=$(command -v "${python_name}")

# Prevent each training process from spawning a large BLAS / OMP thread pool.
# Multi-GPU runs launch multiple Python processes, so keeping these at 1 by
# default avoids OpenBLAS pthread_create failures while still allowing manual
# overrides from the caller environment.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-1}"

local_paths=(
    "${MOEDP_REPO_ROOT}/moe-dp"
    "${MOEDP_REPO_ROOT}/third_party/robosuite"
    "${MOEDP_REPO_ROOT}/third_party/robomimic"
    "${MOEDP_REPO_ROOT}/third_party/mimicgen"
    "${MOEDP_REPO_ROOT}/third_party/robosuite-task-zoo"
    "${MOEDP_REPO_ROOT}/third_party/gym-0.21.0"
)

local_pythonpath=$(IFS=:; echo "${local_paths[*]}")
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${local_pythonpath}:${PYTHONPATH}"
else
    export PYTHONPATH="${local_pythonpath}"
fi

export MOEDP_RUNTIME_READY=1
