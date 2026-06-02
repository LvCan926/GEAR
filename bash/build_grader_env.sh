#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-${PROJECT_ROOT}/vllm}"
MODEL_PATH="${MODEL_PATH:-}"

if [[ ! -d "${VLLM_SOURCE_DIR}" ]]; then
    echo "ERROR: VLLM_SOURCE_DIR not found: ${VLLM_SOURCE_DIR}" >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: set MODEL_PATH to the grader model directory, got: ${MODEL_PATH}" >&2
    exit 1
fi

source ~/.bashrc
eval "$(conda shell.bash hook)"

conda create -n grader python=3.10 -y
conda activate grader

python -m pip install --upgrade pip setuptools wheel

cd "${VLLM_SOURCE_DIR}"
VLLM_TARGET_DEVICE=empty pip install -v -e .
pip install vllm-ascend==0.9.1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

vllm serve "$MODEL_PATH" --port 8001 --host 0.0.0.0 --served-model-name grader
