#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"

conda create -n rl --clone grader -y
conda activate rl

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

cd "${PROJECT_ROOT}"
pip install -r requirements-npu.txt
pip install -v -e .

python -m pip uninstall -y triton triton-ascend
rm -rf ~/.triton/cache/*
