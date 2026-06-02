#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/2022533109/chenyuhan/miniconda3}"
ENV_NAME="${ENV_NAME:-cs290-qwen-vllm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3.6-27B}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${MODEL_DIR}" "${HF_HOME}" "${PROJECT_ROOT}/logs"

eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
set +u
conda activate "${ENV_NAME}"
set -u

echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "MODEL_ID=${MODEL_ID}"
echo "MODEL_DIR=${MODEL_DIR}"

python -m huggingface_hub.commands.huggingface_cli download \
  "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --cache-dir "${HF_HOME}" \
  --max-workers "${HF_MAX_WORKERS:-8}"

date -u +"download_finished_at=%Y-%m-%dT%H:%M:%SZ" > "${MODEL_DIR}/download_metadata.env"
echo "Download complete."
