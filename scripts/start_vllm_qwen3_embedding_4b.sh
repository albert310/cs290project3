#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/test/test1713/miniconda3}"
ENV_NAME="${ENV_NAME:-vllm}"
ENV_PYTHON="${ENV_PYTHON:-${CONDA_ROOT}/envs/${ENV_NAME}/bin/python}"
CUDA_HOME="${CUDA_HOME:-${CONDA_ROOT}/envs/${ENV_NAME}}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/../model/Qwem3-Embedding-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-embedding-4b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
GPU_IDS="${GPU_IDS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_EXISTING_GPU_MEMORY_MIB="${MAX_EXISTING_GPU_MEMORY_MIB:-78000}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/run}"
PID_FILE="${PID_FILE:-${RUN_DIR}/vllm_qwen3_embedding_4b.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm_qwen3_embedding_4b.log}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -d "${MODEL_DIR}" ]] || [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Embedding model files are missing at ${MODEL_DIR}." >&2
  exit 1
fi

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Python executable is missing at ${ENV_PYTHON}." >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${TENSOR_PARALLEL_SIZE}" ]]; then
  echo "GPU_IDS has ${#GPU_ARRAY[@]} ids, but TENSOR_PARALLEL_SIZE = ${TENSOR_PARALLEL_SIZE}." >&2
  exit 1
fi

if [[ "${ALLOW_BUSY_GPUS}" != "1" ]]; then
  for gpu in "${GPU_ARRAY[@]}"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    if [[ "${used}" -gt "${MAX_EXISTING_GPU_MEMORY_MIB}" ]]; then
      echo "GPU ${gpu} already uses ${used} MiB; refusing to start." >&2
      exit 1
    fi
  done
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "Embedding vLLM already appears to be running, pid $(cat "${PID_FILE}")." >&2
  exit 1
fi

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'env_python=%s\n' "${ENV_PYTHON}"
  printf 'model_dir=%s\n' "${MODEL_DIR}"
  printf 'gpu_ids=%s\n' "${GPU_IDS}"
  printf 'port=%s\n' "${PORT}"
} > "${LOG_FILE}"

VLLM_CMD=(
  env
  PATH="${CONDA_ROOT}/envs/${ENV_NAME}/bin:${PATH}" \
  CUDA_HOME="${CUDA_HOME}" \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  VLLM_NO_USAGE_STATS=1 \
  HF_HOME="${PROJECT_ROOT}/hf_cache" \
  PYTHONUNBUFFERED=1 \
  "${ENV_PYTHON}" -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --runner pooling \
  --convert embed \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}"
)

setsid "${VLLM_CMD[@]}" >> "${LOG_FILE}" 2>&1 < /dev/null &

echo "$!" > "${PID_FILE}"
sleep 3
if ! kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "Embedding vLLM exited during startup. Log tail:" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "Started embedding vLLM pid $(cat "${PID_FILE}")."
echo "Log: ${LOG_FILE}"
echo "Models: http://127.0.0.1:${PORT}/v1/models"
