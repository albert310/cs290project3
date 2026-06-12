#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/test/test1713/miniconda3}"
ENV_NAME="${ENV_NAME:-vllm}"
ENV_PYTHON="${ENV_PYTHON:-${CONDA_ROOT}/envs/${ENV_NAME}/bin/python}"
CUDA_HOME="${CUDA_HOME:-${CONDA_ROOT}/envs/${ENV_NAME}}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_EXISTING_GPU_MEMORY_MIB="${MAX_EXISTING_GPU_MEMORY_MIB:-1024}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/run}"
PID_FILE="${PID_FILE:-${RUN_DIR}/vllm_qwen36_27b.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm_qwen36_27b.log}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -d "${MODEL_DIR}" ]] || [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model files are missing at ${MODEL_DIR}." >&2
  echo "Run scripts/download_qwen36_27b.sh first." >&2
  exit 1
fi

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Python executable is missing at ${ENV_PYTHON}." >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
if [[ "${#GPU_ARRAY[@]}" -ne "${WORLD_SIZE}" ]]; then
  echo "GPU_IDS has ${#GPU_ARRAY[@]} ids, but TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE = ${WORLD_SIZE}." >&2
  exit 1
fi

if [[ "${ALLOW_BUSY_GPUS}" != "1" ]]; then
  for gpu in "${GPU_ARRAY[@]}"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    if [[ "${used}" -gt "${MAX_EXISTING_GPU_MEMORY_MIB}" ]]; then
      echo "GPU ${gpu} already uses ${used} MiB; refusing to start." >&2
      echo "Set GPU_IDS to idle cards, or set ALLOW_BUSY_GPUS=1 if you intentionally want to share." >&2
      exit 1
    fi
  done
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "vLLM already appears to be running, pid $(cat "${PID_FILE}")." >&2
  exit 1
fi

cat > "${RUN_DIR}/vllm_qwen36_27b.env" <<EOF
PROJECT_ROOT=${PROJECT_ROOT}
CONDA_ROOT=${CONDA_ROOT}
ENV_NAME=${ENV_NAME}
ENV_PYTHON=${ENV_PYTHON}
CUDA_HOME=${CUDA_HOME}
MODEL_DIR=${MODEL_DIR}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
HOST=${HOST}
PORT=${PORT}
GPU_IDS=${GPU_IDS}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
GDN_PREFILL_BACKEND=${GDN_PREFILL_BACKEND}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
EOF

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'env_python=%s\n' "${ENV_PYTHON}"
  printf 'cuda_home=%s\n' "${CUDA_HOME}"
  printf 'path=%s\n' "${CONDA_ROOT}/envs/${ENV_NAME}/bin:${PATH}"
  printf 'model_dir=%s\n' "${MODEL_DIR}"
  printf 'gpu_ids=%s\n' "${GPU_IDS}"
  printf 'tensor_parallel_size=%s\n' "${TENSOR_PARALLEL_SIZE}"
  printf 'pipeline_parallel_size=%s\n' "${PIPELINE_PARALLEL_SIZE}"
  printf 'gpu_memory_utilization=%s\n' "${GPU_MEMORY_UTILIZATION}"
  printf 'gdn_prefill_backend=%s\n' "${GDN_PREFILL_BACKEND}"
  printf 'max_model_len=%s\n' "${MAX_MODEL_LEN}"
  printf 'max_num_seqs=%s\n' "${MAX_NUM_SEQS:-<default>}"
} > "${LOG_FILE}"

VLLM_CMD=(
  env
  PATH="${CONDA_ROOT}/envs/${ENV_NAME}/bin:${PATH}" \
  CUDA_HOME="${CUDA_HOME}" \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  VLLM_NO_USAGE_STATS=1 \
  HF_ENDPOINT="https://hf-mirror.com" \
  HF_HOME="${PROJECT_ROOT}/hf_cache" \
  PYTHONUNBUFFERED=1 \
  "${ENV_PYTHON}" -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --gdn-prefill-backend "${GDN_PREFILL_BACKEND}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  VLLM_CMD+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

setsid "${VLLM_CMD[@]}" >> "${LOG_FILE}" 2>&1 < /dev/null &

echo "$!" > "${PID_FILE}"
sleep 3
if ! kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "vLLM exited during startup. Log tail:" >&2
  tail -40 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "Started vLLM pid $(cat "${PID_FILE}")."
echo "Log: ${LOG_FILE}"
echo "Health check: scripts/check_vllm_qwen36_27b.sh"
