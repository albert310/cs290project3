#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/run/vllm_qwen36_27b.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No pid file found at ${PID_FILE}."
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" >/dev/null 2>&1; then
  kill "${pid}"
  echo "Stopped vLLM pid ${pid}."
else
  echo "Process ${pid} is not running."
fi
rm -f "${PID_FILE}"

