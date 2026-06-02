#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"

echo "GET ${BASE_URL}/v1/models"
curl -fsS "${BASE_URL}/v1/models"
echo

echo "POST ${BASE_URL}/v1/chat/completions"
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"用一句话介绍上海科技大学。\"}
    ],
    \"max_tokens\": 80,
    \"temperature\": 0
  }"
echo

