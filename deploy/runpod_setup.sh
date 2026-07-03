#!/usr/bin/env bash
# ============================================================
# runpod_setup.sh — Autonomous Multi-Agent Research System
# ============================================================
# Provisions a fresh RunPod GPU pod (RTX 3090/4090, 24GB VRAM)
# from zero to serving TWO AWQ-quantized models via vLLM:
#
#   Port 8000 : Mistral-7B-Instruct AWQ  (supervisor + search
#               + document + DB agents — temperature is set
#               per-request, so one instance serves all)
#   Port 8002 : DeepSeek-Coder-6.7B AWQ  (code agent)
#
# Usage (on the pod):
#   export HF_TOKEN=hf_xxx          # your HuggingFace token
#   export VLLM_API_KEY=changeme    # key your Mac will use
#   bash runpod_setup.sh
#
# Idempotent: safe to run twice. Estimated first run: ~15-25 min
# (dominated by model downloads, ~9GB total).
#
# Ticket: W4-06
# ============================================================

set -euo pipefail

# ── Config (override via env vars if needed) ─────────────────
VLLM_VERSION="${VLLM_VERSION:-0.22.1}"
SUPERVISOR_MODEL="${SUPERVISOR_MODEL:-solidrust/Mistral-7B-Instruct-v0.3-AWQ}"
CODER_MODEL="${CODER_MODEL:-TheBloke/deepseek-coder-6.7B-instruct-AWQ}"
SUPERVISOR_PORT="${SUPERVISOR_PORT:-8000}"
CODER_PORT="${CODER_PORT:-8002}"
MAX_LEN_SUPERVISOR="${MAX_LEN_SUPERVISOR:-16384}"
MAX_LEN_CODER="${MAX_LEN_CODER:-8192}"   # DeepSeek has no GQA — KV cache ~4x larger/token
GPU_UTIL_SUPERVISOR="${GPU_UTIL_SUPERVISOR:-0.45}"
GPU_UTIL_CODER="${GPU_UTIL_CODER:-0.40}"
LOG_DIR="/workspace/logs"
MODEL_CACHE="/workspace/hf-cache"        # /workspace persists across pod restarts

# ── Helpers ──────────────────────────────────────────────────
log()  { echo -e "\033[1;32m[setup]\033[0m $*"; }
fail() { echo -e "\033[1;31m[setup] ERROR:\033[0m $*" >&2; exit 1; }

# ── Preflight checks ─────────────────────────────────────────
log "Running preflight checks..."

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found — is this a GPU pod?"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || fail "GPU not detected"

[[ -n "${HF_TOKEN:-}" ]]      || fail "HF_TOKEN is not set. Run: export HF_TOKEN=hf_xxx"
[[ -n "${VLLM_API_KEY:-}" ]]  || fail "VLLM_API_KEY is not set. Run: export VLLM_API_KEY=<key>"

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Python ${PYTHON_VERSION} detected"

mkdir -p "${LOG_DIR}" "${MODEL_CACHE}"
export HF_HOME="${MODEL_CACHE}"          # cache weights on persistent volume

# ── Install vLLM (skipped if correct version already present) ─
if python3 -c "import vllm" 2>/dev/null && \
   [[ "$(python3 -c 'import vllm; print(vllm.__version__)')" == "${VLLM_VERSION}" ]]; then
  log "vLLM ${VLLM_VERSION} already installed — skipping"
else
  log "Installing vLLM ${VLLM_VERSION} (this takes a few minutes)..."
  pip install --no-cache-dir "vllm==${VLLM_VERSION}"
fi

python3 -c "import vllm" || fail "vLLM failed to import after install"
log "vLLM $(python3 -c 'import vllm; print(vllm.__version__)') ready"

# ── Serve function ───────────────────────────────────────────
# Starts a vLLM OpenAI-compatible server in the background if
# nothing is already listening on the target port (idempotency).
serve_model() {
  local model="$1" port="$2" max_len="$3" gpu_util="$4" name="$5"

  if curl -s -o /dev/null "http://localhost:${port}/health" 2>/dev/null; then
    log "${name} already serving on port ${port} — skipping"
    return 0
  fi

  log "Starting ${name} (${model}) on port ${port}..."
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --quantization awq \
    --port "${port}" \
    --max-model-len "${max_len}" \
    --gpu-memory-utilization "${gpu_util}" \
    --api-key "${VLLM_API_KEY}" \
    > "${LOG_DIR}/vllm_${name}.log" 2>&1 &

  echo $! > "${LOG_DIR}/vllm_${name}.pid"
}

# ── Health-check wait ────────────────────────────────────────
wait_healthy() {
  local port="$1" name="$2" tries=0 max_tries=60   # 60 x 15s = 15 min ceiling
  log "Waiting for ${name} on port ${port} (model download on first run)..."
  until curl -s -o /dev/null -w "" "http://localhost:${port}/health" 2>/dev/null; do
    tries=$((tries + 1))
    [[ ${tries} -ge ${max_tries} ]] && \
      fail "${name} not healthy after 15 min — check ${LOG_DIR}/vllm_${name}.log"
    sleep 15
  done
  log "${name} is healthy ✔"
}

# ── Launch both models ───────────────────────────────────────
# Sequential launch: supervisor first so its memory partition is
# claimed before the coder process starts.
serve_model "${SUPERVISOR_MODEL}" "${SUPERVISOR_PORT}" "${MAX_LEN_SUPERVISOR}" "${GPU_UTIL_SUPERVISOR}" "supervisor"
wait_healthy "${SUPERVISOR_PORT}" "supervisor"

serve_model "${CODER_MODEL}" "${CODER_PORT}" "${MAX_LEN_CODER}" "${GPU_UTIL_CODER}" "coder"
wait_healthy "${CODER_PORT}" "coder"

# ── Smoke tests ──────────────────────────────────────────────
log "Running smoke tests..."

smoke_test() {
  local port="$1" model="$2" name="$3"
  local response
  response=$(curl -s "http://localhost:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${VLLM_API_KEY}" \
    -d "{
      \"model\": \"${model}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: ready\"}],
      \"max_tokens\": 10
    }")
  echo "${response}" | grep -q '"content"' \
    || fail "${name} smoke test failed. Response: ${response}"
  log "${name} smoke test passed ✔"
}

smoke_test "${SUPERVISOR_PORT}" "${SUPERVISOR_MODEL}" "supervisor"
smoke_test "${CODER_PORT}" "${CODER_MODEL}" "coder"

# ── VRAM report ──────────────────────────────────────────────
log "Final GPU memory state:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ── Done ─────────────────────────────────────────────────────
log "============================================================"
log "Setup complete. Endpoints (from inside the pod):"
log "  Supervisor/agents : http://localhost:${SUPERVISOR_PORT}/v1  (${SUPERVISOR_MODEL})"
log "  Code agent        : http://localhost:${CODER_PORT}/v1  (${CODER_MODEL})"
log ""
log "From your Mac, use the RunPod proxy URLs for these ports"
log "and set in your .env:"
log "  VLLM_BASE_URL=<runpod-proxy-url-for-${SUPERVISOR_PORT}>/v1"
log "  VLLM_CODER_BASE_URL=<runpod-proxy-url-for-${CODER_PORT}>/v1"
log "  VLLM_API_KEY=<the key you exported>"
log ""
log "Logs: ${LOG_DIR}/vllm_supervisor.log, ${LOG_DIR}/vllm_coder.log"
log "⚠  Remember to stop the pod when done (~0.35€/hr while running)"
log "============================================================"
