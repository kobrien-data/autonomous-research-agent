#!/usr/bin/env bash
# ============================================================
# stop_vllm.sh — Autonomous Multi-Agent Research System
# ============================================================
# Gracefully stops vLLM processes started by runpod_setup.sh
# WITHOUT terminating the pod. Frees GPU memory (e.g. to swap
# models) while keeping the pod, weights cache, and logs alive.
#
# Usage (on the pod):
#   bash stop_vllm.sh              # stop both models
#   bash stop_vllm.sh supervisor   # stop only the Mistral process
#   bash stop_vllm.sh coder        # stop only the DeepSeek process
#
# NOTE: this does NOT stop pod billing. To stop paying, stop
# the pod itself from the RunPod console once you're done.
#
# Ticket: W7-07
# ============================================================

set -euo pipefail

LOG_DIR="/workspace/logs"
GRACE_SECONDS=15   # time to allow in-flight requests to finish

log()  { echo -e "\033[1;32m[stop]\033[0m $*"; }
warn() { echo -e "\033[1;33m[stop]\033[0m $*"; }

stop_one() {
  local name="$1"
  local pid_file="${LOG_DIR}/vllm_${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    warn "No PID file for '${name}' (${pid_file}) — nothing to stop"
    return 0
  fi

  local pid
  pid=$(cat "${pid_file}")

  # PID file may be stale (process already dead)
  if ! kill -0 "${pid}" 2>/dev/null; then
    warn "'${name}' (PID ${pid}) is not running — removing stale PID file"
    rm -f "${pid_file}"
    return 0
  fi

  # Graceful: SIGTERM lets vLLM finish in-flight requests and
  # release CUDA memory cleanly
  log "Stopping '${name}' (PID ${pid}) with SIGTERM..."
  kill -TERM "${pid}"

  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [[ ${waited} -ge ${GRACE_SECONDS} ]]; then
      warn "'${name}' didn't exit within ${GRACE_SECONDS}s — sending SIGKILL"
      kill -KILL "${pid}" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  rm -f "${pid_file}"
  log "'${name}' stopped ✔"
}

# ── Which model(s) to stop ───────────────────────────────────
TARGET="${1:-all}"

case "${TARGET}" in
  supervisor) stop_one "supervisor" ;;
  coder)      stop_one "coder" ;;
  all)        stop_one "supervisor"; stop_one "coder" ;;
  *)          echo "Usage: bash stop_vllm.sh [supervisor|coder|all]"; exit 1 ;;
esac

# ── Report freed memory ──────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU memory after shutdown:"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
fi

# ── Budget reminder (W7-07 acceptance criterion) ─────────────
echo ""
warn "⚠  vLLM is stopped but THE POD IS STILL BILLING (~0.35€/hr)."
warn "   Stop the pod from the RunPod console if you're done working."
warn "   Model weights are cached on /workspace — restart is fast:"
warn "   bash runpod_setup.sh"
