#!/bin/bash
set -e

# ==========================================================================
# DistLLM Docker Entrypoint
#
# Handles:
#   - Dynamic command selection (coordinator, worker, API server)
#   - Environment variable passthrough
#   - Graceful shutdown on SIGTERM/SIGINT
#   - Configuration from mounted config files
# ==========================================================================

# --- Signal handling ---
cleanup() {
    echo "[entrypoint] Received shutdown signal, terminating..."
    if [ -n "$CHILD_PID" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    echo "[entrypoint] Shutdown complete"
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- Runtime detection ---
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    echo "[entrypoint] Detected $GPU_COUNT NVIDIA GPU(s)"
else
    echo "[entrypoint] No NVIDIA GPU detected (running CPU mode)"
fi

if command -v free &>/dev/null; then
    TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
    echo "[entrypoint] System memory: ${TOTAL_MEM_MB}MB"
fi

# --- Configuration ---
if [ -f "/etc/distllm/distllm.yaml" ]; then
    echo "[entrypoint] Loading configuration from /etc/distllm/distllm.yaml"
    export DISTLLM_CONFIG="/etc/distllm/distllm.yaml"
fi

# --- Role-based entry ---
ROLE="${DISTLLM_ROLE:-coordinator}"

case "$ROLE" in
    coordinator)
        echo "[entrypoint] Starting DistLLM coordinator..."
        exec distllm-coordinator \
            --model "${DISTLLM_MODEL}" \
            --port "${DISTLLM_PORT:-50050}" \
            ${DISTLLM_NODES:+--nodes "$DISTLLM_NODES"} \
            ${DISTLLM_LOCAL:+--local} \
            "$@"
        ;;
    worker)
        echo "[entrypoint] Starting DistLLM worker node..."
        exec distllm-node \
            --node-id "${DISTLLM_NODE_ID:-worker-0}" \
            --model "${DISTLLM_MODEL}" \
            --start-layer "${DISTLLM_START_LAYER:-0}" \
            --end-layer "${DISTLLM_END_LAYER:-3}" \
            --total-layers "${DISTLLM_TOTAL_LAYERS:-8}" \
            --coordinator-host "${DISTLLM_COORDINATOR_HOST}" \
            --coordinator-port "${DISTLLM_COORDINATOR_PORT:-50050}" \
            "$@"
        ;;
    api)
        echo "[entrypoint] Starting DistLLM API server..."
        exec distllm-api \
            --model "${DISTLLM_MODEL}" \
            --port "${DISTLLM_PORT:-8000}" \
            ${DISTLLM_LOCAL:+--local} \
            "$@"
        ;;
    *)
        echo "[entrypoint] Unknown DISTLLM_ROLE '$ROLE'. Falling through to command."
        exec "$@"
        ;;
esac
