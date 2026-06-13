#!/usr/bin/env bash
# docker-entrypoint.sh — unified entrypoint for distllm Docker images.
#
# Routes to coordinator, worker (node), or API server based on DISTLLM_ROLE.
# Falls through to exec "$@" for arbitrary commands.

set -euo pipefail

# ── Graceful shutdown ────────────────────────────────────────────────────────
# Forward SIGTERM/SIGINT to the child process so it can clean up (NCCL, GPU
# memory, open sockets) instead of being hard-killed by Docker.
_term() {
    echo "[entrypoint] Caught signal, shutting down gracefully..."
    if [ -n "${CHILD_PID:-}" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 0
}
trap _term SIGTERM SIGINT

# ── GPU detection ────────────────────────────────────────────────────────────
# Best-effort: print GPU info if nvidia-smi is available, otherwise continue.
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
    echo "[entrypoint] Detected ${GPU_COUNT} GPU(s), first GPU memory: ${GPU_MEM} MiB"
else
    echo "[entrypoint] nvidia-smi not found — GPU detection skipped"
fi

# ── Config file ──────────────────────────────────────────────────────────────
# If the operator mounted a YAML config, surface it so the Python resolver can
# pick it up via the standard config-candidate list (/etc/distllm/config.yaml).
CONFIG_PATH="/etc/distllm/config.yaml"
if [ -f "$CONFIG_PATH" ]; then
    echo "[entrypoint] Found config at ${CONFIG_PATH}"
else
    echo "[entrypoint] No config file at ${CONFIG_PATH} — using env vars / defaults"
fi

# ── Role routing ─────────────────────────────────────────────────────────────
ROLE="${DISTLLM_ROLE:-}"

case "$ROLE" in
    coordinator)
        echo "[entrypoint] Starting coordinator..."
        exec distllm-coordinator "$@"
        ;;
    worker|node)
        echo "[entrypoint] Starting worker node..."
        exec distllm-node "$@"
        ;;
    api)
        echo "[entrypoint] Starting API server..."
        exec distllm-api "$@"
        ;;
    "")
        # No role set — fall through to user-supplied command
        ;;
    *)
        echo "[entrypoint] Unknown DISTLLM_ROLE='${ROLE}', falling through to exec"
        ;;
esac

# ── Fall-through ─────────────────────────────────────────────────────────────
# If DISTLLM_ROLE is unset or unrecognised, exec whatever the user passed.
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "[entrypoint] No DISTLLM_ROLE and no command provided."
    echo "  Set DISTLLM_ROLE to one of: coordinator, worker, api"
    echo "  Or pass a command: docker run distllm <command>"
    exit 1
fi
