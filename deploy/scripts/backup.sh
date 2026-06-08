#!/bin/bash
# ==========================================================================
# DistLLM Backup Script — scheduled backup for KV cache and config
# ==========================================================================
set -e

BACKUP_DIR="${DISTLLM_BACKUP_DIR:-/data/backups}"
RETENTION_DAYS="${DISTLLM_BACKUP_RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/distllm-backup-${TIMESTAMP}"

echo "[backup] Starting backup to ${BACKUP_PATH}"

# Create backup directory
mkdir -p "${BACKUP_PATH}"

# Backup KV cache snapshots
if [ -d "/data/cache" ]; then
    echo "[backup] Backing up KV cache..."
    tar -czf "${BACKUP_PATH}/kv_cache.tar.gz" -C /data/cache .
fi

# Backup configuration
if [ -f "/etc/distllm/distllm.yaml" ]; then
    echo "[backup] Backing up configuration..."
    cp /etc/distllm/distllm.yaml "${BACKUP_PATH}/distllm.yaml"
fi

# Backup API keys if present
if [ -f "/data/keys" ]; then
    echo "[backup] Backing up API keys..."
    cp /data/keys "${BACKUP_PATH}/keys"
fi

# Backup metadata
cat > "${BACKUP_PATH}/metadata.txt" << EOF
Backup Timestamp: ${TIMESTAMP}
DistLLM Version: ${DISTLLM_VERSION:-unknown}
GPU Count: ${GPU_COUNT:-unknown}
Host: $(hostname)
EOF

echo "[backup] Backup complete: ${BACKUP_PATH}"
echo "[backup] Size: $(du -sh "${BACKUP_PATH}" | cut -f1)"

# Rotate old backups
if [ "${RETENTION_DAYS}" -gt 0 ]; then
    echo "[backup] Rotating backups older than ${RETENTION_DAYS} days..."
    find "${BACKUP_DIR}" -name "distllm-backup-*" -type d -mtime "+${RETENTION_DAYS}" -exec rm -rf {} \; 2>/dev/null || true
fi

echo "[backup] Done"
