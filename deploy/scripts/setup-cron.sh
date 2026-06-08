#!/bin/bash
# ==========================================================================
# DistLLM Cron Job Setup — installs scheduled backup and maintenance tasks
# ==========================================================================
set -e

CRON_FILE="/etc/cron.d/distllm-maintenance"
BACKUP_SCRIPT="/app/deploy/scripts/backup.sh"

# Ensure backup script exists
if [ ! -f "${BACKUP_SCRIPT}" ]; then
    echo "[cron] Warning: ${BACKUP_SCRIPT} not found, skipping backup schedule"
    exit 0
fi

# Create cron file
cat > "${CRON_FILE}" << EOF
# DistLLM scheduled maintenance tasks

# Hourly — rotate logs, check disk space
0 * * * * root df -h /data | logger -t distllm-disk

# Daily — backup KV cache and config (at 2am)
0 2 * * * root bash ${BACKUP_SCRIPT} >> /var/log/distllm-backup.log 2>&1

# Weekly — cleanup old logs (Sunday at 3am)
0 3 * * 0 root find /var/log -name "distllm-*.log*" -mtime +30 -delete
EOF

chmod 644 "${CRON_FILE}"
echo "[cron] Maintenance schedule installed at ${CRON_FILE}"
echo "[cron] - Hourly: disk space check"
echo "[cron] - Daily (2am): KV cache and config backup"
echo "[cron] - Weekly (Sunday 3am): Log rotation cleanup"
