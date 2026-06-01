"""Backup & Restore — cluster configuration backup, disaster recovery.

Supports full and incremental backups of coordinator state, node
registrations, model assignments, KV cache checkpoints, and configuration.
Provides restore operations for disaster recovery.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger


@dataclass
class BackupEntry:
    """A single entry in a backup archive."""
    key: str
    data: Any
    timestamp: float
    content_type: str = "application/json"
    compressed: bool = False


@dataclass
class BackupManifest:
    """Metadata about a backup snapshot."""
    backup_id: str
    created_at: float
    size_bytes: int
    entries: int
    backup_type: str  # "full" or "incremental"
    cluster_name: str = ""
    model_name: str = ""
    node_count: int = 0
    version: str = "1.0"


class BackupManager:
    """Manage cluster configuration backups and disaster recovery.

    Usage:
        mgr = BackupManager(backup_dir="./backups")
        mgr.create_full("coordinator-1", config, nodes, models)
        mgr.create_incremental("coordinator-1", config_changes)

        # Restore
        manifest = mgr.list_backups()[-1]
        state = mgr.restore(manifest.backup_id)
    """

    MAX_BACKUPS: int = 20

    def __init__(
        self,
        backup_dir: str = "./backups",
        max_backups: int = 20,
    ) -> None:
        self._backup_dir = Path(backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._max_backups = max_backups

    # ── Backup operations ───────────────────────────────────────────────

    def create_full(
        self,
        cluster_name: str,
        coordinator_config: dict[str, Any],
        node_registrations: list[dict[str, Any]],
        model_assignments: list[dict[str, Any]],
        generation_settings: dict[str, Any] | None = None,
        custom_data: dict[str, Any] | None = None,
    ) -> BackupManifest:
        """Create a full backup of the cluster state."""
        backup_id = f"full-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"

        entries = [
            BackupEntry("coordinator_config", coordinator_config, time.time()),
            BackupEntry("node_registrations", node_registrations, time.time()),
            BackupEntry("model_assignments", model_assignments, time.time()),
        ]
        if generation_settings:
            entries.append(BackupEntry("generation_settings", generation_settings, time.time()))
        if custom_data:
            for key, value in custom_data.items():
                entries.append(BackupEntry(key, value, time.time()))

        manifest = self._write_backup(backup_id, entries, "full", cluster_name)
        self._prune_old()
        return manifest

    def create_incremental(
        self,
        cluster_name: str,
        changes: dict[str, Any],
    ) -> BackupManifest | None:
        """Create an incremental backup (only changed values)."""
        backup_id = f"inc-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"
        entries = [
            BackupEntry(key, value, time.time())
            for key, value in changes.items()
        ]
        if not entries:
            return None
        manifest = self._write_backup(backup_id, entries, "incremental", cluster_name)
        self._prune_old()
        return manifest

    def list_backups(self) -> list[BackupManifest]:
        """List available backups sorted by creation time (newest first)."""
        manifests: list[BackupManifest] = []
        for manifest_file in sorted(
            self._backup_dir.glob("*.manifest.json"),
            reverse=True,
        ):
            try:
                data = json.loads(manifest_file.read_text())
                manifests.append(BackupManifest(**data))
            except Exception:
                continue
        return manifests

    def get_backup(self, backup_id: str) -> dict[str, Any] | None:
        """Restore a specific backup by ID.

        Returns a dict mapping entry keys to their data.
        """
        archive_path = self._backup_dir / f"{backup_id}.backup"
        if not archive_path.exists():
            return None

        try:
            raw = archive_path.read_bytes()
            if raw[:2] == b"\x1f\x8b":  # gzip magic
                raw = gzip.decompress(raw)
            data: dict[str, Any] = {}
            for line in raw.decode("utf-8").strip().split("\n"):
                entry = json.loads(line)
                data[entry["key"]] = entry["data"]
            return data
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            return None

    def restore(self, backup_id: str) -> dict[str, Any] | None:
        """Alias for ``get_backup()``."""
        return self.get_backup(backup_id)

    def restore_latest(self) -> dict[str, Any] | None:
        """Restore the most recent full backup."""
        full_backups = [
            b for b in self.list_backups() if b.backup_type == "full"
        ]
        if not full_backups:
            logger.warning("No full backups available")
            return None
        return self.get_backup(full_backups[0].backup_id)

    def delete_backup(self, backup_id: str) -> bool:
        """Remove a backup and its manifest."""
        for p in [
            self._backup_dir / f"{backup_id}.backup",
            self._backup_dir / f"{backup_id}.manifest.json",
        ]:
            if p.exists():
                p.unlink()
        return True

    def backup_size_bytes(self) -> int:
        """Total size of all backup files."""
        total = 0
        for p in self._backup_dir.glob("*.backup"):
            total += p.stat().st_size
        for p in self._backup_dir.glob("*.manifest.json"):
            total += p.stat().st_size
        return total

    # ── Private helpers ─────────────────────────────────────────────────

    def _write_backup(
        self, backup_id: str, entries: list[BackupEntry],
        backup_type: str, cluster_name: str,
    ) -> BackupManifest:
        archive_path = self._backup_dir / f"{backup_id}.backup"

        lines = []
        for entry in entries:
            lines.append(json.dumps({
                "key": entry.key,
                "data": entry.data,
                "timestamp": entry.timestamp,
                "content_type": entry.content_type,
            }, default=str))
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        compressed = gzip.compress(raw)
        archive_path.write_bytes(compressed)

        node_count = 0
        model_name = ""
        for entry in entries:
            if entry.key == "node_registrations" and isinstance(entry.data, list):
                node_count = len(entry.data)
            if entry.key == "coordinator_config" and isinstance(entry.data, dict):
                model_name = entry.data.get("model_name", "")

        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=time.time(),
            size_bytes=len(compressed),
            entries=len(entries),
            backup_type=backup_type,
            cluster_name=cluster_name,
            model_name=model_name,
            node_count=node_count,
        )
        manifest_path = self._backup_dir / f"{backup_id}.manifest.json"
        manifest_path.write_text(json.dumps({
            "backup_id": manifest.backup_id,
            "created_at": manifest.created_at,
            "size_bytes": manifest.size_bytes,
            "entries": manifest.entries,
            "backup_type": manifest.backup_type,
            "cluster_name": manifest.cluster_name,
            "model_name": manifest.model_name,
            "node_count": manifest.node_count,
            "version": manifest.version,
        }, indent=2))
        logger.info(f"Created {backup_type} backup {backup_id} "
                    f"({len(compressed)} bytes, {len(entries)} entries)")
        return manifest

    def _prune_old(self) -> None:
        manifests = sorted(
            self._backup_dir.glob("*.manifest.json"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(manifests) > self._max_backups:
            oldest = manifests.pop(0)
            backup_id = oldest.stem.replace(".manifest", "")
            self.delete_backup(backup_id)
            logger.info(f"Pruned old backup {backup_id}")


class AutoBackup:
    """Background auto-backup with configurable schedule and retention.

    Runs periodic backups of the coordinator state and prunes old
    backups based on retention policy.

    Usage::

        auto = AutoBackup(
            backup_manager=mgr,
            interval_hours=6,
            retention_days=7,
        )
        auto.start()
        # ... runs in background ...
        auto.stop()
    """

    def __init__(
        self,
        backup_manager: BackupManager,
        interval_hours: float = 6.0,
        retention_days: int = 7,
        get_state_fn: Callable[[], dict] | None = None,
    ):
        self._mgr = backup_manager
        self._interval_s = interval_hours * 3600
        self._retention_days = retention_days
        self._get_state = get_state_fn
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the auto-backup background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="auto-backup",
        )
        self._thread.start()
        logger.info(f"Auto-backup started (interval={self._interval_s}s, retention={self._retention_days}d)")

    def stop(self) -> None:
        """Stop the auto-backup background thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while self._running:
            try:
                self._run_backup()
                self._prune_expired()
            except Exception as e:
                logger.warning(f"Auto-backup failed: {e}")

            # Sleep in small increments for responsive shutdown
            deadline = time.time() + self._interval_s
            while self._running and time.time() < deadline:
                time.sleep(1.0)

    def _run_backup(self) -> None:
        """Run a single backup cycle."""
        if self._get_state:
            state = self._get_state()
            self._mgr.create_full(
                cluster_name=state.get("cluster_name", "auto"),
                coordinator_config=state.get("config", {}),
                node_registrations=state.get("nodes", []),
                model_assignments=state.get("models", []),
            )
        else:
            logger.debug("Auto-backup: no state function configured, skipping")

    def _prune_expired(self) -> None:
        """Remove backups older than retention period."""
        cutoff = time.time() - (self._retention_days * 86400)
        for backup in self._mgr.list_backups():
            if backup.created_at < cutoff:
                self._mgr.delete_backup(backup.backup_id)
                logger.info(f"Pruned expired backup {backup.backup_id}")

    def stats(self) -> dict:
        return {
            "running": self._running,
            "interval_hours": self._interval_s / 3600,
            "retention_days": self._retention_days,
            "total_backups": len(self._mgr.list_backups()),
        }
