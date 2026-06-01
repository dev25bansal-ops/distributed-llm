"""KV cache backup and restore for planned maintenance.

Snapshots KV cache state to disk and restores it after restart,
enabling zero-downtime maintenance windows.

Usage::

    backup = KVBackupManager(backup_dir="/var/lib/distllm/backups")
    backup_id = backup.create_snapshot(kv_cache_manager)
    # ... maintenance ...
    backup.restore(backup_id, kv_cache_manager)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger


@dataclass
class BackupManifest:
    """Manifest for a KV cache backup."""
    backup_id: str
    created_at: float
    num_entries: int
    total_bytes: int
    metadata: dict = field(default_factory=dict)


class KVBackupManager:
    """Manages KV cache snapshots for backup and restore."""

    def __init__(self, backup_dir: str = ".distllm_backups"):
        self._backup_dir = Path(backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: dict[str, BackupManifest] = {}
        self._load_manifests()

    def create_snapshot(
        self,
        kv_cache_manager: Any,
        metadata: dict | None = None,
    ) -> str:
        """Create a snapshot of all KV caches.

        Args:
            kv_cache_manager: KVCacheManager instance.
            metadata: Optional metadata to store with the backup.

        Returns:
            Backup ID.
        """
        backup_id = str(uuid.uuid4())[:8]
        backup_path = self._backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        num_entries = 0

        # Iterate over all caches in the manager
        caches = getattr(kv_cache_manager, "caches", {})
        for request_id, cache in caches.items():
            try:
                cache_data = {
                    "request_id": request_id,
                    "num_layers": getattr(cache, "num_layers", 0),
                    "sequence_length": getattr(cache, "sequence_length", 0),
                }

                # Save cache tensors
                cache_path = backup_path / f"{request_id}.pt"
                cache_list = cache.get_all() if hasattr(cache, "get_all") else []
                if cache_list:
                    torch.save(cache_list, cache_path)
                    cache_data["tensor_file"] = f"{request_id}.pt"
                    total_bytes += os.path.getsize(cache_path)

                # Save metadata
                meta_path = backup_path / f"{request_id}.json"
                with open(meta_path, "w") as f:
                    json.dump(cache_data, f)

                num_entries += 1

            except Exception as e:
                logger.warning(f"Failed to backup cache for {request_id}: {e}")

        # Save manifest
        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=time.time(),
            num_entries=num_entries,
            total_bytes=total_bytes,
            metadata=metadata or {},
        )
        self._save_manifest(manifest)
        self._manifests[backup_id] = manifest

        logger.info(
            f"KV cache snapshot created: {backup_id} "
            f"({num_entries} entries, {total_bytes / (1024**2):.1f} MB)"
        )
        return backup_id

    def restore(
        self,
        backup_id: str,
        kv_cache_manager: Any,
    ) -> int:
        """Restore KV caches from a snapshot.

        Args:
            backup_id: Backup ID from create_snapshot.
            kv_cache_manager: KVCacheManager instance to restore into.

        Returns:
            Number of entries restored.
        """
        manifest = self._manifests.get(backup_id)
        if manifest is None:
            raise ValueError(f"Backup {backup_id} not found")

        backup_path = self._backup_dir / backup_id
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_path}")

        restored = 0
        for meta_file in backup_path.glob("*.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)

                request_id = meta["request_id"]
                tensor_file = meta.get("tensor_file")

                if tensor_file:
                    tensor_path = backup_path / tensor_file
                    if tensor_path.exists():
                        cache_data = torch.load(tensor_path, map_location="cpu", weights_only=True)
                        # Restore into the manager
                        if hasattr(kv_cache_manager, "caches"):
                            # Create a new KVCache and populate it
                            from distllm.core.kv_cache import KVCache
                            cache = KVCache()
                            cache.set_all(cache_data)
                            kv_cache_manager.caches[request_id] = cache
                        restored += 1

            except Exception as e:
                logger.warning(f"Failed to restore {meta_file.name}: {e}")

        logger.info(f"KV cache restored from {backup_id}: {restored} entries")
        return restored

    def list_backups(self) -> list[dict]:
        """List all available backups."""
        return [
            {
                "backup_id": m.backup_id,
                "created_at": m.created_at,
                "num_entries": m.num_entries,
                "total_bytes": m.total_bytes,
                "metadata": m.metadata,
            }
            for m in self._manifests.values()
        ]

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        import shutil

        backup_path = self._backup_dir / backup_id
        if backup_path.exists():
            shutil.rmtree(backup_path)

        manifest = self._manifests.pop(backup_id, None)
        if manifest:
            self._delete_manifest(backup_id)
            return True
        return False

    def _save_manifest(self, manifest: BackupManifest) -> None:
        """Save manifest to disk."""
        manifest_path = self._backup_dir / f"{manifest.backup_id}.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "backup_id": manifest.backup_id,
                "created_at": manifest.created_at,
                "num_entries": manifest.num_entries,
                "total_bytes": manifest.total_bytes,
                "metadata": manifest.metadata,
            }, f, indent=2)

    def _load_manifests(self) -> None:
        """Load all manifests from disk."""
        for manifest_file in self._backup_dir.glob("*.manifest.json"):
            try:
                with open(manifest_file) as f:
                    data = json.load(f)
                manifest = BackupManifest(**data)
                self._manifests[manifest.backup_id] = manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest {manifest_file.name}: {e}")

    def _delete_manifest(self, backup_id: str) -> None:
        """Delete manifest file."""
        manifest_path = self._backup_dir / f"{backup_id}.manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
