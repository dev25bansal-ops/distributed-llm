"""KV cache persistence manager for disk-backed cache storage."""

import time
import threading
from pathlib import Path

import torch
from loguru import logger

from distllm.config.settings import CachePersistenceSettings


class CachePersistenceManager:
    """Manages KV cache persistence to disk.

    Stores caches as .pt files organized by model name.
    Enforces disk size limits and TTL-based cleanup.
    """

    def __init__(self, settings: CachePersistenceSettings):
        self._settings = settings
        self._storage_path = Path(settings.storage_path)
        self._lock = threading.Lock()
        self._dirty_caches: dict[str, bool] = {}
        if settings.enabled:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    def _cache_dir(self, model_name: str) -> Path:
        """Get or create model directory."""
        d = self._storage_path / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cache_path(self, request_id: str, model_name: str) -> Path:
        """Get cache file path."""
        return self._cache_dir(model_name) / f"{request_id}.pt"

    def save(self, request_id: str, model_name: str, kv_cache_dict: dict) -> None:
        """Save a KV cache dict to disk."""
        path = self._cache_path(request_id, model_name)
        torch.save(kv_cache_dict, path)
        with self._lock:
            self._dirty_caches[request_id] = False
        logger.debug(f"Saved cache {request_id} to {path}")

    def load(self, request_id: str, model_name: str) -> Optional[dict]:
        """Load a KV cache dict from disk."""
        path = self._cache_path(request_id, model_name)
        if not path.exists():
            return None
        # Security: weights_only=True prevents arbitrary code execution via pickle
        return torch.load(path, weights_only=True)

    def delete(self, request_id: str, model_name: str) -> bool:
        """Delete a cache file. Returns True if deleted."""
        path = self._cache_path(request_id, model_name)
        if path.exists():
            path.unlink()
            with self._lock:
                self._dirty_caches.pop(request_id, None)
            return True
        return False

    def cleanup(self, max_age_hours: float | None = None) -> int:
        """Remove cache files older than max_age_hours."""
        max_age = max_age_hours or self._settings.ttl_hours
        cutoff = time.time() - (max_age * 3600)
        removed = 0
        if not self._storage_path.exists():
            return 0
        for model_dir in self._storage_path.iterdir():
            if not model_dir.is_dir():
                continue
            for cache_file in model_dir.glob("*.pt"):
                if cache_file.stat().st_mtime < cutoff:
                    cache_file.unlink()
                    removed += 1
        if removed:
            logger.info(f"Cache cleanup: removed {removed} stale files")
        return removed

    def get_disk_usage(self) -> int:
        """Get total disk usage in bytes."""
        total = 0
        if self._storage_path.exists():
            for f in self._storage_path.rglob("*.pt"):
                total += f.stat().st_size
        return total

    def enforce_disk_limit(self) -> int:
        """Delete oldest files until under max_disk_gb limit."""
        max_bytes = int(self._settings.max_disk_gb * 1024**3)
        removed = 0
        while self.get_disk_usage() > max_bytes:
            oldest = None
            oldest_time = float('inf')
            for f in self._storage_path.rglob("*.pt"):
                mtime = f.stat().st_mtime
                if mtime < oldest_time:
                    oldest = f
                    oldest_time = mtime
            if oldest:
                oldest.unlink()
                removed += 1
            else:
                break
        if removed:
            logger.info(f"Disk limit enforcement: removed {removed} files")
        return removed

    def mark_dirty(self, request_id: str) -> None:
        """Mark a cache as dirty (modified since last save)."""
        with self._lock:
            self._dirty_caches[request_id] = True

    def is_dirty(self, request_id: str) -> bool:
        """Check if a cache is dirty."""
        with self._lock:
            return self._dirty_caches.get(request_id, False)
