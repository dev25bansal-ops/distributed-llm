"""KV cache persistence manager for disk-based caching.

Saves and loads KV cache tensors to/from disk with TTL-based expiry
and disk limit enforcement.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from loguru import logger

from distllm.config.settings import CachePersistenceSettings


class CachePersistenceManager:
    """Manages KV cache persistence to disk.

    Stores cache entries as .pt files organized by model directory.
    Supports TTL-based expiry and disk usage limits with LRU eviction.
    """

    def __init__(self, settings: CachePersistenceSettings):
        self._enabled = settings.enabled
        self._storage_path = Path(settings.storage_path)
        self._max_disk_bytes = int(settings.max_disk_gb * 1024**3)
        self._ttl_seconds = settings.ttl_hours * 3600
        self._dirty: set[str] = set()

        if self._enabled:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, request_id: str, model_name: str) -> Path:
        return self._storage_path / model_name / f"{request_id}.pt"

    def save(self, request_id: str, model_name: str, cache_data: dict) -> None:
        """Save cache data to disk."""
        if not self._enabled:
            return

        path = self._get_path(request_id, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, path)
        self._dirty.discard(request_id)
        logger.debug(f"Saved cache to {path}")

    def load(self, request_id: str, model_name: str) -> dict | None:
        """Load cache data from disk, respecting TTL."""
        if not self._enabled:
            return None

        path = self._get_path(request_id, model_name)
        if not path.exists():
            return None

        # Check TTL (ttl_hours=0.0 means expire immediately)
        mtime = path.stat().st_mtime
        age = time.time() - mtime
        if age > self._ttl_seconds:
            logger.debug(f"Cache expired for {request_id} (age={age:.0f}s > ttl={self._ttl_seconds:.0f}s)")
            path.unlink(missing_ok=True)
            return None

        try:
            data = torch.load(path, weights_only=True, map_location="cpu")
            logger.debug(f"Loaded cache from {path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache from {path}: {e}")
            path.unlink(missing_ok=True)
            return None

    def delete(self, request_id: str, model_name: str) -> bool:
        """Delete a cache entry from disk."""
        path = self._get_path(request_id, model_name)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted cache {path}")
            return True
        return False

    def mark_dirty(self, request_id: str) -> None:
        """Mark a cache entry as dirty (needing persistence)."""
        self._dirty.add(request_id)

    def is_dirty(self, request_id: str) -> bool:
        """Check if a cache entry is dirty."""
        return request_id in self._dirty

    def cleanup(self, max_age_hours: float | None = None) -> int:
        """Remove expired cache files.

        Args:
            max_age_hours: Override TTL for this cleanup. Uses configured TTL if None.

        Returns:
            Number of files removed.
        """
        if not self._enabled:
            return 0

        max_age_seconds = (max_age_hours * 3600) if max_age_hours is not None else self._ttl_seconds
        now = time.time()
        removed = 0

        for pt_file in self._storage_path.rglob("*.pt"):
            try:
                age = now - pt_file.stat().st_mtime
                if age > max_age_seconds:
                    pt_file.unlink()
                    removed += 1
            except OSError:
                pass

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired cache files")
        return removed

    def get_disk_usage(self) -> int:
        """Get current disk usage in bytes."""
        total = 0
        for pt_file in self._storage_path.rglob("*.pt"):
            try:
                total += pt_file.stat().st_size
            except OSError:
                pass
        return total

    def enforce_disk_limit(self) -> int:
        """Delete oldest files until under disk limit.

        Returns:
            Number of files removed.
        """
        if not self._enabled:
            return 0

        current_usage = self.get_disk_usage()
        if current_usage <= self._max_disk_bytes:
            return 0

        # Collect all .pt files with their mtime
        files: list[tuple[float, Path]] = []
        for pt_file in self._storage_path.rglob("*.pt"):
            try:
                files.append((pt_file.stat().st_mtime, pt_file))
            except OSError:
                pass

        # Sort oldest first
        files.sort(key=lambda x: x[0])

        removed = 0
        for _, pt_file in files:
            if current_usage <= self._max_disk_bytes:
                break
            try:
                size = pt_file.stat().st_size
                pt_file.unlink()
                current_usage -= size
                removed += 1
            except OSError:
                pass

        if removed > 0:
            logger.info(f"Evicted {removed} cache files to enforce disk limit")
        return removed

    def start_background_compaction(self, interval_s: float = 300.0) -> None:
        """E4: Start background disk compaction job.

        Periodically cleans up expired files and enforces disk limits.
        """
        import threading

        self._compaction_running = True

        def _compaction_loop():
            while getattr(self, '_compaction_running', False):
                try:
                    self.cleanup()
                    self.enforce_disk_limit()
                except Exception as e:
                    logger.warning(f"Background compaction failed: {e}")
                import time
                time.sleep(interval_s)

        self._compaction_thread = threading.Thread(target=_compaction_loop, daemon=True)
        self._compaction_thread.start()
        logger.info(f"Background disk compaction started (interval={interval_s}s)")

    def stop_background_compaction(self) -> None:
        """Stop background disk compaction."""
        self._compaction_running = False
        if hasattr(self, '_compaction_thread'):
            self._compaction_thread.join(timeout=5)
        logger.info("Background disk compaction stopped")
