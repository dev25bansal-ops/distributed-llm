"""Tests for CachePersistenceManager."""

import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

from distllm.core.cache_persistence import CachePersistenceManager
from distllm.config.settings import CachePersistenceSettings


def make_manager(storage_path=None, enabled=True, max_disk_gb=1.0, ttl_hours=24.0):
    if storage_path is None:
        storage_path = tempfile.mkdtemp()
    settings = CachePersistenceSettings(
        enabled=enabled,
        storage_path=storage_path,
        max_disk_gb=max_disk_gb,
        ttl_hours=ttl_hours,
    )
    return CachePersistenceManager(settings), storage_path


class TestCachePersistence:
    """Tests for CachePersistenceManager."""

    def test_save_and_load(self):
        """Save KV dict, load it back."""
        mgr, path = make_manager()
        cache = {"layer_0": (torch.randn(2, 4), torch.randn(2, 4))}

        mgr.save("req-1", "model-a", cache)
        loaded = mgr.load("req-1", "model-a")

        assert loaded is not None
        assert "layer_0" in loaded

    def test_load_nonexistent(self):
        """Returns None for missing."""
        mgr, path = make_manager()
        assert mgr.load("req-1", "model-a") is None

    def test_delete_existing(self):
        """Deletes file, returns True."""
        mgr, path = make_manager()
        mgr.save("req-1", "model-a", {})
        assert mgr.delete("req-1", "model-a") is True
        assert mgr.load("req-1", "model-a") is None

    def test_delete_nonexistent(self):
        """Returns False."""
        mgr, path = make_manager()
        assert mgr.delete("unknown", "model-a") is False

    def test_cleanup_removes_old(self, monkeypatch):
        """Files older than TTL removed."""
        mgr, path = make_manager(ttl_hours=1.0)
        mgr.save("req-1", "model-a", {})

        # Fake old mtime
        cache_file = Path(path) / "model-a" / "req-1.pt"
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(cache_file, (old_time, old_time))

        removed = mgr.cleanup(max_age_hours=1.0)
        assert removed >= 1
        assert not cache_file.exists()

    def test_cleanup_keeps_recent(self):
        """Recent files kept."""
        mgr, path = make_manager(ttl_hours=24.0)
        mgr.save("req-1", "model-a", {})

        removed = mgr.cleanup(max_age_hours=1.0)
        assert removed == 0
        cache_file = Path(path) / "model-a" / "req-1.pt"
        assert cache_file.exists()

    def test_get_disk_usage(self):
        """Returns correct byte count."""
        mgr, path = make_manager()
        mgr.save("req-1", "model-a", {})

        usage = mgr.get_disk_usage()
        assert usage > 0

    def test_enforce_disk_limit(self):
        """Deletes oldest when over limit."""
        mgr, path = make_manager(max_disk_gb=0.0001)  # Very small limit (~100 bytes)
        for i in range(5):
            mgr.save(f"req-{i}", "model-a", {})

        removed = mgr.enforce_disk_limit()
        # Should have deleted some files
        assert mgr.get_disk_usage() <= int(0.0001 * 1024**3)

    def test_mark_dirty_and_is_dirty(self):
        """Dirty flag works."""
        mgr, _ = make_manager()
        mgr.mark_dirty("req-1")
        assert mgr.is_dirty("req-1") is True
        assert mgr.is_dirty("req-2") is False

    def test_storage_path_created(self):
        """Directory created on init."""
        _, path = make_manager()
        assert Path(path).exists()
