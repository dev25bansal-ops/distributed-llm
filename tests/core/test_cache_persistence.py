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


class TestCachePersistenceSave:
    """Save-specific tests: file creation, path correctness, dirty flag."""

    def test_save_creates_file_at_expected_path(self):
        mgr, path = make_manager()
        mgr.save("req-1", "model-a", {"k": "v"})
        expected = Path(path) / "model-a" / "req-1.pt"
        assert expected.exists()
        assert expected.is_file()

    def test_save_creates_model_directory(self):
        mgr, path = make_manager()
        mgr.save("req-1", "model-deep", {"k": "v"})
        assert (Path(path) / "model-deep").is_dir()

    def test_save_with_tensor_produces_non_empty_file(self):
        mgr, path = make_manager()
        tensor_dict = {"layer_0": (torch.randn(4, 8), torch.randn(4, 8))}
        mgr.save("req-1", "model-a", tensor_dict)
        file_size = (Path(path) / "model-a" / "req-1.pt").stat().st_size
        assert file_size > 0

    def test_save_clears_dirty_flag(self):
        mgr, _ = make_manager()
        mgr.mark_dirty("req-1")
        assert mgr.is_dirty("req-1") is True
        mgr.save("req-1", "model-a", {})
        assert mgr.is_dirty("req-1") is False

    def test_save_overwrite_replaces_file(self):
        mgr, path = make_manager()
        mgr.save("req-1", "model-a", {"version": 1})
        mgr.save("req-1", "model-a", {"version": 2})
        loaded = mgr.load("req-1", "model-a")
        assert loaded == {"version": 2}

    def test_save_multiple_models_separate_dirs(self):
        mgr, path = make_manager()
        mgr.save("req-1", "model-a", {"data": "a"})
        mgr.save("req-1", "model-b", {"data": "b"})
        assert (Path(path) / "model-a" / "req-1.pt").exists()
        assert (Path(path) / "model-b" / "req-1.pt").exists()
        assert mgr.load("req-1", "model-a") == {"data": "a"}
        assert mgr.load("req-1", "model-b") == {"data": "b"}


class TestCachePersistenceLoad:
    """Load-specific tests: data fidelity, missing paths, error handling."""

    def test_load_restores_exact_tensor_values(self):
        mgr, _ = make_manager()
        key_t = torch.randn(2, 4)
        val_t = torch.randn(2, 4)
        original = {"layer_0": (key_t, val_t)}
        mgr.save("req-1", "model-a", original)
        loaded = mgr.load("req-1", "model-a")
        loaded_k, loaded_v = loaded["layer_0"]
        assert torch.equal(loaded_k, key_t)
        assert torch.equal(loaded_v, val_t)

    def test_load_restores_nested_dict(self):
        mgr, _ = make_manager()
        original = {
            "layer_0": (torch.randn(2, 4), torch.randn(2, 4)),
            "layer_1": (torch.randn(2, 4), torch.randn(2, 4)),
            "meta": {"seq_len": 10, "model": "test"},
        }
        mgr.save("req-1", "model-a", original)
        loaded = mgr.load("req-1", "model-a")
        assert set(loaded.keys()) == {"layer_0", "layer_1", "meta"}
        assert loaded["meta"] == {"seq_len": 10, "model": "test"}

    def test_load_nonexistent_model_dir_returns_none(self):
        mgr, path = make_manager()
        result = mgr.load("req-1", "nonexistent-model")
        assert result is None

    def test_load_nonexistent_request_id_returns_none(self):
        mgr, path = make_manager()
        mgr.save("req-1", "model-a", {})
        result = mgr.load("req-2", "model-a")
        assert result is None

    def test_load_multiple_requests_independent(self):
        mgr, _ = make_manager()
        mgr.save("req-1", "model-a", {"id": 1})
        mgr.save("req-2", "model-a", {"id": 2})
        assert mgr.load("req-1", "model-a") == {"id": 1}
        assert mgr.load("req-2", "model-a") == {"id": 2}

    def test_load_preserves_dtype_and_shape(self):
        mgr, _ = make_manager()
        t = torch.zeros(3, 5, dtype=torch.float16)
        mgr.save("req-1", "model-a", {"t": t})
        loaded = mgr.load("req-1", "model-a")
        assert loaded["t"].dtype == torch.float16
        assert loaded["t"].shape == (3, 5)


class TestCachePersistenceTTL:
    """TTL-based expiry: expired cache files return None on load."""

    def test_load_returns_none_after_ttl(self):
        mgr, path = make_manager(ttl_hours=0.0)
        mgr.save("req-1", "model-a", {"k": "v"})
        cache_file = Path(path) / "model-a" / "req-1.pt"
        old_time = time.time() - 3600
        os.utime(cache_file, (old_time, old_time))
        loaded = mgr.load("req-1", "model-a")
        assert loaded is None

    def test_load_removes_expired_file(self):
        mgr, path = make_manager(ttl_hours=0.0)
        mgr.save("req-1", "model-a", {"k": "v"})
        cache_file = Path(path) / "model-a" / "req-1.pt"
        old_time = time.time() - 3600
        os.utime(cache_file, (old_time, old_time))
        mgr.load("req-1", "model-a")
        assert not cache_file.exists()

    def test_load_returns_data_within_ttl(self):
        mgr, _ = make_manager(ttl_hours=24.0)
        mgr.save("req-1", "model-a", {"k": "v"})
        loaded = mgr.load("req-1", "model-a")
        assert loaded == {"k": "v"}

    def test_cleanup_removes_expired(self):
        mgr, path = make_manager(ttl_hours=0.0)
        mgr.save("req-1", "model-a", {"k": "v"})
        mgr.save("req-2", "model-a", {"k": "v"})
        cache_file = Path(path) / "model-a" / "req-1.pt"
        old_time = time.time() - 3600
        os.utime(cache_file, (old_time, old_time))
        removed = mgr.cleanup(max_age_hours=0.0)
        assert removed >= 1

    def test_cleanup_skips_recent_when_some_expired(self):
        mgr, path = make_manager(ttl_hours=1.0)
        mgr.save("req-1", "model-a", {"k": "v"})
        mgr.save("req-2", "model-a", {"k": "v"})
        cache_file = Path(path) / "model-a" / "req-1.pt"
        old_time = time.time() - 7200
        os.utime(cache_file, (old_time, old_time))
        removed = mgr.cleanup(max_age_hours=1.0)
        assert removed == 1
        assert (Path(path) / "model-a" / "req-2.pt").exists()


class TestCachePersistenceDiskLimit:
    """Disk limit enforcement: oldest files deleted first."""

    def test_enforce_limit_removes_oldest_first(self):
        mgr, path = make_manager(max_disk_gb=0.000001)
        huge = torch.randn(200, 200, dtype=torch.float32)
        for i in range(4):
            mgr.save(f"req-{i}", "model-a", {"x": huge})
            if i == 0:
                time.sleep(0.05)
        before = mgr.get_disk_usage()
        assert before > 0
        removed = mgr.enforce_disk_limit()
        assert removed > 0
        assert mgr.get_disk_usage() < before

    def test_enforce_limit_under_limit_no_op(self):
        mgr, path = make_manager(max_disk_gb=10.0)
        mgr.save("req-1", "model-a", {})
        removed = mgr.enforce_disk_limit()
        assert removed == 0

    def test_enforce_limit_empty_storage(self):
        mgr, _ = make_manager(max_disk_gb=0.000001)
        removed = mgr.enforce_disk_limit()
        assert removed == 0

    def test_enforce_limit_removes_correct_model(self):
        mgr, path = make_manager(max_disk_gb=0.000001)
        huge = torch.randn(200, 200, dtype=torch.float32)
        mgr.save("req-1", "model-a", {"x": huge})
        time.sleep(0.05)
        mgr.save("req-1", "model-b", {"x": huge})
        removed = mgr.enforce_disk_limit()
        assert removed >= 1
        assert not (Path(path) / "model-a" / "req-1.pt").exists()
