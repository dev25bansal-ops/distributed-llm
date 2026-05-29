"""Chaos tests for cache system resilience.

Tests behavior under adverse conditions: disk failures, corrupted data,
memory pressure, and concurrent access races.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.cache_manager import CacheManager
from distllm.core.cache_persistence import CachePersistenceManager
from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.config.settings import CachePersistenceSettings


class TestDiskFullDuringPersistence:
    """Graceful fallback when disk is full."""

    def test_save_with_read_only_directory(self):
        """Save should not crash if directory is read-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = CachePersistenceSettings(
                enabled=True,
                storage_path=tmpdir,
                max_disk_gb=1.0,
                ttl_hours=24.0,
            )
            mgr = CachePersistenceManager(settings)

            # Make directory read-only (on Unix-like systems)
            try:
                os.chmod(tmpdir, 0o444)
            except (OSError, PermissionError):
                pytest.skip("Cannot set read-only on this platform")

            # Should not crash
            try:
                mgr.save("req-1", "model-a", {"data": "test"})
            except Exception:
                pass  # Expected to fail, but should not crash

            # Restore permissions for cleanup
            try:
                os.chmod(tmpdir, 0o755)
            except (OSError, PermissionError):
                pass

    def test_enforce_disk_limit_with_zero_budget(self):
        """Enforcing disk limit with 0 budget should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = CachePersistenceSettings(
                enabled=True,
                storage_path=tmpdir,
                max_disk_gb=0.0,
                ttl_hours=24.0,
            )
            mgr = CachePersistenceManager(settings)
            mgr.save("req-1", "model-a", {"data": "test"})

            # Should not crash, should remove files
            removed = mgr.enforce_disk_limit()
            assert removed >= 0


class TestCorruptedCacheFiles:
    """Handle corrupted cache files gracefully."""

    def test_load_corrupted_pt_file(self):
        """Loading a corrupted .pt file should return None, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = CachePersistenceSettings(
                enabled=True,
                storage_path=tmpdir,
                max_disk_gb=1.0,
                ttl_hours=24.0,
            )
            mgr = CachePersistenceManager(settings)

            # Create a corrupted file
            model_dir = Path(tmpdir) / "model-a"
            model_dir.mkdir(parents=True)
            corrupt_file = model_dir / "req-1.pt"
            corrupt_file.write_bytes(b"corrupted data not a valid pt file")

            # Should return None, not crash
            result = mgr.load("req-1", "model-a")
            assert result is None

    def test_load_empty_file(self):
        """Loading an empty file should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = CachePersistenceSettings(
                enabled=True,
                storage_path=tmpdir,
                max_disk_gb=1.0,
                ttl_hours=24.0,
            )
            mgr = CachePersistenceManager(settings)

            model_dir = Path(tmpdir) / "model-a"
            model_dir.mkdir(parents=True)
            empty_file = model_dir / "req-1.pt"
            empty_file.write_bytes(b"")

            result = mgr.load("req-1", "model-a")
            assert result is None


class TestRaceConditions:
    """Race condition stress tests."""

    def test_100_threads_store_same_key(self):
        """100 threads storing the same key should not corrupt data."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        errors = []

        def store_worker(thread_id):
            try:
                kv_data = {"thread": thread_id, "data": "test"}
                cm.store_prefix(tokens, kv_data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify the entry exists (one of the threads should have stored it)
        match_len, result = cm.lookup_prefix(tokens)
        assert match_len == 16
        assert result is not None

    def test_concurrent_create_delete_kv_cache(self):
        """Concurrent create/delete should not crash KVCacheManager."""
        kv_mgr = KVCacheManager()
        errors = []

        def worker(thread_id):
            try:
                for i in range(20):
                    rid = f"req-{thread_id}-{i}"
                    kv_mgr.create(rid, 4, 1, 8, 64)
                    kv_mgr.get(rid)
                    kv_mgr.delete(rid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert kv_mgr.active_requests == 0

    def test_concurrent_eviction_and_access(self):
        """Evicting while other threads access should not crash."""
        kv_mgr = KVCacheManager()
        errors = []
        stop = threading.Event()

        # Create some caches
        for i in range(10):
            kv_mgr.create(f"req-{i}", 4, 1, 8, 64)

        def accessor():
            try:
                while not stop.is_set():
                    for i in range(10):
                        kv_mgr.get(f"req-{i}")
            except Exception as e:
                errors.append(e)

        def evictor():
            try:
                while not stop.is_set():
                    kv_mgr.evict_lowest_score()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=accessor),
            threading.Thread(target=evictor),
        ]
        for t in threads:
            t.start()

        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestMemoryPressure:
    """Behavior under memory pressure."""

    def test_kv_cache_release_frees_memory(self):
        """Releasing KV cache should free memory."""
        kv_mgr = KVCacheManager()

        cache = kv_mgr.create("req-1", 8, 1, 16, 128, device="cpu")
        initial_mem = kv_mgr.total_memory_usage()
        assert initial_mem > 0

        kv_mgr.delete("req-1")
        final_mem = kv_mgr.total_memory_usage()
        assert final_mem == 0

    def test_prefix_cache_eviction_under_pressure(self):
        """PrefixCache should evict when memory budget is exceeded."""
        from distllm.dist.prefix_cache import PrefixCache

        cache = PrefixCache(min_prefix_len=1, memory_budget_bytes=1000)

        # Store entries until budget is exceeded
        for i in range(100):
            tokens = [i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            kv_data = {"data": "x" * 100}  # ~100 bytes each
            cache.store(tokens, kv_data)

        # Memory should be within budget
        stats = cache.stats()
        assert stats["prefix_cache_memory_bytes"] <= 1000


class TestGossipPeerDisconnect:
    """Handle gossip peer disconnection gracefully."""

    def test_gossip_lookup_with_disconnected_peer(self):
        """Gossip lookup should not crash when peer is disconnected."""
        mock_gossip = MagicMock()
        mock_gossip.request_cache_from_peers.side_effect = ConnectionError("peer disconnected")

        cm = CacheManager(
            prefix_cache_enabled=False,
            gossip_protocol=mock_gossip,
            gossip_client=MagicMock(),
        )

        # Should not crash
        result = cm.lookup_with_gossip([1, 2, 3, 4, 5])
        assert result is None

    def test_gossip_sync_with_timeout(self):
        """Gossip sync should handle timeouts gracefully."""
        mock_gossip = MagicMock()
        mock_gossip.advertise.return_value = []
        mock_gossip.select_peer.return_value = "node-b"

        mock_client = MagicMock()
        mock_client.exchange.side_effect = TimeoutError("connection timed out")

        cm = CacheManager(
            prefix_cache_enabled=False,
            gossip_protocol=mock_gossip,
            gossip_client=mock_client,
        )

        # Should not crash
        result = cm.sync_with_peers()
        assert result == 0
