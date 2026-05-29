"""P1: Test CacheManager thread safety.

Verifies concurrent lookups, stores, and mixed operations don't crash
or produce inconsistent state.
"""

import threading
import time

import pytest
import torch

from distllm.core.cache_manager import CacheManager


class TestCacheManagerConcurrency:
    """Thread safety tests for CacheManager."""

    def test_concurrent_stores_no_crash(self):
        """10 threads storing different keys should not crash."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        errors = []

        def store_worker(thread_id):
            try:
                for i in range(50):
                    tokens = [thread_id, i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    kv_data = {"thread": thread_id, "seq": i}
                    cm.store_prefix(tokens, kv_data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_lookups_no_crash(self):
        """10 threads doing concurrent lookups should not crash."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)

        # Pre-populate
        for i in range(100):
            tokens = [i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            cm.store_prefix(tokens, {"idx": i})

        errors = []
        results = []

        def lookup_worker(thread_id):
            try:
                for i in range(100):
                    tokens = [i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    match_len, data = cm.lookup_prefix(tokens)
                    results.append((thread_id, i, match_len))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=lookup_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000

    def test_concurrent_mixed_ops_no_crash(self):
        """Mixed stores and lookups from multiple threads."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        errors = []

        def mixed_worker(thread_id):
            try:
                for i in range(50):
                    tokens = [thread_id, i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    if i % 2 == 0:
                        cm.store_prefix(tokens, {"t": thread_id, "i": i})
                    else:
                        cm.lookup_prefix(tokens)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_gossip_lookup_no_crash(self):
        """Concurrent gossip lookups should not deadlock."""
        from unittest.mock import MagicMock

        mock_gossip = MagicMock()
        mock_gossip.request_cache_from_peers.return_value = None

        cm = CacheManager(
            prefix_cache_enabled=True,
            prefix_cache_min_prefix_len=2,
            gossip_protocol=mock_gossip,
            gossip_client=MagicMock(),
        )
        errors = []

        def gossip_worker(thread_id):
            try:
                for i in range(20):
                    tokens = [thread_id, i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    cm.lookup_with_gossip(tokens)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=gossip_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_tier_stats_no_crash(self):
        """Reading tier stats while other threads do lookups."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        errors = []
        stop = threading.Event()

        def lookup_worker():
            try:
                while not stop.is_set():
                    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    cm.lookup_prefix(tokens)
            except Exception as e:
                errors.append(e)

        def stats_reader():
            try:
                while not stop.is_set():
                    cm.get_tier_stats()
                    cm.get_tier_latencies()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=lookup_worker),
            threading.Thread(target=stats_reader),
        ]
        for t in threads:
            t.start()

        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join()

        assert len(errors) == 0
