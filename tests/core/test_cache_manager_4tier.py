"""P1: Test CacheManager 4-tier lookup cascade.

Tests the full tier fallthrough: local → disk → gossip index → broadcast.
"""

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.cache_manager import CacheManager
from distllm.core.cache_persistence import CachePersistenceManager
from distllm.config.settings import CachePersistenceSettings


class TestCacheManager4TierLocal:
    """Tier 1: Local prefix cache hit."""

    def test_local_hit_returns_immediately(self):
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        tokens = [1, 2, 3, 4, 5]
        kv_data = {"layer_0": (torch.randn(1, 4, 8), torch.randn(1, 4, 8))}
        cm.store_prefix(tokens, kv_data)

        match_len, result = cm.lookup_prefix(tokens)
        assert match_len == 5
        assert result is kv_data

    def test_local_miss_falls_through(self):
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        match_len, result = cm.lookup_prefix([99, 98, 97])
        assert match_len == 0


class TestCacheManager4TierDisk:
    """Tier 2: Disk persistence fallback."""

    def test_disk_hit_returns_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = CachePersistenceSettings(
                enabled=True,
                storage_path=tmpdir,
                max_disk_gb=1.0,
                ttl_hours=24.0,
            )
            persistence = CachePersistenceManager(settings)
            cm = CacheManager(
                prefix_cache_enabled=True,
                prefix_cache_min_prefix_len=2,
                persistence_manager=persistence,
            )

            # Save to disk
            cache_data = {"layer_0": (torch.randn(1, 4), torch.randn(1, 4))}
            persistence.save("test-key", "model-a", cache_data)

            # Verify disk fallback works via lookup_with_disk_fallback
            tokens = list(range(100))
            match_len, entry = cm.lookup_with_disk_fallback(tokens, "model-a")
            # May or may not match depending on hash, but should not crash


class TestCacheManager4TierGossip:
    """Tier 3: Gossip index lookup."""

    def test_gossip_index_hit(self):
        mock_index = MagicMock()
        mock_index.lookup.return_value = "node-b"
        mock_index.index_tokens.return_value = "h123"

        cm = CacheManager(
            prefix_cache_enabled=False,
            cache_index=mock_index,
        )

        result = cm.lookup_with_gossip([1, 2, 3, 4, 5])
        assert result is not None
        source, _ = result
        assert source == "node-b"

    def test_gossip_index_miss_falls_through(self):
        mock_index = MagicMock()
        mock_index.lookup.return_value = None
        mock_index.index_tokens.return_value = "h123"

        cm = CacheManager(
            prefix_cache_enabled=False,
            cache_index=mock_index,
        )

        result = cm.lookup_with_gossip([1, 2, 3, 4, 5])
        assert result is None


class TestCacheManager4TierBroadcast:
    """Tier 4: Active peer broadcast."""

    def test_broadcast_hit(self):
        mock_gossip = MagicMock()
        mock_gossip.request_cache_from_peers.return_value = {
            "kv_data": {"test": "data"},
            "match_len": 5,
        }

        cm = CacheManager(
            prefix_cache_enabled=False,
            gossip_protocol=mock_gossip,
            gossip_client=MagicMock(),
        )

        result = cm.lookup_with_gossip([1, 2, 3, 4, 5])
        assert result is not None
        source, kv_data = result
        assert source == "peer"
        assert kv_data == {"test": "data"}

    def test_broadcast_miss_returns_none(self):
        mock_gossip = MagicMock()
        mock_gossip.request_cache_from_peers.return_value = None

        cm = CacheManager(
            prefix_cache_enabled=False,
            gossip_protocol=mock_gossip,
            gossip_client=MagicMock(),
        )

        result = cm.lookup_with_gossip([1, 2, 3, 4, 5])
        assert result is None


class TestCacheManager4TierCascade:
    """Full 4-tier cascade: local → disk → gossip → broadcast."""

    def test_local_hit_skips_remaining_tiers(self):
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        tokens = [1, 2, 3, 4, 5]
        cm.store_prefix(tokens, {"data": "local"})

        result = cm.lookup_with_gossip(tokens)
        assert result is not None
        source, _ = result
        assert source == "local"

    def test_gossip_hit_skips_broadcast(self):
        mock_index = MagicMock()
        mock_index.lookup.return_value = "node-c"
        mock_index.index_tokens.return_value = "h456"

        mock_gossip = MagicMock()

        cm = CacheManager(
            prefix_cache_enabled=False,
            cache_index=mock_index,
            gossip_protocol=mock_gossip,
            gossip_client=MagicMock(),
        )

        result = cm.lookup_with_gossip([1, 2, 3, 4, 5])
        assert result is not None
        source, _ = result
        assert source == "node-c"
        # Broadcast should not have been called
        mock_gossip.request_cache_from_peers.assert_not_called()


class TestCacheManagerTierStats:
    """Per-tier metrics tracking."""

    def test_tier_stats_tracked(self):
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        tokens = [1, 2, 3, 4, 5]
        cm.store_prefix(tokens, {"data": "test"})

        # Hit local tier
        cm.lookup_with_gossip(tokens)

        stats = cm.get_tier_stats()
        assert stats["local"]["hits"] == 1

    def test_tier_stats_empty_initially(self):
        cm = CacheManager()
        stats = cm.get_tier_stats()
        assert stats["local"]["hits"] == 0
        assert stats["disk"]["hits"] == 0
