"""Tests for cross-cluster prefix KV cache index.

Classes under test:
  - CacheDigest              (distllm.cache.cross_cluster_prefix_index)
  - CrossClusterPrefixIndex  (distllm.cache.cross_cluster_prefix_index)
  - CacheGossipProtocol      (distllm.cache.cross_cluster_prefix_index)
"""

from __future__ import annotations

import time

from distllm.cache.cross_cluster_prefix_index import (
    CacheDigest,
    CacheGossipProtocol,
    CrossClusterPrefixIndex,
)


class TestCacheDigest:
    """CacheDigest construction and binary round-trip."""

    def test_to_bytes_from_bytes_round_trip(self) -> None:
        now = time.time()
        digest = CacheDigest(
            cluster_id="us-east-1",
            prefix_hash="abc123",
            model_id="llama-70b",
            ttl=300.0,
            last_access=now,
            reuse_count=3,
            kv_block_ref="s3://bucket/kv/abc123",
        )
        data = digest.to_bytes()
        restored = CacheDigest.from_bytes(data)
        assert restored.cluster_id == "us-east-1"
        assert restored.prefix_hash == "abc123"
        assert restored.model_id == "llama-70b"
        assert restored.reuse_count == 3
        assert restored.ttl == 300.0

    def test_is_expired(self) -> None:
        digest = CacheDigest(
            cluster_id="us-east-1",
            prefix_hash="abc",
            model_id="llama-70b",
            ttl=0.001,
            last_access=time.time() - 10.0,
            reuse_count=1,
        )
        assert digest.is_expired() is True


class TestLookupAndAnnounce:
    """CrossClusterPrefixIndex lookup and announce."""

    def test_announce_and_lookup(self) -> None:
        idx = CrossClusterPrefixIndex(cluster_id="us-east-1")
        announced = idx.announce(
            prefix_hash="abc123",
            model_id="llama-70b",
            kv_block_ref="s3://bucket/kv/abc123",
            ttl=300.0,
        )
        assert announced.cluster_id == "us-east-1"
        assert announced.prefix_hash == "abc123"

        found = idx.lookup(prefix_hash="abc123", model_id="llama-70b")
        assert found is not None
        assert found.reuse_count >= 2  # bumped on lookup

    def test_lookup_missing_returns_none(self) -> None:
        idx = CrossClusterPrefixIndex(cluster_id="us-east-1")
        assert idx.lookup(prefix_hash="nonexistent", model_id="llama-70b") is None

    def test_gossip_message_merge(self) -> None:
        idx = CrossClusterPrefixIndex(cluster_id="us-east-1")
        idx.announce(
            prefix_hash="hash-a",
            model_id="llama-70b",
            kv_block_ref="ref://a",
        )
        msg = idx.build_gossip_message(compact=True)
        assert msg["type"] == "cache_digest"
        assert msg["count"] == 1

        idx2 = CrossClusterPrefixIndex(cluster_id="eu-west-1")
        merged = idx2.process_gossip_message(msg)
        assert merged == 1
        assert idx2.lookup(prefix_hash="hash-a", model_id="llama-70b") is not None
