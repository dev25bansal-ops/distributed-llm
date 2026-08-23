"""Tests for KV cache digest and content-based federated routing."""

import importlib.util
import os
import struct
from typing import Any

import pytest


def _get_module():
    import sys
    import types

    # Stub merkle module dependency
    merkle_mod = types.ModuleType("distllm.dist.merkle")
    merkle_mod.MerkleTree = None
    sys.modules["distllm.dist.merkle"] = merkle_mod

    path = os.path.join("src", "distllm", "dist", "cache_digest.py")
    spec = importlib.util.spec_from_file_location("cache_digest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cache_digest"] = mod

    mod.logger = types.ModuleType("logger")
    mod.logger.debug = lambda *a, **kw: None
    mod.logger.info = lambda *a, **kw: None
    mod.logger.warning = lambda *a, **kw: None

    spec.loader.exec_module(mod)
    return mod


class TestRollingHash:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()

    def test_rolling_hash_empty(self):
        result = self.mod._rolling_hash([])
        assert result == {}

    def test_rolling_hash_short_sequence(self):
        tokens = [1, 2, 3, 4, 5]
        result = self.mod._rolling_hash(tokens, window_size=4)
        assert len(result) >= 1
        for _, h in result.items():
            assert isinstance(h, int)
            assert 0 <= h < self.mod._HASH_MOD

    def test_rolling_hash_deterministic(self):
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        r1 = self.mod._rolling_hash(tokens, window_size=4)
        r2 = self.mod._rolling_hash(tokens, window_size=4)
        assert r1 == r2

    def test_rolling_hash_different_inputs(self):
        t1 = [1, 2, 3, 4, 5]
        t2 = [1, 2, 3, 4, 6]
        r1 = self.mod._rolling_hash(t1, window_size=4)
        r2 = self.mod._rolling_hash(t2, window_size=4)
        assert r1 != r2

    def test_rolling_hash_stride(self):
        tokens = list(range(200))
        result = self.mod._rolling_hash(tokens, window_size=64)
        positions = sorted(result.keys())
        assert len(positions) >= 1
        for i in range(1, len(positions)):
            diff = positions[i] - positions[i - 1]
            assert diff == 16  # stride = window_size // 4


class TestComputePrefixHash:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()

    def test_prefix_hash_deterministic(self):
        tokens = [1, 2, 3]
        h1 = self.mod.compute_prefix_hash(tokens)
        h2 = self.mod.compute_prefix_hash(tokens)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_prefix_hash_differs(self):
        h1 = self.mod.compute_prefix_hash([1, 2, 3])
        h2 = self.mod.compute_prefix_hash([1, 2, 4])
        assert h1 != h2

    def test_empty_prefix_hash(self):
        h = self.mod.compute_prefix_hash([])
        assert len(h) == 64


class TestKVCacheDigest:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.KVCacheDigest = cls.mod.KVCacheDigest

    def test_compute_returns_expected_keys(self):
        digest = self.KVCacheDigest(window_size=128).compute([1, 2, 3, 4, 5])
        assert "version" in digest
        assert "hash" in digest
        assert "prefix_hash" in digest
        assert "length" in digest
        assert "window_size" in digest

    def test_compute_prefix_hash_matches(self):
        tokens = [10, 20, 30, 40]
        digest = self.KVCacheDigest().compute(tokens)
        expected = self.mod.compute_prefix_hash(tokens)
        assert digest["hash"] == expected

    def test_similarity_identical(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        d1 = self.KVCacheDigest(window_size=4).compute(tokens)
        d2 = self.KVCacheDigest(window_size=4).compute(tokens)
        sim = self.KVCacheDigest.similarity(d1, d2)
        assert sim == 1.0

    def test_similarity_no_match(self):
        t1 = [1, 2, 3, 4, 5, 6, 7, 8]
        t2 = [9, 10, 11, 12, 13, 14, 15, 16]
        d1 = self.KVCacheDigest(window_size=4).compute(t1)
        d2 = self.KVCacheDigest(window_size=4).compute(t2)
        sim = self.KVCacheDigest.similarity(d1, d2)
        assert sim == 0.0

    def test_similarity_partial(self):
        t1 = [1, 2, 3, 4, 5, 6, 7, 8]
        t2 = [1, 2, 3, 4, 9, 10, 11, 12]
        d1 = self.KVCacheDigest(window_size=4).compute(t1)
        d2 = self.KVCacheDigest(window_size=4).compute(t2)
        sim = self.KVCacheDigest.similarity(d1, d2)
        assert 0.0 < sim < 1.0

    def test_longest_common_prefix(self):
        t1 = [1, 2, 3, 4, 5, 6, 7, 8]
        t2 = [1, 2, 3, 4, 9, 10, 11, 12]
        d1 = self.KVCacheDigest(window_size=4).compute(t1)
        d2 = self.KVCacheDigest(window_size=4).compute(t2)
        lcp = self.KVCacheDigest.longest_common_prefix_len(d1, d2)
        assert lcp >= 4

    def test_longest_common_prefix_no_match(self):
        t1 = [1, 2, 3, 4]
        t2 = [5, 6, 7, 8]
        d1 = self.KVCacheDigest(window_size=4).compute(t1)
        d2 = self.KVCacheDigest(window_size=4).compute(t2)
        lcp = self.KVCacheDigest.longest_common_prefix_len(d1, d2)
        assert lcp == 0

    def test_longest_common_prefix_full_match(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        d1 = self.KVCacheDigest(window_size=4).compute(tokens)
        d2 = self.KVCacheDigest(window_size=4).compute(tokens)
        lcp = self.KVCacheDigest.longest_common_prefix_len(d1, d2)
        assert lcp == len(tokens)


class TestContentRouter:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.ContentRouter = cls.mod.ContentRouter
        cls.KVCacheDigest = cls.mod.KVCacheDigest

    def test_score_cluster_empty(self):
        router = self.ContentRouter()
        prompt = self.KVCacheDigest().compute([1, 2, 3, 4])
        scores = router.score_cluster(prompt, {}, {})
        assert scores == []

    def test_score_cluster_high_affinity(self):
        router = self.ContentRouter()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        prompt = self.KVCacheDigest(window_size=4).compute(tokens)
        cached = self.KVCacheDigest(window_size=4).compute(tokens)
        scores = router.score_cluster(prompt, {"peer-1": cached}, {"peer-1": 0.2})
        assert len(scores) == 1
        assert scores[0].cluster_id == "peer-1"
        assert scores[0].cache_affinity > 0.5

    def test_score_cluster_prefers_low_load(self):
        router = self.ContentRouter(cache_weight=0.5, load_weight=0.5)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        prompt = self.KVCacheDigest(window_size=4).compute(tokens)
        d_high = self.KVCacheDigest(window_size=4).compute(tokens)
        d_low = self.KVCacheDigest(window_size=4).compute(tokens)
        scores = router.score_cluster(
            prompt,
            {"peer-1": d_high, "peer-2": d_low},
            {"peer-1": 0.9, "peer-2": 0.1},
        )
        assert len(scores) == 2
        # Both have same affinity, but peer-2 has lower load
        assert scores[0].cluster_id == "peer-2"

    def test_route_returns_best(self):
        router = self.ContentRouter()
        tokens = [1, 2, 3, 4]
        prompt = self.KVCacheDigest().compute(tokens)
        cached = self.KVCacheDigest().compute(tokens)
        cid = router.route(prompt, {"best-peer": cached}, {"best-peer": 0.1})
        assert cid == "best-peer"

    def test_route_returns_none_when_empty(self):
        router = self.ContentRouter()
        prompt = self.KVCacheDigest().compute([1, 2, 3])
        cid = router.route(prompt, {}, {})
        assert cid is None

    def test_score_cluster_multiple_with_mixed_affinity(self):
        router = self.ContentRouter(cache_weight=0.7, load_weight=0.3)
        prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        prompt = self.KVCacheDigest(window_size=4).compute(prompt_tokens)

        d_full = self.KVCacheDigest(window_size=4).compute(prompt_tokens)
        d_partial = self.KVCacheDigest(window_size=4).compute([1, 2, 3, 4, 9, 10, 11, 12])
        d_none = self.KVCacheDigest(window_size=4).compute([9, 10, 11, 12, 13, 14, 15, 16])

        scores = router.score_cluster(
            prompt,
            {"full": d_full, "partial": d_partial, "none": d_none},
            {"full": 0.5, "partial": 0.3, "none": 0.1},
        )
        assert scores[0].cluster_id == "full"
        assert scores[1].cache_affinity < scores[0].cache_affinity
        assert scores[2].cache_affinity <= scores[1].cache_affinity


class TestCacheDigestExchange:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.CacheDigestExchange = cls.mod.CacheDigestExchange
        cls.KVCacheDigest = cls.mod.KVCacheDigest

    def test_serialize_deserialize_roundtrip(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        digest = self.KVCacheDigest(window_size=4).compute(tokens)
        digests = {"cluster-1": digest}

        data = self.CacheDigestExchange.serialize(digests)
        assert isinstance(data, bytes)
        assert len(data) > 0

        parsed = self.CacheDigestExchange.deserialize(data)
        assert "cluster-1" in parsed
        assert parsed["cluster-1"]["hash"] == digest["hash"]
        assert parsed["cluster-1"]["length"] == len(tokens)

    def test_serialize_empty(self):
        data = self.CacheDigestExchange.serialize({})
        assert data == bytes([1])

    def test_deserialize_empty(self):
        result = self.CacheDigestExchange.deserialize(b"")
        assert result == {}

    def test_deserialize_truncated(self):
        result = self.CacheDigestExchange.deserialize(b"\x01\x00\x01")
        assert result == {}

    def test_multiple_clusters_roundtrip(self):
        d1 = self.KVCacheDigest(window_size=4).compute([1, 2, 3, 4])
        d2 = self.KVCacheDigest(window_size=4).compute([5, 6, 7, 8])
        digests = {"c1": d1, "c2": d2}

        data = self.CacheDigestExchange.serialize(digests)
        parsed = self.CacheDigestExchange.deserialize(data)

        assert "c1" in parsed
        assert "c2" in parsed
        assert parsed["c1"]["hash"] == d1["hash"]
        assert parsed["c2"]["hash"] == d2["hash"]


def test_module_exports():
    mod = _get_module()
    assert hasattr(mod, "KVCacheDigest")
    assert hasattr(mod, "ContentRouter")
    assert hasattr(mod, "CacheDigestExchange")
    assert hasattr(mod, "compute_prefix_hash")
    assert hasattr(mod, "RouterScore")
    assert hasattr(mod, "CachedPrefixInfo")
    assert hasattr(mod, "_rolling_hash")
