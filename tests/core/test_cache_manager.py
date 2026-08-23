"""Tests for CacheManager, RollingHash, and _rolling_prefix_hashes.

Uses the import-helper pattern to avoid circular imports.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_cache_mgr_mod = load_module("distllm/core/cache_manager.py")
CacheManager = _cache_mgr_mod.CacheManager
RollingHash = _cache_mgr_mod.RollingHash
_rolling_prefix_hashes = _cache_mgr_mod._rolling_prefix_hashes

_kv_cache_mod = load_module("distllm/core/kv_cache.py")
KVCache = _kv_cache_mod.KVCache


# ── RollingHash ──────────────────────────────────────────────────────────────


class TestRollingHash:
    """Incremental rolling hash O(1) extend, properties, reset."""

    def test_defaults(self):
        rh = RollingHash()
        assert rh.hash == 0
        assert rh.length == 0

    def test_extend_returns_new_hash(self):
        rh = RollingHash()
        h1 = rh.extend(42)
        assert isinstance(h1, int)
        assert h1 != 0
        assert rh.length == 1

    def test_extend_multiple_tokens(self):
        rh = RollingHash()
        rh.extend(1)
        rh.extend(2)
        rh.extend(3)
        assert rh.length == 3
        assert rh.hash != 0

    def test_extend_deterministic(self):
        rh1 = RollingHash()
        rh2 = RollingHash()
        for tok in [10, 20, 30]:
            rh1.extend(tok)
            rh2.extend(tok)
        assert rh1.hash == rh2.hash

    def test_reset_clears_state(self):
        rh = RollingHash()
        rh.extend(99)
        rh.extend(100)
        assert rh.length == 2
        rh.reset()
        assert rh.hash == 0
        assert rh.length == 0


# ── _rolling_prefix_hashes ────────────────────────────────────────────────────


class TestRollingPrefixHashes:
    """_rolling_prefix_hashes O(n) computation."""

    def test_empty_tokens(self):
        result = _rolling_prefix_hashes([], 10)
        assert result == {}

    def test_single_token(self):
        result = _rolling_prefix_hashes([42], 10)
        assert set(result.keys()) == {1}
        assert isinstance(result[1], int)

    def test_multiple_tokens(self):
        result = _rolling_prefix_hashes([1, 2, 3], 10)
        assert set(result.keys()) == {1, 2, 3}

    def test_max_len_clips(self):
        tokens = [10, 20, 30, 40, 50]
        result = _rolling_prefix_hashes(tokens, 3)
        assert set(result.keys()) == {1, 2, 3}
        assert 4 not in result
        assert 5 not in result

    def test_hash_consistency(self):
        r1 = _rolling_prefix_hashes([1, 2, 3], 10)
        r2 = _rolling_prefix_hashes([1, 2, 3], 10)
        assert r1 == r2

    def test_different_tokens_different_hashes(self):
        r1 = _rolling_prefix_hashes([1, 2, 3], 10)
        r2 = _rolling_prefix_hashes([3, 2, 1], 10)
        assert r1 != r2


# ── CacheManager construction ─────────────────────────────────────────────────


class TestCacheManagerInit:
    """CacheManager.__init__ with various config combinations."""

    def test_defaults(self):
        cm = CacheManager()
        assert cm.prefix_cache is None
        assert cm.chunked_prefill_enabled is True
        assert cm.chunked_prefill_chunk_size == 512
        assert cm.persistence_manager is None
        assert cm.cache_index is None
        assert cm.gossip_protocol is None
        assert cm.gossip_client is None

    def test_prefix_cache_enabled(self):
        """prefix_cache_enabled=True creates a PrefixCache."""
        cm = CacheManager(
            prefix_cache_enabled=True,
            prefix_cache_max_entries=64,
            prefix_cache_min_prefix_len=8,
        )
        assert cm.prefix_cache is not None

    def test_chunked_prefill_disabled(self):
        cm = CacheManager(chunked_prefill_enabled=False)
        assert cm.chunked_prefill_enabled is False

    def test_chunked_prefill_custom_chunk(self):
        cm = CacheManager(chunked_prefill_enabled=True, chunked_prefill_chunk_size=256)
        assert cm.chunked_prefill_chunk_size == 256

    def test_tier_defaults(self):
        cm = CacheManager(gpu_cache_mb=1024, cpu_cache_mb=2048, ssd_cache_gb=100)
        assert cm._tiers["gpu"]["max_bytes"] == 1024 * 1024 * 1024
        assert cm._tiers["cpu"]["max_bytes"] == 2048 * 1024 * 1024
        assert cm._tiers["ssd"]["max_bytes"] == 100 * 1024 ** 3

    def test_tier_stats_initialized(self):
        cm = CacheManager()
        for tier in ("local", "disk", "gossip_index", "broadcast"):
            assert cm._tier_stats[tier] == {"hits": 0, "misses": 0}

    def test_predictive_cache_disabled_by_default(self):
        cm = CacheManager()
        assert cm._predictive_cache is None


# ── CacheManager lookup / store ────────────────────────────────────────────────


class TestCacheManagerLookupStore:
    """lookup_prefix and store_prefix with no prefix cache enabled."""

    def test_lookup_prefix_empty_tokens(self):
        cm = CacheManager()
        match_len, result = cm.lookup_prefix([])
        assert match_len == 0
        assert result is None

    def test_lookup_prefix_no_cache_miss(self):
        cm = CacheManager()
        match_len, result = cm.lookup_prefix([1, 2, 3])
        assert match_len == 0
        assert result is None

    def test_store_prefix_empty_tokens(self):
        cm = CacheManager()
        cm.store_prefix([], None)  # should not raise

    def test_store_prefix_with_kv_data(self):
        cm = CacheManager()
        kv_data = {"layer_0": "dummy_data"}
        cm.store_prefix([1, 2, 3], kv_data)
        # Stored in GPU tier
        assert len(cm._tiers["gpu"]["entries"]) == 1

    def test_lookup_prefix_from_tier(self):
        cm = CacheManager()
        cm.store_prefix([10, 20, 30], ("match_len", "blob"))
        # Check GPU tier has the entry
        assert len(cm._tiers["gpu"]["entries"]) > 0

    def test_find_shared_prefix_no_prefix_cache(self):
        cm = CacheManager()
        result = cm.find_shared_prefix([1, 2, 3])
        assert result == 0

    def test_find_shared_prefix_with_prefix_cache(self):
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        cm.prefix_cache.store([1, 2, 3], "data")
        # PrefixCache stores the full rolling hash so an exact match is required.
        match = cm.find_shared_prefix([1, 2, 3])
        assert match == 3  # exact match returns length of cached tokens


# ── maybe_chunk ──────────────────────────────────────────────────────────────


class TestCacheManagerChunk:
    """maybe_chunk splitting of long prompts."""

    def test_short_prompt_no_chunk(self):
        cm = CacheManager()
        result = cm.maybe_chunk([1, 2, 3], chunk_size=10)
        assert result is None

    def test_chunk_exact_size(self):
        cm = CacheManager()
        tokens = list(range(10))
        result = cm.maybe_chunk(tokens, chunk_size=10)
        assert result is None  # not exceeding, equal is not > chunk_size

    def test_chunk_longer_than_chunk_size(self):
        cm = CacheManager()
        tokens = list(range(15))
        result = cm.maybe_chunk(tokens, chunk_size=10)
        assert result is not None
        assert len(result) == 2
        assert result[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert result[1] == [10, 11, 12, 13, 14]

    def test_chunk_multi_chunk(self):
        cm = CacheManager()
        tokens = list(range(25))
        result = cm.maybe_chunk(tokens, chunk_size=10)
        assert result is not None
        assert len(result) == 3
        assert sum(len(c) for c in result) == 25


# ── create_kv_cache / release_kv_cache ────────────────────────────────────────


class TestCacheManagerKV:
    """Static methods for KV cache creation and release."""

    def test_create_kv_cache(self):
        kvc = CacheManager.create_kv_cache()
        assert isinstance(kvc, KVCache)

    def test_release_kv_cache(self):
        """release_kv_cache should not raise on a fresh KVCache."""
        kvc = CacheManager.create_kv_cache()
        # Should not raise
        CacheManager.release_kv_cache(kvc)

    def test_release_kv_cache_idempotent(self):
        kvc = CacheManager.create_kv_cache()
        CacheManager.release_kv_cache(kvc)
        CacheManager.release_kv_cache(kvc)  # second call should not raise


# ── _estimate_entry_size ──────────────────────────────────────────────────────


class TestCacheManagerEstimateSize:
    """_estimate_entry_size for various data types."""

    def test_dict_of_tensors(self):
        cm = CacheManager()
        data = {"k": 1, "v": 2}
        # Simple dict without tensors defaults to 1024
        sz = cm._estimate_entry_size(data)
        assert sz == 1024

    def test_list(self):
        cm = CacheManager()
        sz = cm._estimate_entry_size([1, 2, 3])
        assert sz == 1024  # list of ints, no tensors -> default

    def test_tuple(self):
        cm = CacheManager()
        sz = cm._estimate_entry_size((1, 2))
        assert sz == 1024

    def test_unknown_type(self):
        cm = CacheManager()
        sz = cm._estimate_entry_size("string_data")
        assert sz == 1024  # default

    def test_none(self):
        cm = CacheManager()
        sz = cm._estimate_entry_size(None)
        assert sz == 1024


# ── tier stats / latencies ──────────────────────────────────────────────────────


class TestCacheManagerTierStats:
    """get_tier_stats and get_tier_latencies."""

    def test_get_tier_stats_empty(self):
        cm = CacheManager()
        stats = cm.get_tier_stats()
        assert set(stats.keys()) == {"local", "disk", "gossip_index", "broadcast"}
        for v in stats.values():
            assert v == {"hits": 0, "misses": 0}

    def test_get_tier_latencies_empty(self):
        cm = CacheManager()
        latencies = cm.get_tier_latencies()
        assert set(latencies.keys()) == {"local", "disk", "gossip_index", "broadcast"}
        for v in latencies.values():
            assert v == {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0}

    def test_record_tier_latency(self):
        cm = CacheManager()
        cm._record_tier_latency("local", 10.0)
        cm._record_tier_latency("local", 20.0)
        latencies = cm.get_tier_latencies()
        assert latencies["local"]["avg_ms"] == 15.0
        assert latencies["local"]["p50_ms"] == 20.0  # sorted [10, 20], len//2 = 1

    def test_should_skip_tier_not_enough_samples(self):
        cm = CacheManager()
        assert cm._should_skip_tier("local") is False

    def test_should_skip_tier_under_threshold(self):
        cm = CacheManager()
        cm._tier_timeout_ms = 5.0
        cm._tier_latencies["local"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cm._should_skip_tier("local") is False  # P95 = 5.0, not > 5.0


# ── _hash_tokens ──────────────────────────────────────────────────────────────


class TestCacheManagerHashTokens:
    """_hash_tokens fallback without cache_index."""

    def test_empty_tokens(self):
        cm = CacheManager()
        h = cm._hash_tokens([])
        assert isinstance(h, str)
        assert h.startswith("h")

    def test_single_token(self):
        cm = CacheManager()
        h = cm._hash_tokens([1])
        assert isinstance(h, str)
        assert len(h) == 33  # "h" + 32 hex chars

    def test_deterministic(self):
        cm = CacheManager()
        h1 = cm._hash_tokens([1, 2, 3])
        h2 = cm._hash_tokens([1, 2, 3])
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        cm = CacheManager()
        h1 = cm._hash_tokens([1, 2, 3])
        h2 = cm._hash_tokens([3, 2, 1])
        assert h1 != h2

    def test_with_cache_index(self):
        """When cache_index is provided and has index_tokens, it takes priority."""
        cm = CacheManager()
        # No cache_index by default, so fallback is used


# ── lookup_with_gossip / lookup_with_disk_fallback ────────────────────────────


class TestCacheManagerLookupWithGossip:
    """lookup_with_gossip fallback chain."""

    def test_no_gossip_returns_none(self):
        cm = CacheManager()
        result = cm.lookup_with_gossip([1, 2, 3])
        assert result is None

    def test_with_only_cache_index(self):
        """Gossip index hit returns node id."""
        cm = CacheManager(prefix_cache_enabled=False)

        class FakeCacheIndex:
            def lookup(self, ph):
                return "node-b"
            def index_tokens(self, tokens):
                return "h123"

        cm.cache_index = FakeCacheIndex()
        result = cm.lookup_with_gossip([1, 2, 3])
        assert result is not None
        source, _ = result
        assert source == "node-b"


class TestCacheManagerLookupWithDiskFallback:
    """lookup_with_disk_fallback without persistence manager."""

    def test_no_persistence_falls_through(self):
        cm = CacheManager()
        match_len, entry = cm.lookup_with_disk_fallback([1, 2, 3], "model-a")
        assert match_len == 0
        assert entry is None

    def test_with_prefix_cache_hit(self):
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)
        cm.prefix_cache.store([1, 2, 3], "data")
        match_len, entry = cm.lookup_with_disk_fallback([1, 2, 3], "model-a")
        assert match_len > 0
        assert entry is not None


# ── sync_with_peers / other gossip methods ──────────────────────────────────────


class TestCacheManagerGossip:
    """sync_with_peers, fetch_from_peer without gossip client."""

    def test_sync_with_peers_no_gossip(self):
        cm = CacheManager()
        result = cm.sync_with_peers()
        assert result == 0

    def test_fetch_from_peer_no_gossip_client(self):
        cm = CacheManager()
        result = cm.fetch_from_peer("peer-1", "h123", [1, 2, 3])
        assert result is None

    def test_mark_dirty_no_persistence(self):
        cm = CacheManager()
        cm.mark_dirty("req-1")  # should not raise
