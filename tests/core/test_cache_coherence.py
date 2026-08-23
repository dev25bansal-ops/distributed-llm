"""Tests for CacheCoherenceProtocol (vector-clock based cache coherence)."""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/cache_coherence.py")
CacheCoherenceProtocol = _mod.CacheCoherenceProtocol


class TestCacheCoherenceProtocol:
    """Vector-clock based cache coherence for distributed P2P caching."""

    def test_init(self):
        ccp = CacheCoherenceProtocol("node-a")
        assert ccp._node_id == "node-a"
        assert ccp._vector_clocks == {}
        assert ccp._invalidated == {}

    def test_on_store_increments_clock(self):
        ccp = CacheCoherenceProtocol("node-a")
        clock = ccp.on_store("h123")
        assert clock == 1
        assert ccp._vector_clocks["h123"]["node-a"] == 1
        clock2 = ccp.on_store("h123")
        assert clock2 == 2

    def test_on_receive_newer_entry(self):
        ccp = CacheCoherenceProtocol("node-a")
        result = ccp.on_receive("h123", "node-b", 5)
        assert result is True
        assert ccp._vector_clocks["h123"]["node-b"] == 5

    def test_on_receive_older_entry_ignored(self):
        ccp = CacheCoherenceProtocol("node-a")
        ccp.on_receive("h123", "node-b", 10)
        result = ccp.on_receive("h123", "node-b", 5)
        assert result is False
        assert ccp._vector_clocks["h123"]["node-b"] == 10  # unchanged

    def test_is_stale_detects_outdated_remote(self):
        ccp = CacheCoherenceProtocol("node-a")
        ccp.on_store("h123")
        stale = ccp.is_stale("h123", {"node-a": 0})
        assert stale is True

    def test_is_stale_accepts_current(self):
        ccp = CacheCoherenceProtocol("node-a")
        ccp.on_store("h123")
        stale = ccp.is_stale("h123", {"node-a": 1})
        assert stale is False

    def test_invalidate_and_get_invalidated_since(self):
        import time
        ccp = CacheCoherenceProtocol("node-a")
        ccp.invalidate("h123")
        since = time.time() - 10
        result = ccp.get_invalidated_since(since)
        assert "h123" in result

    def test_stats(self):
        ccp = CacheCoherenceProtocol("node-b")
        ccp.on_store("h1")
        ccp.on_store("h2")
        s = ccp.stats()
        assert s["tracked_prefixes"] == 2
        assert s["node_id"] == "node-b"
