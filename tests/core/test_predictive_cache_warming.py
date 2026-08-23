"""Tests for predictive cache warming."""

from __future__ import annotations

from distllm.core.predictive_cache_warming import (
    LRUPrefixTracker,
    MarkovPrefixPredictor,
    PredictiveCacheWarmer,
    ProactiveKVPusher,
)


class TestLRUPrefixTracker:
    def test_record(self):
        lru = LRUPrefixTracker()
        lru.record("hash-a")
        s = lru.get_stats("hash-a")
        assert s is not None
        assert s.access_count == 1

    def test_multiple_accesses(self):
        lru = LRUPrefixTracker()
        lru.record("hash-a")
        lru.record("hash-a")
        s = lru.get_stats("hash-a")
        assert s.access_count == 2

    def test_top_k(self):
        lru = LRUPrefixTracker()
        lru.record("hash-a")
        lru.record("hash-b")
        lru.record("hash-b")
        top = lru.top_k(5)
        assert top[0].hash == "hash-b"

    def test_eviction(self):
        lru = LRUPrefixTracker(max_prefixes=2)
        lru.record("hash-a")
        lru.record("hash-b")
        lru.record("hash-c")
        assert lru.total_prefixes == 2

    def test_stats(self):
        lru = LRUPrefixTracker()
        lru.record("hash-a")
        s = lru.stats
        assert s["total"] >= 1


class TestMarkovPrefixPredictor:
    def test_record(self):
        mp = MarkovPrefixPredictor()
        mp.record("a", "b")
        preds = mp.predict("a")
        assert len(preds) == 1
        assert preds[0][0] == "b"

    def test_most_common_transition(self):
        mp = MarkovPrefixPredictor()
        mp.record("a", "b")
        mp.record("a", "b")
        mp.record("a", "c")
        preds = mp.predict("a")
        assert preds[0][0] == "b"
        assert preds[0][1] == 2 / 3

    def test_transition_probability(self):
        mp = MarkovPrefixPredictor()
        mp.record("a", "b")
        assert mp.transition_probability("a", "b") == 1.0
        assert mp.transition_probability("a", "x") == 0.0

    def test_unknown_prefix(self):
        mp = MarkovPrefixPredictor()
        assert mp.predict("unknown") == []

    def test_top_k(self):
        mp = MarkovPrefixPredictor()
        mp.record("a", "b")
        mp.record("a", "c")
        mp.record("a", "d")
        preds = mp.predict("a", top_k=2)
        assert len(preds) == 2

    def test_max_prefixes(self):
        mp = MarkovPrefixPredictor(max_prefixes=1)
        mp.record("a", "b")
        mp.record("c", "d")  # Should be dropped
        assert "c" not in mp._transitions


class TestProactiveKVPusher:
    def test_should_push_high_prob(self):
        p = ProactiveKVPusher()
        assert p.should_push("hash-a", probability=0.8) is True

    def test_should_not_push_low_prob(self):
        p = ProactiveKVPusher()
        assert p.should_push("hash-a", probability=0.1) is False

    def test_should_not_push_recent(self):
        p = ProactiveKVPusher()
        p.record_push("hash-a", "n1", ["n2"])
        assert p.should_push("hash-a", probability=0.8) is False


class TestPredictiveCacheWarmer:
    def test_record_access(self):
        w = PredictiveCacheWarmer()
        w.record_access("hash-a")
        assert w._tracker.total_prefixes == 1

    def test_record_transition(self):
        w = PredictiveCacheWarmer()
        w.record_access("hash-a", next_hash="hash-b")
        preds = w.predict_next("hash-a")
        assert len(preds) == 1
        assert preds[0][0] == "hash-b"

    def test_get_top_prefixes(self):
        w = PredictiveCacheWarmer()
        w.record_access("hash-a")
        w.record_access("hash-b")
        top = w.get_top_prefixes(5)
        assert len(top) == 2

    def test_should_warm_popular(self):
        w = PredictiveCacheWarmer()
        w.record_access("hash-a")
        w.record_access("hash-a")
        assert w.should_warm("hash-a") is True

    def test_should_not_warm_rare(self):
        w = PredictiveCacheWarmer()
        w.record_access("hash-a")
        assert w.should_warm("hash-a") is False

    def test_maybe_proactive_push(self):
        w = PredictiveCacheWarmer()
        w.record_access("prefix-a", next_hash="prefix-b")
        pushed = w.maybe_proactive_push("prefix-a", source_node="n1", target_nodes=["n2"])
        assert isinstance(pushed, list)

    def test_stats(self):
        w = PredictiveCacheWarmer()
        w.record_access("hash-a")
        s = w.stats
        assert "lru_tracker" in s
        assert "markov_prefixes" in s
