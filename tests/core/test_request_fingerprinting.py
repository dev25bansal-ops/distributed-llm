"""Tests for RequestFingerprinter and FingerprintEntry.

Covers:
- FingerprintEntry dataclass defaults
- RequestFingerprinter: fingerprint generation (deterministic, different prompts differ)
- RequestFingerprinter: mark_in_flight, is_in_flight, clear_in_flight
- RequestFingerprinter: store and lookup with TTL
- RequestFingerprinter: cache eviction at max size
- RequestFingerprinter: wait_for_result with event notification
- RequestFingerprinter: popularity and stats
- RequestFingerprinter: dedup disabled mode
- Thread safety
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

# Use the REAL module directly. This module imports cleanly on its own, and
# calling bootstrap_fake_packages() here globally clobbers sys.modules for
# every test collected afterward (e.g. it replaces the real
# distllm.core.structured_output with a stub, breaking real-module tests like
# test_structured_output_numbers.py).  Loading via the shim is unnecessary.
from distllm.core.request_fingerprinting import FingerprintEntry, RequestFingerprinter


# ---------------------------------------------------------------------------
# FingerprintEntry dataclass
# ---------------------------------------------------------------------------


class TestFingerprintEntry:
    """FingerprintEntry dataclass."""

    def test_defaults(self) -> None:
        entry = FingerprintEntry(
            fingerprint="fp1",
            prompt="hello",
            params_hash="abc",
            request_id="r1",
            created_at=100.0,
        )
        assert entry.response is None
        assert entry.hit_count == 0
        assert entry.last_accessed >= 0

    def test_full_construction(self) -> None:
        entry = FingerprintEntry(
            fingerprint="fp1",
            prompt="hello",
            params_hash="def",
            request_id="r1",
            created_at=200.0,
            response="world",
            hit_count=3,
            last_accessed=300.0,
        )
        assert entry.response == "world"
        assert entry.hit_count == 3
        assert entry.last_accessed == 300.0


# ---------------------------------------------------------------------------
# RequestFingerprinter construction
# ---------------------------------------------------------------------------


class TestRequestFingerprinterConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        fp = RequestFingerprinter()
        assert fp._cache_size == 10000
        assert fp._cache_ttl == 3600.0
        assert fp._enable_dedup is True

    def test_custom_params(self) -> None:
        fp = RequestFingerprinter(cache_size=100, cache_ttl_s=60.0, enable_dedup=False)
        assert fp._cache_size == 100
        assert fp._cache_ttl == 60.0
        assert fp._enable_dedup is False

    def test_dedup_disabled_still_tracks_cache(self) -> None:
        fp = RequestFingerprinter(enable_dedup=False)
        assert fp._cache is not None
        assert fp._in_flight == {}


# ---------------------------------------------------------------------------
# fingerprint generation
# ---------------------------------------------------------------------------


class TestRequestFingerprinterFingerprint:
    """Fingerprint generation."""

    def test_deterministic_same_input(self) -> None:
        fp = RequestFingerprinter()
        f1 = fp.fingerprint("hello", {"temp": 0.7})
        f2 = fp.fingerprint("hello", {"temp": 0.7})
        assert f1 == f2

    def test_different_prompts_differ(self) -> None:
        fp = RequestFingerprinter()
        f1 = fp.fingerprint("hello", {"temp": 0.7})
        f2 = fp.fingerprint("world", {"temp": 0.7})
        assert f1 != f2

    def test_different_params_differ(self) -> None:
        fp = RequestFingerprinter()
        f1 = fp.fingerprint("hello", {"temp": 0.7})
        f2 = fp.fingerprint("hello", {"temp": 0.9})
        assert f1 != f2

    def test_empty_params(self) -> None:
        fp = RequestFingerprinter()
        f1 = fp.fingerprint("hello")
        f2 = fp.fingerprint("hello", None)
        assert f1 == f2

    def test_params_sorted_keys(self) -> None:
        fp = RequestFingerprinter()
        f1 = fp.fingerprint("hello", {"b": 2, "a": 1})
        f2 = fp.fingerprint("hello", {"a": 1, "b": 2})
        assert f1 == f2


# ---------------------------------------------------------------------------
# In-flight tracking
# ---------------------------------------------------------------------------


class TestRequestFingerprinterInFlight:
    """In-flight request deduplication."""

    def test_mark_in_flight(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")
        assert fp.is_in_flight(fprint) is True

    def test_is_in_flight_false_for_unknown(self) -> None:
        fp = RequestFingerprinter()
        assert fp.is_in_flight("unknown") is False

    def test_clear_in_flight(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")
        fp.clear_in_flight(fprint, "r1")
        assert fp.is_in_flight(fprint) is False

    def test_clear_in_flight_partial(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")
        fp.mark_in_flight(fprint, "r2")
        fp.clear_in_flight(fprint, "r1")
        assert fp.is_in_flight(fprint) is True  # r2 still in flight
        fp.clear_in_flight(fprint, "r2")
        assert fp.is_in_flight(fprint) is False

    def test_clear_in_flight_unknown_fingerprint(self) -> None:
        fp = RequestFingerprinter()
        fp.clear_in_flight("unknown", "r1")  # Should not raise

    def test_clear_in_flight_unknown_request_id(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")
        fp.clear_in_flight(fprint, "nonexistent")  # Should not raise
        assert fp.is_in_flight(fprint) is True

    def test_dedup_disabled_skips_in_flight(self) -> None:
        fp = RequestFingerprinter(enable_dedup=False)
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")
        assert fp.is_in_flight(fprint) is False


# ---------------------------------------------------------------------------
# store / lookup
# ---------------------------------------------------------------------------


class TestRequestFingerprinterStoreAndLookup:
    """Cache storage and lookup."""

    def test_store_and_lookup(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello", {"temp": 0.7})
        fp.store(fprint, "r1", "world", prompt="hello", params={"temp": 0.7})
        entry = fp.lookup(fprint)
        assert entry is not None
        assert entry.response == "world"
        assert entry.hit_count == 1

    def test_lookup_miss(self) -> None:
        fp = RequestFingerprinter()
        assert fp.lookup("nonexistent") is None

    def test_lookup_expired_ttl(self) -> None:
        fp = RequestFingerprinter(cache_ttl_s=0.0)
        fprint = fp.fingerprint("hello")
        fp.store(fprint, "r1", "world")
        # TTL is 0, so entry is immediately expired
        assert fp.lookup(fprint) is None

    def test_lookup_refreshes_hit_count(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.store(fprint, "r1", "world")
        fp.lookup(fprint)
        fp.lookup(fprint)
        entry = fp.lookup(fprint)
        assert entry is not None
        assert entry.hit_count == 3

    def test_store_updates_existing(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.store(fprint, "r1", "first")
        fp.store(fprint, "r2", "second")
        entry = fp.lookup(fprint)
        assert entry is not None
        assert entry.response == "second"


# ---------------------------------------------------------------------------
# Cache eviction
# ---------------------------------------------------------------------------


class TestRequestFingerprinterEviction:
    """LRU cache eviction."""

    def test_evicts_oldest_when_full(self) -> None:
        fp = RequestFingerprinter(cache_size=3)
        fp.store("fp1", "r1", "a", prompt="a")
        fp.store("fp2", "r2", "b", prompt="b")
        fp.store("fp3", "r3", "c", prompt="c")
        fp.store("fp4", "r4", "d", prompt="d")
        # fp1 should be evicted
        assert fp.lookup("fp1") is None
        assert fp.lookup("fp4") is not None

    def test_lookup_refreshes_position(self) -> None:
        fp = RequestFingerprinter(cache_size=2)
        fp.store("fp1", "r1", "a", prompt="a")
        fp.store("fp2", "r2", "b", prompt="b")
        fp.lookup("fp1")  # refresh
        fp.store("fp3", "r3", "c", prompt="c")
        assert fp.lookup("fp2") is None  # evicted
        assert fp.lookup("fp1") is not None
        assert fp.lookup("fp3") is not None


# ---------------------------------------------------------------------------
# wait_for_result
# ---------------------------------------------------------------------------


class TestRequestFingerprinterWaitForResult:
    """Wait for in-flight result."""

    def test_wait_for_result_immediate(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        # Result already in in_flight_results
        fp._in_flight_results[fprint] = "cached_result"
        result = fp.wait_for_result(fprint, timeout_s=1.0)
        assert result == "cached_result"

    def test_wait_for_result_timeout(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")
        start = time.time()
        result = fp.wait_for_result(fprint, poll_interval_s=0.01, timeout_s=0.3)
        elapsed = time.time() - start
        assert result is None  # timeout
        assert elapsed >= 0.25  # at least close to timeout

    def test_wait_for_result_not_in_flight(self) -> None:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        result = fp.wait_for_result(fprint, timeout_s=0.1)
        assert result is None

    def test_wait_for_result_event_signalled(self) -> None:
        """F-044 regression: a waiter on an in-flight request must receive the
        actual result when store() publishes it.  store() now writes
        _in_flight_results before signalling, so waiters no longer time out."""
        fp = RequestFingerprinter()
        fprint = fp.fingerprint("hello")
        fp.mark_in_flight(fprint, "r1")

        results: list[str | None] = []

        def waiter() -> None:
            r = fp.wait_for_result(fprint, timeout_s=3.0)
            results.append(r)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        fp.store(fprint, "r1", "completed!")
        t.join(timeout=2.0)

        assert len(results) == 1
        assert results[0] == "completed!"


# ---------------------------------------------------------------------------
# popularity / stats
# ---------------------------------------------------------------------------


class TestRequestFingerprinterPopularityStats:
    """Popularity and stats methods."""

    def test_popularity_empty(self) -> None:
        fp = RequestFingerprinter()
        assert fp.popularity() == []

    def test_popularity_ranking(self) -> None:
        fp = RequestFingerprinter()
        fp.store("fp_a", "r1", "", prompt="a")
        fp.store("fp_b", "r2", "", prompt="b")
        fp.store("fp_c", "r3", "", prompt="c")
        # Increase hit counts
        fp.lookup("fp_b")
        fp.lookup("fp_b")
        fp.lookup("fp_c")
        popular = fp.popularity(top_n=2)
        assert len(popular) == 2
        # fp_b (2 hits) should be first, fp_c (1 hit) second
        assert popular[0][1] == 2
        assert popular[1][1] == 1

    def test_stats(self) -> None:
        fp = RequestFingerprinter(cache_size=100, enable_dedup=True)
        fprint = fp.fingerprint("hello")
        fp.store(fprint, "r1", "world", prompt="hello")
        fp.lookup(fprint)
        fp.mark_in_flight(fprint + "_other", "r2")
        stats = fp.stats()
        assert stats["cache_entries"] == 1
        assert stats["cache_max"] == 100
        assert stats["total_hits"] == 1
        assert stats["in_flight_requests"] == 1
        assert stats["dedup_enabled"] is True

    def test_stats_no_data(self) -> None:
        fp = RequestFingerprinter()
        stats = fp.stats()
        assert stats["cache_entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["in_flight_requests"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestRequestFingerprinterThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_store_and_lookup(self) -> None:
        fp = RequestFingerprinter(cache_size=500)
        errors: list[Exception] = []

        def worker(prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    fprint = fp.fingerprint(f"{prefix}-{i}")
                    fp.store(fprint, f"r{prefix}{i}", f"result_{i}", prompt=f"{prefix}-{i}")
                    fp.lookup(fprint)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("a", 50)),
            threading.Thread(target=worker, args=("b", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = fp.stats()
        assert stats["cache_entries"] == 100

    def test_concurrent_in_flight(self) -> None:
        fp = RequestFingerprinter(cache_size=100)
        errors: list[Exception] = []

        def worker(fprint: str, rid: str) -> None:
            try:
                fp.mark_in_flight(fprint, rid)
                assert fp.is_in_flight(fprint) is True
                fp.clear_in_flight(fprint, rid)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("fp1", "r1")),
            threading.Thread(target=worker, args=("fp1", "r2")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert fp.is_in_flight("fp1") is False
