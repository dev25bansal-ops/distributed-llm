"""Tests for StorageTier, TieredEntry, TieredMemoryPool.

Uses bytes as data (no torch dependency required for core tests).
"""

from __future__ import annotations

import time as time_module
from typing import Any

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_tiered = load_module("distllm/core/advanced_scheduling/tiered_store.py")
StorageTier = _tiered.StorageTier
TieredEntry = _tiered.TieredEntry
TieredMemoryPool = _tiered.TieredMemoryPool
TierStats = _tiered.TierStats


class TestStorageTier:
    """Test suite for StorageTier enum."""

    def test_members(self) -> None:
        assert StorageTier.HOT.value == "hot"
        assert StorageTier.WARM.value == "warm"
        assert StorageTier.COLD.value == "cold"

    def test_all_distinct(self) -> None:
        values = [m.value for m in StorageTier]
        assert len(values) == len(set(values))


class TestTieredEntry:
    """Test suite for TieredEntry dataclass."""

    def test_construction(self) -> None:
        import time
        entry = TieredEntry(
            key="req-1",
            data=b"some_data",
            tier=StorageTier.HOT,
            size_bytes=9,
        )
        assert entry.key == "req-1"
        assert entry.data == b"some_data"
        assert entry.tier == StorageTier.HOT
        assert entry.size_bytes == 9
        assert entry.access_count == 0
        assert entry.pinned is False
        assert isinstance(entry.last_access, float)
        assert isinstance(entry.created_at, float)


class TestTierStats:
    """Test suite for TierStats."""

    def test_construction(self) -> None:
        stats = TierStats(
            tier=StorageTier.HOT,
            entry_count=5,
            used_bytes=1000,
            capacity_bytes=2000,
            hit_count=50,
            miss_count=10,
            promotion_count=3,
            demotion_count=2,
        )
        assert stats.utilization == 0.5
        assert stats.hit_rate == 50 / 60

    def test_zero_capacity(self) -> None:
        stats = TierStats(
            tier=StorageTier.COLD,
            entry_count=0,
            used_bytes=0,
            capacity_bytes=0,
            hit_count=0,
            miss_count=0,
            promotion_count=0,
            demotion_count=0,
        )
        assert stats.utilization == 0.0
        assert stats.hit_rate == 0.0


class TestTieredMemoryPool:
    """Test suite for TieredMemoryPool (no GPU, no NVMe by default)."""

    def test_default_construction(self) -> None:
        pool = TieredMemoryPool(
            gpu_memory_gb=24.0,
            cpu_memory_gb=128.0,
            nvme_path=None,
        )
        assert pool._l1_capacity == int(24.0 * 1e9)
        assert pool._l2_capacity == int(128.0 * 1e9)
        assert pool._l3_capacity == 0
        assert pool._l1_cache == {}
        assert pool._l2_cache == {}
        assert pool._l3_cache == {}
        assert pool._nvme_path is None

    def test_small_capacity_eviction(self) -> None:
        """Test put/get with tiny capacity to trigger eviction paths."""
        pool = TieredMemoryPool(
            gpu_memory_gb=0.000_01,   # ~10 KB L1
            cpu_memory_gb=0.000_1,     # ~100 KB L2
            nvme_path=None,
            promotion_threshold=10,
            demotion_idle_s=99999,
        )
        # Insert data that fits
        data_a = b"a" * 100
        data_b = b"b" * 100
        data_c = b"c" * 100

        assert pool.put("a", data_a, StorageTier.HOT) is True
        assert pool.put("b", data_b, StorageTier.HOT) is True
        assert pool.put("c", data_c, StorageTier.HOT) is True

        # All should be retrievable via get (which checks L3 -> L2 -> L1)
        # Note: eviction happens on put when L1 exceeds capacity.
        # After 3 puts of 100 bytes each, some may have been demoted to L2.
        # get traverses L1 -> L2 -> L3, so all should be found.
        assert pool.get("a") is not None
        assert pool.get("b") is not None
        assert pool.get("c") is not None

    def test_get_missing_key(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        assert pool.get("nonexistent") is None

    def test_put_and_get_l1(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        data = b"test_data"
        assert pool.put("key1", data, StorageTier.HOT) is True
        result = pool.get("key1")
        assert result == data

    def test_put_and_get_l2(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        data = b"warm_data"
        assert pool.put("warm1", data, StorageTier.WARM) is True
        result = pool.get("warm1")
        assert result == data

    def test_put_l3_without_nvme_fails(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        assert pool.put("cold1", b"data", StorageTier.COLD) is False

    def test_l1_lru_eviction_on_overflow(self) -> None:
        """L1 evicts oldest entries to make room."""
        pool = TieredMemoryPool(
            gpu_memory_gb=0.000_001,  # 1000 bytes L1
            nvme_path=None,
        )
        # Each entry ~200 bytes; L1 capacity is 1000 bytes (80% = 800 bytes before aggressive eviction)
        # The _put_l1 evicts while l1_used + size > l1_capacity
        data = b"x" * 200
        for i in range(10):
            pool.put(f"k{i}", data, StorageTier.HOT)

        # Older entries may have been demoted to L2; they should still be findable via get
        for i in range(10):
            assert pool.get(f"k{i}") is not None, f"key k{i} not found"

    def test_promotion_l2_to_l1(self) -> None:
        """Accessing an L2 entry repeatedly promotes it to L1."""
        pool = TieredMemoryPool(
            gpu_memory_gb=1.0,
            cpu_memory_gb=1.0,
            nvme_path=None,
            promotion_threshold=2,
        )

        data = b"promote_me"
        pool.put("promo", data, StorageTier.WARM)

        # First access: hit in L2, access_count becomes 1
        assert pool.get("promo") == data
        assert "promo" in pool._l2_cache

        # Second access: access_count becomes 2 >= promotion_threshold
        assert pool.get("promo") == data
        assert "promo" in pool._l1_cache, "Entry should be promoted to L1"

    def test_promotion_not_with_below_threshold(self) -> None:
        """Entry stays in L2 until access_count reaches threshold."""
        pool = TieredMemoryPool(
            gpu_memory_gb=1.0,
            cpu_memory_gb=1.0,
            nvme_path=None,
            promotion_threshold=10,
        )

        data = b"stay_warm"
        pool.put("warm", data, StorageTier.WARM)

        for _ in range(5):
            pool.get("warm")

        assert "warm" in pool._l2_cache
        assert "warm" not in pool._l1_cache

    def test_evict_demotes_l1_to_l2(self) -> None:
        """evict() demotes entries from L1 to L2 when L1 is full."""
        pool = TieredMemoryPool(
            gpu_memory_gb=0.000_01,   # ~10 KB L1 (tiny)
            cpu_memory_gb=100.0,       # huge L2
            nvme_path=None,
        )

        # Put enough data in L1 to exceed capacity
        chunk = b"x" * 5000  # 5 KB
        for i in range(5):
            pool.put(f"k{i}", chunk, StorageTier.HOT)

        # L1 capacity is ~10000 bytes.  5 * 5000 = 25000 bytes >> 10000.
        # evict() should demote cold entries to L2.
        pool.evict()

        # At least some entries should now be in L2
        assert len(pool._l2_cache) > 0, "No entries demoted to L2"

    def test_clear_empties_all_tiers(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        pool.put("a", b"data", StorageTier.HOT)
        pool.put("b", b"data", StorageTier.WARM)
        pool.clear()

        assert len(pool._l1_cache) == 0
        assert len(pool._l2_cache) == 0
        assert len(pool._l3_cache) == 0
        assert pool._l1_used == 0
        assert pool._l2_used == 0
        assert pool._l3_used == 0

    def test_get_stats(self) -> None:
        pool = TieredMemoryPool(
            gpu_memory_gb=1.0,
            cpu_memory_gb=1.0,
            nvme_path=None,
        )
        pool.put("s1", b"data", StorageTier.HOT)
        pool.get("s1")

        stats = pool.get_stats()
        assert stats["total_entries"] == 1
        assert stats["l1_hot"]["hit_count"] == 1
        assert stats["l1_hot"]["entry_count"] == 1

    def test_estimate_size_bytes(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        assert pool._estimate_size(b"hello") == 5
        assert pool._estimate_size(b"") == 0
        assert pool._estimate_size(bytearray(100)) == 100
        assert pool._estimate_size(memoryview(b"test")) == 4

    def test_estimate_size_unknown_type(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        assert pool._estimate_size([1, 2, 3]) == 0
        assert pool._estimate_size(42) == 0

    def test_start_stop_maintenance(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        assert pool._running is False
        pool.start()
        assert pool._running is True
        pool.stop()
        assert pool._running is False

    def test_request_in_l2_updates_access_count(self) -> None:
        pool = TieredMemoryPool(nvme_path=None)
        pool.put("l2key", b"l2data", StorageTier.WARM)
        pool.get("l2key")
        assert pool._l2_cache["l2key"].access_count == 1
        assert pool._l2_cache["l2key"].last_access > 0
