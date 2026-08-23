"""Tests for interconnect-topology-aware cache tiering."""

from __future__ import annotations

from distllm.core.topology_aware_tiering import (
    MemoryTier,
    TierProperties,
    TopologyCostMatrix,
    TopologyAwareTiering,
)


class TestMemoryTier:
    def test_enum_values(self):
        assert MemoryTier.LOCAL_HBM.value == "hbm"
        assert MemoryTier.NVLINK.value == "nvl"
        assert MemoryTier.CXL.value == "cxl"
        assert MemoryTier.CPU_DRAM.value == "cpu"
        assert MemoryTier.SSD.value == "ssd"


class TestTierProperties:
    def test_create(self):
        p = TierProperties(bandwidth_gb_s=100, latency_us=10, capacity_gb=64, energy_per_gb_nj=5)
        assert p.bandwidth_gb_s == 100
        assert p.latency_us == 10
        assert p.is_local is True


class TestTopologyCostMatrix:
    def test_default_tiers(self):
        m = TopologyCostMatrix()
        hbm = m.get_tier(MemoryTier.LOCAL_HBM)
        ssd = m.get_tier(MemoryTier.SSD)
        assert hbm.latency_us < ssd.latency_us
        assert hbm.bandwidth_gb_s > ssd.bandwidth_gb_s

    def test_migration_cost(self):
        m = TopologyCostMatrix()
        cost = m.migration_cost(MemoryTier.LOCAL_HBM, MemoryTier.CPU_DRAM, size_gb=0.1)
        assert cost["latency_us"] > 0
        assert cost["energy_nj"] > 0
        assert cost["score"] > 0

    def test_access_cost(self):
        m = TopologyCostMatrix()
        hbm = m.access_cost(MemoryTier.LOCAL_HBM)
        ssd = m.access_cost(MemoryTier.SSD)
        assert hbm["latency_us"] < ssd["latency_us"]
        assert "score" in hbm

    def test_tier_order(self):
        m = TopologyCostMatrix()
        order = m.tier_order
        assert order[0] == MemoryTier.LOCAL_HBM
        assert order[-1] == MemoryTier.SSD

    def test_detect_fallback(self):
        m = TopologyCostMatrix.detect()
        hbm = m.get_tier(MemoryTier.LOCAL_HBM)
        assert hbm.bandwidth_gb_s > 0


class TestTopologyAwareTiering:
    def test_init(self):
        t = TopologyAwareTiering()
        assert t._promote_thresh == 0.3
        assert t.stats["tracked_blocks"] == 0

    def test_select_tier_hot_data(self):
        t = TopologyAwareTiering()
        assert t.select_tier(0.9, latency_sensitive=True) == MemoryTier.LOCAL_HBM

    def test_select_tier_warm_data(self):
        t = TopologyAwareTiering()
        tier = t.select_tier(0.6, size_mb=16)
        assert tier in (MemoryTier.NVLINK, MemoryTier.CPU_DRAM)

    def test_select_tier_cold_data(self):
        t = TopologyAwareTiering()
        assert t.select_tier(0.05, size_mb=2048) == MemoryTier.SSD

    def test_select_tier_large_warm(self):
        t = TopologyAwareTiering()
        assert t.select_tier(0.3, size_mb=512) == MemoryTier.CXL

    def test_record_access(self):
        t = TopologyAwareTiering()
        t.record_access("b1", MemoryTier.CPU_DRAM)
        assert t.stats["tracked_blocks"] == 1
        assert t._blocks["b1"]["tier"] == MemoryTier.CPU_DRAM

    def test_should_migrate_hot_block(self):
        t = TopologyAwareTiering()
        t.record_block_size("b1", 16)
        for _ in range(15):
            t.record_access("b1", MemoryTier.CPU_DRAM)
        assert t.should_migrate("b1", MemoryTier.LOCAL_HBM) is True

    def test_should_not_migrate_cold_block(self):
        t = TopologyAwareTiering()
        t.record_block_size("b2", 1024)
        t.record_access("b2", MemoryTier.CPU_DRAM)
        assert t.should_migrate("b2", MemoryTier.LOCAL_HBM) is False

    def test_should_not_migrate_same_tier(self):
        t = TopologyAwareTiering()
        t.record_access("b3", MemoryTier.LOCAL_HBM)
        assert t.should_migrate("b3", MemoryTier.LOCAL_HBM) is False

    def test_remove_block(self):
        t = TopologyAwareTiering()
        t.record_access("b1", MemoryTier.CPU_DRAM)
        t.remove_block("b1")
        assert t.stats["tracked_blocks"] == 0

    def test_record_block_size_creates_entry(self):
        t = TopologyAwareTiering()
        t.record_block_size("b1", 64)
        assert t._blocks["b1"]["size_mb"] == 64
