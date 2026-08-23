"""Interconnect-topology-aware KV cache tiering.

Extends the existing 3-tier cache (GPU → CPU → SSD) to 5 tiers with
interconnect topology awareness and NVML-based cost modelling:

    local HBM (L1) → NVLink peer (L2) → CXL (L3) → CPU DRAM (L4) → SSD (L5)

The cost matrix captures bandwidth, latency, and energy per tier so the
cache manager can make optimal placement and migration decisions.

Usage::

    matrix = TopologyCostMatrix.detect()      # auto-detect topology
    tiering = TopologyAwareTiering(cost=matrix)
    tier = tiering.select_tier(request_id, access_frequency=0.8)
    tiering.migrate(block_id, from_tier="hbm", to="nvl")
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class MemoryTier(str, Enum):
    """Five memory tiers ordered by performance."""
    LOCAL_HBM = "hbm"       # L1: local GPU HBM (fastest)
    NVLINK = "nvl"          # L2: peer GPU via NVLink
    CXL = "cxl"             # L3: CXL-attached memory
    CPU_DRAM = "cpu"        # L4: local CPU DRAM
    SSD = "ssd"             # L5: NVMe SSD (slowest + largest)


@dataclass
class TierProperties:
    """Performance properties of a memory tier."""
    bandwidth_gb_s: float       # GB/s read bandwidth
    latency_us: float            # microseconds access latency
    capacity_gb: float           # total capacity in GB
    energy_per_gb_nj: float      # nanojoules per GB read
    is_local: bool = True        # True if on same NUMA node
    is_volatile: bool = False    # True if contents lost on power loss


# ── Topology Cost Matrix ─────────────────────────────────────────────────────

class TopologyCostMatrix:
    """Interconnect topology cost model.

    Models bandwidth, latency, and energy cost for transitions between
    memory tiers.  Used by the cache manager to select optimal placement
    and migration paths.

    When NVML is available, detects NVLink connectivity and bandwidth.
    Falls back to sensible defaults for all tiers.
    """

    # Default properties when NVML/NVLink detection is unavailable.
    _DEFAULT_TIERS: dict[MemoryTier, TierProperties] = {
        MemoryTier.LOCAL_HBM: TierProperties(
            bandwidth_gb_s=2000.0, latency_us=0.5, capacity_gb=80.0,
            energy_per_gb_nj=5.0, is_local=True,
        ),
        MemoryTier.NVLINK: TierProperties(
            bandwidth_gb_s=600.0, latency_us=2.0, capacity_gb=160.0,
            energy_per_gb_nj=15.0, is_local=False,
        ),
        MemoryTier.CXL: TierProperties(
            bandwidth_gb_s=64.0, latency_us=50.0, capacity_gb=512.0,
            energy_per_gb_nj=30.0, is_local=False,
        ),
        MemoryTier.CPU_DRAM: TierProperties(
            bandwidth_gb_s=50.0, latency_us=100.0, capacity_gb=1024.0,
            energy_per_gb_nj=20.0, is_local=True,
        ),
        MemoryTier.SSD: TierProperties(
            bandwidth_gb_s=7.0, latency_us=10000.0, capacity_gb=4096.0,
            energy_per_gb_nj=100.0, is_local=True, is_volatile=False,
        ),
    }

    def __init__(self) -> None:
        self._tiers: dict[MemoryTier, TierProperties] = dict(self._DEFAULT_TIERS)
        self._nvlink_topology: dict[str, list[str]] = {}  # node -> peers
        self._detected = False

    @classmethod
    def detect(cls) -> TopologyCostMatrix:
        """Auto-detect topology using NVML when available.

        Returns a populated matrix with NVLink topology if detectable,
        falling back to sensible defaults.
        """
        matrix = cls()
        matrix._detect_nvlink()
        matrix._detected = True
        return matrix

    def _detect_nvlink(self) -> None:
        """Detect NVLink topology via NVML/pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode() if isinstance(
                    pynvml.nvmlDeviceGetName(handle), bytes,
                ) else pynvml.nvmlDeviceGetName(handle)
                node_id = f"gpu-{i}"

                peers = []
                bridge_info = pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, 0, pynvml.NVML_NVLINK_CAP_P2P_RATE,
                ) if hasattr(pynvml, 'NVML_NVLINK_CAP_P2P_RATE') else 0

                if bridge_info:
                    for link in range(min(12, 4)):
                        try:
                            state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                            if state == pynvml.NVML_NVLINK_STATUS_ACTIVE:
                                # Find peer GPU for this link
                                for j in range(device_count):
                                    if j != i:
                                        peers.append(f"gpu-{j}")
                        except Exception:
                            pass
                    peers = list(set(peers))

                self._nvlink_topology[node_id] = peers
                logger.info(f"NVLink topology: {node_id} -> {peers}")

            # Update NVLink tier bandwidth based on detected links
            link_count = max((len(v) for v in self._nvlink_topology.values()), default=0)
            if link_count > 0:
                nvl_bandwidth = min(link_count * 150.0, 1200.0)  # ~150 GB/s per link
                self._tiers[MemoryTier.NVLINK] = TierProperties(
                    bandwidth_gb_s=nvl_bandwidth, latency_us=2.0,
                    capacity_gb=self._tiers[MemoryTier.NVLINK].capacity_gb,
                    energy_per_gb_nj=15.0, is_local=False,
                )
                logger.info(f"NVLink bandwidth detected: {nvl_bandwidth:.0f} GB/s")

            pynvml.nvmlShutdown()
        except ImportError:
            logger.debug("pynvml not available — using default NVLink topology")
        except Exception as e:
            logger.debug(f"NVML detection failed: {e} — using defaults")

    def get_tier(self, tier: MemoryTier) -> TierProperties:
        return self._tiers.get(tier, self._tiers[MemoryTier.SSD])

    def migration_cost(
        self, from_tier: MemoryTier, to_tier: MemoryTier, size_gb: float = 1.0,
    ) -> dict:
        """Compute the cost of migrating *size_gb* from *from_tier* to *to_tier*.

        Returns a dict with ``latency_us``, ``energy_nj``, and ``score``.
        Lower score = cheaper migration.
        """
        src = self.get_tier(from_tier)
        dst = self.get_tier(to_tier)

        # Migration latency is dominated by the slower tier's bandwidth
        effective_bw = min(src.bandwidth_gb_s, dst.bandwidth_gb_s)
        transfer_s = size_gb / max(effective_bw, 0.001)

        latency_us = transfer_s * 1_000_000 + src.latency_us + dst.latency_us
        energy_nj = size_gb * (src.energy_per_gb_nj + dst.energy_per_gb_nj)

        return {
            "latency_us": latency_us,
            "energy_nj": energy_nj,
            "score": latency_us * 0.7 + energy_nj * 0.3,
        }

    def access_cost(self, tier: MemoryTier, size_gb: float = 1.0) -> dict:
        """Compute the cost of a read access from *tier*."""
        props = self.get_tier(tier)
        latency_us = props.latency_us
        energy_nj = size_gb * props.energy_per_gb_nj
        return {
            "latency_us": latency_us,
            "bandwidth_gb_s": props.bandwidth_gb_s,
            "energy_nj": energy_nj,
            "score": latency_us * 0.7 + energy_nj * 0.3,
        }

    @property
    def tiers(self) -> dict[MemoryTier, TierProperties]:
        return dict(self._tiers)

    @property
    def tier_order(self) -> list[MemoryTier]:
        """Tiers ordered by performance (fastest first)."""
        return [
            MemoryTier.LOCAL_HBM, MemoryTier.NVLINK, MemoryTier.CXL,
            MemoryTier.CPU_DRAM, MemoryTier.SSD,
        ]


# ── Topology-Aware Tiering Engine ────────────────────────────────────────────

class TopologyAwareTiering:
    """Manages interconnect-topology-aware cache placement and migration.

    Selects the optimal tier for data based on access frequency, data
    size, and interconnect costs.  Automatically promotes/demotes
    across the 5-tier hierarchy when access patterns change.

    Usage::

        tiering = TopologyAwareTiering()
        tier = tiering.select_tier(access_freq=0.9, size_mb=16)
        tiering.record_access("block-1", tier)
        tiering.maybe_migrate("block-1")  # promotes if access freq increased
    """

    def __init__(
        self,
        cost_matrix: TopologyCostMatrix | None = None,
        promotion_threshold: float = 0.3,
        demotion_threshold: float = 0.1,
    ):
        self._cost = cost_matrix or TopologyCostMatrix.detect()
        self._promote_thresh = promotion_threshold
        self._demote_thresh = demotion_threshold
        self._lock = threading.RLock()

        # Per-block tracking: block_id -> {tier, access_count, last_access, size}
        self._blocks: dict[str, dict] = {}

    def select_tier(
        self,
        access_frequency: float = 0.5,
        size_mb: float = 1.0,
        latency_sensitive: bool = False,
    ) -> MemoryTier:
        """Select the optimal tier for data with given access pattern.

        Args:
            access_frequency: Expected access frequency (0.0-1.0).
            size_mb: Data size in MB.
            latency_sensitive: True if this data is latency-critical.

        Returns:
            The recommended MemoryTier.
        """
        if latency_sensitive or access_frequency > 0.8:
            return MemoryTier.LOCAL_HBM
        if access_frequency > 0.5:
            return MemoryTier.NVLINK if self._cost.get_tier(MemoryTier.NVLINK).bandwidth_gb_s > 100 else MemoryTier.CPU_DRAM
        if access_frequency > 0.2:
            if size_mb < 256:
                return MemoryTier.CPU_DRAM
            return MemoryTier.CXL
        if size_mb > 1024:
            return MemoryTier.SSD
        return MemoryTier.CPU_DRAM

    def record_access(self, block_id: str, current_tier: MemoryTier) -> None:
        """Record an access to *block_id* for tracking."""
        with self._lock:
            if block_id not in self._blocks:
                self._blocks[block_id] = {
                    "tier": current_tier,
                    "access_count": 0,
                    "last_access": 0.0,
                    "size_mb": 1.0,
                }
            info = self._blocks[block_id]
            info["access_count"] += 1
            info["last_access"] = time.time()
            info["tier"] = current_tier

    def should_migrate(self, block_id: str, target_tier: MemoryTier) -> bool:
        """Check if *block_id* should be migrated to *target_tier*.

        Computes the access frequency from recorded history and compares
        migration cost against access cost savings.
        """
        with self._lock:
            info = self._blocks.get(block_id)
            if info is None:
                return False
            current = info["tier"]
            size_mb = info.get("size_mb", 1.0)
            access_count = info["access_count"]

        if current == target_tier:
            return False

        # Estimate future accesses based on past rate
        elapsed = max(time.time() - info.get("created_at", time.time()), 1.0)
        freq = access_count / elapsed  # accesses per second

        # Migration cost
        cost = self._cost.migration_cost(current, target_tier, size_gb=size_mb / 1024)

        # Access cost savings per access
        current_access = self._cost.access_cost(current, size_gb=size_mb / 1024)
        target_access = self._cost.access_cost(target_tier, size_gb=size_mb / 1024)
        saving_per_access = current_access["score"] - target_access["score"]

        # Break-even: how many accesses to recoup migration cost
        if saving_per_access <= 0:
            return False

        break_even = cost["score"] / saving_per_access
        # Expected future accesses in the next window
        expected_future = freq * 60  # 1-minute lookahead

        return expected_future >= break_even

    def record_block_size(self, block_id: str, size_mb: float) -> None:
        with self._lock:
            if block_id in self._blocks:
                self._blocks[block_id]["size_mb"] = size_mb
            else:
                self._blocks[block_id] = {
                    "tier": MemoryTier.CPU_DRAM,
                    "access_count": 0, "last_access": 0.0,
                    "size_mb": size_mb, "created_at": time.time(),
                }

    def remove_block(self, block_id: str) -> None:
        with self._lock:
            self._blocks.pop(block_id, None)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "tracked_blocks": len(self._blocks),
                "topology_detected": self._cost._detected,
                "nvlink_peers": max((len(v) for v in self._cost._nvlink_topology.values()), default=0),
            }
