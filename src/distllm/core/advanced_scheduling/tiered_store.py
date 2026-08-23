"""3-Tier GPU Memory Pool with Hierarchical Caching.

Implements a GPU VRAM → CPU RAM → NVMe SSD tiered cache for KV cache
entries. Provides automatic promotion/demotion based on access patterns,
enabling models to handle context lengths far exceeding GPU memory.

Tier Architecture:
- L1 (HOT): GPU VRAM — active KV cache, lowest latency (~1μs)
- L2 (WARM): CPU RAM with pinned memory — evicted cache, medium latency (~100μs)
- L3 (COLD): NVMe SSD — cold cache, highest latency (~10ms) but largest capacity

Access patterns drive automatic tier transitions:
- Frequent access → promote to L1
- Infrequent access → demote to L2, then L3
- LRU eviction within each tier

Usage::

    pool = TieredMemoryPool(
        gpu_memory_gb=24,
        cpu_memory_gb=128,
        nvme_path="/mnt/nvme/kv_cache",
    )
    pool.put("req_1", kv_tensor, tier=StorageTier.HOT)
    data = pool.get("req_1")  # Auto-promotes if in L2/L3
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from loguru import logger

# Magic prefix distinguishing serialized torch.Tensor payloads (with
# dtype/shape metadata) from raw byte payloads written to NVMe (L3).
_TENSOR_MAGIC = b"distllm_tensor\x00\x01"


class StorageTier(str, Enum):
    """Storage tier levels."""
    HOT = "hot"    # GPU VRAM
    WARM = "warm"  # CPU RAM (pinned)
    COLD = "cold"  # NVMe SSD


@dataclass
class TieredEntry:
    """A single entry in the tiered cache."""
    key: str
    data: Any  # torch.Tensor for HOT/WARM, bytes for COLD
    tier: StorageTier
    size_bytes: int
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    pinned: bool = False  # True if in pinned CPU memory


@dataclass
class TierStats:
    """Statistics for a single tier."""
    tier: StorageTier
    entry_count: int
    used_bytes: int
    capacity_bytes: int
    hit_count: int
    miss_count: int
    promotion_count: int
    demotion_count: int

    @property
    def utilization(self) -> float:
        return self.used_bytes / max(self.capacity_bytes, 1)

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / max(total, 1)


class TieredMemoryPool:
    """3-tier hierarchical memory pool for KV cache.

    Manages automatic promotion/demotion of KV cache entries across
    GPU VRAM, CPU RAM (pinned), and NVMe SSD tiers based on access
    patterns and memory pressure.

    Args:
        gpu_memory_gb: GPU VRAM budget for L1 (HOT) tier.
        cpu_memory_gb: CPU RAM budget for L2 (WARM) tier.
        nvme_path: Path for L3 (COLD) tier storage. None disables L3.
        nvme_max_gb: Maximum NVMe storage for L3 tier.
        promotion_threshold: Access count threshold for promotion.
        demotion_idle_s: Seconds of inactivity before demotion.
        block_size: Block size in bytes for alignment.
    """

    def __init__(
        self,
        gpu_memory_gb: float = 24.0,
        cpu_memory_gb: float = 128.0,
        nvme_path: str | None = None,
        nvme_max_gb: float = 500.0,
        promotion_threshold: int = 3,
        demotion_idle_s: float = 60.0,
        block_size: int = 4096,
    ):
        # Tier capacities
        self._l1_capacity = int(gpu_memory_gb * 1e9)
        self._l2_capacity = int(cpu_memory_gb * 1e9)
        self._l3_capacity = int(nvme_max_gb * 1e9) if nvme_path else 0
        self._block_size = block_size
        self._promotion_threshold = promotion_threshold
        self._demotion_idle_s = demotion_idle_s

        # Tier storage (OrderedDict for LRU tracking)
        self._l1_cache: OrderedDict[str, TieredEntry] = OrderedDict()
        self._l2_cache: OrderedDict[str, TieredEntry] = OrderedDict()
        self._l3_cache: OrderedDict[str, TieredEntry] = OrderedDict()

        # Usage tracking
        self._l1_used = 0
        self._l2_used = 0
        self._l3_used = 0

        # NVMe storage
        self._nvme_path: Path | None = Path(nvme_path) if nvme_path else None
        if self._nvme_path:
            self._nvme_path.mkdir(parents=True, exist_ok=True)

        # Stats
        self._stats = {
            tier: {
                "hits": 0, "misses": 0,
                "promotions": 0, "demotions": 0,
            }
            for tier in StorageTier
        }

        # Background maintenance
        self._lock = threading.Lock()
        self._running = False
        self._maintenance_thread: threading.Thread | None = None

        # Pinned memory pool for L2
        self._pinned_pool: dict[str, torch.Tensor] = {}

        logger.info(
            f"TieredMemoryPool initialized: "
            f"L1={gpu_memory_gb:.0f}GB GPU, "
            f"L2={cpu_memory_gb:.0f}GB CPU, "
            f"L3={nvme_max_gb:.0f}GB NVMe"
        )

    def start(self) -> None:
        """Start background maintenance thread."""
        if self._running:
            return
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="tiered-pool-maintenance",
        )
        self._maintenance_thread.start()
        logger.info("TieredMemoryPool maintenance started")

    def stop(self) -> None:
        """Stop background maintenance."""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)

    def get(self, key: str) -> Any | None:
        """Retrieve an entry, auto-promoting through tiers if needed.

        Checks L1 → L2 → L3 in order. On hit, promotes the entry
        one tier closer to L1 if it has been accessed frequently.

        Returns:
            The cached data (torch.Tensor or bytes), or None if not found.
        """
        with self._lock:
            # Check L1 (HOT)
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                entry.last_access = time.time()
                entry.access_count += 1
                self._l1_cache.move_to_end(key)
                self._stats[StorageTier.HOT]["hits"] += 1
                return entry.data

            # Check L2 (WARM)
            if key in self._l2_cache:
                entry = self._l2_cache[key]
                entry.last_access = time.time()
                entry.access_count += 1
                self._stats[StorageTier.WARM]["hits"] += 1

                # Promote to L1 if frequently accessed
                if entry.access_count >= self._promotion_threshold:
                    self._promote(key, StorageTier.WARM, StorageTier.HOT)
                else:
                    self._l2_cache.move_to_end(key)

                return entry.data

            # Check L3 (COLD)
            if key in self._l3_cache:
                entry = self._l3_cache[key]
                entry.last_access = time.time()
                entry.access_count += 1
                self._stats[StorageTier.COLD]["hits"] += 1

                # Load from NVMe
                data = self._load_from_nvme(key)
                if data is None:
                    self._l3_cache.pop(key, None)
                    return None

                # Promote to L2
                self._promote(key, StorageTier.COLD, StorageTier.WARM, data=data)
                return data

            # Not found in any tier
            self._stats[StorageTier.HOT]["misses"] += 1
            return None

    def put(
        self,
        key: str,
        data: Any,
        tier: StorageTier = StorageTier.HOT,
    ) -> bool:
        """Store an entry in the specified tier.

        If the tier is full, evicts the coldest entries to make room.

        Args:
            key: Cache key.
            data: The data to cache (torch.Tensor for HOT/WARM, bytes for COLD).
            tier: Target storage tier.

        Returns:
            True if stored successfully.
        """
        size_bytes = self._estimate_size(data)

        with self._lock:
            if tier == StorageTier.HOT:
                return self._put_l1(key, data, size_bytes)
            elif tier == StorageTier.WARM:
                return self._put_l2(key, data, size_bytes)
            else:
                return self._put_l3(key, data, size_bytes)

    def evict(self, target_bytes: int = 0) -> int:
        """Evict entries to free space.

        Demotes entries from L1 → L2 → L3 based on access patterns.

        Args:
            target_bytes: Bytes to free. If 0, evicts to 80% capacity.

        Returns:
            Bytes freed.
        """
        freed = 0
        with self._lock:
            # Demote coldest L1 entries to L2
            while self._l1_used > self._l1_capacity * 0.8:
                if not self._l1_cache:
                    break
                key, entry = self._l1_cache.popitem(last=False)
                self._l1_used -= entry.size_bytes
                self._demote_to_l2(key, entry)
                freed += entry.size_bytes

            # Demote coldest L2 entries to L3
            while self._l2_used > self._l2_capacity * 0.8:
                if not self._l2_cache:
                    break
                key, entry = self._l2_cache.popitem(last=False)
                self._l2_used -= entry.size_bytes
                self._demote_to_l3(key, entry)
                freed += entry.size_bytes

        return freed

    def _put_l1(self, key: str, data: Any, size_bytes: int) -> bool:
        """Store in L1 (GPU VRAM)."""
        # Evict if needed
        while self._l1_used + size_bytes > self._l1_capacity and self._l1_cache:
            evict_key, evict_entry = self._l1_cache.popitem(last=False)
            self._l1_used -= evict_entry.size_bytes
            self._demote_to_l2(evict_key, evict_entry)

        if self._l1_used + size_bytes > self._l1_capacity:
            logger.warning(f"L1 cache full, cannot store {key} ({size_bytes} bytes)")
            return False

        entry = TieredEntry(
            key=key, data=data, tier=StorageTier.HOT,
            size_bytes=size_bytes,
        )
        self._l1_cache[key] = entry
        self._l1_used += size_bytes
        return True

    def _put_l2(self, key: str, data: Any, size_bytes: int) -> bool:
        """Store in L2 (CPU RAM with pinned memory)."""
        # Try to pin memory for faster GPU transfer
        pinned = False
        if isinstance(data, torch.Tensor) and data.is_cuda:
            try:
                pinned_data = torch.empty_like(data, pin_memory=True)
                pinned_data.copy_(data)
                data = pinned_data
                pinned = True
            except RuntimeError:
                pass

        # Evict if needed
        while self._l2_used + size_bytes > self._l2_capacity and self._l2_cache:
            evict_key, evict_entry = self._l2_cache.popitem(last=False)
            self._l2_used -= evict_entry.size_bytes
            self._demote_to_l3(evict_key, evict_entry)

        entry = TieredEntry(
            key=key, data=data, tier=StorageTier.WARM,
            size_bytes=size_bytes, pinned=pinned,
        )
        self._l2_cache[key] = entry
        self._l2_used += size_bytes
        return True

    def _put_l3(self, key: str, data: Any, size_bytes: int) -> bool:
        """Store in L3 (NVMe SSD)."""
        if self._nvme_path is None:
            return False

        # Evict if needed
        while self._l3_used + size_bytes > self._l3_capacity and self._l3_cache:
            evict_key, _ = self._l3_cache.popitem(last=False)
            self._delete_from_nvme(evict_key)

        # Write to NVMe
        if not self._save_to_nvme(key, data):
            return False

        entry = TieredEntry(
            key=key, data=None, tier=StorageTier.COLD,
            size_bytes=size_bytes,
        )
        self._l3_cache[key] = entry
        self._l3_used += size_bytes
        return True

    def _promote(
        self,
        key: str,
        from_tier: StorageTier,
        to_tier: StorageTier,
        data: Any | None = None,
    ) -> None:
        """Promote an entry from one tier to another."""
        if from_tier == StorageTier.WARM and to_tier == StorageTier.HOT:
            entry = self._l2_cache.pop(key, None)
            if entry is None:
                return
            self._l2_used -= entry.size_bytes
            entry.tier = StorageTier.HOT
            self._put_l1(key, entry.data, entry.size_bytes)
            self._stats[StorageTier.HOT]["promotions"] += 1

        elif from_tier == StorageTier.COLD and to_tier == StorageTier.WARM:
            entry = self._l3_cache.pop(key, None)
            if entry is None:
                return
            self._l3_used -= entry.size_bytes
            entry.data = data
            entry.tier = StorageTier.WARM
            self._put_l2(key, data, entry.size_bytes)
            self._stats[StorageTier.WARM]["promotions"] += 1

    def _demote_to_l2(self, key: str, entry: TieredEntry) -> None:
        """Demote an entry from L1 to L2."""
        entry.tier = StorageTier.WARM
        entry.pinned = False
        # Try to pin for faster future promotion
        if isinstance(entry.data, torch.Tensor):
            try:
                pinned = torch.empty_like(entry.data, pin_memory=True)
                pinned.copy_(entry.data)
                entry.data = pinned
                entry.pinned = True
            except RuntimeError:
                if entry.data.is_cuda:
                    entry.data = entry.data.cpu()

        self._l2_cache[key] = entry
        self._l2_used += entry.size_bytes
        self._stats[StorageTier.WARM]["demotions"] += 1

    def _demote_to_l3(self, key: str, entry: TieredEntry) -> None:
        """Demote an entry from L2 to L3 (NVMe)."""
        if self._nvme_path is None:
            # No L3 — just drop the entry
            self._stats[StorageTier.COLD]["demotions"] += 1
            return

        # Save to NVMe (torch.Tensors persist with dtype/shape metadata so the
        # L3 round-trip returns a real tensor, not opaque bytes).
        if self._save_to_nvme(key, entry.data):
            entry.tier = StorageTier.COLD
            entry.data = None  # Data is on disk
            self._l3_cache[key] = entry
            self._l3_used += entry.size_bytes
            self._stats[StorageTier.COLD]["demotions"] += 1

    def _save_to_nvme(self, key: str, data: Any) -> bool:
        """Save data to NVMe storage.

        ``torch.Tensor`` payloads are written as a magic-prefixed header
        (dtype/shape) followed by the raw bytes so ``_load_from_nvme`` can
        reconstruct the original tensor.  Non-tensor data is written verbatim.
        """
        if self._nvme_path is None:
            return False
        try:
            file_path = self._nvme_path / f"{key}.bin"
            if isinstance(data, torch.Tensor):
                cpu = data.detach().cpu().contiguous()
                meta = {
                    "dtype": str(cpu.dtype).replace("torch.", ""),
                    "shape": list(cpu.shape),
                }
                header = json.dumps(meta).encode("utf-8")
                payload = (
                    _TENSOR_MAGIC
                    + struct.pack(">I", len(header))
                    + header
                    + cpu.numpy().tobytes()
                )
                file_path.write_bytes(payload)
            else:
                file_path.write_bytes(data)
            return True
        except OSError as e:
            logger.warning(f"NVMe write failed for {key}: {e}")
            return False

    def _load_from_nvme(self, key: str) -> Any:
        """Load data from NVMe storage.

        Reconstructs ``torch.Tensor`` payloads saved by ``_save_to_nvme``
        (restoring dtype and shape) so callers receive the original tensor
        rather than opaque bytes.
        """
        if self._nvme_path is None:
            return None
        try:
            file_path = self._nvme_path / f"{key}.bin"
            if not file_path.exists():
                return None
            raw = file_path.read_bytes()
            if raw.startswith(_TENSOR_MAGIC):
                offset = len(_TENSOR_MAGIC)
                (hlen,) = struct.unpack(">I", raw[offset:offset + 4])
                offset += 4
                meta = json.loads(raw[offset:offset + hlen].decode("utf-8"))
                offset += hlen
                dtype = getattr(torch, meta["dtype"], torch.float32)
                tensor = torch.frombuffer(raw[offset:], dtype=dtype)
                return tensor.reshape(meta["shape"]).clone()
            return raw
        except OSError as e:
            logger.warning(f"NVMe read failed for {key}: {e}")
            return None

    def _delete_from_nvme(self, key: str) -> None:
        """Delete data from NVMe storage."""
        if self._nvme_path is None:
            return
        try:
            file_path = self._nvme_path / f"{key}.bin"
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass

    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, bytes):
            return len(data)
        elif isinstance(data, (bytearray, memoryview)):
            return len(data)
        return 0

    def _maintenance_loop(self) -> None:
        """Background maintenance: demote idle entries."""
        while self._running:
            try:
                self._demote_idle_entries()
            except Exception as e:
                logger.debug(f"Maintenance error: {e}")

            deadline = time.time() + 30.0
            while self._running and time.time() < deadline:
                time.sleep(1.0)

    def _demote_idle_entries(self) -> None:
        """Demote entries that haven't been accessed recently."""
        now = time.time()
        with self._lock:
            # Demote idle L1 entries to L2
            idle_keys = [
                key for key, entry in self._l1_cache.items()
                if now - entry.last_access > self._demotion_idle_s
            ]
            for key in idle_keys[:10]:  # Limit per cycle
                entry = self._l1_cache.pop(key, None)
                if entry:
                    self._l1_used -= entry.size_bytes
                    self._demote_to_l2(key, entry)

            # Demote idle L2 entries to L3
            idle_keys = [
                key for key, entry in self._l2_cache.items()
                if now - entry.last_access > self._demotion_idle_s * 2
            ]
            for key in idle_keys[:10]:
                entry = self._l2_cache.pop(key, None)
                if entry:
                    self._l2_used -= entry.size_bytes
                    self._demote_to_l3(key, entry)

    def get_stats(self) -> dict:
        """Return comprehensive tier statistics."""
        with self._lock:
            return {
                "l1_hot": TierStats(
                    tier=StorageTier.HOT,
                    entry_count=len(self._l1_cache),
                    used_bytes=self._l1_used,
                    capacity_bytes=self._l1_capacity,
                    hit_count=self._stats[StorageTier.HOT]["hits"],
                    miss_count=self._stats[StorageTier.HOT]["misses"],
                    promotion_count=self._stats[StorageTier.HOT]["promotions"],
                    demotion_count=self._stats[StorageTier.HOT]["demotions"],
                ).__dict__,
                "l2_warm": TierStats(
                    tier=StorageTier.WARM,
                    entry_count=len(self._l2_cache),
                    used_bytes=self._l2_used,
                    capacity_bytes=self._l2_capacity,
                    hit_count=self._stats[StorageTier.WARM]["hits"],
                    miss_count=self._stats[StorageTier.WARM]["misses"],
                    promotion_count=self._stats[StorageTier.WARM]["promotions"],
                    demotion_count=self._stats[StorageTier.WARM]["demotions"],
                ).__dict__,
                "l3_cold": TierStats(
                    tier=StorageTier.COLD,
                    entry_count=len(self._l3_cache),
                    used_bytes=self._l3_used,
                    capacity_bytes=self._l3_capacity,
                    hit_count=self._stats[StorageTier.COLD]["hits"],
                    miss_count=self._stats[StorageTier.COLD]["misses"],
                    promotion_count=self._stats[StorageTier.COLD]["promotions"],
                    demotion_count=self._stats[StorageTier.COLD]["demotions"],
                ).__dict__,
                "total_entries": len(self._l1_cache) + len(self._l2_cache) + len(self._l3_cache),
                "total_used_gb": (self._l1_used + self._l2_used + self._l3_used) / 1e9,
            }

    def clear(self) -> None:
        """Clear all tiers."""
        with self._lock:
            self._l1_cache.clear()
            self._l2_cache.clear()
            self._l3_cache.clear()
            self._l1_used = 0
            self._l2_used = 0
            self._l3_used = 0
            # Clean up NVMe
            if self._nvme_path and self._nvme_path.exists():
                for f in self._nvme_path.glob("*.bin"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
