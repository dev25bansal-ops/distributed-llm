"""Pluggable eviction policies for PagedAttention block pools.

Provides a common interface for different eviction strategies,
enabling experimentation with LRU, LFU, FIFO, 2Q, ARC, and
ML-driven policies.

Usage::

    from distllm.core.block_eviction_policy import (
        LRUPolicy, LFUPolicy, ARCPolicy, create_policy,
    )

    policy = create_policy("lru")
    policy.record_access(block_id=42)
    victim = policy.pick_victim(block_usage_dict)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


class EvictionPolicy(ABC):
    """Base class for block eviction policies."""

    @abstractmethod
    def record_access(self, block_id: int) -> None:
        """Record that a block was accessed."""

    @abstractmethod
    def record_allocation(self, block_id: int) -> None:
        """Record that a block was newly allocated."""

    @abstractmethod
    def record_free(self, block_id: int) -> None:
        """Record that a block was freed."""

    @abstractmethod
    def pick_victim(self, block_usage: Dict[int, Any]) -> Optional[int]:
        """Pick a block to evict. Returns block_id or None."""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return policy statistics."""


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction.

    Uses an OrderedDict as a doubly-linked list for O(1) access
    updates and O(1) victim selection.
    """

    def __init__(self) -> None:
        self._order: OrderedDict[int, None] = OrderedDict()
        self._hits = 0

    def record_access(self, block_id: int) -> None:
        if block_id in self._order:
            self._order.move_to_end(block_id)
        else:
            self._order[block_id] = None
        self._hits += 1

    def record_allocation(self, block_id: int) -> None:
        self._order[block_id] = None
        self._order.move_to_end(block_id)

    def record_free(self, block_id: int) -> None:
        self._order.pop(block_id, None)

    def pick_victim(self, block_usage: Dict[int, Any]) -> Optional[int]:
        for bid in self._order:
            if bid in block_usage and block_usage[bid].ref_count <= 0:
                return bid
        # If no free-refcount block, pick the oldest
        for bid in self._order:
            if bid in block_usage:
                return bid
        return None

    def stats(self) -> Dict[str, Any]:
        return {"policy": "lru", "tracked": len(self._order), "hits": self._hits}

    def __repr__(self) -> str:
        return f"LRUPolicy(tracked={len(self._order)})"


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction with aging.

    Maintains access counts with exponential decay to prevent
    stale entries from dominating.
    """

    def __init__(self, decay_interval_s: float = 60.0):
        self._counts: Dict[int, int] = {}
        self._last_decay: float = time.time()
        self._decay_interval = decay_interval_s
        self._total_accesses = 0

    def _maybe_decay(self) -> None:
        now = time.time()
        if now - self._last_decay > self._decay_interval:
            for bid in list(self._counts.keys()):
                self._counts[bid] = max(1, self._counts[bid] // 2)
            self._last_decay = now

    def record_access(self, block_id: int) -> None:
        self._maybe_decay()
        self._counts[block_id] = self._counts.get(block_id, 0) + 1
        self._total_accesses += 1

    def record_allocation(self, block_id: int) -> None:
        self._counts[block_id] = 1

    def record_free(self, block_id: int) -> None:
        self._counts.pop(block_id, None)

    def pick_victim(self, block_usage: Dict[int, Any]) -> Optional[int]:
        if not self._counts:
            return None
        # Pick the block with the lowest count
        victim = min(
            (bid for bid in self._counts if bid in block_usage),
            key=lambda b: self._counts.get(b, 0),
            default=None,
        )
        return victim

    def stats(self) -> Dict[str, Any]:
        return {
            "policy": "lfu",
            "tracked": len(self._counts),
            "total_accesses": self._total_accesses,
        }

    def __repr__(self) -> str:
        return f"LFUPolicy(tracked={len(self._counts)}, accesses={self._total_accesses})"


class FIFOPolicy(EvictionPolicy):
    """First-In-First-Out eviction.

    Simple queue-based policy — evicts the oldest allocated block
    regardless of access pattern.  Useful as a baseline.
    """

    def __init__(self) -> None:
        self._queue: OrderedDict[int, None] = OrderedDict()

    def record_access(self, block_id: int) -> None:
        pass  # FIFO ignores access patterns

    def record_allocation(self, block_id: int) -> None:
        self._queue[block_id] = None

    def record_free(self, block_id: int) -> None:
        self._queue.pop(block_id, None)

    def pick_victim(self, block_usage: Dict[int, Any]) -> Optional[int]:
        for bid in self._queue:
            if bid in block_usage:
                return bid
        return None

    def stats(self) -> Dict[str, Any]:
        return {"policy": "fifo", "tracked": len(self._queue)}

    def __repr__(self) -> str:
        return f"FIFOPolicy(tracked={len(self._queue)})"


class TwoQPolicy(EvictionPolicy):
    """2Q eviction policy.

    Maintains two queues:
    - ``A1in``: recent blocks (FIFO, bounded size)
    - ``Am``: frequently accessed blocks (LRU)

    A block enters A1in on first access.  On second access it is
    promoted to Am.  Eviction targets A1in first, then Am.
    """

    def __init__(self, a1in_fraction: float = 0.25, max_blocks: int = 1024):
        self._a1in_limit = max(1, int(max_blocks * a1in_fraction))
        self._a1in: OrderedDict[int, None] = OrderedDict()
        self._am: OrderedDict[int, None] = OrderedDict()
        self._promotions = 0

    def record_access(self, block_id: int) -> None:
        if block_id in self._am:
            self._am.move_to_end(block_id)
        elif block_id in self._a1in:
            # Second access — promote to Am
            del self._a1in[block_id]
            self._am[block_id] = None
            self._promotions += 1
        else:
            self._a1in[block_id] = None
            # Evict from A1in if over limit
            while len(self._a1in) > self._a1in_limit:
                self._a1in.popitem(last=False)

    def record_allocation(self, block_id: int) -> None:
        self._a1in[block_id] = None

    def record_free(self, block_id: int) -> None:
        self._a1in.pop(block_id, None)
        self._am.pop(block_id, None)

    def pick_victim(self, block_usage: Dict[int, Any]) -> Optional[int]:
        # Evict from A1in first
        for bid in self._a1in:
            if bid in block_usage:
                return bid
        # Then from Am (LRU end)
        for bid in self._am:
            if bid in block_usage:
                return bid
        return None

    def stats(self) -> Dict[str, Any]:
        return {
            "policy": "2q",
            "a1in_size": len(self._a1in),
            "am_size": len(self._am),
            "promotions": self._promotions,
        }

    def __repr__(self) -> str:
        return f"TwoQPolicy(a1in={len(self._a1in)}, am={len(self._am)})"


class ARCPolicy(EvictionPolicy):
    """Adaptive Replacement Cache (ARC) eviction.

    Maintains four lists (T1, T2, B1, B2) and adaptively balances
    between recency and frequency based on workload patterns.

    - T1: recent blocks (LRU)
    - T2: frequent blocks (LRU)
    - B1: ghost entries evicted from T1
    - B2: ghost entries evicted from T2
    """

    def __init__(self, max_blocks: int = 1024):
        self._p = 0  # target size for T1
        self._c = max_blocks  # total capacity
        self._t1: OrderedDict[int, None] = OrderedDict()
        self._t2: OrderedDict[int, None] = OrderedDict()
        self._b1: OrderedDict[int, None] = OrderedDict()
        self._b2: OrderedDict[int, None] = OrderedDict()

    def record_access(self, block_id: int) -> None:
        if block_id in self._t1:
            del self._t1[block_id]
            self._t2[block_id] = None
            self._t2.move_to_end(block_id)
        elif block_id in self._t2:
            self._t2.move_to_end(block_id)
        elif block_id in self._b1:
            # Hit in B1 — increase T1 target
            self._p = min(self._c, self._p + max(1, len(self._b2) // max(len(self._b1), 1)))
            self._replace(block_id, in_b1=True)
            del self._b1[block_id]
            self._t2[block_id] = None
        elif block_id in self._b2:
            # Hit in B2 — decrease T1 target
            self._p = max(0, self._p - max(1, len(self._b1) // max(len(self._b2), 1)))
            self._replace(block_id, in_b1=False)
            del self._b2[block_id]
            self._t2[block_id] = None
        else:
            self._replace(block_id, in_b1=False)
            self._t1[block_id] = None
            self._t1.move_to_end(block_id)

    def _replace(self, block_id: int, in_b1: bool) -> None:
        if len(self._t1) > 0 and (
            (in_b1 and len(self._t1) > self._p) or
            (not in_b1 and len(self._t1) >= max(1, self._p))
        ):
            evicted, _ = self._t1.popitem(last=False)
            self._b1[evicted] = None
            if len(self._b1) > self._c:
                self._b1.popitem(last=False)
        elif len(self._t2) > 0:
            evicted, _ = self._t2.popitem(last=False)
            self._b2[evicted] = None
            if len(self._b2) > self._c:
                self._b2.popitem(last=False)

    def record_allocation(self, block_id: int) -> None:
        self._t1[block_id] = None

    def record_free(self, block_id: int) -> None:
        self._t1.pop(block_id, None)
        self._t2.pop(block_id, None)
        self._b1.pop(block_id, None)
        self._b2.pop(block_id, None)

    def pick_victim(self, block_usage: Dict[int, Any]) -> Optional[int]:
        for bid in self._t1:
            if bid in block_usage:
                return bid
        for bid in self._t2:
            if bid in block_usage:
                return bid
        return None

    def stats(self) -> Dict[str, Any]:
        return {
            "policy": "arc",
            "t1_size": len(self._t1),
            "t2_size": len(self._t2),
            "b1_size": len(self._b1),
            "b2_size": len(self._b2),
            "target_t1": self._p,
        }

    def __repr__(self) -> str:
        return (
            f"ARCPolicy(t1={len(self._t1)}, t2={len(self._t2)}, "
            f"b1={len(self._b1)}, b2={len(self._b2)}, p={self._p})"
        )


_POLICY_REGISTRY: Dict[str, Type[EvictionPolicy]] = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "fifo": FIFOPolicy,
    "2q": TwoQPolicy,
    "arc": ARCPolicy,
}


def create_policy(name: str, **kwargs: Any) -> EvictionPolicy:
    """Create an eviction policy by name.

    Supported names: "lru", "lfu", "fifo", "2q", "arc".

    Raises:
        ValueError: If the policy name is unknown.
    """
    cls = _POLICY_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown eviction policy: {name!r}. "
            f"Available: {list(_POLICY_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def list_policies() -> List[str]:
    """Return the names of all available eviction policies."""
    return list(_POLICY_REGISTRY.keys())
