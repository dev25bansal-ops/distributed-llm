"""Cross-Node Prompt Caching and KV Cache Trading System.

Combines a consistent hash ring, a lightweight KV cache slot marketplace,
a learned (RL-based) eviction policy, and a Bloom/Cuckoo filter cascade
for sub-millisecond cache lookups across a distributed cluster.

Components
----------
- :class:`ConsistentHashRing` — Ketama-based sharding with 160 virtual nodes
  per unit of weight.
- :class:`CacheSlot` / :class:`CacheMarket` — Lightweight auction for KV
  cache slots with gossip-based periodic clearing.
- :class:`LearnedEvictionPolicy` — Feature-weighted eviction scoring with
  an online feedback loop that adjusts weights from hit/miss outcomes.
- :class:`BloomFilterCascade` — Three-layer (Bloom → Cuckoo → exact)
  sub-millisecond membership test.
- :class:`GaiaCache` — Top-level orchestrator that ties the four subsystems
  together.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KETAMA_VNODES: Final[int] = 160
_RING_SIZE: Final[int] = 2**32 - 1  # 32-bit hash ring
_BLOOM_DEFAULT_SIZE: Final[int] = 1 << 20  # 1M bits
_BLOOM_DEFAULT_HASHES: Final[int] = 7
_CUCKOO_BUCKETS: Final[int] = 1 << 16  # 65536 buckets
_CUCKOO_FINGERPRINT_BITS: Final[int] = 16
_CUCKOO_MAX_KICKS: Final[int] = 500
_MARKET_CLEAR_INTERVAL: Final[float] = 30.0  # seconds
_DEFAULT_TTL: Final[float] = 3600.0  # 1 hour
_EVICTION_LEARNING_RATE: Final[float] = 0.05
_EVICTION_DISCOUNT: Final[float] = 0.9

# ---------------------------------------------------------------------------
# ConsistentHashRing
# ---------------------------------------------------------------------------


class ConsistentHashRing:
    """Ketama-style consistent hash ring for prefix cache sharding.

    Each real node is mapped to ``160 * weight`` virtual node positions on
    a 32-bit ring.  ``get_node(key)`` walks clockwise from the key's hash
    to find the owning node.

    Thread-safe for concurrent reads; writes use a lock.
    """

    def __init__(self, virtual_node_count: int = _KETAMA_VNODES) -> None:
        self._vnode_count = virtual_node_count
        self._ring: list[tuple[int, str]] = []  # sorted (hash, node_id)
        self._nodes: dict[str, float] = {}  # node_id -> weight
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, weight: float = 1.0) -> None:
        """Add *node_id* to the ring.

        Parameters
        ----------
        node_id :
            Unique node identifier (e.g. ``"node-abc"``).
        weight :
            Relative capacity weight.  Higher values produce more virtual
            nodes and a larger share of the hash space.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        with self._lock:
            if node_id in self._nodes:
                logger.warning("Node {!r} already present; updating weight", node_id)
            self._nodes[node_id] = weight
            self._rebuild_ring_locked()

    def remove_node(self, node_id: str) -> None:
        """Remove *node_id* from the ring."""
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node {node_id!r} not found")
            del self._nodes[node_id]
            self._rebuild_ring_locked()

    def get_node(self, key: str) -> str | None:
        """Return the node that owns *key*, or ``None`` if the ring is empty."""
        if not self._ring:
            return None
        h = self._ketama_hash(key)
        with self._lock:
            # Binary search for the first hash >= h
            idx = self._first_ge(h)
            return self._ring[idx][1]

    def get_nodes(self, key: str, count: int = 1) -> list[str]:
        """Return the *count* distinct nodes responsible for *key*.

        Used for replication / N-way routing.
        """
        if not self._ring or count <= 0:
            return []
        h = self._ketama_hash(key)
        seen: set[str] = set()
        result: list[str] = []
        with self._lock:
            if not self._ring:
                return []
            start = self._first_ge(h)
            for i in range(len(self._ring)):
                node = self._ring[(start + i) % len(self._ring)][1]
                if node not in seen:
                    seen.add(node)
                    result.append(node)
                    if len(result) >= count:
                        break
        return result

    def nodes(self) -> dict[str, float]:
        """Return a copy of the current node map (node_id -> weight)."""
        with self._lock:
            return dict(self._nodes)

    @property
    def size(self) -> int:
        """Number of nodes currently in the ring."""
        with self._lock:
            return len(self._nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ketama_hash(key: str) -> int:
        """MD5-based Ketama hash returning a 32-bit integer."""
        digest = hashlib.md5(key.encode("utf-8")).digest()
        return struct.unpack_from("<I", digest, 0)[0] & 0x7FFFFFFF

    def _rebuild_ring_locked(self) -> None:
        """Rebuild ``self._ring`` from ``self._nodes`` (lock must be held)."""
        new_ring: list[tuple[int, str]] = []
        for node_id, weight in self._nodes.items():
            vnodes = max(1, int(self._vnode_count * weight))
            for v_idx in range(vnodes):
                h = self._ketama_hash(f"{node_id}:{v_idx}")
                new_ring.append((h, node_id))
        new_ring.sort(key=lambda x: x[0])
        self._ring = new_ring

    def _first_ge(self, h: int) -> int:
        """Return index of the first element with hash >= *h*.

        Uses binary search.  Assumes ``self._ring`` is sorted and non-empty.
        """
        # Linear scan for rings up to a few thousand entries; binary search
        # for larger rings.
        ring = self._ring
        lo, hi = 0, len(ring) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if ring[mid][0] < h:
                lo = mid + 1
            else:
                hi = mid
        if ring[lo][0] < h:
            return 0  # wrap around
        return lo


# ---------------------------------------------------------------------------
# CacheSlot & CacheMarket
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSlot:
    """A single KV cache slot listed on the market.

    Attributes
    ----------
    slot_id :
        Globally unique slot identifier.
    node_id :
        The node that owns/hosted this slot.
    prefix_hash :
        Hash of the prompt prefix this slot caches.
    size_mb :
        Estimated size of the cached KV data in mebibytes.
    price :
        Asking price in arbitrary units (e.g. credits, tokens).
    expires_at :
        Unix timestamp after which the listing is stale.
    buyer :
        Node id of the buyer, or empty string if unassigned.
    """

    slot_id: str
    node_id: str
    prefix_hash: str
    size_mb: float
    price: float
    expires_at: float
    buyer: str = ""


class CacheMarket:
    """Lightweight gossip-based auction for KV cache slots.

    Nodes advertise available slots via :meth:`sell`; other nodes discover
    and acquire them via :meth:`buy`.  The market runs a simple first-price
    auction and clears stale entries every ``clear_interval`` seconds.

    Thread-safe.
    """

    def __init__(
        self,
        clear_interval: float = _MARKET_CLEAR_INTERVAL,
        local_node_id: str = "",
    ) -> None:
        self._slots: dict[str, CacheSlot] = {}  # slot_id -> slot
        self._clear_interval = clear_interval
        self._local_node_id = local_node_id
        self._last_clear = time.time()
        self._tx_count = 0  # total completed transactions
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available(
        self,
        prefix_hash: str | None = None,
        max_price: float | None = None,
        min_size_mb: float | None = None,
    ) -> list[CacheSlot]:
        """Return available (unbought, non-expired) slots matching filters.

        Parameters
        ----------
        prefix_hash :
            If given, only slots with this prefix hash are returned.
        max_price :
            If given, only slots whose price does not exceed this value.
        min_size_mb :
            If given, only slots at least this large.
        """
        self._maybe_clear()
        now = time.time()
        result: list[CacheSlot] = []
        with self._lock:
            for slot in self._slots.values():
                if slot.buyer:
                    continue
                if slot.expires_at <= now:
                    continue
                if prefix_hash is not None and slot.prefix_hash != prefix_hash:
                    continue
                if max_price is not None and slot.price > max_price:
                    continue
                if min_size_mb is not None and slot.size_mb < min_size_mb:
                    continue
                result.append(slot)
        return result

    def buy(self, slot_id: str, buyer_node: str) -> CacheSlot | None:
        """Purchase a slot, transferring ownership to *buyer_node*.

        Returns the slot if successful, ``None`` if the slot is already sold
        or expired.
        """
        self._maybe_clear()
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                return None
            if slot.buyer:
                logger.warning("Slot {!r} already bought by {!r}", slot_id, slot.buyer)
                return None
            if slot.expires_at <= time.time():
                return None
            bought = CacheSlot(
                slot_id=slot.slot_id,
                node_id=slot.node_id,
                prefix_hash=slot.prefix_hash,
                size_mb=slot.size_mb,
                price=slot.price,
                expires_at=slot.expires_at,
                buyer=buyer_node,
            )
            self._slots[slot_id] = bought
            self._tx_count += 1
        logger.info(
            "Slot {!r} purchased by {!r} from {!r} for {:.2f}",
            slot_id,
            buyer_node,
            slot.node_id,
            slot.price,
        )
        return bought

    def sell(
        self,
        prefix_hash: str,
        size_mb: float,
        price: float,
        ttl: float = _DEFAULT_TTL,
        node_id: str | None = None,
    ) -> CacheSlot:
        """List a new slot on the market.

        Parameters
        ----------
        prefix_hash :
            Hash of the prompt prefix being cached.
        size_mb :
            Estimated size of the cached data.
        price :
            Asking price.
        ttl :
            Time-to-live in seconds before the listing expires.
        node_id :
            Owning node.  Defaults to ``self._local_node_id``.

        Returns
        -------
        CacheSlot
            The newly created slot.
        """
        if node_id is None:
            node_id = self._local_node_id or "unknown"
        slot_id = self._generate_slot_id(node_id, prefix_hash)
        now = time.time()
        slot = CacheSlot(
            slot_id=slot_id,
            node_id=node_id,
            prefix_hash=prefix_hash,
            size_mb=size_mb,
            price=price,
            expires_at=now + ttl,
        )
        with self._lock:
            self._slots[slot_id] = slot
        logger.debug("Slot {!r} listed by {!r} (size={}MB, price={})", slot_id, node_id, size_mb, price)
        return slot

    def cancel(self, slot_id: str) -> bool:
        """Remove a slot listing.  Returns True if it was removed."""
        with self._lock:
            if slot_id in self._slots:
                del self._slots[slot_id]
                return True
        return False

    def stats(self) -> dict[str, Any]:
        """Return current market statistics."""
        self._maybe_clear()
        with self._lock:
            now = time.time()
            total = len(self._slots)
            available = sum(1 for s in self._slots.values() if not s.buyer and s.expires_at > now)
            sold = sum(1 for s in self._slots.values() if s.buyer)
            expired = sum(1 for s in self._slots.values() if s.expires_at <= now)
        return {
            "total_slots": total,
            "available_slots": available,
            "sold_slots": sold,
            "expired_slots": expired,
            "transactions": self._tx_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_clear(self) -> None:
        """Expire old listings if the clear interval has elapsed."""
        now = time.time()
        if now - self._last_clear < self._clear_interval:
            return
        with self._lock:
            before = len(self._slots)
            self._slots = {
                sid: s
                for sid, s in self._slots.items()
                if s.buyer or s.expires_at > now
            }
            removed = before - len(self._slots)
            self._last_clear = now
        if removed:
            logger.debug("Market cleared {} expired slots", removed)

    @staticmethod
    def _generate_slot_id(node_id: str, prefix_hash: str) -> str:
        """Deterministic but unique-enough slot id."""
        raw = f"{node_id}:{prefix_hash}:{time.monotonic_ns()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LearnedEvictionPolicy
# ---------------------------------------------------------------------------

_EVICTION_FEATURE_NAMES: Final[list[str]] = [
    "lru_recency",
    "semantic_similarity",
    "access_frequency",
    "size",
    "tenant_priority",
]


class LearnedEvictionPolicy:
    """Feature-weighted eviction policy with an online RL feedback loop.

    Each cache entry receives a score — higher means more likely to evict —
    based on a linear combination of features:

    .. math::
        \\text{score} = w_1 \\cdot f_\\text{recency}
        + w_2 \\cdot f_\\text{similarity}
        + w_3 \\cdot f_\\text{freq}
        + w_4 \\cdot f_\\text{size}
        + w_5 \\cdot f_\\text{tenant}

    The weights are adjusted via a lightweight REINFORCE-style update:
    after each eviction, the policy observes whether a **re-access** would
    have been a hit or miss and nudges weights accordingly.

    Thread-safe for concurrent scoring / feedback.
    """

    def __init__(
        self,
        learning_rate: float = _EVICTION_LEARNING_RATE,
        discount: float = _EVICTION_DISCOUNT,
        embedding_dim: int = 128,
    ) -> None:
        self._weights: np.ndarray = np.array([0.4, 0.3, 0.2, 0.05, 0.05], dtype=np.float64)
        self._lr = learning_rate
        self._discount = discount
        self._embedding_dim = embedding_dim

        # Online feedback buffers
        self._total_evictions = 0
        self._total_reaccess_hits = 0
        self._total_reaccess_misses = 0

        self._lock = threading.Lock()

        logger.info(
            "LearnedEvictionPolicy initialised with weights={}",
            self._weights.tolist(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        lru_recency: float,
        semantic_similarity: float,
        access_frequency: float,
        size: float,
        tenant_priority: float,
    ) -> float:
        """Compute an eviction score for a cache entry.

        All feature values should be normalised to ``[0, 1]``.

        Higher scores -> more likely to be evicted.
        """
        features = np.array(
            [lru_recency, semantic_similarity, access_frequency, size, tenant_priority],
            dtype=np.float64,
        )
        with self._lock:
            return float(np.dot(self._weights, features))

    def score_entry(self, entry: dict[str, Any]) -> float:
        """Convenience: extract named features from a dict and score them."""
        return self.score(
            lru_recency=entry.get("lru_recency", 0.0),
            semantic_similarity=entry.get("semantic_similarity", 0.0),
            access_frequency=entry.get("access_frequency", 0.0),
            size=entry.get("size", 0.0),
            tenant_priority=entry.get("tenant_priority", 0.0),
        )

    def record_eviction(
        self,
        features: dict[str, float],
        was_reaccessed: bool,
        was_hit: bool,
    ) -> None:
        """Feed back the outcome of an eviction decision.

        Parameters
        ----------
        features :
            The feature vector of the evicted entry at eviction time.
        was_reaccessed :
            ``True`` if the evicted prefix was requested again
            (i.e. we have an outcome to learn from).
        was_hit :
            If re-accessed, ``True`` if the re-access hit another cache
            entry (good outcome), ``False`` if it was a miss (bad outcome).
        """
        if not was_reaccessed:
            return
        with self._lock:
            self._total_evictions += 1
            if was_hit:
                self._total_reaccess_hits += 1
                reward = 1.0  # evicting this entry was fine
            else:
                self._total_reaccess_misses += 1
                reward = -1.0  # evicting this entry was harmful

            # REINFORCE-style update: increase weights for features that
            # correlated with good outcomes, decrease for bad.
            feat_arr = np.array(
                [
                    features.get("lru_recency", 0.0),
                    features.get("semantic_similarity", 0.0),
                    features.get("access_frequency", 0.0),
                    features.get("size", 0.0),
                    features.get("tenant_priority", 0.0),
                ],
                dtype=np.float64,
            )
            # Scale update by reward and learning rate
            gradient = self._lr * reward * feat_arr
            self._weights = np.clip(self._weights + gradient, 0.0, 1.0)
            # Renormalise so they sum to 1
            total = self._weights.sum()
            if total > 0:
                self._weights /= total

    def select_victim(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Score all entries and return the one with the highest score.

        Parameters
        ----------
        entries :
            List of feature dicts (must include the features expected by
            :meth:`score_entry` plus an ``"entry_id"`` key).
        """
        if not entries:
            return None
        best_entry: dict[str, Any] | None = None
        best_score = -1.0
        for entry in entries:
            s = self.score_entry(entry)
            if s > best_score:
                best_score = s
                best_entry = entry
        return best_entry

    def weights(self) -> dict[str, float]:
        """Return a copy of the current feature weights."""
        with self._lock:
            return dict(zip(_EVICTION_FEATURE_NAMES, self._weights.tolist()))

    def stats(self) -> dict[str, Any]:
        """Return policy statistics."""
        with self._lock:
            total = self._total_evictions
            hits = self._total_reaccess_hits
            misses = self._total_reaccess_misses
        w = self.weights()
        hit_rate = hits / total if total else 0.0
        return {
            "weights": w,
            "total_evictions_with_feedback": total,
            "reaccess_hits": hits,
            "reaccess_misses": misses,
            "reaccess_hit_rate": hit_rate,
        }


# ---------------------------------------------------------------------------
# BloomFilterCascade
# ---------------------------------------------------------------------------


class _CuckooFilter:
    """Cuckoo filter for precise membership tests.

    Uses a bucket-based cuckoo hashing scheme.  Each item is represented by
    a short fingerprint stored in one of two candidate buckets.
    """

    def __init__(
        self,
        num_buckets: int = _CUCKOO_BUCKETS,
        fingerprint_bits: int = _CUCKOO_FINGERPRINT_BITS,
        max_kicks: int = _CUCKOO_MAX_KICKS,
    ) -> None:
        self._num_buckets = num_buckets
        self._fpb = fingerprint_bits
        self._max_kicks = max_kicks
        # Each bucket is a small list of fingerprints (4 slots per bucket)
        self._buckets: list[list[int]] = [[] for _ in range(num_buckets)]
        self._bucket_capacity = 4
        self._count = 0

    def insert(self, item: str) -> bool:
        """Insert *item* into the filter.  Returns True on success."""
        fp = self._fingerprint(item)
        i1 = self._bucket_index(item)
        i2 = self._alt_bucket(i1, fp)

        if len(self._buckets[i1]) < self._bucket_capacity:
            self._buckets[i1].append(fp)
            self._count += 1
            return True
        if len(self._buckets[i2]) < self._bucket_capacity:
            self._buckets[i2].append(fp)
            self._count += 1
            return True

        # Cuckoo kick
        cur_bucket = i1 if random.random() < 0.5 else i2
        for _ in range(self._max_kicks):
            # Evict a random fingerprint from this bucket
            kick_idx = random.randrange(len(self._buckets[cur_bucket]))
            fp, self._buckets[cur_bucket][kick_idx] = (
                self._buckets[cur_bucket][kick_idx],
                fp,
            )
            cur_bucket = self._alt_bucket(cur_bucket, fp)
            if len(self._buckets[cur_bucket]) < self._bucket_capacity:
                self._buckets[cur_bucket].append(fp)
                self._count += 1
                return True
        return False  # Filter full

    def contains(self, item: str) -> bool:
        """Check whether *item* may be in the filter.

        Returns ``True`` if the fingerprint is found in either candidate
        bucket (no false negatives).
        """
        fp = self._fingerprint(item)
        i1 = self._bucket_index(item)
        i2 = self._alt_bucket(i1, fp)
        return fp in self._buckets[i1] or fp in self._buckets[i2]

    def clear(self) -> None:
        """Remove all entries."""
        for b in self._buckets:
            b.clear()
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def _fingerprint(self, item: str) -> int:
        h = hashlib.sha256(item.encode()).digest()
        fp = struct.unpack_from("<I", h, 0)[0]
        mask = (1 << self._fpb) - 1
        return fp & mask if fp != 0 else 1  # fp must not be zero

    def _bucket_index(self, item: str) -> int:
        h = hashlib.sha256(item.encode()).digest()
        return struct.unpack_from("<I", h, 4)[0] % self._num_buckets

    @staticmethod
    def _alt_bucket(bucket: int, fingerprint: int) -> int:
        # xor with hash of fingerprint; same property as original cuckoo
        h = hashlib.sha256(struct.pack("<I", fingerprint)).digest()
        alt = struct.unpack_from("<I", h, 0)[0]
        return (bucket ^ alt) % _CUCKOO_BUCKETS


class BloomFilterCascade:
    """Three-layer cascade for sub-millisecond cache membership lookups.

    Layers
    ------
    1. **Bloom filter** — fast probabilistic check (~100 ns per ``might_contain``).
    2. **Cuckoo filter** — more precise (~500 ns), no false negatives.
    3. **Exact match** — compares the full hash against the index.

    Lookup returns early at the first layer that yields a definitive answer.
    """

    def __init__(
        self,
        bloom_size: int = _BLOOM_DEFAULT_SIZE,
        bloom_hashes: int = _BLOOM_DEFAULT_HASHES,
        cuckoo_buckets: int = _CUCKOO_BUCKETS,
    ) -> None:
        self._bloom_size = bloom_size
        self._bloom_hashes = bloom_hashes
        self._bloom_bits: bytearray = bytearray(bloom_size // 8 + 1)
        self._cuckoo = _CuckooFilter(num_buckets=cuckoo_buckets)
        # Exact index: prefix_hash -> list of node_ids
        self._exact: dict[str, list[str]] = {}
        self._lock = threading.Lock()

        # Accuracy tracking
        self._lookups = 0
        self._bloom_claims = 0  # bloom said "maybe"
        self._cuckoo_claims = 0  # cuckoo said "present"
        self._exact_hits = 0  # exact match confirmed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, prefix_hash: str) -> tuple[bool, list[str]]:
        """Check whether *prefix_hash* is cached.

        Returns ``(found, node_list)``, where ``node_list`` contains the
        nodes that hold the cached value (empty list if not found).

        The lookup short-circuits at the fastest layer that can give a
        definitive answer.
        """
        with self._lock:
            self._lookups += 1

            # Layer 1 — Bloom filter (fast negative elimination)
            if not self._bloom_might_contain_locked(prefix_hash):
                return False, []  # definitely not present
            self._bloom_claims += 1

            # Layer 2 — Cuckoo filter (no false negatives)
            if not self._cuckoo.contains(prefix_hash):
                # Bloom said "maybe" but cuckoo says "no" — false positive
                return False, []
            self._cuckoo_claims += 1

            # Layer 3 — Exact match
            nodes = self._exact.get(prefix_hash, [])
            if nodes:
                self._exact_hits += 1
                return True, list(nodes)
            return False, []

    def add(self, prefix_hash: str, node_ids: list[str]) -> None:
        """Record that *node_ids* cache the given prefix."""
        with self._lock:
            self._bloom_add_locked(prefix_hash)
            self._cuckoo.insert(prefix_hash)
            existing = self._exact.setdefault(prefix_hash, [])
            for nid in node_ids:
                if nid not in existing:
                    existing.append(nid)

    def remove(self, prefix_hash: str, node_id: str | None = None) -> None:
        """Remove *prefix_hash* (or just *node_id*'s entry for that hash)."""
        with self._lock:
            if prefix_hash not in self._exact:
                return
            if node_id is None:
                del self._exact[prefix_hash]
            else:
                nodes = self._exact[prefix_hash]
                if node_id in nodes:
                    nodes.remove(node_id)
                if not nodes:
                    del self._exact[prefix_hash]

    def rebuild(self, current_entries: dict[str, list[str]]) -> None:
        """Rebuild all three filters from a fresh set of entries.

        Parameters
        ----------
        current_entries :
            Mapping from ``prefix_hash`` to list of node ids.
        """
        with self._lock:
            # Reset
            self._bloom_bits = bytearray(self._bloom_size // 8 + 1)
            self._cuckoo.clear()
            self._exact.clear()
            # Repopulate
            for prefix_hash, node_ids in current_entries.items():
                self._bloom_add_locked(prefix_hash)
                self._cuckoo.insert(prefix_hash)
                self._exact[prefix_hash] = list(node_ids)
        logger.info("BloomFilterCascade rebuilt with {} entries", len(current_entries))

    def accuracy(self) -> dict[str, Any]:
        """Return per-layer accuracy / speed statistics."""
        with self._lock:
            total = self._lookups
            bloom_claims = self._bloom_claims
            cuckoo_claims = self._cuckoo_claims
            exact_hits = self._exact_hits
            exact_count = len(self._exact)
        bloom_fpr = (cuckoo_claims - exact_hits) / bloom_claims if bloom_claims else 0.0
        return {
            "total_lookups": total,
            "bloom_positive_rate": bloom_claims / total if total else 0.0,
            "cuckoo_positive_rate": cuckoo_claims / total if total else 0.0,
            "exact_hit_rate": exact_hits / total if total else 0.0,
            "bloom_false_positive_rate": bloom_fpr,
            "exact_entries": exact_count,
        }

    # ------------------------------------------------------------------
    # Bloom internals
    # ------------------------------------------------------------------

    def _bloom_hash(self, item: str, seed: int) -> int:
        h = seed
        for ch in item.encode("utf-8"):
            h = (h * 31337 + ch + seed * 7919) % ((1 << 61) - 1)
        return h % self._bloom_size

    def _bloom_add_locked(self, item: str) -> None:
        for i in range(self._bloom_hashes):
            idx = self._bloom_hash(item, i)
            self._bloom_bits[idx // 8] |= 1 << (idx % 8)

    def _bloom_might_contain_locked(self, item: str) -> bool:
        for i in range(self._bloom_hashes):
            idx = self._bloom_hash(item, i)
            if not (self._bloom_bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True


# ---------------------------------------------------------------------------
# Semantic similarity helpers
# ---------------------------------------------------------------------------

_PREFIX_EMBEDDING_DIM: Final[int] = 128


def _default_prefix_embedding(prefix_hash: str) -> np.ndarray:
    """Deterministic pseudo-embedding for a prefix hash.

    In production this would be replaced with a real embedding model
    (e.g. SentenceTransformer).
    """
    seed = int(hashlib.sha256(prefix_hash.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=_PREFIX_EMBEDDING_DIM).astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


# ---------------------------------------------------------------------------
# CacheEntry — internal to GaiaCache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Internal representation of a cached KV value."""

    key: str
    value: Any
    prefix_hash: str
    size_mb: float
    ttl: float
    stored_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    tenant: str = ""
    eviction_score: float = 0.0


# ---------------------------------------------------------------------------
# GaiaCache
# ---------------------------------------------------------------------------


class GaiaCache:
    """Cross-node prompt caching and KV cache trading system.

    Orchestrates four subsystems:

    * :attr:`ring` — consistent hash ring for prefix sharding
    * :attr:`market` — KV cache slot marketplace
    * :attr:`eviction` — learned eviction policy
    * :attr:`filters` — Bloom/Cuckoo filter cascade

    Usage::

        cache = GaiaCache(local_node_id="node-a")
        cache.add_node("node-a", weight=1.0)
        cache.add_node("node-b", weight=2.0)

        value = cache.get("/prompt/prefix")
        if value is None:
            value = compute_expensive()
            cache.put("/prompt/prefix", value, ttl=3600)

        cache.evict()
        print(cache.stats())
    """

    def __init__(
        self,
        local_node_id: str = "",
        embedding_fn=None,
    ) -> None:
        self._local_node_id = local_node_id or f"node-{uuid_short()}"

        # Subsystems
        self._ring = ConsistentHashRing()
        self._market = CacheMarket(local_node_id=self._local_node_id)
        self._eviction = LearnedEvictionPolicy()
        self._filters = BloomFilterCascade()

        # Local key-value store
        self._store: dict[str, _CacheEntry] = {}
        self._max_local_entries: int = 100_000
        self._max_local_size_mb: float = 1024.0  # 1 GB

        # Current total size (MB)
        self._total_size_mb: float = 0.0

        # Embedding function for semantic similarity (overridable)
        self._embedding_fn = embedding_fn or _default_prefix_embedding

        # Embedding cache for fast cosine similarity
        self._embed_cache: dict[str, np.ndarray] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info("GaiaCache initialised (node={})", self._local_node_id)

    # ------------------------------------------------------------------
    # Hash ring delegation
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, weight: float = 1.0) -> None:
        """Add a node to the consistent hash ring."""
        self._ring.add_node(node_id, weight)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the consistent hash ring."""
        self._ring.remove_node(node_id)

    def get_node(self, key: str) -> str | None:
        """Return the node responsible for *key*."""
        return self._ring.get_node(key)

    def get_nodes(self, key: str, count: int = 1) -> list[str]:
        """Return *count* nodes responsible for *key* (for replication)."""
        return self._ring.get_nodes(key, count)

    # ------------------------------------------------------------------
    # Market delegation
    # ------------------------------------------------------------------

    @property
    def market(self) -> CacheMarket:
        return self._market

    # ------------------------------------------------------------------
    # Core cache operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> tuple[Any, dict[str, Any]] | tuple[None, None]:
        """Look up *key* in the cache.

        Returns
        -------
        ``(value, routing_info)`` if found, or ``(None, None)`` if missed.

        ``routing_info`` is a dict with at least:
          - ``node`` — the node that owns this key
          - ``found`` — whether it was found
          - ``filter_hit`` — whether the filter cascade predicted presence
          - ``source`` — ``"local"`` or ``"market"``
        """
        routing: dict[str, Any] = {
            "node": self._local_node_id,
            "found": False,
            "filter_hit": False,
            "source": "",
        }

        # 1. Quick filter check
        prefix_hash = self._hash_key(key)
        filter_found, node_list = self._filters.lookup(prefix_hash)
        routing["filter_hit"] = filter_found

        # 2. Local lookup
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                # Check TTL
                if time.time() - entry.stored_at > entry.ttl:
                    self._remove_entry_locked(key)
                    logger.debug("Key {!r} expired locally", key)
                else:
                    entry.access_count += 1
                    entry.last_access = time.time()
                    routing["found"] = True
                    routing["source"] = "local"
                    return entry.value, routing

        # 3. Market lookup — ask if any other node is selling this prefix
        available = self._market.list_available(prefix_hash=prefix_hash)
        if available:
            # Try to buy the cheapest available
            cheapest = min(available, key=lambda s: s.price)
            bought = self._market.buy(cheapest.slot_id, self._local_node_id)
            if bought is not None:
                routing["found"] = True
                routing["source"] = "market"
                routing["node"] = bought.node_id
                routing["slot_id"] = bought.slot_id
                # The actual KV data would be fetched from the seller node
                # via the transport layer; here we return a placeholder.
                return None, routing

        return None, None

    def put(
        self,
        key: str,
        value: Any,
        ttl: float = _DEFAULT_TTL,
        tenant: str = "",
        size_mb: float | None = None,
    ) -> None:
        """Store *value* under *key* in the local cache.

        If the cache is full (by entry count or total MB) the least-
        valuable entry is evicted first.

        After storing, the owning node's market is notified so other nodes
        can discover and buy this slot.
        """
        prefix_hash = self._hash_key(key)
        if size_mb is None:
            size_mb = self._estimate_size_mb(value)

        entry = _CacheEntry(
            key=key,
            value=value,
            prefix_hash=prefix_hash,
            size_mb=size_mb,
            ttl=ttl,
            tenant=tenant,
        )

        with self._lock:
            # Evict if we need room
            while (
                len(self._store) >= self._max_local_entries
                or self._total_size_mb + size_mb > self._max_local_size_mb
            ):
                if not self._evict_one_locked():
                    logger.warning("Cache full but no evictable entries found")
                    break

            self._store[key] = entry
            self._total_size_mb += size_mb

        # Update filters
        owner = self._ring.get_node(key) or self._local_node_id
        self._filters.add(prefix_hash, [owner])

        # Advertise on the market
        self._market.sell(
            prefix_hash=prefix_hash,
            size_mb=size_mb,
            price=self._compute_price(size_mb, tenant),
            ttl=ttl,
        )

        logger.debug(
            "Cached key={!r} ({} MB, tenant={!r}, ttl={})",
            key,
            size_mb,
            tenant,
            ttl,
        )

    def evict(self) -> int:
        """Run learned eviction on all entries, removing the highest-scored.

        Returns the number of entries evicted.
        """
        evicted = 0
        with self._lock:
            scored = self._score_all_entries_locked()
            victim = self._eviction.select_victim(scored)
            if victim is not None:
                entry_id = victim.get("entry_id", "")
                if entry_id and entry_id in self._store:
                    entry = self._store[entry_id]
                    self._remove_entry_locked(entry_id)
                    self._filters.remove(entry.prefix_hash, self._local_node_id)

                    # Record feedback (we don't know yet if it will be
                    # re-accessed; the caller / background task should
                    # call record_eviction_outcome later).
                    self._eviction.record_eviction(
                        features=victim,
                        was_reaccessed=False,
                        was_hit=False,
                    )
                    evicted = 1
        if evicted:
            logger.info("Evicted 1 entry via learned policy")
        return evicted

    def record_eviction_outcome(
        self,
        evicted_prefix_hash: str,
        was_reaccessed: bool,
        was_hit: bool,
        features: dict[str, float] | None = None,
    ) -> None:
        """Feed back the outcome of a previous eviction.

        Call this when you know whether the evicted prefix was re-accessed
        and whether the re-access was a cache hit (via another entry).
        """
        if features is None:
            features = {
                "lru_recency": 1.0,
                "semantic_similarity": 0.5,
                "access_frequency": 0.0,
                "size": 0.0,
                "tenant_priority": 0.0,
            }
        self._eviction.record_eviction(
            features=features,
            was_reaccessed=was_reaccessed,
            was_hit=was_hit,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics across all subsystems."""
        with self._lock:
            local_entries = len(self._store)
            local_size_mb = self._total_size_mb
        return {
            "local_node": self._local_node_id,
            "ring": {
                "nodes": self._ring.nodes(),
                "size": self._ring.size,
            },
            "local_cache": {
                "entries": local_entries,
                "size_mb": round(local_size_mb, 2),
                "max_entries": self._max_local_entries,
                "max_size_mb": self._max_local_size_mb,
                "usage_pct": round(
                    local_size_mb / self._max_local_size_mb * 100, 1
                )
                if self._max_local_size_mb
                else 0.0,
            },
            "market": self._market.stats(),
            "eviction": self._eviction.stats(),
            "filters": self._filters.accuracy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _estimate_size_mb(self, value: Any) -> float:
        """Rough size estimation of a cached value in MB."""
        if isinstance(value, (str, bytes)):
            return len(value) / (1024 * 1024)
        if isinstance(value, (list, tuple)):
            return sum(self._estimate_size_mb(v) for v in value)
        if isinstance(value, dict):
            return sum(
                self._estimate_size_mb(k) + self._estimate_size_mb(v)
                for k, v in value.items()
            )
        # Fallback: assume 1 KB
        return 0.001

    def _compute_price(self, size_mb: float, tenant: str) -> float:
        """Simple price function: larger entries cost more; tenants get a tier."""
        base = size_mb * 0.01
        if tenant and tenant.startswith("premium_"):
            return round(base * 1.5, 4)
        return round(base, 4)

    def _extract_features(self, entry: _CacheEntry) -> dict[str, float]:
        """Extract a normalised feature vector for an entry."""
        now = time.time()
        age = now - entry.stored_at
        recency = now - entry.last_access

        # LRU recency: how long since last access (normalised to [0,1])
        max_recency = 3600.0  # 1 hour window
        lru_recency = min(recency / max_recency, 1.0)

        # Access frequency: accesses per second since storage
        freq = entry.access_count / max(age, 1.0)
        access_frequency = min(freq * 10, 1.0)  # scale

        # Size
        size = min(entry.size_mb / 100.0, 1.0)

        # Tenant priority (lower tenant number = higher priority)
        tenant_priority = 0.5  # default
        if entry.tenant:
            if entry.tenant.startswith("premium_"):
                tenant_priority = 0.1  # low eviction chance
            else:
                tenant_priority = 0.7

        # Semantic similarity: cosine distance to other recently accessed
        emb = self._get_embedding(entry.prefix_hash)
        semantic_similarity = self._compute_max_semantic_similarity(
            entry.prefix_hash, emb
        )

        return {
            "lru_recency": lru_recency,
            "semantic_similarity": semantic_similarity,
            "access_frequency": access_frequency,
            "size": size,
            "tenant_priority": tenant_priority,
            "entry_id": entry.key,
            "prefix_hash": entry.prefix_hash,
        }

    def _get_embedding(self, prefix_hash: str) -> np.ndarray:
        if prefix_hash not in self._embed_cache:
            self._embed_cache[prefix_hash] = self._embedding_fn(prefix_hash)
        return self._embed_cache[prefix_hash]

    def _compute_max_semantic_similarity(
        self,
        prefix_hash: str,
        emb: np.ndarray,
    ) -> float:
        """Compute the maximum cosine similarity to other cached prefixes.

        Returns a value in ``[0, 1]`` where higher = more similar to others
        (and therefore more redundant / good eviction candidate).
        """
        max_sim = 0.0
        for other_hash, other_emb in self._embed_cache.items():
            if other_hash == prefix_hash:
                continue
            cos_sim = float(np.dot(emb, other_emb))
            if cos_sim > max_sim:
                max_sim = cos_sim
        return max_sim

    def _score_all_entries_locked(self) -> list[dict[str, Any]]:
        """Build a scored feature list for all local entries."""
        scored: list[dict[str, Any]] = []
        for entry in self._store.values():
            features = self._extract_features(entry)
            features["score"] = self._eviction.score_entry(features)
            scored.append(features)
        return scored

    def _evict_one_locked(self) -> bool:
        """Score all entries and evict the highest-scored one.

        Returns True if an entry was evicted.
        """
        scored = self._score_all_entries_locked()
        victim = self._eviction.select_victim(scored)
        if victim is None:
            return False
        entry_id = victim.get("entry_id", "")
        if not entry_id or entry_id not in self._store:
            return False
        entry = self._store[entry_id]
        self._remove_entry_locked(entry_id)
        self._filters.remove(entry.prefix_hash, self._local_node_id)

        # Record eviction for feedback
        self._eviction.record_eviction(
            features=victim,
            was_reaccessed=False,
            was_hit=False,
        )
        return True

    def _remove_entry_locked(self, key: str) -> None:
        """Remove an entry; lock must be held."""
        entry = self._store.pop(key, None)
        if entry is not None:
            self._total_size_mb -= entry.size_mb


def uuid_short() -> str:
    """Return a short, URL-safe unique identifier (8 chars)."""
    return hashlib.sha256(
        str(time.monotonic_ns()).encode()
    ).hexdigest()[:8]
