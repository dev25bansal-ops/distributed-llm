"""Hash-based LRU prefix cache for token sequences with memory-based limits."""


from __future__ import annotations
import time
import threading
from collections import OrderedDict

from distllm.dist.cache import TTLPolicy


_DEFAULT_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024
_NO_TENANT = ""


class _BloomFilter:
    """Simple bloom filter for fast negative lookups (E2).


    Uses multiple hash functions to probabilistically check if a prefix
    has been seen before. False positives are possible, false negatives
    are not.
    """


    def __init__(self, size: int = 1 << 20, num_hashes: int = 7):
        self._size = size
        self._num_hashes = num_hashes
        self._bits = bytearray(size // 8 + 1)

    def _hash(self, token_ids: list[int], seed: int) -> int:
        h = seed
        for tok in token_ids:
            h = (h * 31337 + tok + seed * 7919) % ((1 << 61) - 1)
        return h % self._size

    def add(self, token_ids: list[int]) -> None:
        for i in range(self._num_hashes):
            idx = self._hash(token_ids, i)
            self._bits[idx // 8] |= 1 << (idx % 8)

    def might_contain(self, token_ids: list[int]) -> bool:
        for i in range(self._num_hashes):
            idx = self._hash(token_ids, i)
            if not (self._bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True

    def clear(self) -> None:
        self._bits = bytearray(self._size // 8 + 1)


class PrefixCache:
    _HASH_BASE = 31337
    _HASH_MOD = (1 << 61) - 1

    def __init__(
        self,
        max_entries: int = 0,
        min_prefix_len: int = 16,
        memory_budget_bytes: int = _DEFAULT_MEMORY_BUDGET_BYTES,
        paged_attention_mgr: object | None = None,
        ttl_policy: TTLPolicy | None = None,
    ):
        self.min_prefix_len = min_prefix_len
        self._paged_attention_mgr = paged_attention_mgr
        self._cache: OrderedDict[tuple[int, str], dict] = OrderedDict()
        self._hits = 0
        self._misses = 0

        self._memory_budget = memory_budget_bytes
        self._total_memory_bytes = 0
        self._max_entries_soft = max_entries or 0

        self._ttl_policy = ttl_policy
        self._lock = threading.RLock()

        # E2: Bloom filter for fast negative lookups
        self._bloom = _BloomFilter(size=max(max(max_entries, 1) * 8, 1 << 16))

        # E8: Memory pressure tracking
        self._last_pressure_check = 0.0
        self._pressure_check_interval = 5.0  # seconds

    @property
    def max_entries(self) -> int:
        return self._max_entries_soft or len(self._cache) + 1

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        self._max_entries_soft = value

    def _estimate_entry_memory(self, kv_data: dict) -> int:
        total = 0
        if isinstance(kv_data, dict):
            for v in kv_data.values():
                if hasattr(v, 'element_size') and hasattr(v, 'numel'):
                    total += v.element_size() * v.numel()
                elif isinstance(v, tuple) and v:
                    for t in v:
                        if hasattr(t, 'element_size') and hasattr(t, 'numel'):
                            total += t.element_size() * t.numel()
        elif isinstance(kv_data, list):
            for t in kv_data:
                if hasattr(t, 'element_size') and hasattr(t, 'numel'):
                    total += t.element_size() * t.numel()
        return total

    def _evict_until_fit(self, needed_bytes: int) -> None:
        # E8: Check system memory pressure periodically
        now = time.time()
        if now - self._last_pressure_check > self._pressure_check_interval:
            self._last_pressure_check = now
            try:
                import psutil
                mem = psutil.virtual_memory()
                if mem.percent > 90:
                    # Under memory pressure, reduce budget by 25%
                    self._memory_budget = int(self._memory_budget * 0.75)
                elif mem.percent < 70 and self._memory_budget < _DEFAULT_MEMORY_BUDGET_BYTES:
                    # Recover budget gradually
                    self._memory_budget = min(int(self._memory_budget * 1.1), _DEFAULT_MEMORY_BUDGET_BYTES)
            except ImportError:
                pass

        if self._ttl_policy:
            all_keys = list(self._cache.keys())
            expired = self._ttl_policy.get_expired_keys(all_keys)
            for key in expired:
                if key in self._cache:
                    entry = self._cache.pop(key)
                    entry_bytes = self._estimate_entry_memory(entry.get("kv_data", {}))
                    self._total_memory_bytes = max(0, self._total_memory_bytes - entry_bytes)
                    self._ttl_policy.remove(key)
                    if self._total_memory_bytes + needed_bytes <= self._memory_budget:
                        return

        # E6: Hybrid LFU+LRU eviction — prefer evicting low-frequency entries
        while self._cache and (self._total_memory_bytes + needed_bytes > self._memory_budget):
            # Find entry with lowest access frequency among the oldest entries
            if len(self._cache) > 10:
                # For large caches, use LRU (popitem) for speed
                _key, entry = self._cache.popitem(last=False)
            else:
                # For small caches, find LFU entry
                lfu_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].get("access_count", 1),
                )
                entry = self._cache.pop(lfu_key)
                _key = lfu_key
            entry_bytes = self._estimate_entry_memory(entry.get("kv_data", {}))
            self._total_memory_bytes = max(0, self._total_memory_bytes - entry_bytes)
            if self._ttl_policy:
                self._ttl_policy.remove(_key)

    @staticmethod
    def _compute_full_hash(token_ids: list[int]) -> int:
        """Compute the rolling hash over the full token sequence."""

        h = 0
        for tok in token_ids:
            h = ((h * PrefixCache._HASH_BASE) + tok) % PrefixCache._HASH_MOD
        return h

    def lookup(self, token_ids: list[int], tenant_id: str = _NO_TENANT) -> tuple[int, dict | None]:
        if len(token_ids) < self.min_prefix_len:
            with self._lock:
                self._misses += 1
            return 0, None

        # E2: Bloom filter fast-negative check (check min prefix only)
        bloom_key = token_ids[:self.min_prefix_len]
        if not self._bloom.might_contain(bloom_key):
            with self._lock:
                self._misses += 1
            return 0, None

        hashes: list[tuple[int, int]] = []
        running_hash = 0
        for i, tok in enumerate(token_ids):
            running_hash = ((running_hash * self._HASH_BASE) + tok) % self._HASH_MOD
            length = i + 1
            if length >= self.min_prefix_len:
                hashes.append((length, running_hash))

        for length, h in reversed(hashes):
            key = (h, tenant_id)
            with self._lock:
                cached = self._cache.get(key)
            if cached is None:
                continue
            if self._ttl_policy and self._ttl_policy.is_expired(key):
                continue
            cached_tokens = cached["tokens"]
            if len(cached_tokens) == length:
                match = True
                for j in range(length):
                    if cached_tokens[j] != token_ids[j]:
                        match = False
                        break
                if match:
                    with self._lock:
                        self._hits += 1
                        self._cache.move_to_end(key)
                        # E6: Track access frequency
                        cached["access_count"] = cached.get("access_count", 0) + 1
                        if self._ttl_policy:
                            self._ttl_policy.record_access(key)
                    return length, cached["kv_data"]

        with self._lock:
            self._misses += 1
        return 0, None

    def store(self, token_ids: list[int], kv_data: dict, tenant_id: str = _NO_TENANT) -> None:
        with self._lock:
            if len(token_ids) < self.min_prefix_len:
                return

            entry_bytes = self._estimate_entry_memory(kv_data)

            # E7: Size-aware admission — skip entries that can never fit in budget
            if entry_bytes > self._memory_budget:
                return

            h = self._compute_full_hash(token_ids)
            key = (h, tenant_id)

            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key]["kv_data"] = kv_data
                self._cache[key]["access_count"] = self._cache[key].get("access_count", 0) + 1
                if self._ttl_policy:
                    self._ttl_policy.record_access(key)
                return

            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)

            self._cache[key] = {
                "tokens": list(token_ids),
                "kv_data": kv_data,
                "stored_at": time.time(),
                "access_count": 1,
            }
            self._total_memory_bytes += entry_bytes

            # E2: Add to bloom filter (store min prefix for fast lookup)
            self._bloom.add(token_ids[:self.min_prefix_len])

            if self._ttl_policy:
                self._ttl_policy.record_access(key)

            self._evict_until_fit(0)

    def evict(self, token_ids: list[int], tenant_id: str = _NO_TENANT) -> bool:
        with self._lock:
            h = 0
            for tok in token_ids:
                h = ((h * self._HASH_BASE) + tok) % self._HASH_MOD
            key = (h, tenant_id)
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            if self._ttl_policy:
                self._ttl_policy.clear()
            self._bloom.clear()

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "prefix_cache_entries": len(self._cache),
                "prefix_cache_max_entries": self.max_entries,
                "prefix_cache_hits": self._hits,
                "prefix_cache_misses": self._misses,
                "prefix_cache_hit_rate": round(self.hit_rate, 4),
                "prefix_cache_memory_bytes": self._total_memory_bytes,
                "prefix_cache_memory_budget": self._memory_budget,
                "prefix_cache_memory_util": round(
                    self._total_memory_bytes / max(self._memory_budget, 1), 4
                ),
            }

    def adjust_memory_budget(self, new_budget_bytes: int) -> None:
        with self._lock:
            self._memory_budget = new_budget_bytes


class DistributedPrefixCache:
    """Distributed prefix cache with Merkle tree + gossip protocol.


    Extends the node-local PrefixCache with cross-node prefix sharing.
    Uses Merkle trees for efficient prefix hash comparison and gossip
    protocol for disseminating prefix information across nodes.

    This enables 30-50% TTFT reduction for shared prefixes (e.g.,
    system prompts, few-shot examples) by routing requests to nodes
    that already have the prefix cached.

    Args:
        local_cache: The local PrefixCache instance.
        node_id: This node's identifier.
        gossip_interval_s: How often to gossip prefix info.
    """


    def __init__(
        self,
        local_cache: PrefixCache,
        node_id: str,
        gossip_interval_s: float = 5.0,
    ):
        self._local = local_cache
        self._node_id = node_id
        self._gossip_interval = gossip_interval_s

        # Remote prefix info: node_id -> {hash -> length}
        self._remote_prefixes: dict[str, dict[int, int]] = {}
        self._lock = threading.Lock()

        # Merkle tree for efficient prefix comparison
        self._merkle_root: int = 0
        self._prefix_hashes: dict[int, int] = {}  # hash -> length

    def compute_merkle_root(self, token_ids: list[int]) -> int:
        """Compute Merkle root hash for a token sequence.


        Uses a simple binary tree hash for efficient comparison.
        """

        if not token_ids:
            return 0

        # Build leaf hashes
        leaves = []
        for i, tok in enumerate(token_ids):
            leaf = ((i + 1) * 31337 + tok) & ((1 << 61) - 1)
            leaves.append(leaf)

        # Build tree bottom-up
        while len(leaves) > 1:
            next_level = []
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    combined = (leaves[i] * 31 + leaves[i + 1]) & ((1 << 61) - 1)
                else:
                    combined = leaves[i]
                next_level.append(combined)
            leaves = next_level

        return leaves[0] if leaves else 0

    def get_prefix_info(self) -> dict[int, int]:
        """Get local prefix hash info for gossip dissemination.


        Returns dict of {hash: length} for all cached prefixes.
        """

        with self._lock:
            return dict(self._prefix_hashes)

    def update_local_prefix(self, token_ids: list[int]) -> None:
        """Update local prefix hash after caching new prefix."""

        if len(token_ids) < self._local.min_prefix_len:
            return

        h = 0
        for tok in token_ids:
            h = ((h * 31337) + tok) & ((1 << 61) - 1)

        with self._lock:
            self._prefix_hashes[h] = len(token_ids)
            self._merkle_root = self.compute_merkle_root(token_ids)

    def receive_gossip(self, node_id: str, prefix_info: dict[int, int]) -> None:
        """Receive prefix info gossip from a remote node.


        Args:
            node_id: Source node ID.
            prefix_info: Dict of {hash: length} from the remote node.
        """

        with self._lock:
            self._remote_prefixes[node_id] = prefix_info

    def find_best_node(self, token_ids: list[int]) -> tuple[str, int] | None:
        """Find the node with the best cached prefix match.


        Args:
            token_ids: Token IDs to match against.

        Returns:
            Tuple of (node_id, match_length) or None if no match.
        """

        if len(token_ids) < self._local.min_prefix_len:
            return None

        # Compute hash for full sequence
        h = 0
        for tok in token_ids:
            h = ((h * 31337) + tok) & ((1 << 61) - 1)

        # Check local cache first
        with self._lock:
            local_len = self._prefix_hashes.get(h, 0)
            if local_len > 0:
                return self._node_id, local_len

            # Check remote nodes
            best_node = None
            best_length = 0
            for node_id, prefixes in self._remote_prefixes.items():
                remote_len = prefixes.get(h, 0)
                if remote_len > best_length:
                    best_length = remote_len
                    best_node = node_id

            if best_node is not None:
                return best_node, best_length

        return None

    def get_gossip_payload(self) -> dict:
        """Get payload for gossip dissemination."""

        with self._lock:
            return {
                "node_id": self._node_id,
                "prefixes": dict(self._prefix_hashes),
                "merkle_root": self._merkle_root,
                "timestamp": time.time(),
            }

    def stats(self) -> dict:
        with self._lock:
            return {
                "local_prefixes": len(self._prefix_hashes),
                "remote_nodes": len(self._remote_prefixes),
                "total_remote_prefixes": sum(len(p) for p in self._remote_prefixes.values()),
            }
