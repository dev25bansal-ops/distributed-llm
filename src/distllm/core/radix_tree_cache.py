"""Trie-based prefix cache using a radix tree for O(k) lookups.

Provides RadixNode (trie node) and RadixTreeCache (cache interface)
for storing and retrieving KV cache entries by token prefix.
"""

from __future__ import annotations

import time
import threading
from typing import Any


class RadixNode:
    """A node in the radix tree trie.

    Each node stores one token value and optional KV data.
    Children are indexed by token ID for O(1) lookup.
    """

    def __init__(self, token: int = -1):
        self.token = token
        self.children: dict[int, RadixNode] = {}
        self.kv_data: Any = None
        self.last_access: float = time.time()
        self.size: int = 0  # Number of descendants with kv_data

    def insert(self, token_ids: list[int], kv_data: Any) -> None:
        """Insert a token sequence with associated KV data."""
        node = self
        for tok in token_ids:
            if tok not in node.children:
                node.children[tok] = RadixNode(token=tok)
            node = node.children[tok]
        node.kv_data = kv_data
        node.last_access = time.time()
        # Update size counts up the tree
        self._update_sizes(token_ids)

    def _update_sizes(self, token_ids: list[int]) -> None:
        """Update size counters along a path.

        Size = count of entries with kv_data in subtree (including self).
        Recomputed bottom-up so parent sizes see refreshed child values.
        """
        # Walk down to the inserted leaf, collecting the path
        path = []
        node = self
        for tok in token_ids:
            if tok not in node.children:
                break
            node = node.children[tok]
            path.append(node)

        # Recompute sizes leaf -> root
        for node in reversed(path):
            count = 0
            if node.kv_data is not None:
                count += 1
            for child in node.children.values():
                count += child.size
            node.size = count

    def lookup(self, token_ids: list[int]) -> tuple[int, Any]:
        """Find the longest prefix match with stored KV data.

        Returns:
            (match_length, kv_data) — match_length is 0 if no KV data found.
        """
        node = self
        best_match = 0
        best_kv = None

        for i, tok in enumerate(token_ids):
            if tok not in node.children:
                break
            node = node.children[tok]
            if node.kv_data is not None:
                node.last_access = time.time()
                best_match = i + 1
                best_kv = node.kv_data

        return best_match, best_kv

    def find_shared_prefix(self, token_ids: list[int]) -> int:
        """Find the longest prefix length that exists in the trie (even without KV data)."""
        node = self
        matched = 0
        for tok in token_ids:
            if tok not in node.children:
                break
            node = node.children[tok]
            matched += 1
        return matched

    def _count_entries(self) -> int:
        """Count total entries with KV data in the subtree."""
        count = 1 if self.kv_data is not None else 0
        for child in self.children.values():
            count += child._count_entries()
        return count

    def _find_lru_leaf(self) -> tuple[float, RadixNode | None]:
        """Find the leaf node with the oldest last_access time."""
        if self.kv_data is not None and not self.children:
            return self.last_access, self

        best_time = float("inf")
        best_node = None
        for child in self.children.values():
            t, n = child._find_lru_leaf()
            if t < best_time:
                best_time = t
                best_node = n
        return best_time, best_node

    def evict_lru(self, max_entries: int) -> int:
        """Evict LRU leaf nodes until count <= max_entries.

        Eviction clears the ``kv_data`` of the least-recently-used leaf
        (structure is preserved so shared prefixes survive).

        Returns:
            Number of entries evicted.
        """
        evicted = 0
        while self._count_entries() > max_entries and self._count_entries() > 0:
            _, lru_leaf = self._find_lru_leaf()
            if lru_leaf is None:
                break
            lru_leaf.kv_data = None
            lru_leaf.size = 0
            evicted += 1
        return evicted

    def _evict_path(self, token_ids: list[int]) -> bool:
        """Evict KV data at the given path. Returns True if entry was found and removed."""
        node = self
        for tok in token_ids:
            if tok not in node.children:
                return False
            node = node.children[tok]
        if node.kv_data is not None:
            node.kv_data = None
            node.size = 0
            return True
        return False

    def clear(self) -> None:
        """Clear all children and KV data."""
        self.children.clear()
        self.kv_data = None
        self.size = 0

    def stats(self) -> dict:
        """Return statistics about the subtree."""
        total_entries = self._count_entries()
        total_nodes = self._count_nodes()
        max_depth = self._max_depth()
        return {
            "total_entries": total_entries,
            "total_nodes": total_nodes,
            "max_depth": max_depth,
        }

    def _count_nodes(self) -> int:
        count = 1
        for child in self.children.values():
            count += child._count_nodes()
        return count

    def _max_depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child._max_depth() for child in self.children.values())


class RadixTreeCache:
    """Trie-based prefix cache implementing the ICacheBackend protocol.

    Uses a RadixNode trie for O(k) lookup where k is the token count.
    Supports memory budget enforcement and LRU eviction.
    """

    def __init__(
        self,
        max_entries: int = 0,
        min_prefix_len: int = 1,
        memory_budget_bytes: int = 512 * 1024 * 1024,
    ):
        self._root = RadixNode()
        self._min_prefix_len = min_prefix_len
        self._max_entries = max_entries
        self._memory_budget = memory_budget_bytes
        self._total_memory_bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def lookup(self, token_ids: list[int]) -> tuple[int, Any]:
        """Lookup longest prefix match in the trie."""
        if len(token_ids) < self._min_prefix_len:
            with self._lock:
                self._misses += 1
            return 0, None

        with self._lock:
            match_len, kv_data = self._root.lookup(token_ids)
            if match_len > 0 and kv_data is not None:
                self._hits += 1
                return match_len, kv_data
            self._misses += 1
            return 0, None

    def store(self, token_ids: list[int], kv_data: Any) -> None:
        """Store KV data at the given token prefix."""
        if len(token_ids) < self._min_prefix_len:
            return

        with self._lock:
            # Estimate memory for this entry
            entry_bytes = self._estimate_entry_memory(kv_data)
            self._total_memory_bytes += entry_bytes
            self._root.insert(token_ids, kv_data)
            self._evict_until_fit()

    def find_shared_prefix(self, token_ids: list[int]) -> int:
        """Find the longest prefix that exists in the trie (even without KV data)."""
        with self._lock:
            return self._root.find_shared_prefix(token_ids)

    def evict(self, token_ids: list[int]) -> bool:
        """Evict the entry at the given token prefix."""
        with self._lock:
            return self._root._evict_path(token_ids)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._root.clear()
            self._hits = 0
            self._misses = 0
            self._total_memory_bytes = 0

    def _estimate_entry_memory(self, kv_data: Any) -> int:
        """Estimate memory usage of a KV data entry."""
        total = 0
        if isinstance(kv_data, dict):
            for v in kv_data.values():
                if hasattr(v, "element_size") and hasattr(v, "numel"):
                    total += v.element_size() * v.numel()
                elif isinstance(v, tuple):
                    for t in v:
                        if hasattr(t, "element_size") and hasattr(t, "numel"):
                            total += t.element_size() * t.numel()
        elif isinstance(kv_data, (list, tuple)):
            for t in kv_data:
                if hasattr(t, "element_size") and hasattr(t, "numel"):
                    total += t.element_size() * t.numel()
        return total

    def _evict_until_fit(self) -> None:
        """Evict LRU entries until within memory budget and entry count."""
        # Evict by entry count
        if self._max_entries > 0:
            self._root.evict_lru(self._max_entries)

        # Evict by memory budget (find and remove LRU leaves)
        attempts = 0
        max_attempts = 1000
        while self._total_memory_bytes > self._memory_budget and attempts < max_attempts:
            _, lru_leaf = self._root._find_lru_leaf()
            if lru_leaf is None or lru_leaf.kv_data is None:
                break
            entry_bytes = self._estimate_entry_memory(lru_leaf.kv_data)
            lru_leaf.kv_data = None
            lru_leaf.size = 0
            self._total_memory_bytes = max(0, self._total_memory_bytes - entry_bytes)
            attempts += 1

    def adjust_memory_budget(self, new_budget_bytes: int) -> None:
        """Adjust memory budget and evict if necessary."""
        with self._lock:
            self._memory_budget = new_budget_bytes
            self._evict_until_fit()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            tree_stats = self._root.stats()
            return {
                "prefix_cache_entries": tree_stats["total_entries"],
                "radix_tree_nodes": tree_stats["total_nodes"],
                "radix_tree_max_depth": tree_stats["max_depth"],
                "prefix_cache_memory_bytes": self._total_memory_bytes,
                "prefix_cache_memory_budget": self._memory_budget,
                "prefix_cache_hit_rate": round(self.hit_rate, 4),
                "prefix_cache_hits": self._hits,
                "prefix_cache_misses": self._misses,
            }
