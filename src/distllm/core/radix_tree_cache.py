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

    Invariants:
      * ``size`` equals the number of nodes with non-None ``kv_data`` in this
        subtree (including self). Maintained incrementally by ``insert`` /
        ``_remove_entry`` so ``_count_entries`` is O(1).
      * After any eviction, a node with no ``kv_data`` and no children is
        pruned from its parent (recursively upward), so LRU-leaf detection
        keeps finding victims and the tree holds no dead weight.
    """

    def __init__(self, token: int = -1):
        self.token = token
        self.children: dict[int, RadixNode] = {}
        self.kv_data: Any = None
        self.last_access: float = time.time()
        self.size: int = 0  # Number of descendants-with-data plus self
        self.parent: RadixNode | None = None

    def insert(self, token_ids: list[int], kv_data: Any) -> None:
        """Insert a token sequence with associated KV data."""
        node = self
        for tok in token_ids:
            child = node.children.get(tok)
            if child is None:
                child = RadixNode(token=tok)
                child.parent = node
                node.children[tok] = child
            node = child
        was_new = node.kv_data is None
        node.kv_data = kv_data
        node.last_access = time.time()
        if was_new:
            # One more live entry in every node on the path to root.
            n: RadixNode | None = node
            while n is not None:
                n.size += 1
                n = n.parent
        # On overwrite the entry count is unchanged (memory accounting for the
        # replaced value is handled by the cache layer in ``store``).

    def lookup(self, token_ids: list[int]) -> tuple[int, Any]:
        """Find the longest prefix match with stored KV data.

        Returns:
            (match_length, kv_data) — match_length is 0 if no KV data found.
        """
        node = self
        best_match = 0
        best_kv = None

        for i, tok in enumerate(token_ids):
            child = node.children.get(tok)
            if child is None:
                break
            node = child
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
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            matched += 1
        return matched

    def _count_entries(self) -> int:
        """Count total entries with KV data in the subtree. O(1)."""
        return self.size

    def _find_lru_leaf(self) -> tuple[float, RadixNode | None]:
        """Find the leaf node with the oldest last_access time.

        A candidate must hold KV data and have no children. Iterative so deep
        token chains cannot hit the recursion limit.
        """
        best_time = float("inf")
        best_node: RadixNode | None = None
        stack: list[RadixNode] = [self]
        while stack:
            node = stack.pop()
            if node.kv_data is not None and not node.children:
                if node.last_access < best_time:
                    best_time = node.last_access
                    best_node = node
            stack.extend(node.children.values())
        return best_time, best_node

    def _collect_evictable(self) -> tuple[list[tuple[float, RadixNode]], int]:
        """Return evictable leaf entries (oldest first) plus live entry count.

        Single pass (O(n log n) with the sort) so callers can batch-evict
        instead of re-walking the tree once per victim. The count covers ALL
        entries with data, including interior nodes that are not themselves
        evictable leaves.
        """
        found: list[tuple[float, RadixNode]] = []
        live = 0
        stack: list[RadixNode] = [self]
        while stack:
            node = stack.pop()
            if node.kv_data is not None:
                live += 1
                if not node.children:
                    found.append((node.last_access, node))
            stack.extend(node.children.values())
        found.sort(key=lambda pair: pair[0])
        return found, live

    def _remove_entry(self) -> Any:
        """Remove this node's KV entry and prune any dead chain left behind.

        Clears ``kv_data``, decrements the running size counter along the
        ancestor path, then detaches this node (and any now-empty ancestors)
        from the tree so LRU-leaf detection keeps working.

        Returns:
            The removed value, or None if the node held no entry.
        """
        if self.kv_data is None:
            return None
        old_value = self.kv_data
        self.kv_data = None

        # Fix running entry counts up the path.
        n: RadixNode | None = self
        while n is not None:
            n.size = max(0, n.size - 1)
            n = n.parent

        # Prune upward: a node with no data and no children is dead weight.
        node = self
        while node.parent is not None and node.kv_data is None and not node.children:
            parent = node.parent
            parent.children.pop(node.token, None)
            node.parent = None
            node = parent

        return old_value

    def evict_lru(self, max_entries: int) -> int:
        """Evict LRU leaf entries until the subtree holds <= max_entries.

        Evicted entries are cleared AND their now-empty leaf chains pruned, so
        repeated shared-prefix evictions never exhaust the victim pool.

        Returns:
            Number of entries evicted.
        """
        limit = max(0, max_entries)
        victims, live = self._collect_evictable()
        remaining = live
        evicted = 0
        for _, node in victims:
            if remaining <= limit:
                break
            if node.kv_data is None:
                continue  # Already removed (defensive; cannot normally happen)
            node._remove_entry()
            remaining -= 1
            evicted += 1
        return evicted

    def _take_path(self, token_ids: list[int]) -> Any:
        """Remove and return the KV entry at the given path (None if absent)."""
        node = self
        for tok in token_ids:
            child = node.children.get(tok)
            if child is None:
                return None
            node = child
        return node._remove_entry()

    def _evict_path(self, token_ids: list[int]) -> bool:
        """Evict KV data at the given path. Returns True if entry was found and removed."""
        return self._take_path(token_ids) is not None

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
        stack = list(self.children.values())
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count

    def _max_depth(self) -> int:
        max_depth = 0
        stack: list[tuple[RadixNode, int]] = [(self, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
            stack.extend((child, depth + 1) for child in node.children.values())
        return max_depth


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
        """Store KV data at the given token prefix.

        Overwriting an existing entry subtracts the replaced value's estimated
        size before adding the new one, keeping the memory counter accurate.
        """
        if len(token_ids) < self._min_prefix_len:
            return

        with self._lock:
            entry_bytes = self._estimate_entry_memory(kv_data)

            # Subtract the replaced entry's size on overwrite.
            replaced_bytes = 0
            node = self._root
            path_complete = True
            for tok in token_ids:
                child = node.children.get(tok)
                if child is None:
                    path_complete = False
                    break
                node = child
            if path_complete and node.kv_data is not None:
                replaced_bytes = self._estimate_entry_memory(node.kv_data)

            self._total_memory_bytes += entry_bytes - replaced_bytes
            if self._total_memory_bytes < 0:
                self._total_memory_bytes = 0

            self._root.insert(token_ids, kv_data)
            self._evict_until_fit()

    def find_shared_prefix(self, token_ids: list[int]) -> int:
        """Find the longest prefix that exists in the trie (even without KV data)."""
        with self._lock:
            return self._root.find_shared_prefix(token_ids)

    def evict(self, token_ids: list[int]) -> bool:
        """Evict the entry at the given token prefix."""
        with self._lock:
            removed = self._root._take_path(token_ids)
            if removed is not None:
                removed_bytes = self._estimate_entry_memory(removed)
                self._total_memory_bytes = max(
                    0, self._total_memory_bytes - removed_bytes
                )
                return True
            return False

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

    def _within_limits(self) -> bool:
        """True when both the entry-count cap and memory budget are satisfied."""
        within_count = self._max_entries <= 0 or self._root.size <= self._max_entries
        within_budget = self._total_memory_bytes <= self._memory_budget
        return within_count and within_budget

    def _evict_until_fit(self) -> None:
        """Evict LRU entries until within memory budget and entry count.

        All evictable leaves are collected in a single traversal and removed
        oldest-first, so evicting k leaves costs O(n + k) instead of re-walking
        the tree once per victim.
        """
        if self._within_limits():
            return

        victims, _live = self._root._collect_evictable()
        for _, node in victims:
            if self._within_limits():
                break
            if node.kv_data is None:
                continue  # Defensive; victims always hold data here
            entry_bytes = self._estimate_entry_memory(node.kv_data)
            node._remove_entry()
            self._total_memory_bytes = max(
                0, self._total_memory_bytes - entry_bytes
            )

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
