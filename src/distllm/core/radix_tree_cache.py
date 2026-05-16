"""RadixTree (trie) based prefix cache for token sequences.

Replaces the flat hash-based LRU with a trie structure (like SGLang's).
Provides:
- O(log n) lookups via trie traversal
- Substring sharing (common prefixes share nodes)
- Cross-request KV cache reuse
- LRU eviction at the node level
"""

import time
from typing import Any, Dict, List, Optional, Tuple


class RadixNode:
    """A node in the radix trie.

    Each node represents a token in the sequence. The path from root to
    any node represents a token prefix. Leaf nodes may store KV cache data.
    """

    __slots__ = ("token", "children", "kv_data", "last_access", "size")

    def __init__(self, token: int = -1):
        self.token = token
        self.children: Dict[int, "RadixNode"] = {}
        self.kv_data: Optional[Any] = None
        self.last_access: float = 0.0
        self.size: int = 0  # Number of descendant leaf nodes with KV data

    @property
    def is_leaf(self) -> bool:
        return self.kv_data is not None

    def touch(self) -> None:
        """Mark this node as recently accessed (for LRU)."""
        self.last_access = time.time()

    def insert(self, token_ids: List[int], kv_data: Any) -> None:
        """Insert a token sequence with KV data into the trie.

        Walks down the trie creating nodes as needed, then stores kv_data
        at the terminal node.

        Args:
            token_ids: Full token sequence.
            kv_data: KV cache data to store.
        """
        node = self
        for tok in token_ids:
            if tok not in node.children:
                node.children[tok] = RadixNode(token=tok)
            node = node.children[tok]

        node.kv_data = kv_data
        node.last_access = time.time()
        node.size = 1
        # Update size counters along the path
        self._increment_size(token_ids)

    def _increment_size(self, token_ids: List[int]) -> None:
        """Increment size counters along the path."""
        node = self
        for tok in token_ids:
            node = node.children[tok]
            node.size += 1

    def lookup(self, token_ids: List[int]) -> Tuple[int, Optional[Any]]:
        """Find the longest cached prefix in the trie.

        Traverses the trie following token_ids, returning the deepest node
        that has KV data stored.

        Args:
            token_ids: Token sequence to search.

        Returns:
            (matched_len, kv_data) where matched_len is the number of
            tokens found in the cache. Returns (0, None) on miss.
        """
        node = self
        best_len = 0
        best_kv = None

        for i, tok in enumerate(token_ids):
            if tok not in node.children:
                break
            node = node.children[tok]
            if node.kv_data is not None:
                best_len = i + 1
                best_kv = node.kv_data
                node.touch()

        if best_len > 0:
            return best_len, best_kv
        return 0, None

    def find_shared_prefix(self, token_ids: List[int]) -> int:
        """Find the length of the shared prefix that exists in the trie.

        Unlike lookup, this finds how many tokens match ANY path in the
        trie, not just those with stored KV data. Useful for substring
        sharing across requests.

        Args:
            token_ids: Token sequence to check.

        Returns:
            Length of the longest matching prefix in the trie.
        """
        node = self
        for i, tok in enumerate(token_ids):
            if tok not in node.children:
                return i
            node = node.children[tok]
        return len(token_ids)

    def evict_lru(self, max_entries: int) -> int:
        """Evict least-recently-used entries until under capacity.

        Does a post-order traversal to find and remove the oldest leaf nodes.

        Args:
            max_entries: Maximum number of KV entries to keep.

        Returns:
            Number of entries evicted.
        """
        count = self._count_entries()
        if count <= max_entries:
            return 0

        evicted = 0
        # Collect all leaf nodes with their access times
        leaves = self._collect_leaves()
        # Sort by access time (oldest first)
        leaves.sort(key=lambda x: x[1])

        # Evict oldest entries
        for token_path, _ in leaves:
            if self._count_entries() <= max_entries:
                break
            if self._evict_path(token_path):
                evicted += 1

        return evicted

    def _count_entries(self) -> int:
        """Count total KV data entries in the subtree."""
        count = 0
        if self.kv_data is not None:
            count += 1
        for child in self.children.values():
            count += child._count_entries()
        return count

    def _collect_leaves(self) -> List[Tuple[List[int], float]]:
        """Collect all leaf nodes as (token_path, access_time)."""
        leaves = []
        self._collect_leaves_recursive([], leaves)
        return leaves

    def _collect_leaves_recursive(self, path: List[int], result: List) -> None:
        if self.kv_data is not None:
            result.append((list(path), self.last_access))
        for tok, child in self.children.items():
            path.append(tok)
            child._collect_leaves_recursive(path, result)
            path.pop()

    def _evict_path(self, token_path: List[int]) -> bool:
        """Remove the KV data at the given token path.

        Returns True if an entry was removed.
        """
        node = self
        for tok in token_path:
            if tok not in node.children:
                return False
            node = node.children[tok]

        if node.kv_data is not None:
            node.kv_data = None
            node.size = 0
            return True
        return False

    def clear(self) -> None:
        """Remove all data from this subtree."""
        self.children.clear()
        self.kv_data = None
        self.size = 0

    def stats(self) -> dict:
        """Get statistics about the trie."""
        return {
            "total_entries": self._count_entries(),
            "total_nodes": self._count_nodes(),
            "max_depth": self._max_depth(),
        }

    def _count_nodes(self) -> int:
        """Count total nodes in the trie."""
        count = 1  # This node
        for child in self.children.values():
            count += child._count_nodes()
        return count

    def _max_depth(self) -> int:
        """Get maximum depth of the trie."""
        if not self.children:
            return 0
        return 1 + max(child._max_depth() for child in self.children.values())


class RadixTreeCache:
    """RadixTree (trie) based prefix cache with LRU eviction.

    Provides O(log n) lookups, substring sharing, and cross-request
    KV cache reuse. Replaces the flat hash-based LRU PrefixCache.

    The trie structure naturally shares common prefixes across requests,
    so if multiple requests share a system prompt or conversation history,
    they share the same trie nodes.
    """

    def __init__(self, max_entries: int = 1024, min_prefix_len: int = 16):
        self.max_entries = max_entries
        self.min_prefix_len = min_prefix_len
        self._root = RadixNode()
        self._hits = 0
        self._misses = 0

    def lookup(self, token_ids: List[int]) -> Tuple[int, Optional[Any]]:
        """Find the longest cached prefix.

        Traverses the trie to find the deepest node with KV data
        that matches a prefix of token_ids.

        Args:
            token_ids: Full sequence of token IDs.

        Returns:
            (matched_len, kv_data) where matched_len is the number of tokens
            that were found in the cache. Returns (0, None) on miss.
        """
        if len(token_ids) < self.min_prefix_len:
            self._misses += 1
            return 0, None

        matched_len, kv_data = self._root.lookup(token_ids)

        if matched_len >= self.min_prefix_len:
            self._hits += 1
            return matched_len, kv_data

        self._misses += 1
        return 0, None

    def store(self, token_ids: List[int], kv_data: Any) -> None:
        """Cache a prefix's KV data.

        Args:
            token_ids: Token sequence to cache.
            kv_data: Precomputed KV cache data (layer_idx -> (k, v) tensors).
        """
        if len(token_ids) < self.min_prefix_len:
            return

        self._root.insert(token_ids, kv_data)

        # Evict if over capacity
        self._root.evict_lru(self.max_entries)

    def find_shared_prefix(self, token_ids: List[int]) -> int:
        """Find how many tokens share a prefix with existing cache entries.

        Useful for cross-request substring sharing: even if a full prefix
        isn't cached, part of it might be, saving recomputation.

        Args:
            token_ids: Token sequence to check.

        Returns:
            Length of the longest shared prefix in the trie.
        """
        return self._root.find_shared_prefix(token_ids)

    def evict(self, token_ids: List[int]) -> bool:
        """Remove a specific prefix from the cache.

        Returns True if the entry was found and removed.
        """
        node = self._root
        for tok in token_ids:
            if tok not in node.children:
                return False
            node = node.children[tok]

        if node.kv_data is not None:
            node.kv_data = None
            return True
        return False

    def clear(self) -> None:
        """Remove all cached entries."""
        self._root.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        trie_stats = self._root.stats()
        return {
            "prefix_cache_entries": trie_stats["total_entries"],
            "prefix_cache_max_entries": self.max_entries,
            "prefix_cache_hits": self._hits,
            "prefix_cache_misses": self._misses,
            "prefix_cache_hit_rate": round(self.hit_rate, 4),
            "radix_tree_nodes": trie_stats["total_nodes"],
            "radix_tree_max_depth": trie_stats["max_depth"],
        }
