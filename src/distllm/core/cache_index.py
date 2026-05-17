"""Prefix-hash-based KV cache index for gossip protocol.

Tracks which nodes hold which cache entries, enabling peer-to-peer
cache lookups across the distributed cluster.
"""



class CacheIndex:
    """Thread-unsafe prefix-hash-based KV cache index.

    Maps prefix hashes to the nodes that hold the corresponding cache entries.
    Supports replication (multiple nodes can hold the same entry).
    """

    def __init__(self):
        # prefix_hash -> set of node_ids
        self._index: dict[str, set[str]] = {}
        # prefix_hash -> entry_ref (e.g., serialized KV cache or disk path)
        self._refs: dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    def index_tokens(self, tokens: list[int]) -> str:
        """Compute polynomial hash of token sequence.

        Uses the same polynomial rolling hash as PrefixCache for consistency.

        Args:
            tokens: List of token IDs.

        Returns:
            String hash of the token sequence.
        """
        # Polynomial rolling hash: same as PrefixCache
        base = 31
        mod = 2**61 - 1  # Mersenne prime
        h = 0
        for t in tokens:
            h = (h * base + t) % mod
        return f"h{h}"

    def store(self, prefix_hash: str, node_id: str, entry_ref: str) -> None:
        """Record that a node holds a cache entry.

        Args:
            prefix_hash: Hash of the token sequence.
            node_id: ID of the node holding the entry.
            entry_ref: Reference to the cache data (serialized or path).
        """
        if prefix_hash not in self._index:
            self._index[prefix_hash] = set()
        self._index[prefix_hash].add(node_id)
        self._refs[prefix_hash] = entry_ref

    def lookup(self, prefix_hash: str) -> str | None:
        """Find a node that holds the given cache entry.

        Args:
            prefix_hash: Hash of the token sequence.

        Returns:
            Node ID holding the entry, or None if not found.
        """
        nodes = self._index.get(prefix_hash)
        if nodes:
            self._hits += 1
            return next(iter(nodes))
        self._misses += 1
        return None

    def lookup_all(self, prefix_hash: str) -> list[str]:
        """Find all nodes that hold the given cache entry.

        Args:
            prefix_hash: Hash of the token sequence.

        Returns:
            List of node IDs (empty if not found).
        """
        nodes = self._index.get(prefix_hash)
        if nodes:
            self._hits += 1
            return list(nodes)
        self._misses += 1
        return []

    def get_ref(self, prefix_hash: str) -> str | None:
        """Get the entry reference for a prefix hash.

        Args:
            prefix_hash: Hash of the token sequence.

        Returns:
            Entry reference string, or None.
        """
        return self._refs.get(prefix_hash)

    def remove(self, prefix_hash: str, node_id: str | None = None) -> None:
        """Evict an entry from the index.

        Args:
            prefix_hash: Hash of the token sequence.
            node_id: If provided, only remove this node's entry.
                     If None, remove all replicas.
        """
        if prefix_hash not in self._index:
            return

        if node_id is None:
            del self._index[prefix_hash]
            self._refs.pop(prefix_hash, None)
        else:
            self._index[prefix_hash].discard(node_id)
            if not self._index[prefix_hash]:
                del self._index[prefix_hash]
                self._refs.pop(prefix_hash, None)

    def clear(self) -> None:
        """Remove all entries from the index."""
        self._index.clear()
        self._refs.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return index statistics.

        Returns:
            Dict with hit_count, miss_count, total_entries, unique_nodes.
        """
        all_nodes: set[str] = set()
        for nodes in self._index.values():
            all_nodes.update(nodes)

        return {
            "hit_count": self._hits,
            "miss_count": self._misses,
            "total_entries": len(self._index),
            "unique_nodes": len(all_nodes),
        }
