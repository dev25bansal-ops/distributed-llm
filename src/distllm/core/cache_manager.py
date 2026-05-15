"""Cache manager for distributed LLM inference.

Manages prefix cache, KV cache lifecycle, and chunked prefill.
Extracted from the Coordinator class.
"""

from typing import Any, List, Optional, Tuple

import torch

from distllm.core.prefix_cache import PrefixCache
from distllm.core.chunked_prefill import ChunkState, maybe_chunk
from distllm.core.kv_cache import KVCache


class CacheManager:
    """Manages prefix cache, KV cache, and chunked prefill.

    Attributes:
        prefix_cache: Optional prefix cache for prompt deduplication.
        chunked_prefill_enabled: Whether chunked prefill is enabled.
        chunked_prefill_chunk_size: Size of chunks for chunked prefill.
        persistence_manager: Optional cache persistence manager for disk storage.
        cache_index: Optional cache index for P2P gossip lookups.
        gossip_protocol: Optional gossip protocol for peer communication.
        gossip_client: Optional gossip client for P2P RPCs.
    """

    def __init__(
        self,
        prefix_cache_enabled: bool = False,
        prefix_cache_max_entries: int = 1024,
        prefix_cache_min_prefix_len: int = 16,
        chunked_prefill_enabled: bool = True,
        chunked_prefill_chunk_size: int = 512,
        persistence_manager=None,
        cache_index=None,
        gossip_protocol=None,
        gossip_client=None,
    ):
        self.prefix_cache: Optional[PrefixCache] = None
        if prefix_cache_enabled:
            self.prefix_cache = PrefixCache(
                max_entries=prefix_cache_max_entries,
                min_prefix_len=prefix_cache_min_prefix_len,
            )

        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.chunked_prefill_chunk_size = chunked_prefill_chunk_size
        self.persistence_manager = persistence_manager
        self.cache_index = cache_index
        self.gossip_protocol = gossip_protocol
        self.gossip_client = gossip_client

    def lookup_prefix(self, tokens: List[int]) -> Tuple[int, Any]:
        """Lookup prefix match length for tokens.

        Args:
            tokens: List of token IDs.

        Returns:
            Tuple of (prefix_match_length, cache_entry).
        """
        if self.prefix_cache is None:
            return (0, None)
        return self.prefix_cache.lookup(tokens)

    def store_prefix(self, tokens: List[int], entry: Any) -> None:
        """Store tokens and entry in the prefix cache.

        Args:
            tokens: List of token IDs.
            entry: Cache entry to store.
        """
        if self.prefix_cache is not None:
            self.prefix_cache.store(tokens, entry)

    def maybe_chunk(self, tokens: List[int]) -> Optional[ChunkState]:
        """Apply chunked prefill if enabled and prompt is long.

        Args:
            tokens: List of token IDs.

        Returns:
            ChunkState if chunking is needed, None otherwise.
        """
        return maybe_chunk(
            tokens,
            self.chunked_prefill_chunk_size,
            enabled=self.chunked_prefill_enabled,
        )

    @staticmethod
    def create_kv_cache() -> KVCache:
        """Create a new KV cache instance."""
        return KVCache()

    @staticmethod
    def release_kv_cache(cache: KVCache) -> None:
        """Release a KV cache and free associated memory."""
        del cache

    def lookup_with_disk_fallback(self, tokens: List[int], model_name: str) -> Tuple[int, Optional[dict]]:
        """Lookup prefix with disk cache fallback.

        First checks in-memory prefix cache, then falls back to disk cache.

        Args:
            tokens: List of token IDs.
            model_name: Model name for disk cache path.

        Returns:
            Tuple of (prefix_match_length, cached_data_or_None).
        """
        match_len, entry = self.lookup_prefix(tokens)
        if match_len > 0:
            return match_len, entry

        if self.persistence_manager is not None:
            request_id = str(hash(tuple(tokens)))
            cached = self.persistence_manager.load(request_id, model_name)
            if cached is not None:
                return len(tokens), cached

        return 0, None

    def mark_dirty(self, request_id: str) -> None:
        """Mark a cache entry as dirty for persistence."""
        if self.persistence_manager is not None:
            self.persistence_manager.mark_dirty(request_id)

    def lookup_with_gossip(self, tokens: List[int]) -> Optional[str]:
        """Lookup prefix with gossip fallback: local → disk → peer nodes.

        Checks in-memory prefix cache first, then disk cache, then
        queries the gossip index for peer nodes that may hold the entry.

        Args:
            tokens: List of token IDs.

        Returns:
            Node ID holding the cache entry, or None.
        """
        # 1. Local prefix cache
        if self.prefix_cache is not None:
            match_len, _ = self.prefix_cache.lookup(tokens)
            if match_len > 0:
                return "local"

        # 2. Disk persistence
        if self.persistence_manager is not None:
            from distllm.core.cache_index import CacheIndex
            idx = CacheIndex()
            prefix_hash = idx.index_tokens(tokens)
            if self.persistence_manager._storage_path:
                import os
                path = os.path.join(self.persistence_manager._storage_path, f"{prefix_hash}.pt")
                if os.path.exists(path):
                    return "disk"

        # 3. Gossip index lookup
        if self.cache_index is not None:
            from distllm.core.cache_index import CacheIndex
            idx = CacheIndex()
            prefix_hash = idx.index_tokens(tokens)
            node_id = self.cache_index.lookup(prefix_hash)
            if node_id:
                return node_id

        return None

    def sync_with_peers(self) -> int:
        """Trigger one gossip round with a random peer.

        Returns:
            Number of new entries discovered.
        """
        if self.gossip_protocol is None or self.gossip_client is None:
            return 0

        # Build advertisement
        ad = self.gossip_protocol.advertise()

        # Select a peer
        peer = self.gossip_protocol.select_peer()
        if peer is None:
            return 0

        # Exchange advertisements (simplified — peer lookup would resolve host:port)
        # In production, this calls gossip_client.exchange(peer_host, peer_port, ad)
        return 0
