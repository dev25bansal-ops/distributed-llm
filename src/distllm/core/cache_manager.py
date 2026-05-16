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
        radix_tree_cache_enabled: bool = False,
    ):
        self.prefix_cache: Optional[PrefixCache] = None
        if prefix_cache_enabled:
            if radix_tree_cache_enabled:
                from distllm.core.radix_tree_cache import RadixTreeCache
                self.prefix_cache = RadixTreeCache(
                    max_entries=prefix_cache_max_entries,
                    min_prefix_len=prefix_cache_min_prefix_len,
                )
            else:
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

    def find_shared_prefix(self, tokens: List[int]) -> int:
        """Find shared prefix length for cross-request substring sharing.

        For RadixTreeCache, this finds how many tokens match any path
        in the trie, even if no KV data is stored at that point.

        Args:
            tokens: List of token IDs.

        Returns:
            Length of the shared prefix.
        """
        if self.prefix_cache is None:
            return 0
        if hasattr(self.prefix_cache, "find_shared_prefix"):
            return self.prefix_cache.find_shared_prefix(tokens)
        # Fallback: use regular lookup
        match_len, _ = self.prefix_cache.lookup(tokens)
        return match_len

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

        Builds an advertisement of local cache entries, sends it to a peer,
        receives the peer's advertisement, and merges missing entries.

        Returns:
            Number of new entries discovered.
        """
        if self.gossip_protocol is None or self.gossip_client is None:
            return 0

        # Build advertisement with local cache hashes
        ad = self.gossip_protocol.advertise()

        # Select a peer
        peer = self.gossip_protocol.select_peer()
        if peer is None:
            return 0

        # Exchange advertisements via gossip client
        # In production, gossip_client.exchange() would:
        # 1. Resolve peer_id to host:port
        # 2. Send our ad, receive peer's ads
        # 3. Process missing entries
        try:
            peer_ad = self.gossip_client.exchange(peer, ad)
            if peer_ad:
                # Process peer's advertisement
                missing = self.gossip_protocol.process_advertisement(peer_ad)

                # Request missing entries
                if missing:
                    request = self.gossip_protocol.build_request(peer, missing)
                    response = self.gossip_client.request_entries(peer, request)
                    merged = self.gossip_protocol.process_response(response)

                    # Store discovered entries in local prefix cache
                    for prefix_hash, entry_ref in response.get("cache_entries", {}).items():
                        self._store_discovered_entry(prefix_hash, entry_ref)

                    return merged
        except Exception:
            # Gossip errors are non-fatal; log and continue
            pass

        return 0

    def _store_discovered_entry(self, prefix_hash: str, entry_ref: str) -> None:
        """Store a cache entry discovered via gossip.

        The entry_ref contains serialized KV cache data or a pointer
        to fetch it from the peer node.

        Args:
            prefix_hash: Hash of the token prefix.
            entry_ref: Reference to the KV cache data.
        """
        if self.gossip_protocol:
            self.gossip_protocol.store_local(prefix_hash, entry_ref)

        # If we have the actual KV data, store in prefix cache
        if self.prefix_cache and entry_ref:
            # In production, this would deserialize and store the KV tensors
            # For now, record the availability for future fetch
            pass
