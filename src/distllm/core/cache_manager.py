"""Cache manager for distributed LLM inference.

Manages prefix cache, KV cache lifecycle, and chunked prefill.
Extracted from the Coordinator class.
"""

import asyncio
import threading
from typing import Any

from loguru import logger

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
        radix_tree_cache_enabled: bool = True,
    ):
        self.prefix_cache = None

        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.chunked_prefill_chunk_size = chunked_prefill_chunk_size
        self.persistence_manager = persistence_manager
        self.cache_index = cache_index
        self.gossip_protocol = gossip_protocol
        self.gossip_client = gossip_client
        self._lock = threading.Lock()

    def lookup_prefix(self, tokens: list[int]) -> tuple[int, Any]:
        if self.prefix_cache is None:
            return (0, None)
        with self._lock:
            return self.prefix_cache.lookup(tokens)

    async def async_lookup_prefix(self, tokens: list[int]) -> tuple[int, Any]:
        """Async variant: runs lookup_prefix in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self.lookup_prefix, tokens)

    def store_prefix(self, tokens: list[int], entry: Any) -> None:
        """Store tokens and entry in the prefix cache.

        Args:
            tokens: List of token IDs.
            entry: Cache entry to store.
        """
        if self.prefix_cache is not None:
            with self._lock:
                self.prefix_cache.store(tokens, entry)

    def find_shared_prefix(self, tokens: list[int]) -> int:
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
        with self._lock:
            if hasattr(self.prefix_cache, "find_shared_prefix"):
                return self.prefix_cache.find_shared_prefix(tokens)
            match_len, _ = self.prefix_cache.lookup(tokens)
            return match_len

    def maybe_chunk(self, tokens: list[int]) -> Any | None:
        """Apply chunked prefill if enabled and prompt is long."""
        return None

    @staticmethod
    def create_kv_cache() -> KVCache:
        """Create a new KV cache instance."""
        return KVCache()

    @staticmethod
    def release_kv_cache(cache: KVCache) -> None:
        """Release a KV cache and free associated memory."""
        del cache

    def _get_cache_index(self):
        return None

    def lookup_with_disk_fallback(self, tokens: list[int], model_name: str) -> tuple[int, dict | None]:
        """Lookup prefix with disk cache fallback.

        First checks in-memory prefix cache, then falls back to disk cache.
        Uses rolling hash matching prefix_cache / CacheIndex semantics.

        Args:
            tokens: List of token IDs.
            model_name: Model name for disk cache path.

        Returns:
            Tuple of (prefix_match_length, cached_data_or_None).
        """
        with self._lock:
            match_len, entry = self.lookup_prefix(tokens)
            if match_len > 0:
                return match_len, entry

            if self.persistence_manager is not None:
                idx = self._get_cache_index()
                if idx is None:
                    return 0, None
                for trim in range(min(len(tokens), 512)):
                    trimmed = tokens[:len(tokens) - trim]
                    prefix_hash = idx.index_tokens(trimmed)
                    key = str(prefix_hash)
                    cached = self.persistence_manager.load(key, model_name)
                    if cached is not None:
                        return len(trimmed), cached

        return 0, None

    def mark_dirty(self, request_id: str) -> None:
        """Mark a cache entry as dirty for persistence."""
        if self.persistence_manager is not None:
            with self._lock:
                self.persistence_manager.mark_dirty(request_id)

    def lookup_with_gossip(self, tokens: list[int]) -> str | None:
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
            idx = self._get_cache_index()
            if idx is None:
                return None
            prefix_hash = idx.index_tokens(tokens)
            if self.persistence_manager._storage_path:
                import os
                path = os.path.join(self.persistence_manager._storage_path, f"{prefix_hash}.pt")
                if os.path.exists(path):
                    return "disk"

        # 3. Gossip index lookup
        if self.cache_index is not None:
            idx = self._get_cache_index()
            if idx is None:
                return None
            prefix_hash = idx.index_tokens(tokens)
            node_id = self.cache_index.lookup(prefix_hash)
            if node_id:
                return node_id

        # 4. Active peer query: gossip cache index missed, broadcast to all peers
        if self.gossip_protocol is not None and self.gossip_client is not None:
            idx = self._get_cache_index()
            if idx is None:
                return None
            prefix_hash = idx.index_tokens(tokens)
            entry = self.gossip_protocol.request_cache_from_peers(prefix_hash, self.gossip_client)
            if entry is not None:
                return "peer"

        return None

    def sync_with_peers(self) -> int:
        """Trigger one gossip round with a random peer.

        Builds an advertisement of local cache entries, sends it to a peer,
        receives the peer's advertisement, and merges missing entries.
        Only fetches KV cache data when bandwidth is available.

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

        try:
            peer_ad = self.gossip_client.exchange(peer, ad)
            if peer_ad:
                # Process peer's advertisement
                missing = self.gossip_protocol.process_advertisement(peer_ad)

                # Request missing entries (bandwidth-aware)
                if missing:
                    request = self.gossip_protocol.build_request(peer, missing)
                    response = self.gossip_client.request_entries(peer, request)
                    merged = self.gossip_protocol.process_response(response)

                    # Store discovered entries
                    for prefix_hash, entry_ref in response.get("cache_entries", {}).items():
                        self._store_discovered_entry(prefix_hash, entry_ref, peer)

                    return merged
        except Exception as e:
            logger.debug("Gossip sync failed: {}", e)

        return 0

    def fetch_from_peer(self, peer_id: str, prefix_hash: str, tokens: list[int]) -> dict | None:
        """Fetch a specific KV cache entry from a peer node.

        Args:
            peer_id: Peer node ID to fetch from.
            prefix_hash: Hash of the prefix to fetch.
            tokens: Original token sequence (for verification).

        Returns:
            KV cache data dict, or None if fetch failed.
        """
        if self.gossip_client is None:
            return None

        try:
            kv_data = self.gossip_client.fetch_kv_cache(peer_id, prefix_hash)
            if kv_data:
                # Verify the fetched cache matches our token sequence
                if self.prefix_cache is not None:
                    self.prefix_cache.store(tokens, kv_data)
                return kv_data
        except Exception as e:
            logger.debug(f"Failed to fetch cache from {peer_id}: {e}")

        return None

    def _store_discovered_entry(
        self, prefix_hash: str, entry_ref: str, source_peer: str
    ) -> None:
        """Store a cache entry discovered via gossip.

        Records the entry in the gossip protocol's index and,
        if we have the actual KV data, stores in the prefix cache.

        Args:
            prefix_hash: Hash of the token prefix.
            entry_ref: Reference to the KV cache data.
            source_peer: Node ID of the source peer.
        """
        if self.gossip_protocol:
            self.gossip_protocol.store_local(prefix_hash, entry_ref)

        # Update cache index with discovered entry location
        if self.cache_index is not None:
            self.cache_index.store(prefix_hash, source_peer, entry_ref)
