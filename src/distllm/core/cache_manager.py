"""Cache manager for distributed LLM inference.

Manages prefix cache, KV cache lifecycle, and chunked prefill.
Extracted from the Coordinator class.
"""

import asyncio
import hashlib
import threading
import time
from typing import Any

import torch
from loguru import logger

from distllm.core.kv_cache import KVCache

# Polynomial rolling hash constants (same as PrefixCache)
_HASH_BASE = 31337
_HASH_MOD = (1 << 61) - 1


class RollingHash:
    """Incremental rolling hash supporting O(1) extend and O(n) recomputation.

    Maintains a running polynomial hash over a token sequence.
    """

    def __init__(self):
        self._hash = 0
        self._length = 0

    def extend(self, token: int) -> int:
        """Append a token and return the new hash. O(1)."""
        self._hash = (self._hash * _HASH_BASE + token) % _HASH_MOD
        self._length += 1
        return self._hash

    @property
    def hash(self) -> int:
        return self._hash

    @property
    def length(self) -> int:
        return self._length

    def reset(self):
        self._hash = 0
        self._length = 0


def _rolling_prefix_hashes(tokens: list[int], max_len: int) -> dict[int, int]:
    """Compute rolling hashes for all prefix lengths 1..max_len in O(n).

    Returns a dict mapping prefix_length → rolling_hash.
    """
    hashes: dict[int, int] = {}
    h = 0
    for i, tok in enumerate(tokens[:max_len]):
        h = (h * _HASH_BASE + tok) % _HASH_MOD
        hashes[i + 1] = h
    return hashes


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
        predictive_cache_enabled: bool = False,
        gpu_cache_mb: int = 512,
        cpu_cache_mb: int = 4096,
        ssd_cache_gb: int = 50,
    ):
        if prefix_cache_enabled:
            from distllm.dist.prefix_cache import PrefixCache
            self.prefix_cache = PrefixCache(
                max_entries=prefix_cache_max_entries,
                min_prefix_len=prefix_cache_min_prefix_len,
            )
        else:
            self.prefix_cache = None

        self._predictive_cache = None
        if predictive_cache_enabled:
            from distllm.dist.predictive_cache import PredictiveCacheManager
            self._predictive_cache = PredictiveCacheManager(
                gpu_memory_bytes=gpu_cache_mb * 1024 * 1024,
                cpu_memory_bytes=cpu_cache_mb * 1024 * 1024,
            )
            self._predictive_cache.start_prefetch_service()
            logger.info(f"Predictive cache enabled: GPU={gpu_cache_mb}MB, CPU={cpu_cache_mb}MB")

        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.chunked_prefill_chunk_size = chunked_prefill_chunk_size
        self.persistence_manager = persistence_manager
        self.cache_index = cache_index
        self.gossip_protocol = gossip_protocol
        self.gossip_client = gossip_client
        self._lock = threading.RLock()
        # Per-tier metrics (C15)
        self._tier_stats: dict[str, dict[str, int]] = {
            "local": {"hits": 0, "misses": 0},
            "disk": {"hits": 0, "misses": 0},
            "gossip_index": {"hits": 0, "misses": 0},
            "broadcast": {"hits": 0, "misses": 0},
        }
        # E3: Adaptive tier latency tracking
        self._tier_latencies: dict[str, list[float]] = {
            "local": [],
            "disk": [],
            "gossip_index": [],
            "broadcast": [],
        }
        self._tier_timeout_ms: float = 5000.0  # Skip tier if P95 > this

        # ── Multi-tier cache orchestration ──────────────────────────────
        # Tiers: GPU (L1, fastest), CPU pinned memory (L2), SSD (L3, largest).
        # Entries start in L1, graduate to L2 on inactivity, demote to L3
        # under pressure.  Promotion (L3→L2, L2→L1) happens on access.
        self._tiers: dict[str, dict[str, Any]] = {
            "gpu": {
                "max_bytes": gpu_cache_mb * 1024 * 1024,
                "used_bytes": 0,
                "entries": {},  # prefix_hash -> (blob, last_access, size)
                "latency_ns": 1_000,       # ~1µs
                "bandwidth_gbps": 900,     # HBM2e
            },
            "cpu": {
                "max_bytes": cpu_cache_mb * 1024 * 1024,
                "used_bytes": 0,
                "entries": {},
                "latency_ns": 100_000,     # ~100µs
                "bandwidth_gbps": 50,      # PCIe 4.0 x16
            },
            "ssd": {
                "max_bytes": ssd_cache_gb * 1024**3,
                "used_bytes": 0,
                "entries": {},
                "latency_ns": 10_000_000,  # ~10ms
                "bandwidth_gbps": 3,       # NVMe
            },
        }
        # Ghost cache: metadata-only tracking of recently evicted blocks
        # to detect thrashing and adjust admission policy.
        self._ghost_cache: dict[str, float] = {}  # prefix_hash -> eviction_time
        self._ghost_cache_ttl: float = 60.0

    def _tier_lookup(self, prefix_hash: str) -> tuple[str | None, Any | None]:
        """Look up *prefix_hash* across tiers L1→L2→L3.

        On a hit in a lower tier, promotes the entry one level up.
        Returns (tier_name, blob) or (None, None) on miss.
        """
        with self._lock:
            for tier_name in ("gpu", "cpu", "ssd"):
                tier = self._tiers[tier_name]
                if prefix_hash in tier["entries"]:
                    blob, last_access, size = tier["entries"][prefix_hash]
                    now = time.time()
                    # Promote: if hit in a lower tier, move to the tier above
                    if tier_name == "ssd":
                        self._tier_store("cpu", prefix_hash, blob, size)
                        self._tiers["ssd"]["entries"].pop(prefix_hash, None)
                        self._tiers["ssd"]["used_bytes"] -= size
                    elif tier_name == "cpu":
                        self._tier_store("gpu", prefix_hash, blob, size)
                        self._tiers["cpu"]["entries"].pop(prefix_hash, None)
                        self._tiers["cpu"]["used_bytes"] -= size
                    else:
                        tier["entries"][prefix_hash] = (blob, now, size)
                    return tier_name, blob
            return None, None

    def _prune_ghost_cache(self) -> None:
        """Drop expired ghost-cache entries (bounded; called under the lock).

        Previously the ghost cache was append-only — every SSD eviction added
        an entry that was never removed, so the dict grew without bound.
        """
        if not self._ghost_cache:
            return
        now = time.time()
        cutoff = now - self._ghost_cache_ttl
        self._ghost_cache = {
            h: t for h, t in self._ghost_cache.items() if t > cutoff
        }
        # Hard cap to bound memory even under heavy eviction churn.
        max_ghost = 100_000
        if len(self._ghost_cache) > max_ghost:
            oldest = sorted(self._ghost_cache.items(), key=lambda kv: kv[1])
            for h, _ in oldest[: len(self._ghost_cache) - max_ghost]:
                self._ghost_cache.pop(h, None)

    def _tier_store(self, tier_name: str, prefix_hash: str, blob: Any, size: int) -> None:
        """Store *blob* in *tier_name*, evicting if necessary.
        Must be called with self._lock held.
        """
        self._prune_ghost_cache()
        tier = self._tiers[tier_name]
        while tier["used_bytes"] + size > tier["max_bytes"] and tier["entries"]:
            victim_hash, (victim_blob, _, victim_size) = min(
                tier["entries"].items(),
                key=lambda kv: self._tier_eviction_score(kv[0], kv[1], tier_name),
            )
            if tier_name == "gpu":
                self._tier_store("cpu", victim_hash, victim_blob, victim_size)
            elif tier_name == "cpu":
                self._tier_store("ssd", victim_hash, victim_blob, victim_size)
            else:
                self._ghost_cache[victim_hash] = time.time()
            tier["entries"].pop(victim_hash, None)
            tier["used_bytes"] -= victim_size
        tier["entries"][prefix_hash] = (blob, time.time(), size)
        tier["used_bytes"] += size

    def _tier_eviction_score(
        self, prefix_hash: str, entry: tuple[Any, float, int], tier_name: str,
    ) -> float:
        """Compute eviction score for a tier entry.

        Lower score = better eviction candidate.
        Factors: recency, frequency (from ghost cache), and size.
        """
        _blob, last_access, size = entry
        now = time.time()
        age = max(now - last_access, 1.0)
        recency = 1.0 / age
        # Entries that were recently evicted from upper tiers (thrashing
        # signal) get a boost to avoid repeated promote/demote cycles.
        ghost_boost = 0.0
        if prefix_hash in self._ghost_cache and (now - self._ghost_cache[prefix_hash]) < self._ghost_cache_ttl:
            ghost_boost = 2.0  # Penalty — was recently evicted
        return recency + ghost_boost + (size / max(tier["max_bytes"], 1))

    def lookup_prefix(self, tokens: list[int]) -> tuple[int, Any]:
        """Lookup prefix across all local and remote tiers.

        Linear fallthrough: predictive → L1/L2/L3 → local prefix cache → remote.
        The multi-tier cache (GPU → CPU → SSD) is checked first for the
        full prefix_hash.  If found, the entry is promoted one level.
        """
        prefix_hash = ""
        if tokens:
            prefix_hash = self._hash_tokens(tokens)

        # 1. Multi-tier cache (GPU L1 → CPU L2 → SSD L3) with promotion
        if prefix_hash:
            _tier, blob = self._tier_lookup(prefix_hash)
            if blob is not None:
                if isinstance(blob, tuple) and len(blob) == 2:
                    # Stored form is (matched_len, kv_data); return the data
                    # itself — every caller treats element 1 as the KV blob.
                    return blob[0], blob[1]
                return len(tokens), blob

        # 2. Predictive cache (ML-based pattern matching)
        if self._predictive_cache is not None:
            predictions = self._predictive_cache.observe_request(tokens)
            if predictions:
                result = self._predictive_cache.lookup(tokens)
                if result and result[0] > 0:
                    return result

        # 3. Local prefix cache (in-memory)
        if self.prefix_cache is not None:
            with self._lock:
                match_len, kv_data = self.prefix_cache.lookup(tokens)
                if match_len > 0 and kv_data is not None:
                    return match_len, kv_data

        # 4. Remote peers via gossip protocol
        return self._lookup_prefix_remote(tokens)

    def _lookup_prefix_remote(self, tokens: list[int]) -> tuple[int, Any]:
        """Try to find a matching prefix in peer nodes' caches via gossip."""
        if self.gossip_protocol is None or not tokens:
            return (0, None)
        try:
            from distllm.dist.cache import CacheIndex
            idx = CacheIndex()
            prefix_hash = idx.index_tokens(tokens[:32])
            result = self.gossip_protocol.request_cache_from_peers(prefix_hash, self.gossip_client)
            if result and result.get("kv_data") is not None:
                match_len = result.get("match_len", 0)
                kv_data = result["kv_data"]
                if match_len > 0 and self.prefix_cache is not None:
                    with self._lock:
                        self.prefix_cache.store(tokens[:match_len], kv_data)
                return match_len, kv_data
        except Exception:
            logger.debug("Remote prefix lookup failed", exc_info=True)
        return (0, None)

    def _estimate_entry_size(self, kv_data: Any) -> int:
        """Estimate memory footprint of a cache entry."""
        if isinstance(kv_data, torch.Tensor):
            return kv_data.element_size() * kv_data.numel()
        if isinstance(kv_data, dict):
            total = 0
            for v in kv_data.values():
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.numel()
            return total if total else 1024
        if isinstance(kv_data, (list, tuple)):
            total = 0
            for v in kv_data:
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.numel()
            return total if total else 1024
        return 1024  # default 1KB for unknown types

    def store_prefix(self, tokens: list[int], kv_data: Any) -> None:
        """Store a prefix in the local cache and advertise via gossip.

        Args:
            tokens: List of token IDs.
            kv_data: Cache entry / KV data to store.
        """
        # Store in multi-tier cache (GPU L1)
        if tokens:
            # NOTE: must hash the FULL token list — the lookup path hashes the
            # entire prompt, so hashing only the first 32 tokens here made the
            # whole tier cache a guaranteed miss for any prompt >32 tokens.
            prefix_hash = self._hash_tokens(tokens)
            size = self._estimate_entry_size(kv_data)
            self._tier_store("gpu", prefix_hash, (len(tokens), kv_data), size)

        # Local prefix cache (in-memory)
        if self.prefix_cache is not None:
            with self._lock:
                self.prefix_cache.store(tokens, kv_data)
        if self.gossip_protocol is not None:
            try:
                from distllm.dist.cache import CacheIndex
                idx = CacheIndex()
                prefix_hash = idx.index_tokens(tokens[:32])
                self.gossip_protocol.advertise_cache(prefix_hash)
            except Exception:
                pass

    async def async_lookup_prefix(self, tokens: list[int]) -> tuple[int, Any]:
        """Async variant: runs lookup_prefix in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self.lookup_prefix, tokens)

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

    def maybe_chunk(self, tokens: list[int], chunk_size: int = 512) -> list[list[int]] | None:
        """Split a long prompt into chunks for chunked prefill.

        Returns a list of token chunks if the prompt exceeds *chunk_size*,
        or None if no chunking is needed.
        """
        if len(tokens) <= chunk_size:
            return None
        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    @staticmethod
    def create_kv_cache() -> KVCache:
        """Create a new KV cache instance."""
        return KVCache()

    @staticmethod
    def release_kv_cache(cache: KVCache) -> None:
        """Release a KV cache and free associated memory.

        Clears the cache's internal data structures and moves tensors to CPU
        so that garbage collection can reclaim GPU memory. Note: the caller
        must also drop their reference to the KVCache object for full cleanup.

        Thread-safe: acquires the cache's internal lock before mutating.
        """
        import torch
        # Acquire the cache's lock before accessing internals to prevent
        # racing with concurrent update() or get() calls.
        cache_lock = getattr(cache, '_lock', None)
        if cache_lock is not None:
            cache_lock.acquire()
        try:
            if hasattr(cache, 'clear'):
                cache.clear()
            if hasattr(cache, 'cache') and isinstance(cache.cache, list):
                for i, (k, v) in enumerate(cache.cache):
                    cache.cache[i] = (k.cpu(), v.cpu())
                cache.cache.clear()
            if hasattr(cache, '_seq_lens'):
                cache._seq_lens.clear()
            if hasattr(cache, '_qsegments'):
                cache._qsegments.clear()
        finally:
            if cache_lock is not None:
                cache_lock.release()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Compute a hash string for a token sequence using the shared CacheIndex."""
        if self.cache_index is not None and hasattr(self.cache_index, 'index_tokens'):
            return self.cache_index.index_tokens(tokens)
        # Fallback: SHA-256
        h = hashlib.sha256()
        for t in tokens:
            h.update(t.to_bytes(4, "little", signed=True))
        return f"h{h.hexdigest()[:32]}"

    def lookup_with_disk_fallback(self, tokens: list[int], model_name: str) -> tuple[int, dict | None]:
        """Lookup prefix with disk cache fallback.

        First checks in-memory prefix cache, then falls back to disk cache.
        Uses pre-computed rolling hashes for O(n) total hash computation
        instead of O(n²) per-candidate recomputation.

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

            if self.persistence_manager is None:
                return 0, None

            # Pre-compute all prefix hashes in one O(n) pass
            max_len = min(len(tokens), 512)
            prefix_hashes = _rolling_prefix_hashes(tokens, max_len)

            # Try candidate lengths in descending order (longest first)
            # to find the best match quickly
            best_match = 0
            best_entry = None

            # Build candidate list: full length, then powers of 2, then fill
            candidates = [max_len]
            step = max_len
            while step > 1:
                step = max(step // 2, 1)
                if step not in candidates and step > best_match:
                    candidates.append(step)
            # Fill in small lengths
            for l in range(1, min(max_len, 16)):
                if l not in candidates:
                    candidates.append(l)

            candidates.sort(reverse=True)

            for length in candidates:
                if length <= best_match:
                    continue
                h = prefix_hashes.get(length)
                if h is None:
                    continue
                key = f"h{h}"
                cached = self.persistence_manager.load(key, model_name)
                if cached is not None:
                    best_match = length
                    best_entry = cached
                    # Since we're going longest-first, first hit is the best
                    break

            if best_match > 0:
                return best_match, best_entry

        return 0, None

    def mark_dirty(self, request_id: str) -> None:
        """Mark a cache entry as dirty for persistence."""
        if self.persistence_manager is not None:
            with self._lock:
                self.persistence_manager.mark_dirty(request_id)

    def lookup_with_gossip(self, tokens: list[int]) -> tuple[str, Any] | None:
        """Lookup prefix with gossip fallback: local → disk → peer nodes.

        Checks in-memory prefix cache first, then disk cache, then
        queries the gossip index for peer nodes that may hold the entry.

        Args:
            tokens: List of token IDs.

        Returns:
            Tuple of (source, kv_data) if found, or None.
            source is "local", "disk", "gossip_index", "broadcast", or a node ID.
        """
        prefix_hash = self._hash_tokens(tokens)

        # 1. Local prefix cache
        if self.prefix_cache is not None:
            match_len, kv_data = self.prefix_cache.lookup(tokens)
            if match_len > 0 and kv_data is not None:
                with self._lock:
                    self._tier_stats["local"]["hits"] += 1
                return ("local", kv_data)

        # 2. Disk persistence
        if self.persistence_manager is not None:
            import os
            storage_path = getattr(self.persistence_manager, '_storage_path', None)
            if storage_path:
                path = os.path.join(storage_path, f"{prefix_hash}.pt")
                if os.path.exists(path):
                    try:
                        kv_data = self.persistence_manager.load(prefix_hash, "")
                        if kv_data is not None:
                            with self._lock:
                                self._tier_stats["disk"]["hits"] += 1
                            return ("disk", kv_data)
                    except Exception:
                        pass

        # 3. Gossip index lookup
        if self.cache_index is not None:
            node_id = self.cache_index.lookup(prefix_hash)
            if node_id:
                with self._lock:
                    self._tier_stats["gossip_index"]["hits"] += 1
                return (node_id, None)

        # 4. Active peer query: gossip cache index missed, broadcast to all peers
        if self.gossip_protocol is not None and self.gossip_client is not None:
            try:
                entry = self.gossip_protocol.request_cache_from_peers(prefix_hash, self.gossip_client)
                if entry is not None:
                    with self._lock:
                        self._tier_stats["broadcast"]["hits"] += 1
                    kv_data = entry.get("kv_data") if isinstance(entry, dict) else None
                    return ("peer", kv_data)
            except Exception:
                logger.debug("Broadcast peer query failed", exc_info=True)

        return None

    def get_tier_stats(self) -> dict[str, dict[str, int]]:
        """Return per-tier hit/miss statistics."""
        with self._lock:
            return {tier: dict(counts) for tier, counts in self._tier_stats.items()}

    def _record_tier_latency(self, tier: str, latency_ms: float) -> None:
        """Record latency for a tier (keeps last 100 samples)."""
        with self._lock:
            samples = self._tier_latencies[tier]
            samples.append(latency_ms)
            if len(samples) > 100:
                samples.pop(0)

    def _should_skip_tier(self, tier: str) -> bool:
        """Check if a tier should be skipped due to high latency."""
        with self._lock:
            samples = self._tier_latencies.get(tier, [])
            if len(samples) < 5:
                return False  # Not enough data
            p95 = sorted(samples)[int(len(samples) * 0.95)]
            return p95 > self._tier_timeout_ms

    def get_tier_latencies(self) -> dict[str, dict[str, float]]:
        """Return per-tier latency statistics (avg, p50, p95)."""
        with self._lock:
            result = {}
            for tier, samples in self._tier_latencies.items():
                if not samples:
                    result[tier] = {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0}
                    continue
                sorted_s = sorted(samples)
                result[tier] = {
                    "avg_ms": sum(sorted_s) / len(sorted_s),
                    "p50_ms": sorted_s[len(sorted_s) // 2],
                    "p95_ms": sorted_s[int(len(sorted_s) * 0.95)],
                }
            return result

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
