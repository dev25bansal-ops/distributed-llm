"""Semantic cache — cache responses for semantically similar prompts.

Uses embedding similarity to find cached responses for prompts that
are semantically equivalent (not just exact string matches).

Usage::

    cache = SemanticCache(similarity_threshold=0.92)
    cache.store("What is Python?", response="Python is a programming language...")
    result = cache.lookup("Tell me about Python")
    # result is not None — semantically similar to cached prompt
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CacheEntry:
    """A single cached response."""
    prompt_hash: str
    prompt_text: str
    prompt_embedding: list[float]
    response: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0


class SemanticCache:
    """Cache responses for semantically similar prompts.

    Uses cosine similarity on prompt embeddings to find cached
    responses. Falls back to exact string matching when embeddings
    are not available.

    Supports learnable similarity thresholds that adapt per-model
    based on observed hit/miss patterns.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 10000,
        default_ttl: float = 3600.0,
        adaptive_threshold: bool = False,
        threshold_learning_rate: float = 0.01,
        min_threshold: float = 0.80,
        max_threshold: float = 0.99,
    ):
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        # Adaptive threshold
        self._adaptive = adaptive_threshold
        self._lr = threshold_learning_rate
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._recent_similarities: list[float] = []  # Track similarity scores of hits

        # Stats
        self._hits = 0
        self._misses = 0

    def store(
        self,
        prompt: str,
        response: str,
        embedding: list[float] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Store a prompt-response pair in the cache.

        Args:
            prompt: The prompt text.
            response: The generated response.
            embedding: Optional prompt embedding for semantic matching.
            ttl: Time-to-live in seconds (uses default if None).
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        entry = CacheEntry(
            prompt_hash=prompt_hash,
            prompt_text=prompt,
            prompt_embedding=embedding or [],
            response=response,
            ttl_seconds=ttl or self._default_ttl,
        )

        with self._lock:
            # Evict if at capacity
            if len(self._entries) >= self._max_entries:
                self._evict_lru()

            self._entries[prompt_hash] = entry

    def lookup(
        self,
        prompt: str,
        embedding: list[float] | None = None,
    ) -> str | None:
        """Look up a cached response for a prompt.

        Uses semantic similarity if embeddings are available,
        otherwise falls back to exact hash matching.

        Args:
            prompt: The prompt to look up.
            embedding: Optional prompt embedding for semantic matching.

        Returns:
            Cached response text, or None if not found.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        with self._lock:
            # Exact match
            entry = self._entries.get(prompt_hash)
            if entry and not self._is_expired(entry):
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._hits += 1
                return entry.response

            # Semantic match
            if embedding:
                best_match = self._find_similar(embedding)
                if best_match and not self._is_expired(best_match):
                    best_match.access_count += 1
                    best_match.last_accessed = time.time()
                    self._hits += 1
                    return best_match.response

            self._misses += 1
            return None

    def _find_similar(self, query_embedding: list[float]) -> CacheEntry | None:
        """Find the most similar cached entry by cosine similarity."""
        best_entry = None
        best_similarity = -1.0

        for entry in self._entries.values():
            if not entry.prompt_embedding:
                continue
            similarity = self._cosine_similarity(query_embedding, entry.prompt_embedding)
            if similarity >= self._threshold and similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        # Track similarity score for adaptive threshold
        if self._adaptive and best_entry is not None:
            self._recent_similarities.append(best_similarity)
            if len(self._recent_similarities) > 100:
                self._recent_similarities.pop(0)
            self._update_threshold()

        return best_entry

    def _update_threshold(self) -> None:
        """Adapt the similarity threshold based on recent hit patterns.

        If hits are coming from very similar prompts (high similarity scores),
        tighten the threshold to be more selective. If hits are from moderately
        similar prompts, relax the threshold slightly.
        """
        if len(self._recent_similarities) < 10:
            return

        avg_sim = sum(self._recent_similarities) / len(self._recent_similarities)
        # Move threshold toward the average similarity of recent hits
        # but stay within [min_threshold, max_threshold]
        target = avg_sim - 0.02  # Slightly below average hit similarity
        delta = (target - self._threshold) * self._lr
        self._threshold = max(
            self._min_threshold,
            min(self._max_threshold, self._threshold + delta),
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _is_expired(self, entry: CacheEntry) -> bool:
        return time.time() > entry.created_at + entry.ttl_seconds

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return
        oldest_key = min(
            self._entries,
            key=lambda k: self._entries[k].last_accessed,
        )
        del self._entries[oldest_key]

    def invalidate(self, prompt: str) -> bool:
        """Remove a specific prompt from the cache."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        with self._lock:
            return self._entries.pop(prompt_hash, None) is not None

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(total, 1),
                "similarity_threshold": self._threshold,
            }


class CacheAwareRouter:
    """Routes requests to nodes that already have the relevant KV cache.

    Tracks which nodes have cached prefixes and routes new requests
    to maximize cache reuse.
    """

    def __init__(self):
        self._node_caches: dict[str, set[str]] = {}  # node_id -> set of prefix hashes
        self._lock = threading.Lock()

    def register_cache_entry(self, node_id: str, prefix_hash: str) -> None:
        """Register that a node has a cached prefix."""
        with self._lock:
            if node_id not in self._node_caches:
                self._node_caches[node_id] = set()
            self._node_caches[node_id].add(prefix_hash)

    def find_best_node(
        self,
        prefix_hash: str,
        available_nodes: list[str],
    ) -> str | None:
        """Find the node that already has the cached prefix.

        Args:
            prefix_hash: Hash of the prompt prefix.
            available_nodes: List of available node IDs.

        Returns:
            Node ID with the cached prefix, or None.
        """
        with self._lock:
            for node_id in available_nodes:
                caches = self._node_caches.get(node_id, set())
                if prefix_hash in caches:
                    return node_id
            return None

    def get_cache_stats(self) -> dict:
        """Return cache distribution across nodes."""
        with self._lock:
            return {
                node_id: len(caches)
                for node_id, caches in self._node_caches.items()
            }
