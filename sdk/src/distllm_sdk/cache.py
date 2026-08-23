"""Client-side response caching for the DistLLM SDK.

Provides TTL-based caches for embeddings, model listings, and
prompt/response pairs.  The ``PromptCache`` supports both exact-match
and semantic-similarity caching for chat completions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# TTL cache (internal)
# ---------------------------------------------------------------------------

class _TTLCache:
    """Simple TTL-backed cache with max-size eviction."""

    def __init__(self, ttl: float, max_entries: int):
        self._ttl = ttl
        self._max_entries = max_entries
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._data[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._data) >= self._max_entries:
            oldest = min(self._data, key=lambda k: self._data[k][0])
            del self._data[oldest]
        self._data[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# General cache config
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    """Configuration for client-side response caching.

    Attributes:
        embedding_ttl: TTL in seconds for cached embedding results.
        model_list_ttl: TTL in seconds for cached model listings.
        max_embedding_entries: Max cached embedding results.
        enabled: Set to False to disable caching entirely.
    """
    embedding_ttl: float = 3600.0
    model_list_ttl: float = 300.0
    max_embedding_entries: int = 10000
    enabled: bool = True


# ---------------------------------------------------------------------------
# Prompt response cache with semantic similarity
# ---------------------------------------------------------------------------

@dataclass
class PromptCacheConfig:
    """Configuration for prompt/response caching.

    Attributes:
        ttl: TTL in seconds for cached responses (default 3600).
        max_entries: Maximum number of cached responses (default 10000).
        similarity_threshold: Minimum Jaccard similarity (0-1) for a
            semantic cache hit.  0.85 means two prompts must share
            85% of their character n-grams to be considered similar.
            Set to ``1.0`` for exact-match only.  Default 0.92.
        ngram_n: Character n-gram size for fingerprinting (default 3).
    """
    ttl: float = 3600.0
    max_entries: int = 10000
    similarity_threshold: float = 0.85
    ngram_n: int = 3


def _ngram_fingerprint(text: str, n: int = 3) -> set[str]:
    """Build a character n-gram set for similarity comparison.

    Example with n=3: ``"hello"`` → ``{"hel", "ell", "llo"}``
    """
    text = text.lower().strip()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets: ``|intersection| / |union|``."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class PromptCache:
    """Cache for chat completion responses with exact and semantic matching.

    Usage::

        from distllm_sdk.cache import PromptCache, PromptCacheConfig

        cache = PromptCache(PromptCacheConfig(ttl=3600, similarity_threshold=0.92))
        client = DistLLMClient(base_url="...", cache=cache)

    Each cache entry stores a (model, messages) → response mapping.
    On lookup, the cache first checks for an exact match; if none found,
    it scans for semantically similar prompts using character n-gram
    fingerprinting.
    """

    def __init__(self, config: PromptCacheConfig | None = None):
        self._config = config or PromptCacheConfig()
        self._exact: dict[str, tuple[float, Any]] = {}
        self._fingerprints: dict[str, set[str]] = {}

    # -- Key helpers ---------------------------------------------------------

    @staticmethod
    def _make_key(model: str, messages: list[dict[str, str]] | tuple) -> str:
        """Create an exact-match key from model + messages."""
        if isinstance(messages, list):
            msg_part = "|".join(f"{m.get('role','')}:{m.get('content','')}" for m in messages)
        else:
            msg_part = str(messages)
        return f"{model}::{msg_part}"

    def _make_fingerprint(self, model: str, messages: list[dict[str, str]] | tuple) -> str:
        """Create a similarity-match key (model + condensed text fingerprint)."""
        text = " ".join(m.get("content", "") for m in (messages if isinstance(messages, list) else []))
        fp = _ngram_fingerprint(text, self._config.ngram_n)
        self._fingerprints[model] = fp
        return model

    # -- Public API ----------------------------------------------------------

    def lookup(self, model: str, messages: list[dict[str, str]]) -> Any | None:
        """Look up a cached response.  Checks exact match first, then semantic."""
        # 1. Exact match
        key = self._make_key(model, messages)
        entry = self._exact.get(key)
        if entry is not None:
            expires_at, value = entry
            if time.monotonic() <= expires_at:
                return value
            del self._exact[key]

        # 2. Semantic match (scan all entries for the same model)
        if self._config.similarity_threshold >= 1.0:
            return None

        query_fp = _ngram_fingerprint(
            " ".join(m.get("content", "") for m in messages),
            self._config.ngram_n,
        )
        if not query_fp:
            return None

        best_score = 0.0
        best_value = None
        now = time.monotonic()
        expired_keys = []

        for ek, (expires_at, value) in self._exact.items():
            if now > expires_at:
                expired_keys.append(ek)
                continue
            if not ek.startswith(f"{model}::"):
                continue
            stored_fp = self._fingerprints.get(ek)
            if stored_fp is None:
                continue
            score = _jaccard_similarity(query_fp, stored_fp)
            if score > best_score:
                best_score = score
                best_value = value

        # Clean expired entries
        for ek in expired_keys:
            self._exact.pop(ek, None)
            self._fingerprints.pop(ek, None)

        if best_score >= self._config.similarity_threshold:
            return best_value
        return None

    def store(self, model: str, messages: list[dict[str, str]], response: Any) -> None:
        """Store a response in the cache."""
        key = self._make_key(model, messages)
        # Evict oldest if full
        if len(self._exact) >= self._config.max_entries:
            oldest = min(self._exact, key=lambda k: self._exact[k][0])
            del self._exact[oldest]
            self._fingerprints.pop(oldest, None)
        self._exact[key] = (time.monotonic() + self._config.ttl, response)
        # Pre-compute fingerprint for future semantic lookups
        text = " ".join(m.get("content", "") for m in messages)
        self._fingerprints[key] = _ngram_fingerprint(text, self._config.ngram_n)

    def clear(self) -> None:
        """Clear all cached responses."""
        self._exact.clear()
        self._fingerprints.clear()

    @property
    def size(self) -> int:
        return len(self._exact)

    def invalidate(self, model: str | None = None, messages: list[dict[str, str]] | None = None) -> None:
        """Invalidate a specific entry, or all entries for *model*."""
        if model and messages:
            key = self._make_key(model, messages)
            self._exact.pop(key, None)
            self._fingerprints.pop(key, None)
        elif model:
            for ek in list(self._exact.keys()):
                if ek.startswith(f"{model}::"):
                    del self._exact[ek]
                    self._fingerprints.pop(ek, None)
