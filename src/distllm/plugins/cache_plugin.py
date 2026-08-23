"""Cache plugin — semantic deduplication and exact-match response caching.

Provides a ``CachePlugin`` that intercepts inference requests to serve cached
responses for identical or semantically similar prompts.  Integrates with the
existing ``SemanticCache`` class for embedding-based similarity and exposes
hit/miss counters that ``MetricsPlugin`` can scrape.

Cache backends
--------------
* **In-memory LRU** (default) — zero-dependency, thread-safe, bounded by
  ``DISTLLM_PLUGIN_CACHE_MAX_ENTRIES``.
* **Redis** (optional) — enabled by setting ``DISTLLM_PLUGIN_CACHE_REDIS_URL``.
  Shares cache across multiple coordinator processes.

Configuration (environment variables)
--------------------------------------
``DISTLLM_PLUGIN_CACHE_ENABLED``
    Set to ``1`` to activate (default: ``0``).
``DISTLLM_PLUGIN_CACHE_TTL``
    Entry time-to-live in seconds (default: ``3600``).
``DISTLLM_PLUGIN_CACHE_MAX_ENTRIES``
    Maximum in-memory entries (default: ``10000``).
``DISTLLM_PLUGIN_CACHE_SIMILARITY_THRESHOLD``
    Cosine-similarity threshold for semantic match (default: ``0.92``).
``DISTLLM_PLUGIN_CACHE_SEMANTIC_ENABLED``
    Set to ``1`` to enable semantic (embedding) caching (default: ``0``).
``DISTLLM_PLUGIN_CACHE_REDIS_URL``
    Redis connection URL for distributed caching (e.g.
    ``redis://localhost:6379/0``).  When unset the plugin uses in-memory only.
``DISTLLM_PLUGIN_CACHE_REDIS_MAX_CONNECTIONS``
    Redis connection pool size (default: ``20``).
``DISTLLM_PLUGIN_CACHE_NORMALIZE``
    Set to ``1`` to normalize prompts before hashing — strips leading/trailing
    whitespace, collapses runs of whitespace, lowercases (default: ``1``).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

from loguru import logger

from distllm.core.plugin_system import PluginBase

# Optional dependencies — degrade gracefully when absent.
try:
    from distllm.core.semantic_cache import SemanticCache
except ImportError:  # pragma: no cover
    SemanticCache = None  # type: ignore[assignment,misc]

try:
    import redis as _redis_mod
except ImportError:
    _redis_mod = None  # type: ignore[assignment]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_prompt(text: str) -> str:
    """Collapse whitespace and lowercase for stable hashing."""
    return " ".join(text.lower().split())


def _build_cache_key(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    scope: str = "",
) -> str:
    """Build a deterministic cache key from prompt + generation parameters.

    The ``scope`` (tenant/key isolation) is part of the key, so a cached
    response for one tenant or key can never be served to another.  Requests
    with no identity share the (empty) scope and behave exactly as before.

    The key is a SHA-256 hex digest of a JSON encoding of the request tuple.
    JSON encoding keeps the mapping injective — a scope or prompt containing a
    delimiter character (e.g. ``|``) cannot collide with a different request.
    """
    raw = json.dumps(
        [scope, prompt, model, temperature, top_p],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _request_scope(request: dict[str, Any]) -> str:
    """Return the tenant/user isolation scope for a request.

    The scope is built from the AUTHENTICATED identity that the request
    dispatcher actually supplies — ``api_key_id`` (per-key) and the server-set
    ``tenant`` (SSO) — so a cached response for one tenant or key can never be
    served to another.  The historical ``tenant_id``/``user_id`` names are
    still honoured for direct callers.  The ``key:``/``tenant:`` prefixes keep
    namespaces distinct even when ids collide across dimensions.
    """
    api_key_id = request.get("api_key_id") or request.get("user_id") or ""
    tenant = request.get("tenant") or request.get("tenant_id") or ""
    if api_key_id and tenant:
        return f"{tenant}:{api_key_id}"
    if api_key_id:
        return f"key:{api_key_id}"
    if tenant:
        return f"tenant:{tenant}"
    return ""


# ── In-memory LRU backend ───────────────────────────────────────────────────

class _LRUCache:
    """Thread-safe bounded LRU cache backed by ``OrderedDict``.

    This is intentionally kept minimal — it stores serialised response
    strings keyed by cache-key hex digests and evicts the least-recently-used
    entry when at capacity.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        # key -> (response_json, created_at, ttl_seconds)
        self._data: OrderedDict[str, tuple[str, float, float]] = OrderedDict()

    def get(self, key: str) -> str | None:
        """Return cached response JSON or ``None`` on miss / expiry."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            response_json, created_at, ttl = entry
            if time.time() > created_at + ttl:
                # Expired — remove lazily.
                del self._data[key]
                return None
            # Promote to most-recently-used.
            self._data.move_to_end(key)
            return response_json

    def put(self, key: str, response_json: str, ttl: float) -> None:
        """Insert or replace an entry."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (response_json, time.time(), ttl)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._data)


# ── Redis backend ────────────────────────────────────────────────────────────

class _RedisCacheBackend:
    """Thin wrapper around Redis for cache storage.

    Uses ``SETEX`` for TTL-aware writes and plain ``GET`` for reads.
    All values are stored as JSON strings.
    """

    KEY_PREFIX = "distllm:cache:plugin:"

    def __init__(
        self,
        url: str,
        max_connections: int = 20,
        socket_timeout: float = 5.0,
    ) -> None:
        if _redis_mod is None:
            raise ImportError("redis package is required for Redis cache backend")
        self._pool = _redis_mod.ConnectionPool.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            retry_on_timeout=True,
            decode_responses=True,
        )
        self._client = _redis_mod.Redis(connection_pool=self._pool)
        # Verify connectivity.
        self._client.ping()
        logger.info(f"CachePlugin: connected to Redis at {url}")

    def get(self, key: str) -> str | None:
        try:
            return self._client.get(f"{self.KEY_PREFIX}{key}")
        except Exception as exc:
            logger.warning(f"CachePlugin Redis GET failed: {exc}")
            return None

    def put(self, key: str, response_json: str, ttl: float) -> None:
        try:
            self._client.setex(
                f"{self.KEY_PREFIX}{key}",
                int(ttl),
                response_json,
            )
        except Exception as exc:
            logger.warning(f"CachePlugin Redis SETEX failed: {exc}")

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(f"{self.KEY_PREFIX}{key}"))
        except Exception as exc:
            logger.warning(f"CachePlugin Redis DEL failed: {exc}")
            return False

    def clear(self) -> None:
        try:
            keys = self._client.keys(f"{self.KEY_PREFIX}*")
            if keys:
                self._client.delete(*keys)
        except Exception as exc:
            logger.warning(f"CachePlugin Redis CLEAR failed: {exc}")

    def disconnect(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ── CachePlugin ──────────────────────────────────────────────────────────────

class CachePlugin(PluginBase):
    """Response caching plugin with exact-match and semantic deduplication.

    On each ``on_request`` hook the plugin computes a cache key from the
    prompt text and generation parameters (model, temperature, top_p) and
    checks the cache.  If a hit is found, the hook returns
    ``{"_cached_response": <response>}`` which upstream middleware can use
    to short-circuit inference.

    On ``on_response`` the plugin stores the response so future identical
    (or semantically similar) requests are served from cache.

    Semantic matching is opt-in (``DISTLLM_PLUGIN_CACHE_SEMANTIC_ENABLED=1``)
    and requires the ``SemanticCache`` class from ``distllm.core`` to be
    importable.  When enabled, the plugin generates an embedding placeholder
    (the prompt text itself, since embedding generation is model-specific)
    and delegates similarity lookup to ``SemanticCache``.

    Metrics
    -------
    The plugin increments counters on the plugin-system context dict so that
    ``MetricsPlugin`` (or any observer) can read them:

    * ``cache_hits`` — total cache hits
    * ``cache_misses`` — total cache misses
    * ``cache_hit_rate`` — rolling hit rate (0.0–1.0)
    * ``cache_entries`` — current number of cached entries
    * ``cache_semantic_hits`` — hits that came from semantic (not exact) match
    """

    # ── PluginBase overrides ──────────────────────────────────────────────

    def name(self) -> str:
        return "cache"

    def version(self) -> str:
        return "1.0.0"

    def on_init(self, context: dict[str, Any]) -> None:
        """Read configuration from environment variables and set up backends."""
        self._enabled = os.environ.get("DISTLLM_PLUGIN_CACHE_ENABLED", "0") == "1"
        if not self._enabled:
            logger.info("CachePlugin: disabled (set DISTLLM_PLUGIN_CACHE_ENABLED=1 to enable)")
            return

        # TTL and capacity.
        self._ttl = self._env_float("DISTLLM_PLUGIN_CACHE_TTL", 3600.0)
        self._max_entries = self._env_int("DISTLLM_PLUGIN_CACHE_MAX_ENTRIES", 10_000)

        # Normalization.
        self._normalize = os.environ.get("DISTLLM_PLUGIN_CACHE_NORMALIZE", "1") == "1"

        # Semantic caching.
        self._semantic_enabled = (
            os.environ.get("DISTLLM_PLUGIN_CACHE_SEMANTIC_ENABLED", "0") == "1"
        )
        self._similarity_threshold = self._env_float(
            "DISTLLM_PLUGIN_CACHE_SIMILARITY_THRESHOLD", 0.92,
        )
        self._semantic_cache: SemanticCache | None = None
        if self._semantic_enabled and SemanticCache is not None:
            self._semantic_cache = SemanticCache(
                similarity_threshold=self._similarity_threshold,
                max_entries=self._max_entries,
                default_ttl=self._ttl,
            )
            logger.info(
                f"CachePlugin: semantic caching enabled "
                f"(threshold={self._similarity_threshold})"
            )
        elif self._semantic_enabled:
            logger.warning(
                "CachePlugin: semantic caching requested but SemanticCache "
                "is not available — falling back to exact match only"
            )
            self._semantic_enabled = False

        # Backend selection: Redis if URL is set, otherwise in-memory LRU.
        redis_url = os.environ.get("DISTLLM_PLUGIN_CACHE_REDIS_URL", "")
        self._redis_backend: _RedisCacheBackend | None = None
        self._lru_backend: _LRUCache | None = None

        if redis_url:
            try:
                max_conn = self._env_int(
                    "DISTLLM_PLUGIN_CACHE_REDIS_MAX_CONNECTIONS", 20,
                )
                self._redis_backend = _RedisCacheBackend(
                    url=redis_url,
                    max_connections=max_conn,
                )
                logger.info(f"CachePlugin: using Redis backend at {redis_url}")
            except Exception as exc:
                logger.warning(
                    f"CachePlugin: Redis init failed ({exc}), "
                    f"falling back to in-memory LRU"
                )
                self._redis_backend = None

        if self._redis_backend is None:
            self._lru_backend = _LRUCache(max_entries=self._max_entries)
            logger.info(
                f"CachePlugin: using in-memory LRU backend "
                f"(max_entries={self._max_entries})"
            )

        # Counters.
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._counter_lock = threading.Lock()

        logger.info(
            f"CachePlugin: ttl={self._ttl}s, max_entries={self._max_entries}, "
            f"normalize={self._normalize}"
        )

    def on_request(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Check the cache for a matching response.

        Returns a dict with ``_cached_response`` on hit so that upstream
        middleware can short-circuit inference, or ``None`` on miss.
        """
        if not self._enabled:
            return None

        prompt = context.get("prompt") or context.get("messages", "")
        if not prompt:
            return None

        # Normalise prompt text for stable hashing.
        lookup_prompt = _normalize_prompt(prompt) if self._normalize else prompt

        model = context.get("model", "")
        temperature = float(context.get("temperature", 1.0))
        top_p = float(context.get("top_p", 1.0))

        cache_key = _build_cache_key(
            lookup_prompt,
            model,
            temperature,
            top_p,
            scope=_request_scope(context),
        )

        # --- Exact-match lookup (primary path) ---
        cached = self._backend_get(cache_key)
        if cached is not None:
            self._record_hit(is_semantic=False)
            logger.debug(f"CachePlugin: exact hit for key={cache_key[:12]}...")
            return {"_cached_response": cached}

        # --- Semantic-match lookup (secondary path) ---
        if self._semantic_cache is not None:
            semantic_result = self._semantic_cache.lookup(
                lookup_prompt,
                embedding=self._prompt_to_embedding(lookup_prompt),
                scope=_request_scope(context),
            )
            if semantic_result is not None:
                self._record_hit(is_semantic=True)
                logger.debug(f"CachePlugin: semantic hit for key={cache_key[:12]}...")
                return {"_cached_response": semantic_result}

        # --- Cache miss ---
        with self._counter_lock:
            self._misses += 1
        return None

    def on_response(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Store the response in the cache for future lookups."""
        if not self._enabled:
            return

        prompt = request.get("prompt") or request.get("messages", "")
        if not prompt:
            return

        # Serialise the response payload.
        response_text = self._extract_response_text(response)
        if not response_text:
            return

        store_prompt = _normalize_prompt(prompt) if self._normalize else prompt
        model = request.get("model", "")
        temperature = float(request.get("temperature", 1.0))
        top_p = float(request.get("top_p", 1.0))

        cache_key = _build_cache_key(
            store_prompt,
            model,
            temperature,
            top_p,
            scope=_request_scope(request),
        )

        # Store in primary backend (LRU or Redis).
        self._backend_put(cache_key, response_text, self._ttl)

        # Store in semantic cache when enabled.
        if self._semantic_cache is not None:
            self._semantic_cache.store(
                prompt=store_prompt,
                response=response_text,
                embedding=self._prompt_to_embedding(store_prompt),
                ttl=self._ttl,
                scope=_request_scope(request),
            )

        logger.debug(f"CachePlugin: stored key={cache_key[:12]}...")

    def on_stop(self, context: dict[str, Any]) -> None:
        """Clean up resources on shutdown."""
        if self._redis_backend is not None:
            self._redis_backend.disconnect()
            logger.info("CachePlugin: Redis connection closed")

    # ── Metrics integration ───────────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        """Return current cache metrics.

        Designed to be called by ``MetricsPlugin`` or any external observer.
        """
        with self._counter_lock:
            total = self._hits + self._misses
            return {
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "cache_hit_rate": self._hits / total if total > 0 else 0.0,
                "cache_entries": self._entry_count(),
                "cache_semantic_hits": self._semantic_hits,
            }

    # ── Private helpers ───────────────────────────────────────────────────

    def _record_hit(self, *, is_semantic: bool) -> None:
        with self._counter_lock:
            self._hits += 1
            if is_semantic:
                self._semantic_hits += 1

    def _backend_get(self, key: str) -> str | None:
        """Read from whichever backend is active."""
        if self._redis_backend is not None:
            raw = self._redis_backend.get(key)
            if raw is not None:
                try:
                    return json.loads(raw).get("response")
                except (json.JSONDecodeError, AttributeError):
                    return raw
            return None
        if self._lru_backend is not None:
            raw = self._lru_backend.get(key)
            if raw is not None:
                try:
                    return json.loads(raw).get("response")
                except (json.JSONDecodeError, AttributeError):
                    return raw
        return None

    def _backend_put(self, key: str, response: str, ttl: float) -> None:
        """Write to whichever backend is active."""
        payload = json.dumps({"response": response}, default=str)
        if self._redis_backend is not None:
            self._redis_backend.put(key, payload, ttl)
        if self._lru_backend is not None:
            self._lru_backend.put(key, payload, ttl)

    def _entry_count(self) -> int:
        if self._lru_backend is not None:
            return self._lru_backend.size()
        # Redis entry count is expensive; return -1 to signal "unknown".
        return -1

    @staticmethod
    def _extract_response_text(response: dict[str, Any]) -> str:
        """Pull the textual content out of a response dict.

        Handles common response shapes: ``{"text": ...}``,
        ``{"choices": [{"text": ...}]}`` (OpenAI-style), and
        ``{"response": ...}``.
        """
        if "text" in response:
            return str(response["text"])
        if "response" in response:
            return str(response["response"])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("message", {}).get("content", ""))
        return ""

    @staticmethod
    def _prompt_to_embedding(prompt: str) -> list[float]:
        """Generate a simple character-frequency embedding for semantic matching.

        This is a lightweight placeholder that avoids requiring an external
        embedding model at the plugin level.  The ``SemanticCache`` class
        compares embeddings via cosine similarity, and character-frequency
        vectors give reasonable results for short, domain-specific prompts.

        For production use, override this method or pass pre-computed
        embeddings through the request context under the ``embedding`` key.
        """
        # 128-dim vector: frequency of each ASCII code-point modulo 128.
        vec = [0.0] * 128
        for ch in prompt.encode("utf-8", errors="replace"):
            vec[ch % 128] += 1.0
        # L2-normalise.
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # ── Env helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, str(default)))
        except (ValueError, TypeError):
            return default
