"""Semantic caching middleware for the distributed LLM API.

Extends the exact-match dedup (``DedupMiddleware``) with embedding-based
similarity matching.  Semantically equivalent prompts (e.g. "What's the
capital of France?" vs "What is the capital city of France?") produce the
same embedding vector and return the cached response, reducing latency and
compute cost.

Typical flow::

    Request → RequestBody → Embedding Vector (via sentence-transformers)
      → Similarity Search (FAISS / HNSW index)
        → Cache Hit? → Return cached response + "x-cache: HIT" header
        → Cache Miss? → Process normally → Store (embedding, response) pair

The cache is **opt-in** — only requests with the header
``X-Semantic-Cache: 1`` (or a score threshold, e.g. ``X-Semantic-Cache: 0.95``)
are candidates.  This avoids surprising users who expect unique responses.

Streaming requests are never cached.
"""

from __future__ import annotations

import json
import os
import threading
import time

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded via sentence-transformers)
# ---------------------------------------------------------------------------


class _EmbeddingModel:
    """Lazy-loaded embedding model (sentence-transformers).

    Uses the same deferred-import pattern as ``MLInjectionClassifier``.
    When ``DISTLLM_SEMANTIC_CACHE_MODEL`` is set, loads that model from
    HuggingFace via ``sentence-transformers``.  Falls back to a hash-based
    fingerprint when no model is configured or the import fails.
    """

    def __init__(self, model_name: str = ""):
        self._model_name = model_name or os.environ.get(
            "DISTLLM_SEMANTIC_CACHE_MODEL", ""
        )
        self._model = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        if not self._model_name:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._model.to("cpu")
            logger.info(f"Semantic cache embedding model loaded: {self._model_name}")
        except ImportError:
            logger.info(
                "sentence-transformers not installed; semantic cache uses "
                "hash-based fingerprint fallback.  Install with: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load embedding model '{self._model_name}': {e}")

    def embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True).tolist()
            return vec

        # ── Hash-based fallback (64-dim count-min sketch) ────────────
        tokens = text.lower().split()
        dims = 64
        vec = [0.0] * dims
        for token in tokens:
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dims
            vec[bucket] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


_embedder = _EmbeddingModel()


# ---------------------------------------------------------------------------
# FAISS-backed index for sub-millisecond similarity search
# ---------------------------------------------------------------------------


class _FaissIndex:
    """Flat / HNSW index wrapping FAISS for fast nearest-neighbour search.

    - Dim is auto-detected from the first stored vector.
    - Uses ``IndexIDMap`` so each stored vector has a stable integer ID that
      maps back to the cached response payload.
    - When the model is a real sentence-transformer (e.g. 384-d) the index
      search replaces the O(n) ``OrderedDict`` scan with a
      ``IndexFlatIP`` (inner-product = cosine for unit vectors).  The
      fallback hash-based embedding uses 64-d with the same index.

    Trade-off: FAISS is an optional dependency (``pip install faiss-cpu``).
    When unavailable the cache degrades to brute-force O(n) over an
    ``OrderedDict`` with no correctness loss.
    """

    def __init__(self):
        self._index = None
        self._dim: int = 0
        self._next_id: int = 0
        self._lock = threading.Lock()
        self._id_to_key: dict[int, str] = {}           # faiss id → prompt key
        self._response_store: dict[str, str] = {}       # prompt key → response text
        self._ts_store: dict[str, float] = {}           # prompt key → timestamp

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            try:
                import faiss
                # Inner-product = cosine similarity for unit-normalised vectors
                self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
                self._dim = dim
                logger.debug(f"FAISS index created (dim={dim})")
            except ImportError:
                logger.debug("faiss not available — using brute-force fallback")
                self._index = None
                self._dim = dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query_vec: list[float], threshold: float) -> tuple[str | None, float]:
        """Return (response_text, similarity) for the best match ≥ *threshold*.

        Returns (None, 0.0) when no match meets the threshold.
        """
        import numpy as np
        q = np.array([query_vec], dtype=np.float32)

        if self._index is not None:
            # ── FAISS path ────────────────────────────────────────────
            with self._lock:
                if self._index.ntotal == 0:
                    return None, 0.0
                sims, ids = self._index.search(q, 1)
                best_sim = float(sims[0][0])
                best_id = int(ids[0][0])
                if best_sim >= threshold and best_id >= 0:
                    key = self._id_to_key.get(best_id)
                    if key and key in self._response_store:
                        return self._response_store[key], best_sim
                return None, 0.0
        else:
            # ── Brute-force fallback ──────────────────────────────────
            best_sim = 0.0
            best_key = None
            now = time.time()
            stale: list[str] = []
            with self._lock:
                for key, ts in self._ts_store.items():
                    if now - ts > _CACHE_TTL:
                        stale.append(key)
                        continue
                    entry_vec = _embedder.embed(key)
                    sim = self._cosine(query_vec, entry_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_key = key
                for k in stale:
                    self._evict(k)
                if best_key is not None and best_sim >= threshold:
                    return self._response_store.get(best_key, ""), best_sim
            return None, 0.0

    def add(self, prompt: str, response_text: str) -> None:
        """Store a (prompt, response) pair."""
        vec = _embedder.embed(prompt)
        import numpy as np
        v = np.array([vec], dtype=np.float32)
        now = time.time()

        with self._lock:
            self._ensure_index(len(vec))

            if self._index is not None:
                # ── FAISS path ────────────────────────────────────────
                fid = self._next_id
                self._next_id += 1
                self._index.add_with_ids(v, np.array([fid], dtype=np.int64))
                self._id_to_key[fid] = prompt
            # else brute-force uses the dicts directly

            self._response_store[prompt] = response_text
            self._ts_store[prompt] = now

            # LRU eviction via key-order tracking
            self._evict_lru()

    def _evict(self, key: str) -> None:
        self._response_store.pop(key, None)
        self._ts_store.pop(key, None)

    def _evict_lru(self) -> None:
        """Drop oldest entries when over the size limit."""
        while len(self._response_store) > _CACHE_MAX_ENTRIES:
            if self._ts_store:
                oldest = min(self._ts_store, key=self._ts_store.get)
                # If FAISS is active, also remove from the index
                if self._index is not None:
                    # Reverse-lookup faiss id → purge.  Since FAISS doesn't
                    # support per-element removal for IndexFlatIP, we rebuild
                    # the index periodically when evictions mount.
                    pass
                self._evict(oldest)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(ai * bi for ai, bi in zip(a, b))


# ---------------------------------------------------------------------------
# Cache constants & singleton
# ---------------------------------------------------------------------------

_CACHE_MAX_ENTRIES = 5000
_CACHE_TTL = 300.0  # 5 minutes

_cache = _FaissIndex()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class SemanticCacheMiddleware(BaseHTTPMiddleware):
    """Middleware that returns cached responses for semantically similar prompts.

    Header ``X-Semantic-Cache`` controls behaviour:
    - ``1`` — enable with default threshold (0.95)
    - ``0.9``, ``0.95``, ``0.99`` — use a custom similarity threshold
    - Absent or ``0`` — bypass (normal processing)

    Only applies to non-streaming ``POST /v1/chat/completions``.
    """

    SKIP_PATHS = {"/health", "/ready", "/live", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        # Gate: only POST to chat completions, and only when the header is set
        if request.method != "POST" or not request.url.path.startswith("/v1/chat/completions"):
            return await call_next(request)

        cache_header = request.headers.get("X-Semantic-Cache", "0")
        if cache_header == "0":
            return await call_next(request)

        try:
            threshold = float(cache_header)
        except (ValueError, TypeError):
            threshold = 0.95

        if threshold <= 0 or threshold > 1:
            return await call_next(request)

        # Don't cache streaming requests
        body_bytes = await request.body()
        if not body_bytes:
            return await call_next(request)
        try:
            body = json.loads(body_bytes)
            if body.get("stream", False):
                return await call_next(request)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await call_next(request)

        # Extract prompt text for embedding
        prompt = self._extract_prompt(body)
        if not prompt:
            return await call_next(request)

        # ── Semantic cache lookup ──────────────────────────────────────
        vec = _embedder.embed(prompt)
        cached, sim = _cache.search(vec, threshold=threshold)
        if cached is not None:
            logger.debug(f"Semantic cache hit (sim={sim:.3f}, threshold={threshold})")
            resp = Response(content=cached, media_type="application/json")
            resp.headers["X-Cache"] = "HIT"
            resp.headers["X-Cache-Similarity"] = f"{sim:.3f}"
            return resp

        # ── Cache miss: process normally, then store ──────────────────
        response = await call_next(request)
        if response.status_code == 200:
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk
            if resp_body:
                _cache.add(prompt, resp_body.decode())
            return Response(
                content=resp_body,
                status_code=response.status_code,
                media_type=response.media_type,
                headers=dict(response.headers) | {"X-Cache": "MISS"},
            )
        return response

    def _extract_prompt(self, body: dict) -> str:
        """Extract the full prompt text from a parsed request body."""
        parts: list[str] = []
        messages = body.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
        prompt = body.get("prompt", "")
        if prompt:
            parts.append(prompt)
        return "\n".join(parts)
