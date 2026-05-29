"""Routing extensions — LRU cache, semantic router, speculative pre-warming.

Provides advanced routing capabilities that plug into the core ModelRouter:

- **LRUModelCache**: Cost-aware LRU eviction for multi-model GPU memory.
- **SemanticRouter**: Embedding-similarity-based routing.
- **SpeculativePreWarmer**: Pre-warms models based on routing patterns.
- **RoutingMetrics**: Prometheus counters and histograms for routing decisions.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 4.3  Cost-Aware LRU Model Cache
# ---------------------------------------------------------------------------

@dataclass
class _ModelSlot:
    """Tracks a loaded model in the LRU cache."""
    name: str
    memory_gb: float
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0


class LRUModelCache:
    """Cost-aware LRU eviction for models sharing GPU memory.

    Integrates with ModelRouter by providing ``available_models()`` that
    returns only currently-loaded models, and ``on_route()`` to update
    LRU timestamps.

    Args:
        total_memory_gb: Total GPU memory available for models.
        load_model_fn: Callback ``(model_name) -> memory_gb`` to load a model.
        unload_model_fn: Callback ``(model_name) -> None`` to unload a model.
        cost_fn: Optional ``(model_name) -> float`` returning per-query cost.

    Usage::

        cache = LRUModelCache(24.0, load_fn, unload_fn)
        cache.register("codellama", memory_gb=8.0)
        cache.register("mathgpt", memory_gb=6.0)
        cache.register("llama3", memory_gb=12.0)

        # Before routing
        available = cache.available_models()

        # After routing — ensure model is loaded
        cache.ensure_loaded("codellama")
    """

    def __init__(
        self,
        total_memory_gb: float,
        load_model_fn: Callable[[str], float] | None = None,
        unload_model_fn: Callable[[str], None] | None = None,
        cost_fn: Callable[[str], float] | None = None,
    ) -> None:
        self._total_memory = total_memory_gb
        self._load_fn = load_model_fn
        self._unload_fn = unload_model_fn
        self._cost_fn = cost_fn
        self._slots: dict[str, _ModelSlot] = {}
        self._loaded: dict[str, _ModelSlot] = {}
        self._lock = threading.Lock()
        self._evictions = 0

    def register(self, name: str, memory_gb: float, cost_per_query: float = 0.0) -> None:
        """Register a model with its memory footprint."""
        with self._lock:
            self._slots[name] = _ModelSlot(name=name, memory_gb=memory_gb)

    def available_models(self) -> list[str]:
        """Return names of currently loaded models."""
        with self._lock:
            return list(self._loaded.keys())

    def total_memory_used(self) -> float:
        with self._lock:
            return self._total_memory_used_unlocked()

    def _total_memory_used_unlocked(self) -> float:
        return sum(s.memory_gb for s in self._loaded.values())

    def available_memory(self) -> float:
        if self._total_memory <= 0:
            return float("inf")
        return max(0.0, self._total_memory - self._total_memory_used_unlocked())

    def _available_memory_unlocked(self) -> float:
        if self._total_memory <= 0:
            return float("inf")
        return max(0.0, self._total_memory - self._total_memory_used_unlocked())

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def on_route(self, name: str) -> None:
        """Update LRU timestamp when a model is used for inference."""
        with self._lock:
            if name in self._loaded:
                self._loaded[name].last_used_at = time.time()
                self._loaded[name].use_count += 1

    def ensure_loaded(self, name: str) -> bool:
        """Ensure a model is loaded, evicting LRU models if needed.

        Returns True if the model is loaded (or was already loaded).
        """
        if name in self._loaded:
            return True

        slot = self._slots.get(name)
        if slot is None:
            logger.warning(f"LRUModelCache: model '{name}' not registered")
            return False

        with self._lock:
            # Evict until we have enough memory
            needed = slot.memory_gb
            while self._available_memory_unlocked() < needed and self._loaded:
                evict_name = self._pick_eviction_candidate()
                if evict_name is None:
                    break
                self._evict(evict_name)

            if self._available_memory_unlocked() < needed:
                logger.warning(
                    f"LRUModelCache: cannot fit '{name}' "
                    f"({needed:.1f}GB needed, "
                    f"{self._available_memory_unlocked():.1f}GB free)"
                )
                return False

            # Load the model
            if self._load_fn:
                try:
                    actual_gb = self._load_fn(name)
                    slot.memory_gb = actual_gb or slot.memory_gb
                except (RuntimeError, OSError, ValueError) as e:
                    logger.error(f"LRUModelCache: failed to load '{name}': {e}")
                    return False

            slot.last_used_at = time.time()
            slot.use_count += 1
            self._loaded[name] = slot
            logger.info(
                f"LRUModelCache: loaded '{name}' "
                f"({slot.memory_gb:.1f}GB, {len(self._loaded)} models active)"
            )
            return True

    def _pick_eviction_candidate(self) -> str | None:
        """Pick the LRU model to evict, preferring cheaper models."""
        if not self._loaded:
            return None

        best_name = None
        best_score = float("inf")

        for name, slot in self._loaded.items():
            # Score: lower is better to evict
            # Prefer: older usage, lower cost, smaller memory
            age = time.time() - slot.last_used_at
            cost = self._cost_fn(name) if self._cost_fn else 0.0
            score = age * 10.0 - cost * 100.0 - slot.memory_gb

            if score < best_score:
                best_score = score
                best_name = name

        return best_name

    def _evict(self, name: str) -> None:
        """Evict a model from GPU memory."""
        slot = self._loaded.pop(name, None)
        if slot is None:
            return
        self._evictions += 1
        if self._unload_fn:
            try:
                self._unload_fn(name)
            except (RuntimeError, OSError) as e:
                logger.warning(f"LRUModelCache: error unloading '{name}': {e}")
        logger.info(f"LRUModelCache: evicted '{name}' ({slot.memory_gb:.1f}GB freed)")

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_memory_gb": self._total_memory,
                "used_memory_gb": round(self._total_memory_used_unlocked(), 2),
                "available_memory_gb": round(self._available_memory_unlocked(), 2),
                "loaded_models": list(self._loaded.keys()),
                "registered_models": list(self._slots.keys()),
                "total_evictions": self._evictions,
            }


# ---------------------------------------------------------------------------
# 4.4  Semantic Router (Embedding Similarity)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_embed(text: str, dim: int = 384) -> list[float]:
    """Fallback embedding using feature hashing (no ML dependency).

    For production, replace with sentence-transformers or similar.
    """
    import hashlib
    vec = [0.0] * dim
    text_lower = text.lower()
    for n in (2, 3, 4):
        for i in range(len(text_lower) - n + 1):
            ngram = text_lower[i:i + n].encode("utf-8")
            h = int(hashlib.sha256(ngram).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass
class SemanticRoute:
    """A semantic routing rule with example embeddings."""
    name: str
    target_model: str
    examples: list[str]
    embeddings: list[list[float]] = field(default_factory=list)
    threshold: float = 0.6


class SemanticRouter:
    """Embedding-similarity-based router.

    Pre-computes embeddings for example queries per route and selects
    the model whose examples are most similar to the input.

    Args:
        embed_fn: Optional embedding function.  Defaults to feature hashing.
        similarity_threshold: Default threshold for a match.

    Usage::

        sr = SemanticRouter()
        sr.add_route("code", "codellama", [
            "write a function to sort a list",
            "implement a binary search tree",
            "debug this Python script",
        ])
        sr.add_route("math", "mathgpt", [
            "solve the integral of x^2",
            "prove that sqrt(2) is irrational",
        ])

        model = sr.route("create a linked list implementation")
        # model == "codellama"
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._embed_fn = embed_fn or _simple_embed
        self._default_threshold = similarity_threshold
        self._routes: dict[str, SemanticRoute] = {}

    def add_route(
        self,
        name: str,
        target_model: str,
        examples: list[str],
        threshold: float | None = None,
    ) -> None:
        """Add a semantic route with example queries.

        Args:
            name: Route name.
            target_model: Model to route to.
            examples: Example queries that belong to this route.
            threshold: Similarity threshold (overrides default).
        """
        embeddings = [self._embed_fn(ex) for ex in examples]
        self._routes[name] = SemanticRoute(
            name=name,
            target_model=target_model,
            examples=examples,
            embeddings=embeddings,
            threshold=threshold or self._default_threshold,
        )

    def route(
        self,
        text: str,
        available_models: list[str] | None = None,
    ) -> tuple[str | None, str, float]:
        """Route text to the best-matching semantic route.

        Args:
            text: Query text.
            available_models: Filter for currently-loaded models.

        Returns:
            Tuple of (model_name, route_name, similarity).
            model_name is None if no route exceeds its threshold.
        """
        query_emb = self._embed_fn(text)
        best_model = None
        best_route = ""
        best_sim = 0.0

        for route_name, sr in self._routes.items():
            if available_models and sr.target_model not in available_models:
                continue

            # Average similarity across all examples
            sims = [_cosine_similarity(query_emb, emb) for emb in sr.embeddings]
            avg_sim = sum(sims) / len(sims) if sims else 0.0

            if avg_sim >= sr.threshold and avg_sim > best_sim:
                best_sim = avg_sim
                best_model = sr.target_model
                best_route = route_name

        return best_model, best_route, best_sim

    @property
    def routes(self) -> dict[str, SemanticRoute]:
        return dict(self._routes)


# ---------------------------------------------------------------------------
# 4.5  Speculative Model Pre-Warmer
# ---------------------------------------------------------------------------

class SpeculativePreWarmer:
    """Pre-warms models based on routing patterns.

    Tracks recent routing decisions and predicts the next likely model
    to pre-load it before the request arrives.

    Args:
        warm_fn: Callback ``(model_name) -> None`` to pre-warm a model.
        history_size: Number of recent routing decisions to track.
        min_confidence: Minimum transition probability to trigger pre-warm.

    Usage::

        warmer = SpeculativePreWarmer(warm_fn=cache.ensure_loaded)
        warmer.record("codellama")  # code query routed
        warmer.predict_and_warm()    # pre-warms mathgpt if pattern learned
    """

    def __init__(
        self,
        warm_fn: Callable[[str], None] | None = None,
        history_size: int = 100,
        min_confidence: float = 0.3,
    ) -> None:
        self._warm_fn = warm_fn
        self._history_size = history_size
        self._min_confidence = min_confidence
        self._history: collections.deque[str] = collections.deque(maxlen=history_size)
        self._transitions: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._pre_warms = 0

    def record(self, model: str) -> None:
        """Record a routing decision."""
        with self._lock:
            if self._history:
                prev = self._history[-1]
                if prev != model:
                    if prev not in self._transitions:
                        self._transitions[prev] = {}
                    self._transitions[prev][model] = (
                        self._transitions[prev].get(model, 0) + 1
                    )
            self._history.append(model)

    def predict_next(self, current_model: str) -> str | None:
        """Predict the next likely model based on transition probabilities."""
        with self._lock:
            transitions = self._transitions.get(current_model, {})
            if not transitions:
                return None

            total = sum(transitions.values())
            if total == 0:
                return None

            best_model = None
            best_prob = 0.0

            for model, count in transitions.items():
                prob = count / total
                if prob >= self._min_confidence and prob > best_prob:
                    best_prob = prob
                    best_model = model

            return best_model

    def predict_and_warm(self) -> str | None:
        """Predict next model and pre-warm it if a warm_fn is set."""
        if not self._history:
            return None

        current = self._history[-1]
        predicted = self.predict_next(current)

        if predicted and self._warm_fn:
            self._warm_fn(predicted)
            self._pre_warms += 1
            logger.debug(
                f"SpeculativePreWarmer: pre-warming '{predicted}' "
                f"(after '{current}')"
            )

        return predicted

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "history_size": len(self._history),
                "transitions": dict(self._transitions),
                "pre_warms": self._pre_warms,
            }


# ---------------------------------------------------------------------------
# 5.2  Routing Prometheus Metrics
# ---------------------------------------------------------------------------

class RoutingMetrics:
    """Prometheus-compatible metrics for routing decisions.

    Lazily imports prometheus_client so the module works without it.

    Usage::

        metrics = RoutingMetrics()
        metrics.record_decision("code-route", "codellama", "code", 0.85)
        metrics.record_fallback()
        metrics.record_latency(1.2)
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {
            "decisions": 0,
            "fallbacks": 0,
            "unavailable": 0,
            "bypasses": 0,
        }
        self._latencies: list[float] = []
        self._by_model: dict[str, int] = {}
        self._by_rule: dict[str, int] = {}
        self._lock = threading.Lock()
        self._prometheus = self._init_prometheus()

    def _init_prometheus(self) -> bool:
        """Try to initialize Prometheus counters."""
        try:
            from prometheus_client import Counter, Histogram, CollectorRegistry

            # Use a dedicated registry to avoid duplicate metric errors
            registry = CollectorRegistry()

            self._decision_counter = Counter(
                "distllm_router_decisions_total",
                "Routing decisions",
                ["rule_name", "target_model", "workload_type", "confidence_bucket"],
                registry=registry,
            )
            self._latency_histogram = Histogram(
                "distllm_router_latency_ms",
                "Routing latency in milliseconds",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
                registry=registry,
            )
            self._fallback_counter = Counter(
                "distllm_router_fallbacks_total",
                "Default fallback count",
                registry=registry,
            )
            self._unavailable_counter = Counter(
                "distllm_router_model_unavailable_total",
                "Routed model not loaded",
                ["target_model"],
                registry=registry,
            )
            self._bypass_counter = Counter(
                "distllm_router_bypasses_total",
                "Router bypass via header",
                registry=registry,
            )
            return True
        except ImportError:
            self._decision_counter = None
            self._latency_histogram = None
            self._fallback_counter = None
            self._unavailable_counter = None
            self._bypass_counter = None
            return False

    def record_decision(
        self,
        rule_name: str,
        target_model: str,
        workload_type: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Record a routing decision."""
        with self._lock:
            self._counters["decisions"] += 1
            self._by_model[target_model] = self._by_model.get(target_model, 0) + 1
            self._by_rule[rule_name] = self._by_rule.get(rule_name, 0) + 1

        if self._prometheus:
            bucket = "high" if confidence >= 0.7 else "medium" if confidence >= 0.4 else "low"
            self._decision_counter.labels(
                rule_name=rule_name,
                target_model=target_model,
                workload_type=workload_type,
                confidence_bucket=bucket,
            ).inc()

    def record_fallback(self) -> None:
        """Record a default fallback."""
        with self._lock:
            self._counters["fallbacks"] += 1
        if self._fallback_counter:
            self._fallback_counter.inc()

    def record_unavailable(self, target_model: str) -> None:
        """Record a routing to an unavailable model."""
        with self._lock:
            self._counters["unavailable"] += 1
        if self._unavailable_counter:
            self._unavailable_counter.labels(target_model=target_model).inc()

    def record_bypass(self) -> None:
        """Record a router bypass via header."""
        with self._lock:
            self._counters["bypasses"] += 1
        if self._bypass_counter:
            self._bypass_counter.inc()

    def record_latency(self, latency_ms: float) -> None:
        """Record routing latency."""
        with self._lock:
            self._latencies.append(latency_ms)
            if len(self._latencies) > 10000:
                self._latencies = self._latencies[-5000:]
        if self._latency_histogram:
            self._latency_histogram.observe(latency_ms)

    @property
    def stats(self) -> dict:
        """Return routing metrics summary."""
        with self._lock:
            lats = self._latencies[-1000:] if self._latencies else []
            return {
                "total_decisions": self._counters["decisions"],
                "total_fallbacks": self._counters["fallbacks"],
                "total_unavailable": self._counters["unavailable"],
                "total_bypasses": self._counters["bypasses"],
                "by_model": dict(self._by_model),
                "by_rule": dict(self._by_rule),
                "latency_p50_ms": (
                    round(sorted(lats)[len(lats) // 2], 2) if lats else 0.0
                ),
                "latency_p99_ms": (
                    round(sorted(lats)[int(len(lats) * 0.99)], 2) if lats else 0.0
                ),
                "prometheus_enabled": self._prometheus,
            }
