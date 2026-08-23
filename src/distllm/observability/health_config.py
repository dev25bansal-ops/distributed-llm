"""Health configuration and subsystem health checks for distributed LLM.

Provides a layered health monitoring system that integrates with the
existing observability and plugin infrastructure:

- **DeepHealthProbe** — Generates a short completion, verifies output,
  measures time-to-first-token (TTFT) and tokens-per-second throughput.
- **KVCacheHealthCheck** — Monitors KV cache block pool utilization
  and defragmentation status.
- **HealthCache** — TTL-based cache that prevents health check storms
  by caching results for a configurable duration (default 5 s).
- **CascadingHealthCheck** — Checks the coordinator → nodes → backends
  dependency hierarchy; a coordinator failure cascades into degraded
  status for all downstream nodes.
- **ClusterHealthAggregator** — Combines all health signals into a
  composite score (0.0 – 1.0) with a textual status.
- **HealthConfigurator** — Top-level orchestrator that ties all
  subsystems together with start/stop lifecycle and enables the
  ``HealthPlugin`` by default.

Typical usage::

    configurator = HealthConfigurator(
        generate_fn=backend.generate,
        block_pool=paged_attention.pool,
        coordinator_check_fn=coordinator.is_healthy,
        backend_registry=backend_registry,
    )
    await configurator.start()
    # ...
    status = configurator.get_aggregated_status()
    await configurator.stop()
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# HealthResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthResult:
    """Result of a single health probe.

    Attributes:
        success: Whether the probe succeeded.
        latency_ms: Total round-trip latency of the probe in milliseconds.
        ttft_ms: Time-to-first-token in milliseconds (0 if not applicable).
        tokens_per_sec: Generated tokens per second (0 if not applicable).
        timestamp: Unix timestamp of when the probe completed.
        error: Optional error message when *success* is ``False``.
    """

    success: bool
    latency_ms: float
    ttft_ms: float
    tokens_per_sec: float
    timestamp: float
    error: str | None = None


# ---------------------------------------------------------------------------
# DeepHealthProbe
# ---------------------------------------------------------------------------

class DeepHealthProbe:
    """Generates a short completion to measure model health and performance.

    Periodically runs a lightweight inference probe (default every 60 s)
    that generates a short completion, verifies the output is non-empty,
    and records time-to-first-token (TTFT) and tokens-per-second throughput.

    The probe uses a caller-supplied *generate_fn* so it works with any
    backend adapter that supports a text-generation interface.

    Probe lifecycle is managed externally (typically by
    :class:`HealthConfigurator`).
    """

    DEFAULT_PROMPT = "Hello, write a one-sentence greeting."
    DEFAULT_MAX_TOKENS = 50
    DEFAULT_INTERVAL_S = 60.0

    def __init__(
        self,
        generate_fn: Callable[..., Any],
        *,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        """Initialise the deep health probe.

        Args:
            generate_fn: A callable that accepts
                ``(prompt, max_tokens, temperature, ...)`` and returns
                ``(text: str, num_tokens: int, generation_time_ms: float)``.
            prompt: The probe prompt (default: a short greeting request).
            max_tokens: Maximum tokens to generate for the probe.
            interval_s: Seconds between automatic probe runs
                (default 60.0).
        """
        self._generate_fn = generate_fn
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._interval_s = interval_s

        self._lock = threading.Lock()
        self._last_result: HealthResult | None = None
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- Public API ---------------------------------------------------------

    @property
    def last_result(self) -> HealthResult | None:
        """Most recent probe result, or ``None`` if no probe has run yet."""
        with self._lock:
            return self._last_result

    @property
    def interval_s(self) -> float:
        """Probe interval in seconds."""
        return self._interval_s

    def probe(self) -> HealthResult:
        """Execute a single deep probe synchronously and return the result.

        This method can be called directly at any time.  When the probe is
        running in background mode, calling this method manually will not
        interfere with the background loop.

        Returns:
            A :class:`HealthResult` containing timing and throughput metrics.
        """
        start_ts = time.time()
        start_mono = time.monotonic()
        ttft_ms = 0.0
        tokens_per_sec = 0.0
        success = False
        error: str | None = None

        try:
            # Call the generation function; assume signature similar to
            # BackendAdapter.generate(prompt, max_tokens, temperature, ...)
            result = self._generate_fn(
                prompt=self._prompt,
                max_tokens=self._max_tokens,
                temperature=0.0,  # deterministic for consistency
            )

            elapsed_ms = (time.monotonic() - start_mono) * 1000.0

            # Unpack result — supports both tuple and dict return styles
            if isinstance(result, tuple):
                text, num_tokens, gen_time_ms = result
                ttft_ms = gen_time_ms if isinstance(gen_time_ms, (int, float)) else 0.0
            elif isinstance(result, dict):
                text = result.get("text", "")
                num_tokens = result.get("num_tokens", 0)
                ttft_ms = result.get("ttft_ms", 0.0)
            else:
                text = str(result)
                num_tokens = max(1, len(text) // 4)

            # Verify output is non-empty
            if text and text.strip():
                generation_time_s = max(elapsed_ms / 1000.0, 0.001)
                tokens_per_sec = num_tokens / generation_time_s
                success = True
            else:
                error = "probe returned empty output"

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_mono) * 1000.0
            error = str(e)

        result_obj = HealthResult(
            success=success,
            latency_ms=round(elapsed_ms, 2),
            ttft_ms=round(ttft_ms, 2),
            tokens_per_sec=round(tokens_per_sec, 2),
            timestamp=time.time(),
            error=error,
        )

        with self._lock:
            self._last_result = result_obj

        if success:
            logger.debug(
                f"DeepHealthProbe: success, latency={result_obj.latency_ms:.1f}ms, "
                f"ttft={result_obj.ttft_ms:.1f}ms, "
                f"tokens/sec={result_obj.tokens_per_sec:.1f}"
            )
        else:
            logger.warning(
                f"DeepHealthProbe: failed ({error}), "
                f"latency={result_obj.latency_ms:.1f}ms"
            )

        return result_obj

    def start(self) -> None:
        """Start the background probe loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="deep-health-probe",
        )
        self._thread.start()
        logger.info(f"DeepHealthProbe: started (interval={self._interval_s}s)")

    def stop(self) -> None:
        """Stop the background probe loop."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("DeepHealthProbe: stopped")

    # -- Internal loop ------------------------------------------------------

    def _loop(self) -> None:
        """Background loop that runs probes at the configured interval."""
        consecutive_errors = 0
        max_backoff_s = 120.0

        while not self._stop_event.is_set():
            try:
                # Run the probe
                result = self.probe()
                if result.success:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

                # Wait for next interval (with backoff on failure)
                wait = self._interval_s
                if consecutive_errors > 0:
                    wait = min(wait * (1.5 ** min(consecutive_errors, 5)), max_backoff_s)

                if self._stop_event.wait(wait):
                    break

            except Exception as exc:
                consecutive_errors += 1
                logger.error(f"DeepHealthProbe: loop error ({exc})")
                backoff = min(consecutive_errors * 2.0, max_backoff_s)
                if self._stop_event.wait(backoff):
                    break


# ---------------------------------------------------------------------------
# KVCacheHealthCheck
# ---------------------------------------------------------------------------

class KVCacheHealthCheck:
    """Monitors KV cache block pool utilisation and defragmentation status.

    Uses a caller-supplied *pool_stats_fn* (or duck-typed block pool with
    a ``stats()`` method) to obtain current usage data.  Alerts (via
    logger.warning) when utilisation exceeds 80%.

    The ``check()`` method returns a dictionary with the following keys:

    * ``total_blocks`` — total blocks in the pool
    * ``used_blocks`` — currently allocated blocks
    * ``free_blocks`` — currently free blocks
    * ``utilization`` — fraction of blocks in use (0.0 – 1.0)
    * ``high_utilization`` — ``True`` when util > 80 %
    * ``defrag_status`` — defragmentation status dict (if available)
    * ``healthy`` — ``True`` when util <= 80 % (or when no pool is available)
    """

    HIGH_UTILIZATION_THRESHOLD = 0.80

    def __init__(
        self,
        pool_stats_fn: Callable[[], dict[str, Any]] | None = None,
        defrag_status_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Initialise the KV cache health check.

        Args:
            pool_stats_fn: Callable returning a dict with ``total_blocks``,
                ``used_blocks``, ``free_blocks`` keys (e.g. ``BlockPool.stats``).
                When ``None``, the probe attempts to import and call
                ``PagedAttentionManager.pool.stats()`` at runtime.
            defrag_status_fn: Optional callable returning defragmentation
                status (e.g. ``{enabled, running, progress, ...}``).
                When ``None``, defrag reporting is skipped.
        """
        self._pool_stats_fn = pool_stats_fn
        self._defrag_status_fn = defrag_status_fn

    # -- Public API ---------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Run a KV cache health check and return a detailed status dict.

        Returns:
            Status dictionary with utilisation metrics and health indicator.
        """
        pool_stats = self._resolve_pool_stats()
        if pool_stats is None:
            return {
                "healthy": True,
                "total_blocks": 0,
                "used_blocks": 0,
                "free_blocks": 0,
                "utilization": 0.0,
                "high_utilization": False,
                "defrag_status": None,
                "note": "no_kv_cache_pool_available",
            }

        total = pool_stats.get("total_blocks", 0)
        used = pool_stats.get("used_blocks", 0)
        free = pool_stats.get("free_blocks", 0)
        utilization = used / max(total, 1)

        defrag: dict[str, Any] | None = None
        if self._defrag_status_fn is not None:
            try:
                defrag = self._defrag_status_fn()
            except Exception as exc:
                defrag = {"error": str(exc)}

        high_util = utilization > self.HIGH_UTILIZATION_THRESHOLD

        status: dict[str, Any] = {
            "total_blocks": total,
            "used_blocks": used,
            "free_blocks": free,
            "utilization": round(utilization, 4),
            "high_utilization": high_util,
            "defrag_status": defrag,
            "healthy": not high_util,
        }

        if high_util:
            logger.warning(
                f"KVCacheHealthCheck: high utilisation "
                f"({utilization:.1%} > {self.HIGH_UTILIZATION_THRESHOLD:.0%}), "
                f"used={used}, total={total}"
            )

        return status

    # -- Internal helpers ---------------------------------------------------

    def _resolve_pool_stats(self) -> dict[str, Any] | None:
        """Return pool stats from the configured callable or runtime lookup."""
        if self._pool_stats_fn is not None:
            try:
                return self._pool_stats_fn()
            except Exception as exc:
                logger.debug(f"KVCacheHealthCheck: pool_stats_fn failed ({exc})")
                return None

        # Runtime duck-type lookup: try common import paths
        for mod_path, attr_path in [
            ("distllm.dist.block_pool", "get_default_pool"),
            ("distllm.dist.attention", "PagedAttentionManager"),
        ]:
            try:
                mod = __import__(mod_path, fromlist=[attr_path])
                obj = getattr(mod, attr_path, None)
                if obj is None:
                    continue
                # If it's a class, try to get a global/singleton instance
                if isinstance(obj, type):
                    continue  # skip classes, look for instances
                if hasattr(obj, "stats"):
                    return obj.stats()
            except Exception:
                continue

        # Try resolving from application state
        try:
            from distllm.api.api_state import g as _global_state

            mgr = getattr(_global_state, "paged_attention_manager", None)
            if mgr is not None and hasattr(mgr, "pool") and hasattr(mgr.pool, "stats"):
                return mgr.pool.stats()
        except Exception:
            pass

        return None


# ---------------------------------------------------------------------------
# HealthCache
# ---------------------------------------------------------------------------

class HealthCache:
    """TTL-based cache for health check results.

    Prevents health check storms by caching results for a configurable
    duration (default 5 seconds).  Thread-safe.

    Usage::

        cache = HealthCache(default_ttl=5)
        cache.set("deep_probe", result)
        cached = cache.get("deep_probe")  # returns result or None
        cache.invalidate("deep_probe")
    """

    def __init__(self, default_ttl: float = 5.0) -> None:
        """Initialise the health cache.

        Args:
            default_ttl: Default time-to-live in seconds for cached entries
                (default 5.0).
        """
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._store: dict[str, _CacheEntry] = {}

    # -- Public API ---------------------------------------------------------

    @property
    def default_ttl(self) -> float:
        """Default TTL for cached entries in seconds."""
        return self._default_ttl

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key.

        Args:
            key: Cache key.

        Returns:
            The cached value, or ``None`` if the key does not exist or
            has expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (overrides the default).
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + max(effective_ttl, 0.0),
            )

    def invalidate(self, key: str) -> None:
        """Remove a single entry from the cache.

        Args:
            key: Cache key to invalidate.
        """
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries evicted.
        """
        now = time.monotonic()
        with self._lock:
            expired = [k for k, v in self._store.items() if now > v.expires_at]
            for k in expired:
                del self._store[k]
            return len(expired)

    @property
    def size(self) -> int:
        """Number of entries currently in the cache (including expired)."""
        with self._lock:
            return len(self._store)


@dataclass
class _CacheEntry:
    """Internal cache entry holding a value and its expiration timestamp."""
    value: Any
    expires_at: float


# ---------------------------------------------------------------------------
# CascadingHealthCheck
# ---------------------------------------------------------------------------

class CascadingHealthCheck:
    """Hierarchical health check for the coordinator → nodes → backends chain.

    Checks dependencies in order:

    1. **Coordinator** — if the coordinator is unreachable, all nodes and
       backends are reported as ``degraded`` regardless of their actual state.
    2. **Nodes** — each registered worker node is probed for health.
    3. **Backends** — each registered backend adapter is checked.

    The ``get_status()`` method returns a hierarchical dictionary that
    reflects this dependency chain.
    """

    def __init__(
        self,
        coordinator_check_fn: Callable[[], bool] | None = None,
        node_check_fn: Callable[[str], dict[str, Any]] | None = None,
        backend_check_fn: Callable[[str], dict[str, Any]] | None = None,
        *,
        node_ids: list[str] | None = None,
        backend_ids: list[str] | None = None,
    ) -> None:
        """Initialise the cascading health check.

        Args:
            coordinator_check_fn: Callable returning ``True`` when the
                coordinator is healthy.
            node_check_fn: Callable ``(node_id) -> status_dict`` for
                individual node probes.
            backend_check_fn: Callable ``(backend_name) -> status_dict``
                for individual backend probes.
            node_ids: List of node IDs to probe.
            backend_ids: List of backend adapter names to probe.
        """
        self._coordinator_check_fn = coordinator_check_fn
        self._node_check_fn = node_check_fn
        self._backend_check_fn = backend_check_fn
        self._node_ids = node_ids or []
        self._backend_ids = backend_ids or []

    # -- Public API ---------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Run the full cascading health check.

        Returns:
            A hierarchical dictionary::

                {
                    "coordinator": {"healthy": bool, ...},
                    "nodes": {"healthy": bool, "degraded": bool,
                              "items": {node_id: status_dict}},
                    "backends": {"healthy": bool, "degraded": bool,
                                 "items": {backend_id: status_dict}},
                    "degraded": bool,   # True when any upstream is down
                }
        """
        coordinator_down = False
        coordinator_status: dict[str, Any] = {"healthy": False, "error": "no_check_fn"}

        if self._coordinator_check_fn is not None:
            try:
                coord_healthy = self._coordinator_check_fn()
                coordinator_status = {
                    "healthy": coord_healthy,
                }
                if not coord_healthy:
                    coordinator_down = True
            except Exception as exc:
                coordinator_status = {"healthy": False, "error": str(exc)}
                coordinator_down = True
        else:
            coordinator_status = {"healthy": True, "note": "no_check_fn_available"}

        # Nodes — when coordinator is down, mark all nodes as degraded
        nodes_status = self._check_nodes(degraded=coordinator_down)

        # Backends — when coordinator or any node is down, degrade backends
        any_node_down = coordinator_down or not nodes_status.get("healthy", True)
        backends_status = self._check_backends(degraded=any_node_down)

        overall_degraded = coordinator_down or not nodes_status.get("healthy", True)

        return {
            "coordinator": coordinator_status,
            "nodes": nodes_status,
            "backends": backends_status,
            "degraded": overall_degraded,
            "timestamp": time.time(),
        }

    # -- Internal helpers ---------------------------------------------------

    def _check_nodes(self, degraded: bool = False) -> dict[str, Any]:
        """Check all registered nodes."""
        if degraded or not self._node_check_fn:
            return {
                "healthy": not degraded,
                "degraded": degraded,
                "items": {},
            }

        items: dict[str, dict[str, Any]] = {}
        all_healthy = True

        for node_id in self._node_ids:
            try:
                status = self._node_check_fn(node_id)
                items[node_id] = status
                if not status.get("healthy", False):
                    all_healthy = False
            except Exception as exc:
                items[node_id] = {"healthy": False, "error": str(exc)}
                all_healthy = False

        return {
            "healthy": all_healthy,
            "degraded": not all_healthy,
            "items": items,
        }

    def _check_backends(self, degraded: bool = False) -> dict[str, Any]:
        """Check all registered backends."""
        if degraded or not self._backend_check_fn:
            return {
                "healthy": not degraded,
                "degraded": degraded,
                "items": {},
            }

        items: dict[str, dict[str, Any]] = {}
        all_healthy = True

        for backend_id in self._backend_ids:
            try:
                status = self._backend_check_fn(backend_id)
                items[backend_id] = status
                if not status.get("healthy", False):
                    all_healthy = False
            except Exception as exc:
                items[backend_id] = {"healthy": False, "error": str(exc)}
                all_healthy = False

        return {
            "healthy": all_healthy,
            "degraded": not all_healthy,
            "items": items,
        }


# ---------------------------------------------------------------------------
# ClusterHealthAggregator
# ---------------------------------------------------------------------------

class ClusterHealthAggregator:
    """Combines multiple health signals into a composite cluster score.

    The aggregate score is a weighted combination of all registered health
    signals, normalised to the range 0.0 – 1.0:

    * **> 0.9** → ``healthy``
    * **> 0.7** → ``degraded``
    * **≤ 0.7** → ``critical``

    Weights can be customised per signal to reflect the relative
    importance of each component.
    """

    # Default thresholds
    HEALTHY_THRESHOLD = 0.9
    DEGRADED_THRESHOLD = 0.7

    def __init__(self) -> None:
        """Initialise the aggregator with an empty signal registry."""
        self._lock = threading.Lock()
        self._signals: dict[str, _SignalEntry] = {}

    # -- Public API ---------------------------------------------------------

    def register_signal(
        self,
        name: str,
        probe_fn: Callable[[], Any],
        weight: float = 1.0,
        *,
        is_healthy_fn: Callable[[Any], bool] | None = None,
        score_fn: Callable[[Any], float] | None = None,
    ) -> None:
        """Register a health signal for aggregation.

        Args:
            name: Unique signal name.
            probe_fn: Zero-argument callable that returns a health status
                value (dict, bool, HealthResult, etc.).
            weight: Relative weight of this signal in the aggregate score.
            is_healthy_fn: Optional callable that returns ``True`` when
                *probe_fn*'s return value indicates health.  When not
                provided, the default heuristic extracts ``.success``,
                ``.get("healthy")``, or the bare boolean value.
            score_fn: Optional callable that converts the probe result to
                a float in [0.0, 1.0].  When not provided, the default
                heuristic maps ``is_healthy_fn`` results to 1.0 or 0.0.

        Raises:
            ValueError: If *name* is already registered.
        """
        if not name:
            raise ValueError("signal name must not be empty")

        with self._lock:
            if name in self._signals:
                raise ValueError(f"signal '{name}' is already registered")
            self._signals[name] = _SignalEntry(
                name=name,
                probe_fn=probe_fn,
                weight=weight,
                is_healthy_fn=is_healthy_fn,
                score_fn=score_fn,
            )

    def unregister_signal(self, name: str) -> None:
        """Remove a previously registered signal.

        Args:
            name: Signal name to remove.

        Raises:
            KeyError: If *name* is not registered.
        """
        with self._lock:
            del self._signals[name]

    def aggregate(self) -> dict[str, Any]:
        """Run all registered signals and compute the composite score.

        Returns:
            Dictionary with keys:

            * ``score`` (float, 0.0 – 1.0)
            * ``status`` (str: ``"healthy"``, ``"degraded"``, or ``"critical"``)
            * ``signals`` (dict of signal_name → signal result)
            * ``timestamp`` (float)
        """
        with self._lock:
            signal_snapshots = list(self._signals.items())

        signal_results: dict[str, Any] = {}
        total_weight = 0.0
        weighted_score = 0.0

        for name, entry in signal_snapshots:
            try:
                raw = entry.probe_fn()
                signal_results[name] = raw

                if entry.score_fn is not None:
                    score = entry.score_fn(raw)
                else:
                    score = 1.0 if self._default_is_healthy(raw) else 0.0

                weighted_score += entry.weight * score
                total_weight += entry.weight

            except Exception as exc:
                logger.error(
                    f"ClusterHealthAggregator: signal '{name}' failed ({exc})"
                )
                signal_results[name] = {"error": str(exc)}
                weighted_score += entry.weight * 0.0
                total_weight += entry.weight

        aggregate_score = weighted_score / max(total_weight, 1.0)
        status = self._classify(aggregate_score)

        return {
            "score": round(aggregate_score, 4),
            "status": status,
            "signals": signal_results,
            "timestamp": time.time(),
        }

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _classify(score: float) -> str:
        """Map a numeric score to a status string."""
        if score > ClusterHealthAggregator.HEALTHY_THRESHOLD:
            return "healthy"
        if score > ClusterHealthAggregator.DEGRADED_THRESHOLD:
            return "degraded"
        return "critical"

    @staticmethod
    def _default_is_healthy(value: Any) -> bool:
        """Heuristic to determine health from a raw probe result.

        Supports:
        - ``HealthResult`` / dataclass with ``.success`` attribute
        - ``dict`` with a ``"healthy"`` or ``"success"`` key
        - ``bool`` directly
        """
        if hasattr(value, "success"):
            return bool(value.success)
        if isinstance(value, dict):
            return bool(value.get("healthy", value.get("success", False)))
        if isinstance(value, bool):
            return value
        return True


@dataclass
class _SignalEntry:
    """Internal registry entry for a single health signal."""
    name: str
    probe_fn: Callable[[], Any]
    weight: float
    is_healthy_fn: Callable[[Any], bool] | None
    score_fn: Callable[[Any], float] | None


# ---------------------------------------------------------------------------
# HealthConfigurator
# ---------------------------------------------------------------------------

class HealthConfigurator:
    """Top-level orchestrator for all health monitoring subsystems.

    Combines :class:`DeepHealthProbe`, :class:`KVCacheHealthCheck`,
    :class:`HealthCache`, :class:`CascadingHealthCheck`, and
    :class:`ClusterHealthAggregator` into a unified interface with
    start/stop lifecycle management.

    By default, enables the ``HealthPlugin`` if it is available in the
    plugin system.

    Typical usage::

        configurator = HealthConfigurator(generate_fn=backend.generate)
        await configurator.start()
        # ... application runs ...
        aggregated = configurator.get_aggregated_status()
        await configurator.stop()
    """

    def __init__(
        self,
        generate_fn: Callable[..., Any] | None = None,
        pool_stats_fn: Callable[[], dict[str, Any]] | None = None,
        defrag_status_fn: Callable[[], dict[str, Any]] | None = None,
        coordinator_check_fn: Callable[[], bool] | None = None,
        node_check_fn: Callable[[str], dict[str, Any]] | None = None,
        backend_check_fn: Callable[[str], dict[str, Any]] | None = None,
        *,
        node_ids: list[str] | None = None,
        backend_ids: list[str] | None = None,
        probe_interval_s: float = 60.0,
        cache_ttl_s: float = 5.0,
        enable_health_plugin: bool = True,
        plugin_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the health configurator.

        Args:
            generate_fn: Callable passed to :class:`DeepHealthProbe`.
            pool_stats_fn: Callable passed to :class:`KVCacheHealthCheck`.
            defrag_status_fn: Callable for defrag status.
            coordinator_check_fn: Callable for coordinator health.
            node_check_fn: Callable for per-node health.
            backend_check_fn: Callable for per-backend health.
            node_ids: List of node IDs for cascading checks.
            backend_ids: List of backend names for cascading checks.
            probe_interval_s: Interval for deep health probes (default 60).
            cache_ttl_s: TTL for health cache (default 5).
            enable_health_plugin: Whether to enable the ``HealthPlugin``
                (default ``True``).
            plugin_context: Context dict passed to the plugin lifecycle
                methods (``on_init``, ``on_start``, ``on_stop``).
        """
        # Subsystems
        self._deep_probe = DeepHealthProbe(
            generate_fn=generate_fn,
            interval_s=probe_interval_s,
        ) if generate_fn else None

        self._kv_cache_check = KVCacheHealthCheck(
            pool_stats_fn=pool_stats_fn,
            defrag_status_fn=defrag_status_fn,
        )

        self._cache = HealthCache(default_ttl=cache_ttl_s)

        self._cascading = CascadingHealthCheck(
            coordinator_check_fn=coordinator_check_fn,
            node_check_fn=node_check_fn,
            backend_check_fn=backend_check_fn,
            node_ids=node_ids,
            backend_ids=backend_ids,
        )

        self._aggregator = ClusterHealthAggregator()
        self._register_default_signals()

        # Plugin integration
        self._enable_health_plugin = enable_health_plugin
        self._plugin_context = plugin_context or {}
        self._health_plugin_instance: Any = None

        # Lifecycle
        self._lock = threading.Lock()
        self._running = False

    # -- Properties ---------------------------------------------------------

    @property
    def deep_probe(self) -> DeepHealthProbe | None:
        """Underlying :class:`DeepHealthProbe` instance."""
        return self._deep_probe

    @property
    def kv_cache_check(self) -> KVCacheHealthCheck:
        """Underlying :class:`KVCacheHealthCheck` instance."""
        return self._kv_cache_check

    @property
    def cache(self) -> HealthCache:
        """Underlying :class:`HealthCache` instance."""
        return self._cache

    @property
    def cascading(self) -> CascadingHealthCheck:
        """Underlying :class:`CascadingHealthCheck` instance."""
        return self._cascading

    @property
    def aggregator(self) -> ClusterHealthAggregator:
        """Underlying :class:`ClusterHealthAggregator` instance."""
        return self._aggregator

    @property
    def is_running(self) -> bool:
        """Whether the configurator is currently running."""
        return self._running

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start all health monitoring subsystems.

        Starts the deep health probe background loop and, if enabled,
        activates the ``HealthPlugin``.
        """
        if self._running:
            logger.warning("HealthConfigurator: already running")
            return

        self._running = True

        # Start deep health probe background loop
        if self._deep_probe is not None:
            self._deep_probe.start()

        # Enable the HealthPlugin
        if self._enable_health_plugin:
            self._enable_plugin()

        logger.info("HealthConfigurator: started")

    async def stop(self) -> None:
        """Stop all health monitoring subsystems."""
        if not self._running:
            return

        self._running = False

        # Stop deep health probe
        if self._deep_probe is not None:
            self._deep_probe.stop()

        # Disable the HealthPlugin
        if self._enable_health_plugin and self._health_plugin_instance is not None:
            self._disable_plugin()

        # Clear cache
        self._cache.clear()

        logger.info("HealthConfigurator: stopped")

    # -- Health queries -----------------------------------------------------

    def get_deep_probe_result(self) -> HealthResult | None:
        """Return the most recent deep probe result (cached if available).

        Runs a fresh probe only when no cached result exists.
        """
        cached = self._cache.get("deep_probe")
        if cached is not None:
            return cached

        if self._deep_probe is not None:
            result = self._deep_probe.probe()
            self._cache.set("deep_probe", result)
            return result

        return None

    def get_kv_cache_status(self) -> dict[str, Any]:
        """Return the latest KV cache health status (cached)."""
        cached = self._cache.get("kv_cache")
        if cached is not None:
            return cached

        status = self._kv_cache_check.check()
        self._cache.set("kv_cache", status)
        return status

    def get_cascading_status(self) -> dict[str, Any]:
        """Return the hierarchical dependency health status (cached)."""
        cached = self._cache.get("cascading")
        if cached is not None:
            return cached

        status = self._cascading.get_status()
        self._cache.set("cascading", status)
        return status

    def get_aggregated_status(self) -> dict[str, Any]:
        """Return the composite cluster health score and status.

        Shorthand for ``self.aggregator.aggregate()`` with caching.
        """
        cached = self._cache.get("aggregated")
        if cached is not None:
            return cached

        result = self._aggregator.aggregate()
        self._cache.set("aggregated", result)
        return result

    def get_full_report(self) -> dict[str, Any]:
        """Return a comprehensive health report combining all subsystems.

        This method bypasses the cache to ensure a fresh, complete picture.
        """
        deep_result = self._deep_probe.probe() if self._deep_probe else None
        kv_status = self._kv_cache_check.check()
        cascade_status = self._cascading.get_status()
        aggregated = self._aggregator.aggregate()

        return {
            "deep_probe": {
                "result": dataclasses.asdict(deep_result) if deep_result else None,
            },
            "kv_cache": kv_status,
            "cascading": cascade_status,
            "aggregated": aggregated,
            "configurator_running": self._running,
            "timestamp": time.time(),
        }

    # -- Plugin integration -------------------------------------------------

    def _register_default_signals(self) -> None:
        """Register default health signals with the aggregator."""
        # Deep probe signal
        if self._deep_probe is not None:
            def deep_probe_signal() -> HealthResult | None:
                return self._deep_probe.last_result

            self._aggregator.register_signal(
                name="deep_probe",
                probe_fn=deep_probe_signal,
                weight=2.0,
            )

        # KV cache signal
        def kv_cache_signal() -> dict[str, Any]:
            return self._kv_cache_check.check()

        self._aggregator.register_signal(
            name="kv_cache",
            probe_fn=kv_cache_signal,
            weight=1.0,
        )

        # Cascading signal
        def cascading_signal() -> dict[str, Any]:
            return self._cascading.get_status()

        self._aggregator.register_signal(
            name="cascading",
            probe_fn=cascading_signal,
            weight=3.0,  # highest weight — coordinator/nodes/backends
            score_fn=self._cascading_score_fn,
        )

    @staticmethod
    def _cascading_score_fn(status: dict[str, Any]) -> float:
        """Convert cascading status to a score in [0.0, 1.0]."""
        if status.get("degraded", False):
            return 0.3

        coordinator = status.get("coordinator", {})
        nodes = status.get("nodes", {})
        backends = status.get("backends", {})

        scores = [
            1.0 if coordinator.get("healthy", False) else 0.0,
            1.0 if nodes.get("healthy", True) else 0.0,
            1.0 if backends.get("healthy", True) else 0.0,
        ]
        return sum(scores) / len(scores)

    def _enable_plugin(self) -> None:
        """Find and enable the ``HealthPlugin`` via the plugin system."""
        try:
            from distllm.api.server import state as _server_state

            ps = getattr(_server_state, "plugin_system", None)
            if ps is None:
                logger.debug(
                    "HealthConfigurator: plugin_system not available, "
                    "skipping HealthPlugin enable"
                )
                return

            # Check if already registered
            existing = ps.get_plugin("health")
            if existing is not None and existing.instance is not None:
                self._health_plugin_instance = existing.instance
                logger.debug(
                    "HealthConfigurator: HealthPlugin already registered"
                )
                return

            # Dynamically register and initialise
            from distllm.plugins.health_plugin import HealthPlugin

            plugin_inst = HealthPlugin()
            plugin_inst.on_init(self._plugin_context)
            ps.register(plugin_inst)
            plugin_inst.on_start(self._plugin_context)
            self._health_plugin_instance = plugin_inst
            logger.info("HealthConfigurator: HealthPlugin enabled")

        except Exception as exc:
            logger.debug(
                f"HealthConfigurator: could not enable HealthPlugin ({exc})"
            )

    def _disable_plugin(self) -> None:
        """Disable the ``HealthPlugin`` if it was enabled by us."""
        if self._health_plugin_instance is None:
            return

        try:
            self._health_plugin_instance.on_stop(self._plugin_context)
        except Exception as exc:
            logger.warning(
                f"HealthConfigurator: error stopping HealthPlugin ({exc})"
            )
        self._health_plugin_instance = None
        logger.info("HealthConfigurator: HealthPlugin disabled")


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_default_health_configurator(
    generate_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> HealthConfigurator:
    """Create a :class:`HealthConfigurator` pre-configured with sensible defaults.

    This factory attempts to auto-discover the coordinator, nodes,
    backends, and KV cache pool from the application's global state.
    All arguments override the auto-discovered values.

    Args:
        generate_fn: Callable for deep health probes.
        **kwargs: Additional arguments forwarded to ``HealthConfigurator``.

    Returns:
        A fully configured :class:`HealthConfigurator` instance.
    """
    # Attempt auto-discovery of components from global state
    coordinator_check_fn: Callable[[], bool] | None = None
    node_ids: list[str] | None = None
    backend_ids: list[str] | None = None
    pool_stats_fn: Callable[[], dict[str, Any]] | None = None
    node_check_fn: Callable[[str], dict[str, Any]] | None = None
    backend_check_fn: Callable[[str], dict[str, Any]] | None = None

    try:
        from distllm.api.api_state import g as _global_state

        # Coordinator
        coord = getattr(_global_state, "coordinator", None)
        if coord is not None and hasattr(coord, "is_healthy"):
            coordinator_check_fn = coord.is_healthy

        # Nodes
        pipeline = getattr(_global_state, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "nodes"):
            node_ids = list(pipeline.nodes.keys())
            if hasattr(pipeline, "check_node_health"):
                node_check_fn = pipeline.check_node_health

        # KV cache pool
        paged_mgr = getattr(_global_state, "paged_attention_manager", None)
        if paged_mgr is not None and hasattr(paged_mgr, "pool"):
            pool_stats_fn = paged_mgr.pool.stats

    except Exception:
        pass

    # Try to discover backends from the registry
    try:
        from distllm.backends.registry import BackendRegistry

        registry = BackendRegistry()
        all_plugins = registry.list_plugins() if hasattr(registry, "list_plugins") else {}
        backend_ids = list(all_plugins.keys())

        if hasattr(registry, "check_health"):
            backend_check_fn = registry.check_health
    except Exception:
        pass

    # Build the configurator with discovered values, overridden by kwargs
    config_kwargs: dict[str, Any] = {
        "generate_fn": generate_fn,
        "coordinator_check_fn": coordinator_check_fn,
        "pool_stats_fn": pool_stats_fn,
        "node_ids": node_ids,
        "backend_ids": backend_ids,
        "node_check_fn": node_check_fn,
        "backend_check_fn": backend_check_fn,
    }
    config_kwargs.update(kwargs)

    return HealthConfigurator(**config_kwargs)


__all__ = [
    "CascadingHealthCheck",
    "ClusterHealthAggregator",
    "DeepHealthProbe",
    "HealthCache",
    "HealthConfigurator",
    "HealthResult",
    "KVCacheHealthCheck",
    "create_default_health_configurator",
]
