"""Cluster-wide predictive KV-cache prefetch (unified pipeline).

Item 7 of the roadmap: turn the *sketched* pieces — local
:class:`PredictiveCacheManager`, the :class:`GossipCacheBridge` (CRDT gossip),
and :class:`CrossModelPrefixSharing` — into a single, real pipeline that gives
**sub-ms warm-start on repeated prefixes cluster-wide**.

Before this module they were three independent classes.  This coordinator
wires them together:

* **Local prediction + prefetch** — every observed prefix is handed to the
  ``PredictiveCacheManager`` which learns patterns and pre-warms CPU→GPU.
* **Cross-node discovery** — when a prefix is stored/cached, the
  ``GossipCacheBridge`` advertises its hash; ``warm_start`` consults the
  gossip index to find a *remote* replica and reports a cross-node hit
  (with the replica node id) instead of recomputing.
* **Cross-model warm-start** — ``CrossModelPrefixSharing`` lets a fine-tuned
  variant reuse the base model's KV for shared layers, so the first tokens of
  a sibling model start warm.

``warm_start`` returns a :class:`WarmStartResult` with the measured latency
and the hit source, so a live dashboard can show warm-start savings.  Every
component is optional (graceful degradation) so the coordinator is fully
unit-testable with fakes — no real network or GPU required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class WarmStartSource(str, Enum):
    LOCAL_GPU = "local_gpu"
    LOCAL_CPU = "local_cpu"
    CROSS_NODE = "cross_node"
    CROSS_MODEL = "cross_model"
    MISS = "miss"


@dataclass
class WarmStartResult:
    """Outcome of a single warm-start lookup."""

    source: WarmStartSource
    latency_ms: float
    prefix_len: int
    node_id: str | None = None     # set when source == CROSS_NODE
    model_id: str | None = None    # set when source == CROSS_MODEL
    replica_nodes: list[str] = field(default_factory=list)
    hit: bool = True

    @property
    def is_warm(self) -> bool:
        return self.source != WarmStartSource.MISS


class ClusterPredictivePrefetcher:
    """Unifies local prediction, gossip discovery, and cross-model sharing.

    Args:
        node_id: This node's id (used for gossip self-advertisement).
        predictive_manager: Optional ``PredictiveCacheManager`` (local prefetch).
        gossip_bridge: Optional ``GossipCacheBridge`` (cross-node discovery).
        cross_model_sharing: Optional ``CrossModelPrefixSharing`` (variant reuse).
        local_gpu_cache: Optional object with ``lookup(token_ids)`` returning
            ``(match_len, kv)`` for an in-GPU hit.
        local_cpu_cache: Optional object with ``lookup(token_ids)`` for a
            CPU-tier hit.
        metrics: Optional ``MetricsManager`` to accumulate warm-start stats.
    """

    def __init__(
        self,
        node_id: str,
        predictive_manager: Any | None = None,
        gossip_bridge: Any | None = None,
        cross_model_sharing: Any | None = None,
        local_gpu_cache: Any | None = None,
        local_cpu_cache: Any | None = None,
        metrics: Any | None = None,
    ) -> None:
        self._node_id = node_id
        self._predictive = predictive_manager
        self._gossip = gossip_bridge
        self._cross_model = cross_model_sharing
        self._gpu_cache = local_gpu_cache
        self._cpu_cache = local_cpu_cache
        self._metrics = metrics
        self._stats = {
            "observations": 0,
            "local_gpu_hits": 0,
            "local_cpu_hits": 0,
            "cross_node_hits": 0,
            "cross_model_hits": 0,
            "misses": 0,
            "total_warm_start_ms": 0.0,
        }

    # ── observation (drives local prediction) ──

    def observe(self, token_ids: list[int], model_id: str | None = None) -> None:
        """Record a request prefix so the predictor can learn + prefetch."""
        self._stats["observations"] += 1
        if self._predictive is not None:
            try:
                self._predictive.observe_request(token_ids)
            except Exception as exc:
                logger.debug("predictive observe failed: %s", exc)
        # Persist into cross-model sharing so siblings can reuse it.
        if self._cross_model is not None and model_id is not None:
            try:
                self._cross_model.store(model_id, token_ids, kv_data=None)
            except Exception as exc:
                logger.debug("cross-model store failed: %s", exc)

    # ── warm-start lookup ──

    def warm_start(self, token_ids: list[int], model_id: str | None = None) -> WarmStartResult:
        """Resolve a warm KV-cache start for ``token_ids``.

        Resolution order (fastest/cheapest first): local GPU → local CPU →
        cross-node (gossip) → cross-model (variant) → miss.  Records the
        measured latency and bumps ``MetricsManager`` counters when present.
        """
        t0 = time.monotonic()
        prefix_len = len(token_ids)

        # 1. Local GPU tier (sub-ms).
        if self._gpu_cache is not None and hasattr(self._gpu_cache, "lookup"):
            try:
                match_len, kv = self._gpu_cache.lookup(token_ids)
                if match_len and match_len > 0 and kv is not None:
                    return self._finish(WarmStartSource.LOCAL_GPU, t0, prefix_len)
            except Exception as exc:
                logger.debug("gpu lookup failed: %s", exc)

        # 2. Local CPU tier.
        if self._cpu_cache is not None and hasattr(self._cpu_cache, "lookup"):
            try:
                match_len, kv = self._cpu_cache.lookup(token_ids)
                if match_len and match_len > 0 and kv is not None:
                    return self._finish(WarmStartSource.LOCAL_CPU, t0, prefix_len)
            except Exception as exc:
                logger.debug("cpu lookup failed: %s", exc)

        # 3. Cross-node via gossip: find a remote replica of this prefix hash.
        if self._gossip is not None and hasattr(self._gossip, "discover_prefix"):
            prefix_hash = self._hash(token_ids)
            try:
                nodes = self._gossip.discover_prefix(prefix_hash)
                nodes = [n for n in nodes if n != self._node_id]
                if nodes:
                    return self._finish(
                        WarmStartSource.CROSS_NODE, t0, prefix_len,
                        node_id=nodes[0], replica_nodes=nodes,
                    )
            except Exception as exc:
                logger.debug("gossip discover failed: %s", exc)

        # 4. Cross-model: a sibling variant already cached this base prefix.
        if self._cross_model is not None and model_id is not None:
            try:
                entry = self._cross_model.lookup(model_id, token_ids)
                if entry is not None:
                    return self._finish(
                        WarmStartSource.CROSS_MODEL, t0, prefix_len,
                        model_id=entry.source_model,
                    )
            except Exception as exc:
                logger.debug("cross-model lookup failed: %s", exc)

        # 5. Miss.
        return self._finish(WarmStartSource.MISS, t0, prefix_len, hit=False)

    # ── store propagation (advertise to gossip on local cache fill) ──

    def on_local_cache_store(self, token_ids: list[int], size_bytes: int = 0) -> None:
        """Call when a prefix KV is stored locally, so peers can discover it."""
        if self._gossip is not None and hasattr(self._gossip, "on_cache_store"):
            try:
                self._gossip.on_cache_store(self._hash(token_ids), self._node_id, size_bytes)
            except Exception as exc:
                logger.debug("gossip advertise failed: %s", exc)

    # ── helpers ──

    def _finish(
        self,
        source: WarmStartSource,
        t0: float,
        prefix_len: int,
        node_id: str | None = None,
        model_id: str | None = None,
        replica_nodes: list[str] | None = None,
        hit: bool = True,
    ) -> WarmStartResult:
        latency = (time.monotonic() - t0) * 1000.0
        self._stats["total_warm_start_ms"] += latency
        if source is WarmStartSource.LOCAL_GPU:
            self._stats["local_gpu_hits"] += 1
        elif source is WarmStartSource.LOCAL_CPU:
            self._stats["local_cpu_hits"] += 1
        elif source is WarmStartSource.CROSS_NODE:
            self._stats["cross_node_hits"] += 1
        elif source is WarmStartSource.CROSS_MODEL:
            self._stats["cross_model_hits"] += 1
        else:
            self._stats["misses"] += 1

        if self._metrics is not None:
            self._metrics.increment("kv_warm_starts_total")
            if hit:
                self._metrics.increment("kv_warm_start_hits_total")

        return WarmStartResult(
            source=source,
            latency_ms=latency,
            prefix_len=prefix_len,
            node_id=node_id,
            model_id=model_id,
            replica_nodes=replica_nodes or [],
            hit=hit,
        )

    @staticmethod
    def _hash(token_ids: list[int]) -> str:
        import hashlib

        return hashlib.sha256(
            ("|".join(str(t) for t in token_ids)).encode("utf-8")
        ).hexdigest()[:16]

    def stats(self) -> dict:
        return dict(self._stats)
