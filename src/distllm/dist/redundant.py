"""Redundant Speculative Parallelism for unreliable home networks.

Runs the same pipeline stage on multiple redundant peers and uses
the first response. This turns latency variance into a feature:
a slow node doesn't slow down the pipeline because a faster
redundant node handles the request.

Usage:
    executor = RedundantExecutor(pipeline, redundancy=2)
    output = executor.run_pipeline(input_ids, node_kv_caches, request_id)
"""

from __future__ import annotations

import time
from typing import Any

from distllm._optional_deps import optional_import

torch = optional_import("torch")
from loguru import logger

from distllm.errors.types import NodeUnreachableError


class RedundantExecutor:
    """Wraps a PipelineOrchestrator with redundant parallel execution.

    For each pipeline stage, instead of sending to a single node,
    fans out to ``redundancy`` nodes that cover the same layer range.
    Uses the first response; remaining requests are discarded.

    Args:
        pipeline: The PipelineOrchestrator instance.
        redundancy: Number of redundant peers per stage.
        timeout_s: Per-request timeout.
    """

    def __init__(self, pipeline, redundancy: int = 1, timeout_s: float = 30.0):
        self._pipeline = pipeline
        self._redundancy = max(1, redundancy)
        self._timeout = timeout_s

    @property
    def enabled(self) -> bool:
        return self._redundancy > 1

    def _find_redundant_nodes(
        self, node_id: str,
    ) -> list[tuple[str, Any]]:
        """Find redundant nodes covering the same layer range as *node_id*."""
        target = self._pipeline._nodes.get(node_id)
        if target is None:
            return [(node_id, None)]

        results = [(node_id, target)]
        if not self.enabled:
            return results

        for nid, node in self._pipeline._nodes.items():
            if nid == node_id:
                continue
            if not getattr(node, 'is_healthy', False):
                continue
            if node.start_layer == target.start_layer and node.end_layer == target.end_layer:
                results.append((nid, node))
                if len(results) >= self._redundancy:
                    break

        return results

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
        draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """Run the pipeline with optional redundant execution.

        When redundancy is 1, behaves identically to the standard pipeline.
        When >1, fans out each stage to redundant peers and uses the
        first response.
        """
        if not self.enabled:
            return self._run_standard(input_ids, node_kv_caches, request_id, draft_tokens)

        return self._run_redundant(input_ids, node_kv_caches, request_id, draft_tokens)

    def _run_standard(self, input_ids, node_kv_caches, request_id, draft_tokens):
        """Non-redundant path — delegates to standard pipeline."""
        return self._pipeline.run_pipeline(
            input_ids, node_kv_caches, request_id,
        )

    def _run_redundant(self, input_ids, node_kv_caches, request_id, draft_tokens):
        """Redundant path — send each stage to multiple peers, use first response.

        Groups nodes by identical layer range: each distinct range is ONE
        logical stage, and replicas covering that range are its candidates.
        The range is applied exactly once — to whichever replica answers
        first — so redundant fan-out never pushes the hidden state through
        the same layers twice.  Fan-out is concurrent and bounded by
        ``timeout_s``.
        """
        with self._pipeline._lock:
            node_order = list(self._pipeline._node_order)
            nodes = dict(self._pipeline._nodes)

        if not node_order:
            raise RuntimeError("No healthy nodes in pipeline")

        # Eligible nodes for NEW work (healthy, not draining).  Reuse the
        # orchestrator's serving filter — the single source of truth shared
        # by the sequential and micro-batched paths — so `admin drain`
        # means the same thing on the redundant path.  Doubles lacking the
        # helper get the inline equivalent.
        serving_fn = getattr(self._pipeline, "serving_node_ids", None)
        if callable(serving_fn):
            serving_ids = set(serving_fn())
        else:
            draining: set[str] = set()
            resource_mgr = getattr(self._pipeline, "_resource_mgr", None)
            if resource_mgr is not None and hasattr(
                resource_mgr, "get_draining_nodes"
            ):
                try:
                    draining = set(resource_mgr.get_draining_nodes())
                except Exception:
                    draining = set()
            serving_ids = {
                nid for nid in node_order
                if nid in nodes
                and getattr(nodes[nid], "is_healthy", True)
                and nid not in draining
            }

        # Collapse replicas: one logical stage per distinct layer range
        # (preserving layer order). A same-range sibling is a stage
        # candidate, never a stage of its own — running it as its own
        # stage would apply the same layers again and corrupt the output.
        stages: list[str] = []
        seen_ranges: set[tuple[int, int]] = set()
        for nid in node_order:
            if nid not in serving_ids:
                continue
            node = nodes.get(nid)
            if node is None:
                continue
            key = (node.start_layer, node.end_layer)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            stages.append(nid)

        if not stages:
            raise RuntimeError("No healthy nodes in pipeline")

        current_hidden = input_ids

        for i, primary_id in enumerate(stages):
            candidates = [
                (rid, rnode)
                for rid, rnode in self._find_redundant_nodes(primary_id)
                if rnode is not None and rid in serving_ids
            ]

            winner = self._fan_out(
                current_hidden, candidates, node_kv_caches, request_id,
            )

            if winner is None:
                primary = nodes[primary_id]
                raise NodeUnreachableError(
                    node_id=primary_id,
                    host=primary.host,
                    port=primary.port,
                    original_error=Exception(
                        f"no redundant node succeeded within {self._timeout}s"
                    ),
                )

            best_id, current_hidden = winner
            if len(candidates) > 1:
                logger.debug(
                    f"Stage {i}: used replica {best_id} of {len(candidates)} "
                    f"candidates covering layers "
                    f"{nodes[primary_id].start_layer}-{nodes[primary_id].end_layer}"
                )

        return current_hidden

    def _fan_out(self, hidden_states, candidates, node_kv_caches, request_id):
        """Dispatch one stage to all *candidates* concurrently.

        Returns ``(replica_id, output)`` for the first replica that
        completes with a non-None result, or ``None`` when every candidate
        failed (or the per-stage deadline from ``timeout_s`` expired with
        no success).  Losing requests are discarded.
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        if not candidates:
            return None

        deadline = time.monotonic() + self._timeout
        pool = ThreadPoolExecutor(max_workers=len(candidates))
        try:
            futures = {
                pool.submit(
                    self._forward_once,
                    rnode,
                    hidden_states,
                    node_kv_caches.get(rid),
                    request_id,
                ): rid
                for rid, rnode in candidates
            }
            pending = set(futures)
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(
                    pending, return_when=FIRST_COMPLETED, timeout=remaining,
                )
                for fut in done:
                    rid = futures[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        logger.debug(f"Redundant node {rid} failed: {e}")
                        continue
                    if out is None:
                        continue
                    return rid, out
            return None
        finally:
            pool.shutdown(wait=False)

    def _forward_once(self, node, hidden_states, kv_cache, request_id):
        """Send one forward request to *node*.

        Mirrors PipelineOrchestrator.run_pipeline's transport selection so
        redundant fan-out uses the same injected/pooled clients and the
        same cluster auth secret as the standard path.
        """
        client = getattr(node, "client", None)
        if client is not None and hasattr(client, "forward"):
            return client.forward(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                request_id=request_id,
            )

        from distllm.dist.node_client import forward_request

        return forward_request(
            host=node.host,
            port=node.port,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            request_id=request_id,
            cluster_key=getattr(self._pipeline, "_cluster_key", None),
            use_tls=getattr(self._pipeline, "_use_tls", False),
            ca_cert=getattr(self._pipeline, "_ca_cert", None),
        )

    def get_node_groups(self) -> list[list[str]]:
        """Return node groups where each group covers a layer range."""
        if not self.enabled:
            return [list(self._pipeline._node_order)]

        groups = []
        seen = set()
        for nid in self._pipeline._node_order:
            if nid in seen:
                continue
            group = [nid]
            seen.add(nid)
            target = self._pipeline._nodes[nid]
            for other_id in self._pipeline._node_order:
                if other_id in seen:
                    continue
                other = self._pipeline._nodes[other_id]
                if (other.start_layer == target.start_layer
                        and other.end_layer == target.end_layer):
                    group.append(other_id)
                    seen.add(other_id)
                    if len(group) >= self._redundancy:
                        break
            groups.append(group)
        return groups


class StateReplicationEngine:
    """Maintains a hot standby pipeline with replicated KV cache state.

    Continuously replicates KV cache state from active nodes to standby
    nodes so that on failure, the standby can take over with exact cache
    state in <100ms (RTO).

    Architecture:
        Active Node A ──replicate──► Standby Node A'
        Active Node B ──replicate──► Standby Node B'

    On failure:
        1. Detect failure (circuit breaker trip)
        2. Promote standby to active
        3. Resume from replicated KV cache (no re-computation)

    Args:
        pipeline: PipelineOrchestrator instance.
        replication_interval_s: How often to replicate state.
        max_replication_lag_s: Max acceptable replication lag.
    """

    def __init__(
        self,
        pipeline,
        replication_interval_s: float = 1.0,
        max_replication_lag_s: float = 5.0,
    ):
        self._pipeline = pipeline
        self._interval = replication_interval_s
        self._max_lag = max_replication_lag_s
        self._standby_map: dict[str, str] = {}  # active_id -> standby_id
        self._replica_state: dict[str, dict] = {}  # standby_id -> state
        self._last_replication: dict[str, float] = {}  # standby_id -> timestamp
        self._running = False

    def register_standby(self, active_id: str, standby_id: str) -> None:
        """Register a standby node for an active node."""
        self._standby_map[active_id] = standby_id
        self._replica_state[standby_id] = {
            "kv_cache": None,
            "hidden_state": None,
            "last_request_id": None,
        }
        self._last_replication[standby_id] = 0
        logger.info(f"State replication: {active_id} → {standby_id}")

    def replicate_state(
        self,
        active_id: str,
        kv_cache: list | None,
        hidden_state: torch.Tensor | None,
        request_id: str,
    ) -> None:
        """Replicate state from active to standby.

        Called after each successful forward pass on the active node.
        """
        standby_id = self._standby_map.get(active_id)
        if standby_id is None:
            return

        state = self._replica_state.get(standby_id)
        if state is None:
            return

        # Deep copy KV cache to avoid aliasing
        if kv_cache is not None:
            state["kv_cache"] = [
                (k.clone().cpu(), v.clone().cpu()) for k, v in kv_cache
            ]
        if hidden_state is not None:
            state["hidden_state"] = hidden_state.clone().cpu()
        state["last_request_id"] = request_id
        self._last_replication[standby_id] = time.time()

    def get_replica_state(self, active_id: str) -> dict | None:
        """Get the replicated state for an active node's standby.

        Used during failover to restore the standby's KV cache.
        """
        standby_id = self._standby_map.get(active_id)
        if standby_id is None:
            return None
        return self._replica_state.get(standby_id)

    def promote_standby(self, active_id: str) -> str | None:
        """Promote a standby node to active.

        Returns the standby node ID, or None if no standby available.
        """
        standby_id = self._standby_map.pop(active_id, None)
        if standby_id is None:
            return None

        state = self._replica_state.pop(standby_id, None)
        if state is not None:
            logger.info(f"Promoted standby {standby_id} for failed node {active_id}")

            # Restore KV cache to the promoted node's cache
            if state["kv_cache"] is not None:
                self._pipeline._nodes[standby_id].kv_cache = state["kv_cache"]

        self._last_replication.pop(standby_id, None)
        return standby_id

    def get_replication_lag(self, active_id: str) -> float:
        """Get replication lag in seconds for an active node's standby."""
        standby_id = self._standby_map.get(active_id)
        if standby_id is None:
            return float("inf")
        last = self._last_replication.get(standby_id, 0)
        return time.time() - last

    def is_healthy(self, active_id: str) -> bool:
        """Check if replication for an active node is healthy."""
        lag = self.get_replication_lag(active_id)
        return lag <= self._max_lag

    @property
    def standby_count(self) -> int:
        return len(self._standby_map)

    @property
    def active_pairs(self) -> dict[str, str]:
        return dict(self._standby_map)
