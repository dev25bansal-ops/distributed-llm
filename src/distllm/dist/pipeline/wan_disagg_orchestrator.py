"""WAN disaggregated orchestrator — separates prefill (compute-heavy) from
decode (latency-sensitive) across geographically distant nodes.

In a disaggregated setup, prefill nodes run in cheap regions with abundant
compute, while decode nodes sit close to end users to minimise time-to-first-
token and per-token latency. KV caches are streamed from the prefill node to
the decode node over WAN links using :mod:`httpx.AsyncClient`.

Key design decisions:

- **Two-phase routing**: ``route_request`` independently selects the nearest
  prefill node (lowest latency) and the nearest decode node (lowest latency
  to the client), then pairs them.
- **Streaming KV transfer**: Uses ``httpx.AsyncClient`` to stream KV cache
  data between the selected prefill and decode endpoints.
- **Token accumulation**: An inner ``TokenAccumulator`` batches decode tokens
  before they are sent over the WAN link, amortising round-trip costs.
- **WAN timeout fallback**: If measured WAN latency exceeds
  ``wan_timeout_ms`` the orchestrator falls back to local-only inference
  (single node handles both phases).
- **Adaptive batch sizing**: The accumulator's minimum batch size adjusts
  based on measured RTT — higher latency increases the batch target.

Usage::

    config = WanDisaggConfig(
        prefill_endpoints=["us-central-1:8001", "us-east-1:8001"],
        decode_endpoints=["us-west-1:8002", "eu-west-1:8002"],
    )
    orchestator = WanDisaggOrchestrator(config)
    result = await orchestrator.route_request("What is the capital of France?", "llama-3-8b")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.dist.pipeline.token_accumulator import TokenAccumulator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WanDisaggConfig:
    """Configuration for the WAN disaggregated prefill/decode orchestrator.

    Attributes:
        prefill_endpoints: List of ``host:port`` strings for prefill nodes
            (compute-heavy, in cheap regions).
        decode_endpoints: List of ``host:port`` strings for decode nodes
            (latency-sensitive, near users).
        wan_timeout_ms: Maximum acceptable WAN round-trip time in
            milliseconds. When measured latency exceeds this value the
            orchestrator falls back to local inference.
        min_batch_tokens: Minimum number of decode tokens to accumulate
            before sending a batch over the WAN link.
        max_accumulation_tokens: Hard upper bound on the number of decode
            tokens buffered before a forced flush.
        flush_interval_s: Wall-clock interval (seconds) for time-based
            token flushes, used to cap interactive latency.
    """

    prefill_endpoints: tuple[str, ...] = ()
    decode_endpoints: tuple[str, ...] = ()
    wan_timeout_ms: float = 5000.0
    min_batch_tokens: int = 32
    max_accumulation_tokens: int = 512
    flush_interval_s: float = 0.1

    def __post_init__(self) -> None:
        if self.wan_timeout_ms < 0:
            raise ValueError(f"wan_timeout_ms must be >= 0, got {self.wan_timeout_ms}")
        if self.min_batch_tokens < 1:
            raise ValueError(
                f"min_batch_tokens must be >= 1, got {self.min_batch_tokens}"
            )
        if self.max_accumulation_tokens < self.min_batch_tokens:
            raise ValueError(
                f"max_accumulation_tokens ({self.max_accumulation_tokens}) must be "
                f">= min_batch_tokens ({self.min_batch_tokens})"
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class WanDisaggOrchestrator:
    """Orchestrates geographically separate prefill and decode nodes.

    Routes a prompt to the nearest prefill node for the compute-heavy
    prefill phase, then streams the resulting KV cache to the nearest
    decode node for the latency-sensitive decode phase.

    The orchestrator maintains an inner :class:`TokenAccumulator` that
    batches decode tokens before WAN transfer. The batch size is adaptive:
    it starts at ``config.min_batch_tokens`` and grows when measured RTT
    is high, shrinking when the link is fast.

    Args:
        config: Disaggregated WAN configuration.

    Attributes:
        config: The configuration used by this orchestrator.
        accumulator: The inner token accumulator for batching decode tokens.
    """

    def __init__(self, config: WanDisaggConfig) -> None:
        self.config = config
        self.accumulator = TokenAccumulator(
            min_batch_size=config.min_batch_tokens,
            max_tokens=config.max_accumulation_tokens,
            flush_interval_s=config.flush_interval_s,
        )

        # Registered endpoints with metadata.
        self._prefill_nodes: dict[str, _NodeInfo] = {}
        self._decode_nodes: dict[str, _NodeInfo] = {}

        for ep in config.prefill_endpoints:
            self._prefill_nodes[ep] = _NodeInfo(endpoint=ep)
        for ep in config.decode_endpoints:
            self._decode_nodes[ep] = _NodeInfo(endpoint=ep)

        # Metrics.
        self._wan_latencies: list[float] = []
        self._tokens_per_rtt: list[float] = []
        self._fallback_count: int = 0
        self._last_prefill_node: str | None = None
        self._last_decode_node: str | None = None

        logger.info(
            "WanDisaggOrchestrator initialised: {} prefill nodes, {} decode nodes",
            len(self._prefill_nodes),
            len(self._decode_nodes),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_request(
        self,
        prompt: str,
        model: str,
        client_location: str | None = None,
    ) -> dict[str, Any]:
        """Route a prompt through the disaggregated prefill/decode pipeline.

        The routing strategy is:

        1. Select the nearest prefill node (by configured endpoint order,
           preferring lower latency — a production system would use
           geo-lookup or real-time probes).
        2. Select the nearest decode node (again by latency preference).
        3. Measure the WAN round-trip between them.
        4. If WAN latency ``> wan_timeout_ms``, fall back to local inference
           on the prefill node.
        5. Otherwise, run prefill on the prefill node, stream the KV cache
           to the decode node, and run decode there.

        Args:
            prompt: The input text prompt.
            model: Model identifier (e.g. ``"llama-3-8b"``).
            client_location: Optional hint about the client's geographic
                location, used for decode-node affinity.

        Returns:
            A dictionary with keys:

            - ``"output"``: Generated text or tensors (simplified).
            - ``"prefill_node"``: The endpoint used for prefill.
            - ``"decode_node"``: The endpoint used for decode.
            - ``"wan_latency_ms"``: Measured WAN latency in ms.
            - ``"fallback"``: Whether local-only fallback was used.
        """
        # 1. Select nodes.
        prefill_node = self._select_nearest(self._prefill_nodes, client_location)
        decode_node = self._select_nearest(self._decode_nodes, client_location)

        self._last_prefill_node = prefill_node.endpoint
        self._last_decode_node = decode_node.endpoint

        # 2. Measure WAN latency between the selected pair.
        wan_latency = await self._measure_wan_rtt(prefill_node, decode_node)
        self._wan_latencies.append(wan_latency)

        logger.debug(
            "WAN RTT {:.0f} ms (prefill={}, decode={})",
            wan_latency,
            prefill_node.endpoint,
            decode_node.endpoint,
        )

        # 3. Fallback if WAN is too slow.
        if wan_latency > self.config.wan_timeout_ms:
            self._fallback_count += 1
            logger.warning(
                "WAN latency {:.0f} ms exceeds threshold {:.0f} ms — "
                "falling back to local inference",
                wan_latency,
                self.config.wan_timeout_ms,
            )
            result = await self._run_local(prompt, model, prefill_node)
            result["fallback"] = True
            return result

        # 4. Disaggregated execution.
        result = await self._run_disaggregated(
            prompt, model, prefill_node, decode_node, wan_latency
        )
        result["fallback"] = False
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> dict[str, Any]:
        """Snapshot of orchestrator performance metrics.

        Returns:
            Dictionary with:

            - ``wan_latency_ms``: List of measured WAN RTT values.
            - ``avg_wan_latency_ms``: Average WAN latency or 0.
            - ``tokens_per_rtt``: List of token-per-RTT measurements.
            - ``avg_tokens_per_rtt``: Average or 0.
            - ``fallback_count``: Total number of local fallbacks.
            - ``prefill_node``: Last used prefill endpoint or ``None``.
            - ``decode_node``: Last used decode endpoint or ``None``.
        """
        return {
            "wan_latency_ms": list(self._wan_latencies),
            "avg_wan_latency_ms": (
                sum(self._wan_latencies) / len(self._wan_latencies)
                if self._wan_latencies
                else 0.0
            ),
            "tokens_per_rtt": list(self._tokens_per_rtt),
            "avg_tokens_per_rtt": (
                sum(self._tokens_per_rtt) / len(self._tokens_per_rtt)
                if self._tokens_per_rtt
                else 0.0
            ),
            "fallback_count": self._fallback_count,
            "prefill_node": self._last_prefill_node,
            "decode_node": self._last_decode_node,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_nearest(
        self,
        nodes: dict[str, _NodeInfo],
        hint: str | None = None,
    ) -> _NodeInfo:
        """Select the nearest node from a set of candidates.

        In a production deployment this method would consult a geo-index
        or live latency probes. This implementation returns the first
        healthy node or a deterministic round-robin choice.

        Args:
            nodes: Map of endpoint to node info.
            hint: Optional geo-hint (currently unused; reserved for
                future latency-aware selection).

        Returns:
            The selected node info.

        Raises:
            RuntimeError: If no nodes are available.
        """
        if not nodes:
            raise RuntimeError("No nodes available for selection")

        # Prefer the first endpoint as a simple heuristic.
        # A real implementation would use geo-distance or live probes.
        candidates = sorted(nodes.values(), key=lambda n: n.endpoint)
        selected = candidates[0]
        logger.debug("Selected node {} (hint={})", selected.endpoint, hint)
        return selected

    async def _measure_wan_rtt(
        self,
        prefill: _NodeInfo,
        decode: _NodeInfo,
    ) -> float:
        """Measure the round-trip latency between a prefill and decode node.

        Sends a lightweight probe (via ``httpx.AsyncClient``) from the
        prefill to the decode node and returns the measured RTT in
        milliseconds.

        Args:
            prefill: The prefill node info.
            decode: The decode node info.

        Returns:
            Measured RTT in milliseconds. Returns a large sentinel value
            if the probe fails.
        """
        import httpx

        probe_url = f"http://{decode.endpoint}/ping"
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.config.wan_timeout_ms / 1000) as client:
                resp = await client.get(probe_url)
                resp.raise_for_status()
        except Exception:
            logger.warning("WAN probe to {} failed — using timeout sentinel", probe_url)
            return self.config.wan_timeout_ms + 1.0

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return elapsed_ms

    async def _run_disaggregated(
        self,
        prompt: str,
        model: str,
        prefill: _NodeInfo,
        decode: _NodeInfo,
        wan_latency: float,
    ) -> dict[str, Any]:
        """Execute the disaggregated prefill-then-decode pipeline.

        Steps:

        1. Run prefill on *prefill* node (obtain KV cache).
        2. Stream KV cache over WAN to *decode* node.
        3. Run decode on *decode* node, accumulating tokens through the
           inner :class:`TokenAccumulator`.
        4. Adapt the accumulator's batch size based on measured RTT.

        Args:
            prompt: Input text prompt.
            model: Model identifier.
            prefill: Selected prefill node.
            decode: Selected decode node.
            wan_latency: Measured WAN RTT in ms.

        Returns:
            Result dictionary with ``output``, node identifiers, and
            WAN latency information.
        """
        import httpx

        # --- Phase 1: Prefill ---
        logger.debug("Running prefill on {}", prefill.endpoint)
        prefill_url = f"http://{prefill.endpoint}/prefill"

        t0 = time.monotonic()
        async with httpx.AsyncClient(
            timeout=self.config.wan_timeout_ms / 1000
        ) as client:
            prefill_resp = await client.post(
                prefill_url,
                json={"prompt": prompt, "model": model},
            )
            prefill_resp.raise_for_status()
            prefill_data = prefill_resp.json()

        prefill_ms = (time.monotonic() - t0) * 1000.0
        logger.debug("Prefill completed in {:.0f} ms", prefill_ms)

        # --- Phase 2: Stream KV cache to decode node ---
        logger.debug("Streaming KV cache to {}", decode.endpoint)
        kv_url = f"http://{decode.endpoint}/kv"

        t0 = time.monotonic()
        async with httpx.AsyncClient(
            timeout=self.config.wan_timeout_ms / 1000
        ) as client:
            kv_resp = await client.post(
                kv_url,
                json={
                    "request_id": prefill_data.get("request_id", ""),
                    "kv_cache": prefill_data.get("kv_cache", {}),
                    "model": model,
                },
            )
            kv_resp.raise_for_status()

        kv_ms = (time.monotonic() - t0) * 1000.0
        logger.debug("KV transfer completed in {:.0f} ms", kv_ms)

        # --- Phase 3: Decode (accumulated) ---
        logger.debug("Running decode on {}", decode.endpoint)
        decode_url = f"http://{decode.endpoint}/decode"

        # Compute adaptive batch target: scale with measured RTT.
        adaptive_min = self._adaptive_batch_target(wan_latency)

        tokens: list[str] = []
        done = False
        while not done:
            if self.accumulator.should_flush(adaptive_min=adaptive_min):
                batch = self.accumulator.flush()
                if batch:
                    t0 = time.monotonic()
                    async with httpx.AsyncClient(
                        timeout=self.config.wan_timeout_ms / 1000
                    ) as client:
                        decode_resp = await client.post(
                            decode_url,
                            json={
                                "tokens": batch,
                                "model": model,
                            },
                        )
                        decode_resp.raise_for_status()
                        decode_data = decode_resp.json()

                    rtt_ms = (time.monotonic() - t0) * 1000.0
                    self._tokens_per_rtt.append(len(batch))
                    logger.debug(
                        "Decode batch of {} tokens in {:.0f} ms ({:.1f} tok/s)",
                        len(batch),
                        rtt_ms,
                        len(batch) / (rtt_ms / 1000) if rtt_ms > 0 else float("inf"),
                    )

                    tokens.extend(decode_data.get("tokens", []))
                    done = decode_data.get("done", False)

            # If the accumulator isn't ready yet, yield briefly to the
            # event loop. In a real system this would be driven by an
            # incoming stream of decode tokens.
            if not done:
                await asyncio_sleep(0)
                # Simulate a decode token arriving (the production loop
                # would be driven by the model backend pushing tokens).
                self.accumulator.add(0)

        # Flush any remaining tokens.
        if self.accumulator.buffer_size > 0:
            remaining = self.accumulator.flush()
            if remaining:
                async with httpx.AsyncClient(
                    timeout=self.config.wan_timeout_ms / 1000
                ) as client:
                    flush_resp = await client.post(
                        decode_url,
                        json={"tokens": remaining, "model": model},
                    )
                    flush_resp.raise_for_status()
                    flush_data = flush_resp.json()
                    tokens.extend(flush_data.get("tokens", []))

        output_text = "".join(tokens) if tokens else ""

        return {
            "output": output_text,
            "prefill_node": prefill.endpoint,
            "decode_node": decode.endpoint,
            "wan_latency_ms": wan_latency,
            "prefill_time_ms": prefill_ms,
            "kv_transfer_time_ms": kv_ms,
        }

    async def _run_local(
        self,
        prompt: str,
        model: str,
        node: _NodeInfo,
    ) -> dict[str, Any]:
        """Fallback: run both prefill and decode on a single node.

        Args:
            prompt: Input text prompt.
            model: Model identifier.
            node: The node that handles both phases.

        Returns:
            Result dictionary with output and node information.
        """
        import httpx

        logger.warning("Running local inference on {} (fallback)", node.endpoint)
        url = f"http://{node.endpoint}/infer"

        async with httpx.AsyncClient(
            timeout=self.config.wan_timeout_ms / 1000
        ) as client:
            resp = await client.post(
                url,
                json={"prompt": prompt, "model": model},
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "output": data.get("output", ""),
            "prefill_node": node.endpoint,
            "decode_node": node.endpoint,
            "wan_latency_ms": 0.0,
        }

    @staticmethod
    def _adaptive_batch_target(wan_latency_ms: float) -> int:
        """Compute an adaptive minimum batch size from measured WAN latency.

        The strategy is:

        - Latency ``<= 50`` ms (fast link): keep the configured minimum.
        - Latency ``50-200`` ms (moderate): double the minimum.
        - Latency ``> 200`` ms (slow link): quadruple the minimum.

        Args:
            wan_latency_ms: Measured WAN round-trip time in milliseconds.

        Returns:
            Recommended minimum batch size.
        """
        if wan_latency_ms <= 50:
            return 32
        elif wan_latency_ms <= 200:
            return 64
        else:
            return 128


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass
class _NodeInfo:
    """Lightweight information about a registered node endpoint."""

    endpoint: str
    healthy: bool = True
    last_seen: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Module-level helper (avoid star-import pollution)
# ---------------------------------------------------------------------------

try:
    from asyncio import sleep as asyncio_sleep
except ImportError:  # pragma: no cover
    import asyncio

    asyncio_sleep = asyncio.sleep


__all__ = [
    "WanDisaggConfig",
    "WanDisaggOrchestrator",
]
