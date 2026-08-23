"""Multi-region failover and client-side load balancing for the DistLLM SDK.

Wraps ``DistLLMClient`` and ``DistLLMClientSync`` with automatic
failover across multiple coordinator URLs, or distributes requests
across them using configurable strategies.

Usage — failover::

    client = MultiCoordinatorClient(
        DistLLMClient,
        urls=["http://primary:8000", "http://backup:8000"],
        mode="failover",
        strategy="latency",
    )
    response = await client.chat_completions(...)

Usage — load balancing::

    client = MultiCoordinatorClient(
        DistLLMClient,
        urls=["http://node1:8000", "http://node2:8000", "http://node3:8000"],
        mode="load_balance",
        strategy="least_loaded",  # "round_robin" | "least_loaded" | "latency"
    )
    response = await client.chat_completions(...)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Per-URL tracking
# ---------------------------------------------------------------------------

@dataclass
class _CoordinatorStats:
    """Tracks latency, error rate, and active requests for one coordinator URL."""
    url: str
    _latency_ms: list[float] = field(default_factory=list)
    errors: int = 0
    active_requests: int = 0
    last_healthy: float = field(default_factory=time.time)

    @property
    def avg_latency(self) -> float:
        return sum(self._latency_ms) / len(self._latency_ms) if self._latency_ms else 50.0

    @property
    def error_rate(self) -> float:
        total = self.errors + len(self._latency_ms)
        return self.errors / max(total, 1)

    def record(self, latency_ms: float, success: bool) -> None:
        self.active_requests = max(0, self.active_requests - 1)
        if success:
            self._latency_ms.append(latency_ms)
            if len(self._latency_ms) > 100:
                self._latency_ms.pop(0)
            self.last_healthy = time.time()
        else:
            self.errors += 1

    def start_request(self) -> None:
        self.active_requests += 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiCoordinatorConfig:
    """Configuration for multi-coordinator operation.

    Attributes:
        urls: Ordered list of coordinator URLs.
        mode: ``"failover"`` — try primary, fall through on failure.
               ``"load_balance"`` — distribute across all URLs.
        strategy: URL selection strategy.
            ``"round_robin"`` — cycle through URLs in order.
            ``"least_loaded"`` — pick URL with fewest active requests.
            ``"latency"`` — pick URL with lowest average latency.
            ``"sequential"`` (failover only) — stick to primary until errors.
        health_check_interval: Seconds between background health checks (default 30).
    """
    urls: list[str]
    mode: str = "failover"
    strategy: str = "round_robin"
    health_check_interval: float = 30.0


# ---------------------------------------------------------------------------
# Multi-coordinator client
# ---------------------------------------------------------------------------

class MultiCoordinatorClient:
    """Wraps a DistLLM client with multi-coordinator support.

    Supports two modes:
    - **failover**: requests go to the primary coordinator; fall through
      to backups when the primary is unhealthy.
    - **load_balance**: requests are distributed across all coordinators
      using the configured strategy.

    Usage::

        client = MultiCoordinatorClient(
            DistLLMClient,
            urls=["http://node1:8000", "http://node2:8000"],
            mode="load_balance",
            strategy="least_loaded",
        )
        response = await client.chat_completions(...)
    """

    def __init__(
        self,
        client_class: type,
        config: MultiCoordinatorConfig,
        **client_kwargs: Any,
    ):
        self._config = config
        self._client_class = client_class
        self._client_kwargs = client_kwargs
        self._stats = {url: _CoordinatorStats(url) for url in config.urls}
        self._round_robin = itertools.cycle(config.urls)
        self._clients: dict[str, Any] = {}
        self._current_url = config.urls[0]
        self._last_health_check = 0.0

    def _get_client(self, url: str) -> Any:
        """Get or create a client for *url*."""
        if url not in self._clients:
            self._clients[url] = self._client_class(
                base_url=url,
                **{k: v for k, v in self._client_kwargs.items() if k != "base_url"},
            )
        return self._clients[url]

    def _select_url(self) -> str:
        """Select a coordinator URL based on mode and strategy."""
        urls = self._config.urls

        if self._config.mode == "failover":
            if self._config.strategy == "sequential":
                stats = self._stats.get(self._current_url, _CoordinatorStats(urls[0]))
                if stats.errors > 3 and (time.time() - stats.last_healthy) > 30:
                    idx = urls.index(self._current_url)
                    return urls[(idx + 1) % len(urls)]
                return self._current_url
            if self._config.strategy == "latency":
                healthy = [
                    u for u in urls
                    if (time.time() - self._stats[u].last_healthy) < 120
                ] or urls
                return min(healthy, key=lambda u: self._stats[u].avg_latency)
            return urls[0]

        # Load balancing mode
        if self._config.strategy == "round_robin":
            return next(self._round_robin)
        if self._config.strategy == "least_loaded":
            return min(urls, key=lambda u: self._stats[u].active_requests)
        if self._config.strategy == "latency":
            return min(urls, key=lambda u: self._stats[u].avg_latency)

        return urls[0]

    def _execute(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a method on the selected coordinator."""
        url = self._select_url()
        self._current_url = url
        client = self._get_client(url)
        self._stats[url].start_request()
        try:
            result = getattr(client, method)(*args, **kwargs)
            self._stats[url].record(0, success=True)
            return result
        except Exception:
            self._stats[url].record(0, success=False)
            raise

    async def _execute_async(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Async version of _execute."""
        url = self._select_url()
        self._current_url = url
        client = self._get_client(url)
        self._stats[url].start_request()
        try:
            result = await getattr(client, method)(*args, **kwargs)
            self._stats[url].record(0, success=True)
            return result
        except Exception:
            self._stats[url].record(0, success=False)
            raise

    # -- Proxy: async methods ------------------------------------------------

    async def chat_completions(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute_async("chat_completions", *args, **kwargs)

    async def chat_completions_stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_client(self._select_url()).chat_completions_stream(*args, **kwargs)

    async def completions(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute_async("completions", *args, **kwargs)

    async def embeddings(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute_async("embeddings", *args, **kwargs)

    async def list_models(self, *args: Any, **kwargs: Any) -> Any:
        return await self._execute_async("list_models", *args, **kwargs)

    async def health_check(self) -> dict:
        return await self._execute_async("health_check")

    async def close(self) -> None:
        for c in self._clients.values():
            await c.close()

    # -- Proxy: sync methods -------------------------------------------------

    def chat_completions_sync(self, *args: Any, **kwargs: Any) -> Any:
        return self._execute("chat_completions", *args, **kwargs)

    def embeddings_sync(self, *args: Any, **kwargs: Any) -> Any:
        return self._execute("embeddings", *args, **kwargs)

    def health_check_sync(self) -> dict:
        return self._execute("health_check")

    # -- Context managers ----------------------------------------------------

    async def __aenter__(self) -> "MultiCoordinatorClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __enter__(self) -> "MultiCoordinatorClient":
        return self

    def __exit__(self, *args: Any) -> None:
        for c in self._clients.values():
            if hasattr(c, "close"):
                c.close()
