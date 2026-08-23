"""DaaS (Draft-as-a-Service) Marketplace — P2P marketplace for draft model tokens.

Allows providers to register compute resources for speculative / draft decoding
and lets consumers discover, rate, and purchase draft tokens from the best
available provider.

Classes:
    DaaSProviderInfo
        Describes a provider's hardware, pricing, model, and location.

    DaaSConsumer
        Queries the marketplace, selects the best provider using a reputation
        system, and purchases tokens.

    MarketplaceIntegration
        Orchestrates provider registration, peer discovery, heartbeats,
        and reputation lookups across the DaaS network.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class DaaSProviderInfo:
    """Specification of a draft-token provider in the DaaS marketplace.

    Attributes:
        provider_id: Globally unique identifier for this provider.
        host: IP or hostname where the provider's gRPC endpoint listens.
        port: Port number of the gRPC endpoint.
        hardware: Hardware identifier (e.g. ``"cpu"``, ``"cuda:0"``,
            ``"cuda:1"``).
        model_name: Name of the draft model served (e.g.
            ``"distilgpt2"``, ``"llama-68m"``).
        quantization: Quantization scheme (e.g. ``"fp16"``, ``"int8"``,
            ``"int4"``).
        price_per_token: Cost per draft token in the marketplace's base
            currency unit.
        min_batch: Minimum number of draft tokens a consumer must purchase
            per transaction.
        geo_location: Optional (latitude, longitude) tuple for proximity-
            aware discovery.
        supported_domains: List of domain names the provider explicitly
            supports (e.g. ``["code", "chat", "translation"]``). An empty
            list means all domains.
    """

    provider_id: str
    host: str
    port: int

    hardware: str = "cpu"
    model_name: str = ""
    quantization: str = "fp16"
    price_per_token: float = 0.0
    min_batch: int = 1
    geo_location: tuple[float, float] | None = None
    supported_domains: list[str] = field(default_factory=list)

    # Internal bookkeeping — populated by the integration layer.
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 0.0


class DaaSConsumer:
    """Consumer-facing client for the DaaS marketplace.

    Provides methods to query available providers, select the best match
    using an integrated reputation system, and purchase draft tokens.

    Args:
        integration: A :class:`MarketplaceIntegration` instance that backs
            provider discovery and reputation data.
    """

    def __init__(self, integration: MarketplaceIntegration) -> None:
        self._integration = integration

    # ── Public API ────────────────────────────────────────────────────────

    def query_marketplace(
        self,
        max_price: float,
        min_quality: float,
        max_latency_ms: float,
    ) -> list[DaaSProviderInfo]:
        """Query the marketplace for providers matching the given criteria.

        Filters all discovered providers by:

        * **Price**: ``price_per_token <= max_price``
        * **Quality** (reputation): ``score >= min_quality``
        * **Latency**: ``latency_ms <= max_latency_ms``

        Args:
            max_price: Maximum price per token the consumer is willing to
                pay.
            min_quality: Minimum reputation score required (0.0 – 1.0).
            max_latency_ms: Maximum acceptable round-trip latency in
                milliseconds.

        Returns:
            A list of :class:`DaaSProviderInfo` objects matching all
            criteria, sorted by descending reputation score.
        """
        all_providers = self._integration.discover_providers()
        matches: list[DaaSProviderInfo] = []

        for p in all_providers:
            if p.price_per_token > max_price:
                continue
            if p.latency_ms > max_latency_ms:
                continue
            rep = self._integration.get_reputation(p.provider_id)
            score = rep.get("score", 0.0)
            if score < min_quality:
                continue
            matches.append(p)

        matches.sort(key=lambda p: self._integration.get_reputation(p.provider_id).get("score", 0.0), reverse=True)
        return matches

    def select_best(
        self,
        providers: list[DaaSProviderInfo],
    ) -> DaaSProviderInfo | None:
        """Select the best provider from a list using the reputation system.

        The selection considers:

        1. Reputation score (higher is better).
        2. Latency (lower is better) — used as a tie-breaker for providers
           with identical scores.

        Args:
            providers: A list of candidate providers, typically obtained
                via :meth:`query_marketplace`.

        Returns:
            The best :class:`DaaSProviderInfo`, or ``None`` if the list is
            empty.
        """
        if not providers:
            return None

        def _key(p: DaaSProviderInfo) -> tuple[float, float]:
            rep = self._integration.get_reputation(p.provider_id)
            score = rep.get("score", 0.0)
            return (-score, p.latency_ms)

        return min(providers, key=_key)

    def purchase_tokens(
        self,
        provider: DaaSProviderInfo,
        count: int,
    ) -> str:
        """Purchase draft tokens from a specific provider.

        The actual purchase is delegated to the backing integration. The
        call returns a transaction ID that can be used for later auditing
        or billing reconciliation.

        Args:
            provider: The selected provider to purchase from.
            count: Number of draft tokens to purchase (will be rounded up
                to ``provider.min_batch`` if smaller).

        Returns:
            A transaction ID (UUID hex string).
        """
        actual_count = max(count, provider.min_batch)
        cost = actual_count * provider.price_per_token
        logger.info(
            "Purchasing {} tokens from {} @ ${:.6f}/token = ${:.6f}",
            actual_count,
            provider.provider_id,
            provider.price_per_token,
            cost,
        )
        return self._integration._record_purchase(provider.provider_id, actual_count, cost)  # noqa: SLF001


class MarketplaceIntegration:
    """Central orchestrator for the DaaS marketplace.

    Manages provider registration, peer discovery, heartbeats, reputation
    tracking, and purchase recording.

    This is the primary class that both providers and consumers interact
    with to join and use the marketplace.

    Args:
        discovery_timeout: Default timeout (in seconds) for peer discovery
            queries.
    """

    def __init__(self, discovery_timeout: float = 5.0) -> None:
        self._providers: dict[str, DaaSProviderInfo] = {}
        self._reputation: dict[str, dict[str, Any]] = {}
        self._purchase_log: list[dict[str, Any]] = []
        self._default_timeout = discovery_timeout

    # ── Provider Registration ─────────────────────────────────────────────

    def register_provider(self, info: DaaSProviderInfo) -> None:
        """Register a provider in the local marketplace registry.

        If a provider with the same ``provider_id`` already exists its
        entry is updated with the new information and a fresh
        ``last_seen`` timestamp.

        Args:
            info: A fully-populated :class:`DaaSProviderInfo` describing
                the provider.

        Raises:
            ValueError: If ``provider_id``, ``host``, or ``port`` are
                empty / zero.
        """
        if not info.provider_id:
            raise ValueError("provider_id is required")
        if not info.host:
            raise ValueError("host is required")
        if not (0 < info.port < 65536):
            raise ValueError("port must be in 1..65535")

        info.last_seen = time.time()
        self._providers[info.provider_id] = info

        # Initialise a neutral reputation entry for newcomers.
        if info.provider_id not in self._reputation:
            self._reputation[info.provider_id] = {
                "score": 0.5,
                "uptime": 100.0,
                "acceptance_rate": 1.0,
                "latency_p50": 0.0,
            }

        logger.info(
            "Registered DaaS provider {} @ {}:{} ({})",
            info.provider_id,
            info.host,
            info.port,
            info.hardware,
        )

    # ── Provider Discovery ────────────────────────────────────────────────

    def discover_providers(
        self,
        timeout: float | None = None,
    ) -> list[DaaSProviderInfo]:
        """Discover all currently-registered providers.

        Returns a snapshot of the provider registry.  Entries whose
        ``last_seen`` timestamp is older than *timeout* seconds are
        considered stale and are **not** returned (the caller may re-check
        later).

        Args:
            timeout: Staleness threshold in seconds.  Defaults to the
                instance's ``discovery_timeout``.

        Returns:
            A list of non-stale :class:`DaaSProviderInfo` objects.
        """
        threshold = time.time() - (timeout if timeout is not None else self._default_timeout)
        return [
            p
            for p in self._providers.values()
            if p.last_seen >= threshold
        ]

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def heartbeat(self, provider_id: str) -> bool:
        """Record a heartbeat for the given provider.

        Updates the provider's ``last_seen`` timestamp and bumps its
        reputation score slightly (up to a maximum of 1.0).

        Args:
            provider_id: The provider to heartbeat.

        Returns:
            ``True`` if the provider was found and updated, ``False`` if
            the provider is unknown.
        """
        info = self._providers.get(provider_id)
        if info is None:
            return False

        info.last_seen = time.time()

        rep = self._reputation.setdefault(provider_id, {
            "score": 0.5,
            "uptime": 100.0,
            "acceptance_rate": 1.0,
            "latency_p50": 0.0,
        })
        rep["score"] = min(1.0, rep["score"] + 0.005)
        return True

    # ── Reputation ────────────────────────────────────────────────────────

    def get_reputation(self, provider_id: str) -> dict[str, Any]:
        """Retrieve the full reputation report for a provider.

        Args:
            provider_id: The provider to look up.

        Returns:
            A dictionary with these keys:

            * **score** (``float``) — Composite reputation in 0.0–1.0.
            * **uptime** (``float``) — Uptime percentage.
            * **acceptance_rate** (``float``) — Fraction of purchase
              requests accepted.
            * **latency_p50** (``float``) — Median round-trip latency in
              milliseconds.

            If the provider has no recorded reputation a neutral report
            is returned.
        """
        if provider_id not in self._reputation:
            return {"score": 0.5, "uptime": 100.0, "acceptance_rate": 1.0, "latency_p50": 0.0}
        return dict(self._reputation[provider_id])

    # ── Internal purchase recording ───────────────────────────────────────

    def _record_purchase(
        self,
        provider_id: str,
        count: int,
        cost: float,
    ) -> str:
        """Record a completed token purchase and return a transaction ID.

        Updates the provider's acceptance rate and logs the purchase
        for auditing purposes.

        Args:
            provider_id: The provider tokens were purchased from.
            count: Number of tokens purchased.
            cost: Total cost of the purchase.

        Returns:
            A transaction ID string.
        """
        tx_id = uuid.uuid4().hex

        rep = self._reputation.get(provider_id)
        if rep is not None:
            # Bump acceptance rate — bounded between 0.0 and 1.0.
            rep["acceptance_rate"] = min(1.0, rep["acceptance_rate"] + 0.01)

        record: dict[str, Any] = {
            "tx_id": tx_id,
            "provider_id": provider_id,
            "count": count,
            "cost": cost,
            "timestamp": time.time(),
        }
        self._purchase_log.append(record)
        logger.debug("Purchase recorded: tx_id={}, provider={}, tokens={}, cost={:.6f}", tx_id, provider_id, count, cost)
        return tx_id
