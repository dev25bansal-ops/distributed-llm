"""Remote draft model registry with gossip-based discovery.

Maintains a local registry of remote draft model endpoints discovered
through the P2P gossip protocol.  Each draft is uniquely identified
by a *draft_id* (typically the publishing node's identifier) and
carries metadata such as endpoint URL, model name, hardware class,
and capability flags.

The registry participates in the gossip protocol by building
advertisement messages that carry known draft endpoints, and by
processing incoming advertisements to merge remote registries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass(frozen=True)
class DraftCapabilities:
    """Capability flags and metadata for a draft model."""

    max_batch_size: int = 1
    max_candidates: int = 16
    supports_penalties: bool = False
    supports_logprobs: bool = False
    supports_async: bool = True
    hardware: str = "cpu"
    quantized: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftModelInfo:
    """Immutable snapshot of a registered draft model."""

    draft_id: str
    endpoint: str
    model_name: str
    capabilities: DraftCapabilities = field(default_factory=DraftCapabilities)
    last_seen: float = 0.0
    avg_latency_ms: float = 0.0
    acceptance_rate: float = 0.0
    total_requests: int = 0
    total_errors: int = 0


@dataclass
class _DraftEntry:
    """Mutable bookkeeping for a known draft model."""

    draft_id: str
    endpoint: str
    model_name: str
    capabilities: DraftCapabilities
    last_seen: float
    avg_latency_ms: float = 0.0
    acceptance_rate: float = 0.0
    total_requests: int = 0
    total_errors: int = 0

    def to_info(self) -> DraftModelInfo:
        return DraftModelInfo(
            draft_id=self.draft_id,
            endpoint=self.endpoint,
            model_name=self.model_name,
            capabilities=self.capabilities,
            last_seen=self.last_seen,
            avg_latency_ms=self.avg_latency_ms,
            acceptance_rate=self.acceptance_rate,
            total_requests=self.total_requests,
            total_errors=self.total_errors,
        )


@dataclass
class DraftRequest:
    """Describes a speculative-decoding request that needs a draft model."""

    model_name: str | None = None
    required_hardware: str | None = None
    min_candidates: int = 1
    max_latency_ms: float = float("inf")
    prefers_quantized: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RemoteDraftRegistry:
    """Registry of remote draft models discovered via gossip protocol.

    Provides registration, lookup, and gossip-based advertisement of
    remote speculative-decoding endpoints.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(
        self,
        node_id: str,
        gossip_protocol: Any | None = None,
        stale_threshold_s: float = 120.0,
        max_entries: int = 256,
    ):
        self._node_id = node_id
        self._gossip = gossip_protocol
        self._stale_threshold_s = stale_threshold_s
        self._max_entries = max_entries
        self._entries: dict[str, _DraftEntry] = {}
        self._lock = threading.Lock()

        # Wire into gossip protocol if provided.
        if self._gossip is not None:
            logger.info(
                "RemoteDraftRegistry wired to gossip protocol "
                "(node_id={})", node_id,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        draft_id: str,
        endpoint: str,
        model_name: str,
        capabilities: DraftCapabilities | None = None,
    ) -> None:
        """Register or refresh a remote draft model.

        Args:
            draft_id: Unique identifier for the draft model (e.g. node id).
            endpoint: URL of the draft inference endpoint.
            model_name: Name of the draft model (e.g. "llama-68m").
            capabilities: Optional capability flags; defaults to empty.
        """
        cap = capabilities or DraftCapabilities()
        now = time.time()

        with self._lock:
            existing = self._entries.get(draft_id)
            if existing is not None:
                existing.endpoint = endpoint
                existing.model_name = model_name
                existing.capabilities = cap
                existing.last_seen = now
                logger.debug("Refreshed draft {!r} at {}", draft_id, endpoint)
            else:
                self._evict_if_full()
                self._entries[draft_id] = _DraftEntry(
                    draft_id=draft_id,
                    endpoint=endpoint,
                    model_name=model_name,
                    capabilities=cap,
                    last_seen=now,
                )
                logger.info(
                    "Registered draft {!r} (model={}, endpoint={})",
                    draft_id, model_name, endpoint,
                )

    def unregister(self, draft_id: str) -> bool:
        """Remove a draft model from the registry.

        Returns:
            True if the draft was found and removed, False otherwise.
        """
        with self._lock:
            if draft_id not in self._entries:
                logger.warning("Attempted to unregister unknown draft {!r}", draft_id)
                return False
            del self._entries[draft_id]
            logger.info("Unregistered draft {!r}", draft_id)
            return True

    def find_best(self, request: DraftRequest) -> DraftModelInfo | None:
        """Return the best draft model matching *request*.

        Selection criteria (applied in order):
            1. Not stale (``last_seen`` within ``stale_threshold_s``).
            2. Meets minimum candidate requirement.
            3. Meets latency requirement.
            4. Matches ``required_hardware`` if set.
            5. Matches ``prefers_quantized`` if set.
            6. Scored by ``(1.0 / max(avg_latency_ms, 1)) * (1.0 - error_ratio)``.

        Returns:
            A ``DraftModelInfo`` snapshot, or ``None`` if no match.
        """
        now = time.time()
        with self._lock:
            candidates: list[tuple[float, _DraftEntry]] = []
            for entry in self._entries.values():
                if (now - entry.last_seen) > self._stale_threshold_s:
                    continue
                if entry.capabilities.max_candidates < request.min_candidates:
                    continue
                if entry.avg_latency_ms > request.max_latency_ms:
                    continue
                if (
                    request.required_hardware is not None
                    and entry.capabilities.hardware != request.required_hardware
                ):
                    continue
                if (
                    request.prefers_quantized is not None
                    and entry.capabilities.quantized != request.prefers_quantized
                ):
                    continue

                total = entry.total_requests or 1
                error_ratio = entry.total_errors / total
                score = (1.0 / max(entry.avg_latency_ms, 1.0)) * (1.0 - error_ratio)
                candidates.append((score, entry))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1].to_info()

    def list_available(self) -> list[DraftModelInfo]:
        """Return snapshots of all registered drafts (including stale)."""
        with self._lock:
            return [e.to_info() for e in self._entries.values()]

    def list_healthy(self) -> list[DraftModelInfo]:
        """Return snapshots of non-stale drafts only."""
        now = time.time()
        with self._lock:
            return [
                e.to_info()
                for e in self._entries.values()
                if (now - e.last_seen) <= self._stale_threshold_s
            ]

    # ------------------------------------------------------------------
    # Gossip integration
    # ------------------------------------------------------------------

    def gossip_advertisement(self) -> dict:
        """Build a gossip message body with known draft endpoints.

        The returned dict is designed to be embedded in a
        :class:`~distllm.dist.p2p.gossip.GossipProtocol` advertisement
        or sent as a standalone message.

        Returns:
            A dict with keys ``node_id``, ``type`` (``"draft_registry"``),
            ``drafts`` (list of serialised entries), and ``timestamp``.
        """
        now = time.time()
        with self._lock:
            drafts_serialised: list[dict[str, Any]] = []
            for entry in self._entries.values():
                if (now - entry.last_seen) > self._stale_threshold_s:
                    continue
                cap = entry.capabilities
                drafts_serialised.append({
                    "draft_id": entry.draft_id,
                    "endpoint": entry.endpoint,
                    "model_name": entry.model_name,
                    "capabilities": {
                        "max_batch_size": cap.max_batch_size,
                        "max_candidates": cap.max_candidates,
                        "supports_penalties": cap.supports_penalties,
                        "supports_logprobs": cap.supports_logprobs,
                        "supports_async": cap.supports_async,
                        "hardware": cap.hardware,
                        "quantized": cap.quantized,
                    },
                    "avg_latency_ms": round(entry.avg_latency_ms, 1),
                    "acceptance_rate": round(entry.acceptance_rate, 4),
                })

            return {
                "type": "draft_registry",
                "node_id": self._node_id,
                "drafts": drafts_serialised,
                "timestamp": now,
            }

    def process_gossip_message(self, msg: dict) -> int:
        """Process an incoming draft-registry gossip message.

        Merges remote entries into the local registry, preferring the
        newer ``last_seen`` timestamp when a conflict exists.

        Args:
            msg: The deserialized gossip message (should have ``type``
                 ``"draft_registry"`` and a ``drafts`` list).

        Returns:
            Number of drafts that were newly registered or refreshed.
        """
        if not isinstance(msg, dict):
            return 0
        if msg.get("type") != "draft_registry":
            return 0

        remote_drafts: list[dict[str, Any]] = msg.get("drafts", [])
        if not remote_drafts:
            return 0

        count = 0
        now = time.time()
        with self._lock:
            for entry_dict in remote_drafts:
                draft_id = entry_dict.get("draft_id")
                if not draft_id:
                    continue

                cap_dict = entry_dict.get("capabilities", {})
                capabilities = DraftCapabilities(
                    max_batch_size=cap_dict.get("max_batch_size", 1),
                    max_candidates=cap_dict.get("max_candidates", 16),
                    supports_penalties=cap_dict.get("supports_penalties", False),
                    supports_logprobs=cap_dict.get("supports_logprobs", False),
                    supports_async=cap_dict.get("supports_async", True),
                    hardware=cap_dict.get("hardware", "cpu"),
                    quantized=cap_dict.get("quantized", False),
                )

                existing = self._entries.get(draft_id)
                if existing is not None:
                    # Only update if the remote entry is newer.
                    if entry_dict.get("timestamp", 0) > existing.last_seen:
                        existing.endpoint = entry_dict.get("endpoint", existing.endpoint)
                        existing.model_name = entry_dict.get("model_name", existing.model_name)
                        existing.capabilities = capabilities
                        existing.last_seen = now
                        existing.avg_latency_ms = entry_dict.get("avg_latency_ms", existing.avg_latency_ms)
                        existing.acceptance_rate = entry_dict.get("acceptance_rate", existing.acceptance_rate)
                        count += 1
                else:
                    self._evict_if_full()
                    self._entries[draft_id] = _DraftEntry(
                        draft_id=draft_id,
                        endpoint=entry_dict.get("endpoint", ""),
                        model_name=entry_dict.get("model_name", ""),
                        capabilities=capabilities,
                        last_seen=now,
                        avg_latency_ms=entry_dict.get("avg_latency_ms", 0.0),
                        acceptance_rate=entry_dict.get("acceptance_rate", 0.0),
                    )
                    count += 1

        if count:
            logger.debug(
                "Processed gossip: {} draft entries updated/added", count,
            )
        return count

    def record_success(
        self,
        draft_id: str,
        latency_ms: float,
        tokens_speculated: int = 0,
        tokens_accepted: int = 0,
    ) -> None:
        """Record a successful draft inference call.

        Updates rolling averages for latency and acceptance rate.
        """
        with self._lock:
            entry = self._entries.get(draft_id)
            if entry is None:
                return
            entry.total_requests += 1
            entry.last_seen = time.time()
            entry.avg_latency_ms = (
                entry.avg_latency_ms * 0.85 + latency_ms * 0.15
            )
            if tokens_speculated > 0:
                rate = tokens_accepted / tokens_speculated
                entry.acceptance_rate = (
                    entry.acceptance_rate * 0.85 + rate * 0.15
                )

    def record_error(self, draft_id: str) -> None:
        """Record a failed draft inference call."""
        with self._lock:
            entry = self._entries.get(draft_id)
            if entry is None:
                return
            entry.total_errors += 1
            entry.total_requests += 1
            entry.last_seen = time.time()

    def prune_stale(self) -> int:
        """Remove entries that have exceeded ``stale_threshold_s``.

        Returns:
            Number of entries pruned.
        """
        now = time.time()
        cutoff = now - self._stale_threshold_s
        with self._lock:
            stale = [did for did, e in self._entries.items() if e.last_seen < cutoff]
            for did in stale:
                del self._entries[did]
            if stale:
                logger.info("Pruned {} stale draft entries", len(stale))
            return len(stale)

    def clear(self) -> None:
        """Remove all entries from the registry."""
        with self._lock:
            self._entries.clear()
        logger.info("Cleared all draft entries")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_full(self) -> None:
        """Evict the stalest entry when ``_max_entries`` is exceeded.

        Assumes the lock is held.
        """
        if len(self._entries) < self._max_entries:
            return
        stalest: str | None = None
        stalest_ts = float("inf")
        for did, entry in self._entries.items():
            if entry.last_seen < stalest_ts:
                stalest_ts = entry.last_seen
                stalest = did
        if stalest is not None:
            stale_entry = self._entries.pop(stalest)
            logger.debug(
                "Evicted stale draft {!r} (last_seen={})",
                stalest, stale_entry.last_seen,
            )
