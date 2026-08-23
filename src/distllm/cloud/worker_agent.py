"""DistLLM Cloud worker-agent — hybrid control-plane join client (SCAFFOLD).

.. warning::

    **THIS IS A SCAFFOLD (Strategy E13).** The DistLLM Cloud managed
    control-plane SaaS does **not** exist. This module is the thin client a
    customer *worker* would run to register with a hosted cloud coordinator so
    that the customer keeps compute + data on their own infra (the data plane)
    while DistLLM Cloud hosts only the control plane. See
    ``docs/HYBRID_CONTROL_PLANE.md`` for the architecture and honest-gaps
    section.

What this scaffold DOES do
--------------------------
* Gather a worker's local capabilities using the EXISTING platform types:
    - registered inference backends from
      :class:`distllm.backends.registry.BackendRegistry`, and
    - a topology/link descriptor from
      :class:`distllm.core.placement.LinkInfo`.
* Build a registration payload (control metadata ONLY — no prompts, weights, or
  KV cache; those never leave the data plane).
* POST it to a cloud coordinator over an mTLS-*ready* HTTP channel (bearer auth
  today; mutual TLS is the productionization step).

What this scaffold does NOT do
------------------------------
* There is no live cloud coordinator to accept the registration.
* No real mTLS handshake, no scheduling, no billing wiring.
* Network calls are best-effort and return a clear scaffold-shaped result;
  tests mock the HTTP layer so nothing hits the network.

Dependency-light: ``httpx`` is imported lazily and only when an actual POST is
attempted, so importing this module (and calling ``register_worker`` with a
mocked poster) never requires network libraries.
"""

from __future__ import annotations

import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

try:  # loguru is used throughout the repo; degrade gracefully if absent.
    from loguru import logger
except Exception:  # pragma: no cover - trivial fallback
    import logging

    logger = logging.getLogger(__name__)


__all__ = [
    "WorkerCapabilities",
    "RegistrationResult",
    "collect_capabilities",
    "register_worker",
]


# ── Capability descriptor (control metadata only) ──────────────────────────


@dataclass
class WorkerCapabilities:
    """Control-plane metadata describing a customer worker.

    This is the ONLY thing that crosses the trust boundary at registration
    time. It intentionally contains no prompts, weights, KV cache, or training
    data — only capability metadata the hosted control plane needs to schedule.

    Attributes:
        worker_id: Stable id for this worker (defaults to a random uuid).
        hostname: Local hostname (informational; stays within customer naming).
        region: Logical region/zone label used by placement scoring.
        backends: Names of inference backends registered locally
            (from :class:`~distllm.backends.registry.BackendRegistry`).
        gpu_count: Number of local accelerators advertised.
        latency_ms: Advertised coordinator→worker link latency (placement input).
        bandwidth_gbps: Advertised link bandwidth (placement input).
        labels: Free-form customer labels (e.g. ``{"tier": "sovereign"}``).
    """

    worker_id: str = field(default_factory=lambda: f"worker-{uuid.uuid4().hex[:12]}")
    hostname: str = ""
    region: str = "default"
    backends: list[str] = field(default_factory=list)
    gpu_count: int = 0
    latency_ms: float = 0.0
    bandwidth_gbps: float = 10.0
    labels: dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the control-metadata registration payload.

        NOTE: deliberately excludes any content/data-plane fields — this is the
        contract that preserves sovereignty (see docs/HYBRID_CONTROL_PLANE.md).
        """
        return {
            "schema": "distllm.cloud.worker_registration/v1",
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "region": self.region,
            "backends": list(self.backends),
            "gpu_count": self.gpu_count,
            "link": {
                "latency_ms": self.latency_ms,
                "bandwidth_gbps": self.bandwidth_gbps,
            },
            "labels": dict(self.labels),
        }

    def to_link_info(self) -> Any:
        """Build a :class:`distllm.core.placement.LinkInfo` from these caps.

        Shows the mapping onto the EXISTING placement type the hosted control
        plane's scheduler would consume. Imported lazily to keep this module
        dependency-light.
        """
        from distllm.core.placement import LinkInfo

        return LinkInfo(
            node_id=self.worker_id,
            region=self.region,
            latency_ms=self.latency_ms,
            bandwidth_gbps=self.bandwidth_gbps,
        )


@dataclass
class RegistrationResult:
    """Result of a (scaffolded) worker registration attempt.

    Attributes:
        ok: Whether the control plane acknowledged the registration.
        worker_id: The worker id that was (or would be) registered.
        coordinator_url: The cloud coordinator that was targeted.
        status: One of ``"registered"`` (mock/live ack) or ``"scaffold"``
            (no live coordinator / offline — the default honest state).
        detail: Human-readable note.
        payload: The exact control-metadata payload that was sent.
        response: Whatever the (mocked or real) poster returned, if any.
    """

    ok: bool
    worker_id: str
    coordinator_url: str
    status: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)
    response: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "worker_id": self.worker_id,
            "coordinator_url": self.coordinator_url,
            "status": self.status,
            "detail": self.detail,
            "payload": self.payload,
            "response": self.response,
        }


# ── Capability collection ──────────────────────────────────────────────────


def collect_capabilities(
    *,
    region: str = "default",
    gpu_count: int | None = None,
    latency_ms: float = 0.0,
    bandwidth_gbps: float = 10.0,
    labels: dict[str, str] | None = None,
) -> WorkerCapabilities:
    """Gather local worker capabilities using EXISTING platform types.

    Reuses :class:`distllm.backends.registry.BackendRegistry` to discover which
    inference backends this worker can serve. All values are control metadata —
    nothing here is prompt/weight/KV content.

    Args:
        region: Logical placement region for this worker.
        gpu_count: Override GPU count; defaults to ``DISTLLM_GPU_COUNT`` env or 0.
        latency_ms: Advertised coordinator→worker latency (placement input).
        bandwidth_gbps: Advertised link bandwidth (placement input).
        labels: Optional free-form customer labels.

    Returns:
        A populated :class:`WorkerCapabilities`.
    """
    backends: list[str] = []
    try:
        from distllm.backends.registry import BackendRegistry

        backends = [p.name for p in BackendRegistry().list_backends()]
    except Exception as e:  # registry optional at scaffold time
        logger.debug(f"[cloud.worker_agent SCAFFOLD] backend discovery skipped: {e}")

    if gpu_count is None:
        try:
            gpu_count = int(os.environ.get("DISTLLM_GPU_COUNT", "0"))
        except ValueError:
            gpu_count = 0

    return WorkerCapabilities(
        hostname=_safe_hostname(),
        region=region,
        backends=backends,
        gpu_count=gpu_count,
        latency_ms=latency_ms,
        bandwidth_gbps=bandwidth_gbps,
        labels=labels or {},
    )


def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover
        return platform.node() or "unknown"


# ── Registration (SCAFFOLD) ────────────────────────────────────────────────


def register_worker(
    coordinator_url: str,
    auth_token: str,
    *,
    capabilities: WorkerCapabilities | None = None,
    poster: Callable[[str, dict[str, Any], dict[str, str]], Any] | None = None,
    timeout: float = 5.0,
) -> RegistrationResult:
    """Register this worker with a DistLLM Cloud coordinator (SCAFFOLD).

    The worker *dials out* to the hosted control plane and sends its capability
    metadata over an mTLS-ready, bearer-authenticated channel. Only control
    metadata + (later) billing meters cross the boundary — never prompts,
    weights, or KV cache.

    This is a **scaffold**: there is no live cloud coordinator. When no
    ``poster`` is injected and ``httpx`` is unavailable or the endpoint is
    unreachable, this returns a clear ``status="scaffold"`` result instead of
    raising — so it is safe to call offline. Tests inject a ``poster`` to mock
    the HTTP layer (no network).

    Args:
        coordinator_url: Base URL of the cloud coordinator
            (e.g. ``https://cloud.distllm.ai``).
        auth_token: Bearer token identifying the tenant/worker (Argon2-hashed
            server-side per M3).
        capabilities: Pre-collected capabilities; if ``None`` they are gathered
            via :func:`collect_capabilities`.
        poster: Optional callable ``(url, json_payload, headers) -> response``
            used to send the request. Injected by tests to avoid the network.
            When omitted, a lazy ``httpx`` POST is attempted (best-effort).
        timeout: Network timeout in seconds for the real ``httpx`` path.

    Returns:
        A :class:`RegistrationResult`. ``status`` is ``"registered"`` if the
        control plane (mock or live) acknowledged, else ``"scaffold"``.
    """
    if not coordinator_url:
        raise ValueError("coordinator_url is required")
    if not auth_token:
        raise ValueError("auth_token is required")

    caps = capabilities or collect_capabilities()
    payload = caps.to_payload()
    payload["registered_at"] = time.time()

    endpoint = coordinator_url.rstrip("/") + "/api/v1/cloud/workers/register"
    headers = {
        "Content-Type": "application/json",
        # mTLS-ready: bearer auth today; mutual TLS is the productionization step.
        "Authorization": f"Bearer {auth_token}",
        "X-DistLLM-Worker-Id": caps.worker_id,
    }

    # ── Injected poster path (tests / custom transports) ──
    if poster is not None:
        try:
            resp = poster(endpoint, payload, headers)
            return RegistrationResult(
                ok=True,
                worker_id=caps.worker_id,
                coordinator_url=coordinator_url,
                status="registered",
                detail=(
                    "SCAFFOLD: registration acknowledged by injected poster "
                    "(no real cloud SaaS exists)."
                ),
                payload=payload,
                response=resp,
            )
        except Exception as e:
            return RegistrationResult(
                ok=False,
                worker_id=caps.worker_id,
                coordinator_url=coordinator_url,
                status="scaffold",
                detail=f"SCAFFOLD: injected poster raised ({e}); nothing sent.",
                payload=payload,
            )

    # ── Best-effort real path (lazy httpx; expected to fail offline) ──
    try:
        import httpx  # lazy: not a hard dependency of this scaffold

        with httpx.Client(timeout=timeout) as client:
            r = client.post(endpoint, json=payload, headers=headers)
            return RegistrationResult(
                ok=r.status_code < 400,
                worker_id=caps.worker_id,
                coordinator_url=coordinator_url,
                status="registered" if r.status_code < 400 else "scaffold",
                detail=(
                    f"SCAFFOLD: POST returned {r.status_code}. Note: DistLLM "
                    "Cloud is not a real SaaS yet."
                ),
                payload=payload,
                response={"status_code": r.status_code},
            )
    except Exception as e:
        logger.info(
            "[cloud.worker_agent SCAFFOLD] no live coordinator at "
            f"{endpoint}: {e}"
        )
        return RegistrationResult(
            ok=False,
            worker_id=caps.worker_id,
            coordinator_url=coordinator_url,
            status="scaffold",
            detail=(
                "SCAFFOLD: no live DistLLM Cloud coordinator reachable "
                "(the managed control-plane SaaS is not built). "
                "Capabilities were collected and a payload prepared but not "
                "delivered."
            ),
            payload=payload,
        )
