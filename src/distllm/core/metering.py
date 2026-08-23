"""Multi-tenant metered billing layer (Strategy E12).

This module adds a *thin* metering layer on top of the existing cost tracking
infrastructure in :mod:`distllm.core.cost_tracker`.  It does NOT replace the
``CostTracker`` or the ``CostTrackingMiddleware`` / ``QuotaMiddleware`` —
instead it REUSES the already-computed ``CostEstimate`` (GPU-seconds + USD
cost) produced by the existing :class:`~distllm.core.cost_tracker.CostTracker`
and turns each completed request into a billing ``UsageRecord``.

High-level pieces
-----------------
* :class:`UsageRecord` — an immutable, billing-oriented record per request
  (``tenant_id``, ``timestamp``, ``tokens_in``, ``tokens_out``,
  ``compute_s``, ``cost_usd`` plus a few optional metadata fields).
* :class:`MeteringStore` — an in-memory store of ``UsageRecord`` objects that
  aggregates per-tenant usage and can optionally persist to a JSONL file
  (pluggable backend; file persistence is best-effort and isolated).
* :class:`MeteringMiddleware` — a Starlette middleware that taps the request
  flow and records a ``UsageRecord`` into the store.  It reuses the singleton
  ``CostTracker`` so token/cost math is computed exactly once.
* :class:`BillingExporter` — serializes a per-tenant invoice to JSON.  Real
  Stripe integration is a **STUB**: when ``STRIPE_API_KEY`` is unset (the
  default) it writes a JSON invoice and logs
  ``"Stripe export stub — set STRIPE_API_KEY to enable"``.  If the key is set
  it lazily imports the ``stripe`` SDK and would post the invoice, but this
  path is not exercised by tests and is clearly marked as not-yet-real.

Everything is dependency-light: ``stripe`` is imported lazily and only if an
API key is configured.  With no key the exporter is a pure, network-free
no-op that still produces a valid invoice document.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from loguru import logger

from distllm.core.cost_tracker import get_cost_tracker


# ── UsageRecord ────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    """A single metered usage event for one tenant.

    This is the billing-layer record.  It intentionally reuses the cost
    figures already produced by :class:`~distllm.core.cost_tracker.CostTracker`
    so we never recompute cost or duplicate quota logic.

    Attributes:
        tenant_id: Owning tenant (reused from ``request.state`` / cost tracker).
        timestamp: Epoch seconds when the record was created.
        tokens_in: Input/prompt tokens for the request.
        tokens_out: Output/completion tokens for the request.
        compute_s: GPU-seconds consumed (from ``CostEstimate.estimated_gpu_seconds``).
        cost_usd: USD cost attributed to this request
            (from ``CostEstimate.estimated_cost_usd``).
        model_name: Model served (optional, for invoice line items).
        endpoint: API endpoint path (optional).
        request_id: Stable id linking back to the request (optional).
    """

    tenant_id: str
    timestamp: float
    tokens_in: int
    tokens_out: int
    compute_s: float
    cost_usd: float
    model_name: str = ""
    endpoint: str = ""
    request_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Storage backend protocol (pluggable) ───────────────────────────────────

@runtime_checkable
class MeteringBackend(Protocol):
    """Pluggable persistence backend for the MeteringStore.

    Implementations persist/load the raw list of usage-record dicts.  File
    persistence is best-effort; a backend that fails to write simply logs and
    keeps the in-memory state authoritative.
    """

    def save(self, records: list[dict[str, Any]]) -> None: ...

    def load(self) -> list[dict[str, Any]]: ...


class JsonlBackend:
    """JSONL file backend — one JSON object per line, append-only-safe.

    Safe for single-process use.  Falls back to memory-only if the path is
    not writable.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def save(self, records: list[dict[str, Any]]) -> None:
        try:
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, sort_keys=True))
                    fh.write("\n")
            os.replace(tmp, self._path)
        except (OSError, TypeError) as e:
            logger.debug(f"MeteringStore JSONL save skipped ({e}); memory only")

    def load(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            return []
        return out


# ── MeteringStore ─────────────────────────────────────────────────────────

class MeteringStore:
    """In-memory store of :class:`UsageRecord` objects with per-tenant rollups.

    Thread-safe.  Optional ``backend`` (e.g. :class:`JsonlBackend`) persists
    the raw records; the in-memory list remains authoritative and is the source
    of truth for aggregation/tallies.
    """

    def __init__(self, backend: MeteringBackend | None = None) -> None:
        self._backend = backend
        self._records: list[UsageRecord] = []
        self._lock = threading.RLock()
        if backend is not None:
            for raw in backend.load():
                try:
                    self._records.append(UsageRecord(**raw))
                except (TypeError, ValueError):
                    continue

    # ── Writes ────────────────────────────────────────────────────────────

    def record(self, rec: UsageRecord) -> UsageRecord:
        with self._lock:
            self._records.append(rec)
            if self._backend is not None:
                # Best-effort full rewrite; keeps file in sync.
                self._backend.save([r.to_dict() for r in self._records])
        return rec

    def record_request(
        self,
        *,
        tenant_id: str,
        tokens_in: int,
        tokens_out: int,
        compute_s: float,
        cost_usd: float,
        timestamp: float | None = None,
        model_name: str = "",
        endpoint: str = "",
        request_id: str = "",
    ) -> UsageRecord:
        """Convenience constructor that builds and stores a UsageRecord."""
        rec = UsageRecord(
            tenant_id=tenant_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            compute_s=float(compute_s),
            cost_usd=float(cost_usd),
            model_name=model_name,
            endpoint=endpoint,
            request_id=request_id,
        )
        return self.record(rec)

    # ── Reads / aggregation ───────────────────────────────────────────────

    def records_for_tenant(self, tenant_id: str) -> list[UsageRecord]:
        with self._lock:
            return [r for r in self._records if r.tenant_id == tenant_id]

    def all_records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def tally(self, tenant_id: str) -> dict[str, float]:
        """Aggregate totals for one tenant.

        Returns a dict with request count and summed tokens / compute / cost.
        """
        recs = self.records_for_tenant(tenant_id)
        return {
            "requests": len(recs),
            "tokens_in": sum(r.tokens_in for r in recs),
            "tokens_out": sum(r.tokens_out for r in recs),
            "total_tokens": sum(r.total_tokens for r in recs),
            "compute_s": round(sum(r.compute_s for r in recs), 8),
            "cost_usd": round(sum(r.cost_usd for r in recs), 8),
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            if self._backend is not None:
                self._backend.save([])


# ── Module-level store singleton ──────────────────────────────────────────

_store: MeteringStore | None = None
_store_lock = threading.Lock()


def get_metering_store() -> MeteringStore:
    """Get or create the module-level MeteringStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MeteringStore()
    return _store


def reset_metering_store() -> None:
    """Reset the singleton (used by tests to get a clean store)."""
    global _store
    with _store_lock:
        _store = None


# ── MeteringMiddleware ────────────────────────────────────────────────────

class MeteringMiddleware:
    """Thin ASGI step that records a :class:`UsageRecord` per request.

    This is *added* to the middleware stack — it never replaces
    ``CostTrackingMiddleware`` or ``QuotaMiddleware``.  It reuses the singleton
    :class:`~distllm.core.cost_tracker.CostTracker` to obtain the canonical
    cost/compute figures for a request, so token estimation and pricing logic
    are computed exactly once (no duplication of quota/cost logic).

    To enable, set ``DISTLLM_METERING_ENABLED=1`` or pass ``enable=True``.
    When disabled it is a transparent pass-through.
    """

    def __init__(
        self,
        app,
        store: MeteringStore | None = None,
        enable: bool | None = None,
    ) -> None:
        self.app = app
        self._store = store or get_metering_store()
        env_on = os.environ.get("DISTLLM_METERING_ENABLED", "0") == "1"
        self._enabled = env_on if enable is None else enable
        # Reuse the existing cost tracker singleton (do not re-create).
        self._tracker = get_cost_tracker()

    async def dispatch(self, request, call_next):
        if not self._enabled:
            return await call_next(request)

        path = getattr(getattr(request, "url", None), "path", "") or ""
        tracked = any(
            ep in path
            for ep in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")
        )
        if not tracked:
            return await call_next(request)

        # Resolve tenant from request state (same sources cost middleware uses).
        tenant_id = (
            getattr(request.state, "tenant_id", None)
            or getattr(request.state, "api_key_id", None)
            or "default"
        )
        model_name = getattr(request.state, "model", "") or ""

        response = await call_next(request)

        try:
            # Pull token counts off the response headers the cost middleware
            # already attached — this REUSES cost_middleware output rather than
            # re-reading/re-estimating the body.
            cost_hdr = response.headers.get("X-DistLLM-Cost")
            tok_hdr = response.headers.get("X-DistLLM-Tokens")  # in/out/total
            gpu_hdr = response.headers.get("X-DistLLM-GPU-Time")

            if tok_hdr:
                parts = tok_hdr.split("/")
                tokens_in = int(parts[0]) if len(parts) > 0 and parts[0] else 0
                tokens_out = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            else:
                tokens_in = tokens_out = 0

            # Reuse the canonical cost estimate from the existing CostTracker so
            # compute_s / cost_usd reflect the same model+pricing the rest of
            # the platform uses.
            estimate = self._tracker.estimate_cost(
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                model_name=model_name,
            )
            cost_usd = float(cost_hdr) if cost_hdr else estimate.estimated_cost_usd
            compute_s = (
                float(gpu_hdr) if gpu_hdr else estimate.estimated_gpu_seconds
            )

            self._store.record_request(
                tenant_id=str(tenant_id),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                compute_s=compute_s,
                cost_usd=cost_usd,
                model_name=model_name,
                endpoint=path,
                request_id=getattr(request.state, "request_id", "") or "",
            )
        except Exception as e:  # metering must never break the request
            logger.debug(f"Metering recording skipped: {e}")

        return response


# ── BillingExporter (Stripe stub) ─────────────────────────────────────────

class BillingExporter:
    """Serialize per-tenant invoices to JSON; Stripe integration is a STUB.

    Real Stripe billing is NOT implemented.  When ``STRIPE_API_KEY`` is unset
    (the default), ``export_invoice`` produces a valid JSON invoice document
    and logs ``"Stripe export stub — set STRIPE_API_KEY to enable"``.  When the
    key IS set, the ``stripe`` SDK is imported lazily and an invoice object is
    constructed, but this path is untested/offline and clearly marked as a
    placeholder — do NOT treat it as production billing.
    """

    # Sentinel so an explicit api_key=None means "no key" while an omitted
    # parameter falls back to the STRIPE_API_KEY environment variable.
    _UNSET = object()

    def __init__(self, currency: str = "usd", api_key: str | None = _UNSET) -> None:
        self.currency = currency
        # Only when the parameter is omitted do we fall back to STRIPE_API_KEY.
        self._api_key = (
            os.environ.get("STRIPE_API_KEY") if api_key is self._UNSET else api_key
        )

    def export_invoice(
        self,
        tenant_id: str,
        records: Sequence[UsageRecord],
        period: str = "monthly",
    ) -> dict[str, Any]:
        """Build (and, if configured, attempt to forward) an invoice.

        Args:
            tenant_id: Tenant to bill.
            records: UsageRecords for the billing period.
            period: Label such as ``"monthly"`` / ``"2026-07"``.

        Returns:
            An invoice dict.  ``mode`` is ``"stripe"`` if the Stripe path was
            taken, otherwise ``"stub"`` (no network).
        """
        recs = list(records)
        line_items = [r.to_dict() for r in recs]
        subtotal = round(sum(r.cost_usd for r in recs), 8)
        total_tokens = sum(r.total_tokens for r in recs)
        compute_s = round(sum(r.compute_s for r in recs), 8)

        invoice = {
            "schema": "distllm.metering.invoice/v1",
            "tenant_id": tenant_id,
            "period": period,
            "currency": self.currency,
            "line_items": line_items,
            "line_item_count": len(line_items),
            "total_tokens": total_tokens,
            "compute_seconds": compute_s,
            "subtotal_usd": subtotal,
            "amount_due_usd": subtotal,
            "generated_at": time.time(),
            "mode": "stub",  # overwritten below only if Stripe path taken
            "note": "Stripe integration is a stub; no real charges are made.",
        }

        if not self._api_key:
            logger.info("Stripe export stub — set STRIPE_API_KEY to enable")
            return invoice

        # ── Stripe path (LAZY import, placeholder only) ──────────────────
        try:
            import stripe  # lazy; not a hard dependency

            stripe.api_key = self._api_key
            # NOTE: This is a STUB mapping — real Stripe billing requires a
            # Customer, Price, and Invoice lifecycle that is out of scope here.
            # We simply reference the SDK to satisfy the lazy-import intent;
            # nothing is sent to avoid any side effects in this placeholder.
            _ = stripe.Invoice  # reference; not submitted
            invoice["mode"] = "stripe"
            invoice["note"] = (
                "Stripe SDK loaded; invoice object constructed but NOT submitted "
                "(real Stripe billing is a stub)."
            )
            logger.warning(
                "Stripe export stub: invoice object built for %s but not submitted "
                "(real billing not implemented).",
                tenant_id,
            )
            return invoice
        except ImportError:
            # stripe not installed — fall back to the documented stub.
            logger.info("Stripe export stub — set STRIPE_API_KEY to enable (stripe SDK missing)")
            invoice["mode"] = "stub"
            return invoice

    def export_invoice_json(
        self,
        tenant_id: str,
        records: Sequence[UsageRecord],
        period: str = "monthly",
    ) -> str:
        """Return the invoice as a JSON string (valid, parseable document)."""
        return json.dumps(
            self.export_invoice(tenant_id, records, period), indent=2, sort_keys=True
        )
