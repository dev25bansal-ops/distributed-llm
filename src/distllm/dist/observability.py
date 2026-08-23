"""Unified observability for the distributed inference layer.

Integrates the existing :class:`DistLLMPrometheusExporter` with the
distributed layer's subsystem-specific metrics (pipeline, federation,
recovery, NCCL) and the :class:`Tracer` for OpenTelemetry tracing.

This is a convenience aggregator — import it once at the coordinator's
startup to wire all observability paths::

    from distllm.dist.observability import enable_dist_observability

    observability = enable_dist_observability(
        exporter=prometheus_exporter,
        service_name="distllm-coordinator",
        enable_tracing=True,
        enable_pipeline_metrics=True,
        enable_federation_metrics=True,
        enable_recovery_metrics=True,
        enable_nccl_metrics=True,
    )

    # Metrics are now being collected and exposed via /metrics.
    # Traces are being exported to the configured OTel collector.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def enable_dist_observability(
    exporter: Any | None = None,
    service_name: str = "distllm-node",
    enable_tracing: bool = True,
    enable_pipeline_metrics: bool = True,
    enable_federation_metrics: bool = True,
    enable_recovery_metrics: bool = True,
    enable_nccl_metrics: bool = True,
    otlp_endpoint: str = "http://localhost:4318/v1/traces",
    **kwargs: Any,
) -> dict[str, Any]:
    """Initialize all observability paths for the distributed layer.

    Args:
        exporter: An optional ``DistLLMPrometheusExporter`` instance.
            When ``None``, metrics registration is skipped (caller may
            register their own).
        service_name: Service name for tracing.
        enable_tracing: Start the OpenTelemetry :class:`Tracer` and
            instrument pipeline, federation, and recovery.
        enable_pipeline_metrics: Register pipeline-specific Prometheus
            metrics on *exporter*.
        enable_federation_metrics: Register federation-specific metrics.
        enable_recovery_metrics: Register recovery-specific metrics.
        enable_nccl_metrics: Register NCCL transport metrics.
        otlp_endpoint: OTel collector endpoint for trace export.
        **kwargs: Passed through to the :class:`Tracer` constructor.

    Returns:
        Dict with keys ``"tracer"`` and ``"metrics"`` for optional
        further wiring.
    """
    result: dict[str, Any] = {"tracer": None, "metrics": {}}

    # ── Tracing ───────────────────────────────────────────────────────

    if enable_tracing:
        try:
            from distllm.dist.tracing import Tracer

            tracer = Tracer(
                service_name=service_name,
                otlp_endpoint=otlp_endpoint,
                **{k: v for k, v in kwargs.items()
                   if k in ("console_export", "resource_attributes")},
            )
            tracer.instrument_all()
            result["tracer"] = tracer
            logger.info("Distributed-layer tracing enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize tracing: {e}")

    # ── Pipeline metrics ──────────────────────────────────────────────

    if exporter is not None and enable_pipeline_metrics:
        try:
            _add_pipeline_metrics(exporter)
            result["metrics"]["pipeline"] = True
        except Exception as e:
            logger.warning(f"Failed to register pipeline metrics: {e}")

    # ── Federation metrics ────────────────────────────────────────────

    if exporter is not None and enable_federation_metrics:
        try:
            _add_federation_metrics(exporter)
            result["metrics"]["federation"] = True
        except Exception as e:
            logger.warning(f"Failed to register federation metrics: {e}")

    # ── Recovery metrics ──────────────────────────────────────────────

    if exporter is not None and enable_recovery_metrics:
        try:
            _add_recovery_metrics(exporter)
            result["metrics"]["recovery"] = True
        except Exception as e:
            logger.warning(f"Failed to register recovery metrics: {e}")

    # ── NCCL metrics ──────────────────────────────────────────────────

    if exporter is not None and enable_nccl_metrics:
        try:
            _add_nccl_metrics(exporter)
            result["metrics"]["nccl"] = True
        except Exception as e:
            logger.warning(f"Failed to register NCCL metrics: {e}")

    return result


# ── Per-subsystem metric registration ─────────────────────────────────


def _add_pipeline_metrics(exporter: Any) -> None:
    """Register pipeline-stage-level metrics."""
    r = exporter.registry
    try:
        from prometheus_client import Gauge, Histogram

        exporter._pipeline_stage_latency = Histogram(
            "distllm_pipeline_stage_latency_ms",
            "Per-stage latency in milliseconds",
            ["stage_id", "schedule"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=r,
        )
        exporter._pipeline_overlap_efficiency = Gauge(
            "distllm_pipeline_overlap_efficiency_percent",
            "Pipeline compute-communication overlap efficiency",
            registry=r,
        )
        exporter._pipeline_micro_batches = Gauge(
            "distllm_pipeline_micro_batches",
            "Number of micro-batches in current schedule",
            registry=r,
        )
        exporter._pipeline_warmup = Gauge(
            "distllm_pipeline_warmup_ratio",
            "Warmup micro-batches / total micro-batches",
            registry=r,
        )
    except ImportError:
        logger.warning("prometheus_client not installed — pipeline metrics unavailable")


def _add_federation_metrics(exporter: Any) -> None:
    """Register federation-specific metrics."""
    r = exporter.registry
    try:
        from prometheus_client import Counter, Gauge, Histogram

        exporter._fed_forwards_total = Counter(
            "distllm_federation_forwards_total",
            "Total federated request forwards",
            ["result"],
            registry=r,
        )
        exporter._fed_peers = Gauge(
            "distllm_federation_peers",
            "Number of discovered peer clusters",
            registry=r,
        )
        exporter._fed_cache_digests = Gauge(
            "distllm_federation_cache_digests",
            "Number of remote cache digests held",
            registry=r,
        )
        exporter._fed_heartbeat_latency = Histogram(
            "distllm_federation_heartbeat_latency_ms",
            "Heartbeat round-trip latency",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=r,
        )
        exporter._fed_spillovers_total = Counter(
            "distllm_federation_spillovers_total",
            "Total request spillovers to peer clusters",
            registry=r,
        )
    except ImportError:
        logger.warning("prometheus_client not installed — federation metrics unavailable")


def _add_recovery_metrics(exporter: Any) -> None:
    """Register recovery-specific metrics."""
    r = exporter.registry
    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Additional recovery metrics beyond the core ones in exporter.py.
        exporter._recovery_checkpoint_bytes = Gauge(
            "distllm_recovery_checkpoint_bytes",
            "Total bytes stored in checkpoints",
            registry=r,
        )
        exporter._recovery_redistributions = Counter(
            "distllm_recovery_redistributions_total",
            "Total layer redistributions performed",
            registry=r,
        )
        exporter._recovery_drill_total = Counter(
            "distllm_recovery_drill_total",
            "Total recovery drills executed",
            ["result"],
            registry=r,
        )
        exporter._recovery_drill_duration = Histogram(
            "distllm_recovery_drill_duration_ms",
            "Recovery drill duration",
            buckets=[10, 50, 100, 500, 1000, 5000, 10000],
            registry=r,
        )
    except ImportError:
        logger.warning("prometheus_client not installed — recovery metrics unavailable")


def _add_nccl_metrics(exporter: Any) -> None:
    """Register NCCL transport-level metrics."""
    r = exporter.registry
    try:
        from prometheus_client import Counter, Gauge

        exporter._nccl_bytes_sent = Counter(
            "distllm_nccl_bytes_total",
            "Total bytes transferred via NCCL",
            ["comm_type"],
            registry=r,
        )
        exporter._nccl_ops_total = Counter(
            "distllm_nccl_ops_total",
            "Total NCCL operations performed",
            ["comm_type", "result"],
            registry=r,
        )
        exporter._nccl_active_ops = Gauge(
            "distllm_nccl_active_ops",
            "Currently active NCCL operations",
            registry=r,
        )
        exporter._nccl_preemptions_total = Counter(
            "distllm_nccl_preemptions_total",
            "Total NCCL operation preemptions",
            registry=r,
        )
    except ImportError:
        logger.warning("prometheus_client not installed — NCCL metrics unavailable")
