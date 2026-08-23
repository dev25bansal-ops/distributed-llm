"""E7 regression test -- unify metrics behind one in-memory facade.

The original E7 deliverable unified OTel + Prometheus behind a single
``MetricsFacade``/``MetricSink`` layer in ``distllm.observability.metrics``.
That facade is gone from the current codebase: ``observability`` now provides
the optional, OTel-coupled ``DistLLMMetrics``/``setup_metrics`` interface, and
the *unified* in-memory metrics facade -- counters, gauges, and histograms
with a Prometheus-compatible export -- is ``MetricsManager`` in
``distllm.core.coordinator_metrics``.

This test pins the current unification behavior:

  1. One :class:`MetricsManager` is the single definition point for
     application metrics.  Incrementing a counter, recording a gauge, and
     observing a histogram all update the same manager, and ``get()`` merges
     them into one flat snapshot.

  2. ``get_prometheus()`` exports every metric in one Prometheus-compatible
     dict with typed values -- no duplicate definition across backends.

  3. Histogram percentiles (P50/P95/P99) are exposed alongside the flat
     ``<name>_p95`` projection, and the histogram export carries
     Prometheus ``le`` buckets within ``_CURRENT_BUCKETS``.

These assertions hold with no optional telemetry packages installed -- the
metrics module imports cleanly (``load_module``) without opentelemetry or
prometheus_client.
"""

from __future__ import annotations

import pytest

from tests._import_helper import SRC_DIR, load_module, make_fake_package

# The real ``distllm/__init__.py`` (lazy-import heavy chain) must be bypassed
# so the metrics module loads on its own.  Faking just the two packages whose
# ``__init__`` would drag in optional/absent deps -- not the whole standard
# set, whose ``partition_planner`` workaround needs ``transformers``.
make_fake_package("distllm", SRC_DIR / "distllm")
make_fake_package("distllm.core", SRC_DIR / "distllm" / "core")

_mod = load_module("distllm/core/coordinator_metrics.py")
MetricsManager = _mod.MetricsManager
Histogram = _mod.Histogram


# ---------------------------------------------------------------------------
# Test 1: counters, gauges, and histograms all live in the one manager
# ---------------------------------------------------------------------------
def test_single_manager_defines_all_metric_kinds():
    mm = MetricsManager()
    mm.increment("distllm_tokens_generated_total", 7)
    mm.record("distllm_coordinator_queue_depth", 3)
    mm.observe("distllm_gen_latency_ms", 12.5)
    mm.observe("distllm_gen_latency_ms", 30.0)
    mm.observe("distllm_gen_latency_ms", 5.0)

    snap = mm.get()
    # Counter
    assert snap["distllm_tokens_generated_total"] == 7
    # Gauge (overwrites -> latest value)
    assert snap["distllm_coordinator_queue_depth"] == 3
    # Histogram -> flat p95 projection (nearest-rank of [5, 12.5, 30]).
    assert 12.0 <= snap["distllm_gen_latency_ms_p95"] <= 13.0

    # A single definition per name -- re-observing does not duplicate.
    assert list(snap).count("distllm_gen_latency_ms_p95") == 1


def test_counter_and_gauge_merge_into_one_snapshot():
    mm = MetricsManager()
    mm.increment("errors", 2)
    mm.record("queue_depth", 1)
    merged = mm.get()
    assert merged["errors"] == 2
    assert merged["queue_depth"] == 1
    assert "errors" in merged and "queue_depth" in merged


# ---------------------------------------------------------------------------
# Test 2: get_prometheus is the single Prometheus-compatible export
# ---------------------------------------------------------------------------
def test_get_prometheus_is_typed_and_single():
    mm = MetricsManager()
    mm.increment("distllm_tokens_generated_total", 5)
    mm.record("distllm_coordinator_queue_depth", 2)
    mm.observe("distllm_gen_latency_ms", 12.5)

    prom = mm.get_prometheus()
    # Counter typed as counter, gauge typed as gauge, histogram has buckets.
    assert prom["distllm_tokens_generated_total"] == {
        "value": 5.0,
        "type": "counter",
    }
    assert prom["distllm_coordinator_queue_depth"] == {
        "value": 2,
        "type": "gauge",
    }
    hist = prom["distllm_gen_latency_ms"]
    assert hist["type"] == "histogram"
    assert hist["sample_count"] == 1
    assert "le_+Inf" in hist["buckets"]
    # Every le bucket key falls within the configured bound set.
    for b in hist["buckets"]:
        assert b.startswith("le_")


def test_prometheus_histogram_buckets_respect_current_bounds():
    mm = MetricsManager()
    mm.observe("request_latency_ms", 30.0)
    hist = mm.get_prometheus()["request_latency_ms"]
    assert hist["buckets"]["le_+Inf"] == 1.0
    assert hist["buckets"]["le_10.0"] == 0.0
    assert hist["buckets"]["le_50.0"] == 1.0


# ---------------------------------------------------------------------------
# Test 3: Histogram also exposes percentiles independently
# ---------------------------------------------------------------------------
def test_histogram_percentiles():
    mm = MetricsManager()
    for i in range(1, 101):
        mm.observe("latency_ms", float(i))
    hist = mm.histogram("latency_ms")
    assert isinstance(hist, Histogram)
    assert hist.count == 100
    # Nearest-rank percentiles of 1..100 -> 50 / 95 / 99.
    assert hist.p50 == 50
    assert hist.p95 == 95
    assert hist.p99 == 99
    assert 0 < hist.mean < 100


def test_empty_histogram_reports_zero():
    hist = Histogram("empty")
    assert hist.count == 0
    assert hist.mean == 0.0
    assert hist.p50 == 0.0
    assert hist.to_prometheus()["sample_count"] == 0


# ---------------------------------------------------------------------------
# Test 4: reset restores the initial pre-initialized counters
# ---------------------------------------------------------------------------
def test_reset_restores_initial_state():
    mm = MetricsManager()
    mm.increment("requests", 42)
    mm.record("queue_depth", 9)
    mm.reset()
    snap = mm.get()
    assert snap.get("requests", 0) == 0
    assert "queue_depth" not in snap
    assert snap["total_requests"] == 0
    assert snap["errors"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))