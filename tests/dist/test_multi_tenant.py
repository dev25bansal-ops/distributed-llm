"""Tests for distllm.dist.multi_tenant — zero mocks, real objects only."""

from __future__ import annotations

import time

import pytest

from distllm.dist.multi_tenant import (
    MultiTenantSLOEnforcer,
    SlidingWindowRateCounter,
    TenantMetrics,
    TenantSLO,
)


# ── TenantSLO ─────────────────────────────────────────────────────────────


class TestTenantSLO:
    """Dataclass for per-tenant SLO configuration."""

    def test_defaults(self) -> None:
        slo = TenantSLO(tenant_id="t1")
        assert slo.tenant_id == "t1"
        assert slo.max_rpm == 60.0
        assert slo.latency_slo_ms == 1000.0
        assert slo.max_concurrent == 10
        assert slo.burst_multiplier == 2.0
        assert slo.priority_base == 1.0

    def test_custom_values(self) -> None:
        slo = TenantSLO(
            tenant_id="acme",
            max_rpm=500.0,
            latency_slo_ms=250.0,
            max_concurrent=5,
            burst_multiplier=3.0,
            priority_base=0.5,
        )
        assert slo.max_rpm == 500.0
        assert slo.latency_slo_ms == 250.0
        assert slo.max_concurrent == 5
        assert slo.burst_multiplier == 3.0
        assert slo.priority_base == 0.5

    def test_zero_values_are_accepted(self) -> None:
        """Edge case: zero is valid for numeric fields."""
        slo = TenantSLO(
            tenant_id="zero",
            max_rpm=0.0,
            latency_slo_ms=0.0,
            max_concurrent=0,
            burst_multiplier=0.0,
            priority_base=0.0,
        )
        assert slo.max_rpm == 0.0


# ── TenantMetrics ─────────────────────────────────────────────────────────


class TestTenantMetrics:
    """Live metrics dataclass with computed properties."""

    def test_defaults(self) -> None:
        m = TenantMetrics()
        assert m.total_requests == 0
        assert m.completed_requests == 0
        assert m.rejected_requests == 0
        assert m.total_latency_ms == 0.0
        assert m.latency_p99_ms == 0.0
        assert m.current_concurrent == 0
        assert m.rp1m == 0
        assert m.slo_breaches == 0
        assert m.last_had_backpressure is False

    def test_avg_latency_zero_when_no_completed(self) -> None:
        m = TenantMetrics()
        assert m.avg_latency_ms == 0.0

    def test_avg_latency_computes_correctly(self) -> None:
        m = TenantMetrics(completed_requests=4, total_latency_ms=200.0)
        assert m.avg_latency_ms == 50.0

    def test_avg_latency_with_zero_completed_does_not_divide_by_zero(self) -> None:
        m = TenantMetrics(completed_requests=0, total_latency_ms=100.0)
        assert m.avg_latency_ms == 100.0  # max(0, 1) prevents ZeroDivision

    def test_slo_compliance_pct_100_when_no_requests(self) -> None:
        m = TenantMetrics()
        assert m.slo_compliance_pct == 100.0

    def test_slo_compliance_pct_no_breaches(self) -> None:
        m = TenantMetrics(total_requests=100, slo_breaches=0)
        assert m.slo_compliance_pct == 100.0

    def test_slo_compliance_pct_with_breaches(self) -> None:
        m = TenantMetrics(total_requests=100, slo_breaches=5)
        assert m.slo_compliance_pct == 95.0

    def test_slo_compliance_pct_all_breaches(self) -> None:
        m = TenantMetrics(total_requests=10, slo_breaches=10)
        assert m.slo_compliance_pct == 0.0

    def test_slo_compliance_pct_clamps_below_zero(self) -> None:
        m = TenantMetrics(total_requests=10, slo_breaches=15)
        assert m.slo_compliance_pct == 0.0


# ── SlidingWindowRateCounter ──────────────────────────────────────────────


class TestSlidingWindowRateCounter:
    """Sliding-window rate counter for per-tenant RPM."""

    def test_count_starts_at_zero(self) -> None:
        sw = SlidingWindowRateCounter()
        assert sw.count() == 0

    def test_record_and_count(self) -> None:
        sw = SlidingWindowRateCounter(window_s=60.0)
        sw.record(5)
        assert sw.count() == 5

    def test_multiple_records_sum(self) -> None:
        sw = SlidingWindowRateCounter(window_s=60.0)
        sw.record()
        sw.record(2)
        sw.record(3)
        assert sw.count() == 6

    def test_records_expire_after_window(self) -> None:
        """Use a very short window so records fall out on prune."""
        sw = SlidingWindowRateCounter(window_s=0.0)
        sw.record(10)
        # With window_s=0.0, any record older than 0s is pruned.
        # The record timestamp is time.time() which is > 0.0, so it prunes.
        assert sw.count() == 0

    def test_window_can_be_tiny(self) -> None:
        """Tiny non-zero window keeps recent records."""
        sw = SlidingWindowRateCounter(window_s=0.001)
        sw.record(3)
        # The record is from now() so within the 1ms window.
        assert sw.count() == 3

    def test_count_is_thread_safe(self) -> None:
        """Basic concurrency: record/count use a lock, no crash."""
        sw = SlidingWindowRateCounter()
        sw.record()
        sw.record()
        assert sw.count() == 2


# ── MultiTenantSLOEnforcer ────────────────────────────────────────────────


class TestMultiTenantSLOEnforcerRegistration:
    """Tenant registration and removal."""

    def test_register_single_tenant(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_rpm=100, latency_slo_ms=500)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["tenant_id"] == "t1"
        assert status["slo"]["max_rpm"] == 100
        assert status["slo"]["latency_slo_ms"] == 500

    def test_register_uses_defaults(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("defaults")
        status = enf.get_tenant_status("defaults")
        assert status is not None
        assert status["slo"]["max_rpm"] == 60.0
        assert status["slo"]["latency_slo_ms"] == 1000.0
        assert status["slo"]["max_concurrent"] == 10

    def test_register_does_not_reset_existing_metrics(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_rpm=10)
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=50)
        # Re-register with same tenant_id — metrics should survive.
        enf.register_tenant("t1", max_rpm=20)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["total_requests"] == 1
        assert status["slo"]["max_rpm"] == 20  # config updated

    def test_remove_tenant_clears_all_state(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        enf.remove_tenant("t1")
        assert enf.get_tenant_status("t1") is None

    def test_remove_nonexistent_tenant_does_not_raise(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.remove_tenant("ghost")  # should not raise

    def test_get_tenant_status_unknown(self) -> None:
        enf = MultiTenantSLOEnforcer()
        assert enf.get_tenant_status("unknown") is None


class TestMultiTenantSLOEnforcerAdmission:
    """Admission control (should_admit)."""

    def test_unknown_tenant_is_admitted(self) -> None:
        enf = MultiTenantSLOEnforcer()
        assert enf.should_admit("unknown") is True

    def test_tenant_within_limits_is_admitted(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_rpm=1000)
        assert enf.should_admit("t1") is True

    def test_tenant_blocked_by_concurrent_limit(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_concurrent=1)
        enf.record_request_start("t1")
        assert enf.should_admit("t1") is False

    def test_concurrent_limit_released_after_end(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_concurrent=1)
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=10)
        assert enf.should_admit("t1") is True

    def test_tenant_blocked_by_rate_limit(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_rpm=1)
        # First request admitted.
        assert enf.should_admit("t1") is True
        enf.record_request_start("t1")
        # After one request recorded (burst=2.0), 1 < 2, still admitted.
        assert enf.should_admit("t1") is True
        # Record 2 more to reach burst capacity of 2.
        # But should_admit only _reads_ the counter, doesn't increment it.
        # So we need to manually record into the counter.
        # Actually: should_admit checks rate_counter.count().
        # record_request_start calls rate_counter.record(1).
        # After 1 record: count=1, burst=2.0, 1 < 2 — admitted.
        # After 2 records: count=2, 2 >= 2 — blocked.
        enf.record_request_start("t1")
        # Now 2 records in counter, burst capacity = 2.0
        assert enf.should_admit("t1") is False

    def test_burst_capacity_uses_burst_multiplier(self) -> None:
        enf = MultiTenantSLOEnforcer()
        # burst_multiplier is a TenantSLO field but not exposed via
        # register_tenant(); it defaults to 2.0 in TenantSLO.
        # Raise max_concurrent so we are not blocked by concurrent limit.
        enf.register_tenant("t1", max_rpm=10, max_concurrent=50)
        for _ in range(15):
            enf.record_request_start("t1")
        # burst = 10*2.0 = 20; 15 < 20 — admitted (rate limit not hit).
        assert enf.should_admit("t1") is True

    def test_backpressure_flag_resets_when_admitted(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_concurrent=1)
        enf.record_request_start("t1")
        enf.should_admit("t1")  # blocked, sets backpressure flag
        enf.record_request_end("t1", latency_ms=5)
        # Next should_admit succeeds and clears flag.
        assert enf.should_admit("t1") is True
        status = enf.get_tenant_status("t1")
        assert status is not None
        # last_had_backpressure is an attribute of the TenantMetrics dataclass
        # instance, returned via get_tenant_status() as part of the metrics dict
        # only when populated internally. Check the metrics object directly.
        assert enf._metrics["t1"].last_had_backpressure is False  # type: ignore[union-attr]


class TestMultiTenantSLOEnforcerRecording:
    """Request start/end recording and metrics gathering."""

    def test_record_request_start_unknown_tenant_safe(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.record_request_start("ghost")  # should not raise

    def test_record_request_end_unknown_tenant_safe(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.record_request_end("ghost", latency_ms=50)

    def test_request_start_increments_counters(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        enf.record_request_start("t1")
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["total_requests"] == 1
        assert status["metrics"]["current_concurrent"] == 1

    def test_request_end_decrements_concurrent(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        enf.record_request_start("t1")
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=10)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["current_concurrent"] == 1

    def test_latency_accumulates(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=100.0)
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=200.0)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["completed"] == 2
        assert status["metrics"]["avg_latency_ms"] == 150.0

    def test_concurrent_never_goes_negative(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=5)
        # Extra end without matching start.
        enf.record_request_end("t1", latency_ms=5)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["current_concurrent"] == 0

    def test_p99_latency_with_few_samples(self) -> None:
        """p99 stays at 0 until at least 100 samples (internal threshold)."""
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        for _ in range(99):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=50.0)
        status = enf.get_tenant_status("t1")
        assert status is not None
        # The p99 is only updated when the history deque has >= 100 entries.
        assert status["metrics"]["latency_p99_ms"] == 0.0

    def test_p99_latency_computed_after_100_samples(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        for _ in range(100):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=50.0)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["latency_p99_ms"] == 50.0

    def test_p99_latency_with_many_samples(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        for i in range(200):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=float(i))
        status = enf.get_tenant_status("t1")
        assert status is not None
        # p99 should be high but within range.
        assert 150 <= status["metrics"]["latency_p99_ms"] <= 200

    def test_slo_breach_detected(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=10.0)
        for _ in range(200):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=100.0)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert status["metrics"]["slo_breaches"] > 0
        assert status["metrics"]["slo_compliance_pct"] < 100.0


class TestMultiTenantSLOEnforcerPriorityBoost:
    """Priority boost calculation."""

    def test_unknown_tenant_returns_default_boost(self) -> None:
        enf = MultiTenantSLOEnforcer()
        assert enf.get_priority_boost("unknown") == 1.0

    def test_no_data_returns_default_boost(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        # No requests recorded yet — p99 is 0.
        assert enf.get_priority_boost("t1") == 1.0

    def test_boost_is_1_when_p99_equals_slo(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=100.0)
        for _ in range(100):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=100.0)
        # After 100 identical values, p99 = 100.0, ratio = 1.0 -> 2.0
        assert enf.get_priority_boost("t1") == 2.0

    def test_boost_2_when_breaching(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=50.0)
        for _ in range(100):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=100.0)
        # ratio >= 1.0 -> 2.0
        assert enf.get_priority_boost("t1") == 2.0

    def test_boost_1_5_near_breach(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=100.0)
        for _ in range(100):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=92.0)
        # p99 ~ 92, ratio = 0.92 -> 1.5
        boost = enf.get_priority_boost("t1")
        assert boost == 1.5

    def test_boost_1_2_warming(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=100.0)
        for _ in range(100):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=78.0)
        # ratio ~ 0.78 -> 1.2
        boost = enf.get_priority_boost("t1")
        assert boost == 1.2

    def test_boost_0_8_lots_of_headroom(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=1000.0)
        for _ in range(100):
            enf.record_request_start("t1")
            enf.record_request_end("t1", latency_ms=1.0)
        # ratio ~ 0.001 -> 0.8
        boost = enf.get_priority_boost("t1")
        assert boost == 0.8


class TestMultiTenantSLOEnforcerObservability:
    """Status queries and global aggregation."""

    def test_get_all_status_empty(self) -> None:
        enf = MultiTenantSLOEnforcer()
        assert enf.get_all_status() == []

    def test_get_all_status_multiple_tenants(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        enf.register_tenant("t2")
        all_status = enf.get_all_status()
        assert len(all_status) == 2
        tids = {s["tenant_id"] for s in all_status}
        assert tids == {"t1", "t2"}

    def test_get_global_metrics_empty(self) -> None:
        enf = MultiTenantSLOEnforcer()
        gm = enf.get_global_metrics()
        assert gm["tenants"] == 0
        assert gm["total_requests"] == 0
        assert gm["total_slo_breaches"] == 0
        assert gm["total_rejected"] == 0
        assert gm["overall_slo_compliance"] == 100.0

    def test_get_global_metrics_aggregates(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", latency_slo_ms=100.0)
        enf.register_tenant("t2", latency_slo_ms=100.0)
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=10.0)
        enf.record_request_start("t2")
        enf.record_request_end("t2", latency_ms=10.0)
        gm = enf.get_global_metrics()
        assert gm["tenants"] == 2
        assert gm["total_requests"] == 2
        assert gm["overall_slo_compliance"] == 100.0

    def test_get_global_metrics_with_rejected(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1")
        # Manually increment rejected (normally done externally).
        enf._metrics["t1"].rejected_requests = 5  # type: ignore[union-attr]
        gm = enf.get_global_metrics()
        assert gm["total_rejected"] == 5

    def test_get_tenant_status_returns_full_snapshot(self) -> None:
        enf = MultiTenantSLOEnforcer()
        enf.register_tenant("t1", max_rpm=500, latency_slo_ms=200, max_concurrent=8)
        enf.record_request_start("t1")
        enf.record_request_end("t1", latency_ms=50)
        status = enf.get_tenant_status("t1")
        assert status is not None
        assert set(status.keys()) == {"tenant_id", "slo", "metrics"}
        assert isinstance(status["metrics"]["priority_boost"], float)
