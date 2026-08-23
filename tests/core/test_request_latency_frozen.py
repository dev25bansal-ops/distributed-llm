"""Regression: completed-request latency must freeze, not grow with wall clock.

F-022: ``elapsed_ms`` was computed live as ``(time.time() - enqueued_at)`` even
for COMPLETED rows, so a request that finished fast would later be reported as
overdue once wall-clock advanced — corrupting SLA compliance percentiles.
The fix caps elapsed at completion time.
"""

from __future__ import annotations

import time

from distllm.core.request_latency import RequestLatencyTracker


class TestCompletedLatencyFrozen:
    def test_elapsed_frozen_after_complete(self):
        tracker = RequestLatencyTracker(default_sla_ms=100.0)
        tracker.register("r1", sla_ms=200.0)
        time.sleep(0.01)
        tracker.complete("r1")

        first = tracker.get_metrics("r1")["elapsed_ms"]
        time.sleep(0.05)
        second = tracker.get_metrics("r1")["elapsed_ms"]
        assert second <= first + 5, "completed elapsed_ms must not grow with wall clock"

    def test_completed_fast_request_never_overdue(self):
        tracker = RequestLatencyTracker(default_sla_ms=100.0)
        tracker.register("fast", sla_ms=200.0)
        time.sleep(0.005)
        tracker.complete("fast")

        time.sleep(0.05)
        metrics = tracker.get_metrics("fast")
        assert metrics["is_overdue"] is False, "a completed fast request must stay compliant"
        assert metrics["elapsed_ms"] < 200.0

    def test_sla_compliance_stable_over_time(self):
        tracker = RequestLatencyTracker(default_sla_ms=100.0)
        for i in range(5):
            tracker.register(f"r{i}", sla_ms=500.0)
            time.sleep(0.002)
            tracker.complete(f"r{i}")

        p1 = tracker.get_sla_percentiles()
        time.sleep(0.05)
        p2 = tracker.get_sla_percentiles()
        assert p1["sla_compliance_pct"] == p2["sla_compliance_pct"] == 100.0