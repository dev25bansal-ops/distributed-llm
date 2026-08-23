"""Tests for StreamingCostTracker and StreamingCostState.

Covers:
- StreamingCostState dataclass defaults and field updates
- record_input, record_output_token
- Cost calculation (separate input/output rates)
- Cloud comparison and savings
- Throughput metrics (ttft_ms, tokens_per_second, avg_tokens_per_second)
- to_token_event and to_final_summary format
- StreamingCostTracker: start_tracking, record_token, finish_tracking
- StreamingCostTracker: get_active_state, get_stats
- StreamingCostTracker: completed list bounded at 1000
- Thread safety
- Module-level singleton helpers
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/streaming_cost.py")
StreamingCostState = _mod.StreamingCostState
StreamingCostTracker = _mod.StreamingCostTracker
get_streaming_cost_tracker = _mod.get_streaming_cost_tracker
reset_streaming_cost_tracker = _mod.reset_streaming_cost_tracker


# ---------------------------------------------------------------------------
# StreamingCostState
# ---------------------------------------------------------------------------


class TestStreamingCostStateDefaults:
    """Default field values."""

    def test_defaults(self) -> None:
        state = StreamingCostState()
        assert state.request_id == ""
        assert state.model_name == ""
        assert state.input_tokens == 0
        assert state.output_tokens == 0
        assert state.total_tokens == 0
        assert state.cumulative_cost == 0.0
        assert state.cumulative_savings == 0.0
        assert state.ttft_ms == 0.0
        assert state.tokens_per_second == 0.0
        assert state.avg_tokens_per_second == 0.0


# ---------------------------------------------------------------------------
# record_input / record_output_token
# ---------------------------------------------------------------------------


class TestStreamingCostStateRecord:
    """Recording input and output tokens."""

    def test_record_input(self) -> None:
        state = StreamingCostState(input_cost_per_token=0.0001, output_cost_per_token=0.0002)
        state.record_input(100)
        assert state.input_tokens == 100
        assert state.total_tokens == 100
        assert state.cumulative_input_cost == 0.01  # 100 * 0.0001

    def test_record_output_token_first(self) -> None:
        state = StreamingCostState(start_time=time.time())
        state.record_input(100)
        time.sleep(0.01)
        state.record_output_token()
        assert state.output_tokens == 1
        assert state.total_tokens == 101
        assert state.first_token_time > 0
        assert state.ttft_ms > 0

    def test_record_output_token_multiple(self) -> None:
        state = StreamingCostState(start_time=time.time())
        state.record_input(10)
        for _ in range(5):
            state.record_output_token()
        assert state.output_tokens == 5
        assert state.total_tokens == 15


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


class TestStreamingCostStateCost:
    """Cost calculation with separate input/output rates."""

    def test_cost_after_input_only(self) -> None:
        state = StreamingCostState(
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        state.record_input(10)
        assert state.cumulative_input_cost == 0.01
        assert state.cumulative_output_cost == 0.0
        assert state.cumulative_cost == 0.01

    def test_cost_after_output_tokens(self) -> None:
        state = StreamingCostState(
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        state.record_input(10)
        state.record_output_token()
        assert state.cumulative_output_cost == 0.002
        assert state.cumulative_cost == 0.012

    def test_cost_with_equal_rates(self) -> None:
        state = StreamingCostState(
            input_cost_per_token=0.0005,
            output_cost_per_token=0.0005,
        )
        state.record_input(100)
        for _ in range(50):
            state.record_output_token()
        assert state.cumulative_cost == pytest.approx(0.075)  # 100*0.0005 + 50*0.0005

    def test_cost_with_zero_rates(self) -> None:
        state = StreamingCostState()
        state.record_input(100)
        state.record_output_token()
        assert state.cumulative_cost == 0.0


# ---------------------------------------------------------------------------
# Cloud comparison / savings
# ---------------------------------------------------------------------------


class TestStreamingCostStateCloud:
    """Cloud cost comparison and savings."""

    def test_cloud_cost_comparison(self) -> None:
        state = StreamingCostState(
            input_cost_per_token=0.0001,
            output_cost_per_token=0.0002,
            cloud_input_cost_per_token=0.001,
            cloud_output_cost_per_token=0.002,
        )
        state.record_input(100)
        for _ in range(50):
            state.record_output_token()

        # Our cost: 100*0.0001 + 50*0.0002 = 0.01 + 0.01 = 0.02
        assert state.cumulative_cost == pytest.approx(0.02)
        # Cloud cost: 100*0.001 + 50*0.002 = 0.10 + 0.10 = 0.20
        assert state.cumulative_cloud_cost == pytest.approx(0.20)
        # Savings: 0.20 - 0.02 = 0.18
        assert state.cumulative_savings == pytest.approx(0.18)

    def test_zero_cloud_cost_no_savings(self) -> None:
        state = StreamingCostState(
            input_cost_per_token=0.0001,
            output_cost_per_token=0.0002,
        )
        state.record_input(100)
        assert state.cumulative_savings == 0.0

    def test_negative_savings_when_cloud_cheaper(self) -> None:
        state = StreamingCostState(
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            cloud_input_cost_per_token=0.0001,
            cloud_output_cost_per_token=0.0002,
        )
        state.record_input(100)
        # savings = cloud_cost - our_cost = negative
        assert state.cumulative_savings < 0


# ---------------------------------------------------------------------------
# Throughput / timing
# ---------------------------------------------------------------------------


class TestStreamingCostStateTiming:
    """TTFT and throughput metrics."""

    def test_ttft_recorded_on_first_token(self) -> None:
        state = StreamingCostState(start_time=time.time() - 0.1)  # started 100ms ago
        state.record_input(10)
        state.record_output_token()
        assert state.ttft_ms > 50  # should be ~100ms
        assert state.ttft_ms < 500  # sanity check

    def test_ttft_only_set_once(self) -> None:
        state = StreamingCostState(start_time=time.time())
        state.record_input(10)
        state.record_output_token()
        first_ttft = state.ttft_ms
        time.sleep(0.01)
        state.record_output_token()  # second token should not change ttft
        assert state.ttft_ms == first_ttft

    def test_tokens_per_second_needs_two_tokens(self) -> None:
        state = StreamingCostState(start_time=time.time())
        state.record_input(10)
        state.record_output_token()
        assert state.tokens_per_second == 0.0  # only 1 output token

    def test_tokens_per_second_after_multiple(self) -> None:
        state = StreamingCostState(start_time=time.time() - 1.0)  # started 1s ago
        state.record_input(10)
        state.record_output_token()
        state.record_output_token()
        state.record_output_token()
        assert state.tokens_per_second > 0
        assert state.avg_tokens_per_second > 0


# ---------------------------------------------------------------------------
# to_token_event / to_final_summary
# ---------------------------------------------------------------------------


class TestStreamingCostStateEventSummary:
    """Output format of cost event and summary."""

    def test_to_token_event_keys(self) -> None:
        state = StreamingCostState(input_cost_per_token=0.001, output_cost_per_token=0.002)
        state.record_input(100)
        state.record_output_token()
        event = state.to_token_event()
        assert "cost" in event
        assert "savings" in event
        assert "timing" in event
        assert "cumulative_usd" in event["cost"]
        assert "per_token_usd" in event["cost"]
        assert "tokens" in event["cost"]
        assert "tps" in event["cost"]
        assert "ttft_ms" in event["timing"]
        assert "elapsed_ms" in event["timing"]

    def test_to_final_summary_keys(self) -> None:
        state = StreamingCostState(input_cost_per_token=0.001, output_cost_per_token=0.002)
        state.record_input(100)
        for _ in range(5):
            state.record_output_token()
        summary = state.to_final_summary()
        assert summary["prompt_tokens"] == 100
        assert summary["completion_tokens"] == 5
        assert summary["total_tokens"] == 105
        assert "cost_usd" in summary
        assert "cost_per_token_usd" in summary
        assert "cloud_cost_usd" in summary
        assert "savings_usd" in summary
        assert "ttft_ms" in summary
        assert "total_duration_ms" in summary
        assert "avg_tokens_per_second" in summary


# ---------------------------------------------------------------------------
# StreamingCostTracker
# ---------------------------------------------------------------------------


class TestStreamingCostTrackerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        tracker = StreamingCostTracker()
        assert tracker._default_cost_per_token == 0.0
        assert tracker._active == {}
        assert tracker._completed == []

    def test_custom_default_cost(self) -> None:
        tracker = StreamingCostTracker(default_cost_per_token=0.0005)
        assert tracker._default_cost_per_token == 0.0005


class TestStreamingCostTrackerStart:
    """Start tracking a streaming request."""

    def test_start_tracking_basic(self) -> None:
        tracker = StreamingCostTracker()
        state = tracker.start_tracking(
            request_id="r1",
            input_tokens=100,
            model_name="gpt-4",
            cost_per_token=0.002,
        )
        assert state.request_id == "r1"
        assert state.input_tokens == 100
        assert state.output_cost_per_token == 0.002
        assert state.input_cost_per_token == 0.002  # falls back to cost_per_token
        assert state.cumulative_cost == 0.20  # 100 * 0.002

    def test_start_tracking_separate_rates(self) -> None:
        tracker = StreamingCostTracker()
        state = tracker.start_tracking(
            request_id="r1",
            input_tokens=100,
            model_name="gpt-4",
            input_cost_per_token=0.001,
            cost_per_token=0.003,  # output rate
            cloud_input_cost_per_token=0.01,
            cloud_cost_per_token=0.03,
        )
        assert state.input_cost_per_token == 0.001
        assert state.output_cost_per_token == 0.003
        assert state.cloud_input_cost_per_token == 0.01
        assert state.cloud_output_cost_per_token == 0.03
        assert state.cumulative_cost == 0.10  # 100 * 0.001

    def test_start_tracking_registers_active(self) -> None:
        tracker = StreamingCostTracker()
        tracker.start_tracking("r1", 100)
        assert "r1" in tracker._active
        assert tracker.get_active_state("r1") is not None

    def test_get_active_state_unknown(self) -> None:
        tracker = StreamingCostTracker()
        assert tracker.get_active_state("nonexistent") is None


class TestStreamingCostTrackerRecordToken:
    """Recording tokens during streaming."""

    def test_record_token(self) -> None:
        tracker = StreamingCostTracker()
        tracker.start_tracking("r1", 100, cost_per_token=0.001)
        event = tracker.record_token("r1")
        assert event is not None
        assert event["cost"]["tokens"] == 101

    def test_record_token_unknown_request(self) -> None:
        tracker = StreamingCostTracker()
        assert tracker.record_token("nonexistent") is None

    def test_record_token_updates_state(self) -> None:
        tracker = StreamingCostTracker()
        tracker.start_tracking("r1", 100, cost_per_token=0.001)
        for _ in range(3):
            tracker.record_token("r1")
        state = tracker.get_active_state("r1")
        assert state is not None
        assert state.output_tokens == 3


class TestStreamingCostTrackerFinish:
    """Finishing tracking."""

    def test_finish_tracking(self) -> None:
        tracker = StreamingCostTracker()
        tracker.start_tracking("r1", 100, cost_per_token=0.001)
        tracker.record_token("r1")
        summary = tracker.finish_tracking("r1")
        assert summary is not None
        assert summary["total_tokens"] == 101
        assert "cost_usd" in summary
        # Should be removed from active
        assert tracker.get_active_state("r1") is None

    def test_finish_tracking_unknown(self) -> None:
        tracker = StreamingCostTracker()
        assert tracker.finish_tracking("nonexistent") is None

    def test_finish_tracking_appends_to_completed(self) -> None:
        tracker = StreamingCostTracker()
        tracker.start_tracking("r1", 10)
        tracker.finish_tracking("r1")
        assert len(tracker._completed) == 1

    def test_completed_list_capped(self) -> None:
        tracker = StreamingCostTracker()
        for i in range(1500):
            tracker.start_tracking(f"r{i}", 1)
            tracker.finish_tracking(f"r{i}")
        # Should be trimmed to last 500
        assert len(tracker._completed) <= 1000


class TestStreamingCostTrackerGetStats:
    """Aggregate stats."""

    def test_get_stats_empty(self) -> None:
        tracker = StreamingCostTracker()
        stats = tracker.get_stats()
        assert stats["active_streams"] == 0
        assert stats["completed_count"] == 0
        assert stats["total_tokens_tracked"] == 0
        assert stats["total_cost_tracked"] == 0.0

    def test_get_stats_with_active_and_completed(self) -> None:
        tracker = StreamingCostTracker()
        tracker.start_tracking("active-1", 50, cost_per_token=0.001)
        tracker.record_token("active-1")
        tracker.start_tracking("done-1", 10, cost_per_token=0.001)
        tracker.record_token("done-1")
        tracker.finish_tracking("done-1")

        stats = tracker.get_stats()
        assert stats["active_streams"] == 1
        assert stats["completed_count"] == 1
        assert stats["total_tokens_tracked"] >= 11


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestStreamingCostTrackerThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_start_and_finish(self) -> None:
        tracker = StreamingCostTracker()
        errors: list[Exception] = []

        def run_request(rid: str) -> None:
            try:
                tracker.start_tracking(rid, 100, cost_per_token=0.001)
                for _ in range(5):
                    tracker.record_token(rid)
                tracker.finish_tracking(rid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_request, args=(f"r{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = tracker.get_stats()
        assert stats["completed_count"] == 20


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


class TestStreamingCostTrackerSingleton:
    """Module-level singleton helpers."""

    def test_get_singleton(self) -> None:
        reset_streaming_cost_tracker()
        t1 = get_streaming_cost_tracker()
        t2 = get_streaming_cost_tracker()
        assert t2 is t1

    def test_reset_singleton(self) -> None:
        reset_streaming_cost_tracker()
        t1 = get_streaming_cost_tracker()
        reset_streaming_cost_tracker()
        t2 = get_streaming_cost_tracker()
        assert t2 is not t1
