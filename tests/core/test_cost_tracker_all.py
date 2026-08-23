"""Comprehensive tests for CostTracker, StreamingCostTracker, and usage metering.

Converted from script-style standalone test to proper pytest format.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time

import pytest

from distllm.core.cost_tracker import (
    GPU_COST_PER_HOUR,
    CLOUD_API_COST_PER_M_TOKENS,
    CostBudget,
    CostTracker,
    _estimate_throughput,
    _match_cloud_api,
    get_cost_tracker,
    reset_cost_tracker,
)


class TestCostTracker:
    """Cost estimation and tracking."""

    # ── C6: Regex-based throughput estimation ────────────────────────────

    @pytest.mark.parametrize("gpu,model,expected", [
        ("A100-80GB", "llama-70b", 2000 * 0.15),
        ("A100-80GB", "llama-3.1-8b", 2000),
        ("A100-80GB", "mixtral-8x7b", 2000 * 0.25),
        ("H100", "qwen-72b", 3000 * 0.15),
        ("H100", "mistral-13b", 3000 * 0.5),
        ("RTX-4090", "tiny-model", 1200),
        ("H100", "gpt-4o-70b-custom", 3000 * 0.15),
    ])
    def test_throughput_estimation(self, gpu: str, model: str, expected: float):
        assert _estimate_throughput(gpu, model) == expected

    # ── C7: Cloud API matching ───────────────────────────────────────────

    @pytest.mark.parametrize("model,expected", [
        ("llama-3.1-70b", "llama-3.1-70b"),
        ("mistral-70b", "llama-3.1-70b"),
        ("deepseek-v3", "deepseek-v3"),
        ("qwen-72b", "llama-3.1-70b"),
        ("claude-3.5-sonnet", "claude-3.5-sonnet"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-4o", "gpt-4o"),
        ("claude-3-haiku", "claude-3-haiku"),
        ("unknown-7b", "llama-3.1-8b"),
    ])
    def test_cloud_api_matching(self, model: str, expected: str):
        assert _match_cloud_api(model) == expected

    # ── C8: Updated pricing ──────────────────────────────────────────────

    def test_gpt4o_mini_in_pricing(self):
        assert "gpt-4o-mini" in CLOUD_API_COST_PER_M_TOKENS

    def test_claude_35_sonnet_in_pricing(self):
        assert "claude-3.5-sonnet" in CLOUD_API_COST_PER_M_TOKENS

    def test_claude_3_haiku_in_pricing(self):
        assert "claude-3-haiku" in CLOUD_API_COST_PER_M_TOKENS

    def test_deepseek_v3_in_pricing(self):
        assert "deepseek-v3" in CLOUD_API_COST_PER_M_TOKENS

    # ── C1+C2: Period boundary resets ────────────────────────────────────

    def test_hourly_cost_structure(self):
        ct = CostTracker()
        ct.record_request("t1", 100, 50, 100)
        hour_cost = ct._hourly_costs.get("t1")
        assert hour_cost is not None, "hourly cost should exist"
        assert isinstance(hour_cost, tuple), "should be a tuple"
        assert len(hour_cost) == 2, "should have (cost, period_start)"
        assert hour_cost[0] > 0, "cost should be positive"
        assert hour_cost[1] > 0, "period_start should be positive"

    # ── C9: Running aggregates ───────────────────────────────────────────

    def test_running_aggregates(self):
        reset_cost_tracker()
        ct = CostTracker()
        ct.record_request("t1", 1000, 500, 1000)
        ct.record_request("t1", 2000, 1000, 2000)
        summary = ct.get_cost_summary()
        assert summary["total_requests_tracked"] == 2
        assert summary["total_cost_usd"] > 0
        assert summary["avg_cost_per_request"] > 0

    def test_O1_summary_performance(self):
        ct = CostTracker()
        start = time.perf_counter()
        for _ in range(1000):
            ct.get_cost_summary()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"{elapsed*1000:.1f}ms for 1000 calls (threshold 100ms)"

    # ── C10: Monthly budget check ────────────────────────────────────────

    def test_monthly_budget_blocks(self):
        ct = CostTracker()
        budget = CostBudget(max_cost_per_month=0.001)
        ct.set_budget("t2", budget)
        ct.record_request("t2", 10000, 5000, 10000)
        allowed, reason = ct.check_budget("t2", 1.0)
        assert not allowed, f"budget should block, reason={reason}"

    # ── C15: Cost validation ─────────────────────────────────────────────

    def test_negative_input_clamped(self):
        ct = CostTracker()
        est = ct.estimate_cost(-100, -50)
        assert est.input_tokens == 0, "negative input should clamp to 0"
        assert est.output_tokens == 0, "negative output should clamp to 0"

    # ── C14: Singleton reset ─────────────────────────────────────────────

    def test_singleton_reset(self):
        reset_cost_tracker()
        t1 = get_cost_tracker()
        reset_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is not t2

    # ── estimate_cost accuracy ───────────────────────────────────────────

    def test_estimate_cost_reasonable(self):
        ct = CostTracker(default_gpu_type="A100-80GB")
        est = ct.estimate_cost(1000, 500, "llama-70b")
        assert est.estimated_cost_usd > 0
        assert 0.001 < est.estimated_cost_usd < 0.01, f"${est.estimated_cost_usd:.6f}"
        assert est.cloud_api_name == "llama-3.1-70b"
        assert est.cloud_total_cost > 0

    # ── Tenant isolation ─────────────────────────────────────────────────

    def test_tenant_isolation(self):
        ct = CostTracker()
        ct.record_request("tenant-a", 100, 50, 100)
        ct.record_request("tenant-b", 200, 100, 200)
        s1 = ct.get_cost_summary("tenant-a")
        s2 = ct.get_cost_summary("tenant-b")
        assert s1["tenant_id"] == "tenant-a"
        assert s2["tenant_id"] == "tenant-b"
        assert s1["cost_last_hour"] != s2["cost_last_hour"]


class TestStreamingCostTracker:
    """Streaming cost tracking and lifecycle."""

    def test_input_output_cost_rates(self):
        from distllm.core.streaming_cost import (
            StreamingCostTracker,
            get_streaming_cost_tracker,
            reset_streaming_cost_tracker,
        )
        sct = StreamingCostTracker()
        state = sct.start_tracking(
            "req-1", 100, "llama-70b", "A100-80GB",
            cost_per_token=0.000002,
            cloud_cost_per_token=0.000001,
            input_cost_per_token=0.0000015,
            cloud_input_cost_per_token=0.0000005,
        )
        state.record_input(100)
        for _ in range(3):
            state.record_output_token()

        assert state.input_cost_per_token == 0.0000015
        assert state.output_cost_per_token == 0.000002
        assert state.cumulative_input_cost > 0
        assert state.cumulative_output_cost > 0
        assert abs(state.cumulative_cost - (state.cumulative_input_cost + state.cumulative_output_cost)) < 1e-15
        assert state.cloud_input_cost_per_token == 0.0000005
        assert state.cloud_output_cost_per_token == 0.000001

    def test_full_lifecycle(self):
        from distllm.core.streaming_cost import StreamingCostTracker
        sct = StreamingCostTracker()
        state = sct.start_tracking("req-2", 50, "model", "gpu", cost_per_token=0.001)
        state.record_input(50)
        for _ in range(10):
            state.record_output_token()
        event = state.to_token_event()
        assert "cost" in event
        assert "savings" in event
        assert "timing" in event

        summary = sct.finish_tracking("req-2")
        assert summary is not None
        assert "cost_usd" in summary
        assert "savings_usd" in summary

        stats = sct.get_stats()
        assert "active_streams" in stats
        assert "total_cost_tracked" in stats

    def test_singleton_reset(self):
        from distllm.core.streaming_cost import (
            get_streaming_cost_tracker,
            reset_streaming_cost_tracker,
        )
        reset_streaming_cost_tracker()
        st1 = get_streaming_cost_tracker()
        reset_streaming_cost_tracker()
        st2 = get_streaming_cost_tracker()
        assert st1 is not st2


class TestTokenEstimation:
    """Token estimation middleware."""

    def test_hello_tokens_reasonable(self):
        from distllm.api.cost_middleware import _estimate_tokens
        est = _estimate_tokens("Hello, how are you?")
        assert 3 <= est <= 8, f"got {est}"

    def test_long_text_more_tokens(self):
        from distllm.api.cost_middleware import _estimate_tokens
        short = _estimate_tokens("Hello, how are you?")
        long_text = (
            "This is a longer piece of text with multiple sentences. "
            "It should have more tokens."
        )
        long_est = _estimate_tokens(long_text)
        assert long_est > short, f"short={short}, long={long_est}"

    def test_code_tokens_positive(self):
        from distllm.api.cost_middleware import _estimate_tokens
        code = "def foo():\n    return 42\n\ndef bar():\n    return 'hello world'"
        est = _estimate_tokens(code)
        assert est > 0, f"got {est}"


class TestUsageMeter:
    """SQLite-backed usage metering."""

    def test_wal_mode_enabled(self):
        import os
        from distllm.core.usage_meter import UsageMeter
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            meter = UsageMeter(storage_path=db_path, use_sqlite=True)
            result = meter._conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0] == "wal", f"got {result[0]}"
            assert meter._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
            meter._conn.close()
