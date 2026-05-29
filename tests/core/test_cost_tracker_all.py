"""Comprehensive verification of all CostTracker fixes."""

import time

errors = []

def check(name, condition, msg=""):
    if not condition:
        errors.append(f"FAIL: {name} - {msg}")
        print(f"  FAIL: {name} - {msg}")
    else:
        print(f"  PASS: {name}")


# ── cost_tracker.py ──────────────────────────────────────────────────────────

from distllm.core.cost_tracker import (
    CostTracker, CostEstimate, CostBudget,
    get_cost_tracker, reset_cost_tracker,
    _estimate_throughput, _match_cloud_api,
    GPU_COST_PER_HOUR, CLOUD_API_COST_PER_M_TOKENS,
)

print("\n=== C6: Regex-based throughput estimation ===")
check("70b model", _estimate_throughput("A100-80GB", "llama-70b") == 2000 * 0.15)
check("8b model", _estimate_throughput("A100-80GB", "llama-3.1-8b") == 2000)
check("mixtral", _estimate_throughput("A100-80GB", "mixtral-8x7b") == 2000 * 0.25)
check("72b model", _estimate_throughput("H100", "qwen-72b") == 3000 * 0.15)
check("13b model", _estimate_throughput("H100", "mistral-13b") == 3000 * 0.5)
check("no size", _estimate_throughput("RTX-4090", "tiny-model") == 1200)
check("gpt-4o-70b", _estimate_throughput("H100", "gpt-4o-70b-custom") == 3000 * 0.15)

print("\n=== C7: Cloud API matching ===")
check("llama-70b", _match_cloud_api("llama-3.1-70b") == "llama-3.1-70b")
check("mistral-70b", _match_cloud_api("mistral-70b") == "llama-3.1-70b")
check("deepseek", _match_cloud_api("deepseek-v3") == "deepseek-v3")
check("qwen-72b", _match_cloud_api("qwen-72b") == "llama-3.1-70b")
check("claude-3.5", _match_cloud_api("claude-3.5-sonnet") == "claude-3.5-sonnet")
check("gpt-4o-mini", _match_cloud_api("gpt-4o-mini") == "gpt-4o-mini")
check("gpt-4o", _match_cloud_api("gpt-4o") == "gpt-4o")
check("haiku", _match_cloud_api("claude-3-haiku") == "claude-3-haiku")
check("unknown-7b", _match_cloud_api("unknown-7b") == "llama-3.1-8b")

print("\n=== C8: Updated pricing ===")
check("gpt-4o-mini", "gpt-4o-mini" in CLOUD_API_COST_PER_M_TOKENS)
check("claude-3.5-sonnet", "claude-3.5-sonnet" in CLOUD_API_COST_PER_M_TOKENS)
check("claude-3-haiku", "claude-3-haiku" in CLOUD_API_COST_PER_M_TOKENS)
check("deepseek-v3", "deepseek-v3" in CLOUD_API_COST_PER_M_TOKENS)

print("\n=== C1+C2: Period boundary resets ===")
ct = CostTracker()
ct.record_request("t1", 100, 50, 100)
hour_cost = ct._hourly_costs.get("t1")
check("hourly is tuple", isinstance(hour_cost, tuple))
check("hourly len 2", len(hour_cost) == 2 if hour_cost else False)
check("hourly cost > 0", hour_cost[0] > 0 if hour_cost else False)
check("hourly has period_start", hour_cost[1] > 0 if hour_cost else False)

print("\n=== C9: Running aggregates ===")
reset_cost_tracker()
ct = CostTracker()
ct.record_request("t1", 1000, 500, 1000)
ct.record_request("t1", 2000, 1000, 2000)
summary = ct.get_cost_summary()
check("total_requests", summary["total_requests_tracked"] == 2)
check("total_cost > 0", summary["total_cost_usd"] > 0)
check("avg_cost > 0", summary["avg_cost_per_request"] > 0)

# Performance check
start = time.perf_counter()
for _ in range(1000):
    ct.get_cost_summary()
elapsed = time.perf_counter() - start
check("O(1) summary perf", elapsed < 0.1, f"{elapsed*1000:.1f}ms for 1000 calls")

print("\n=== C10: Monthly budget check ===")
ct2 = CostTracker()
budget = CostBudget(max_cost_per_month=0.001)
ct2.set_budget("t2", budget)
ct2.record_request("t2", 10000, 5000, 10000)
allowed, reason = ct2.check_budget("t2", 1.0)
check("monthly budget blocks", not allowed, f"allowed={allowed}, reason={reason}")

print("\n=== C15: Cost validation ===")
ct3 = CostTracker()
est = ct3.estimate_cost(-100, -50)
check("negative input clamped", est.input_tokens == 0)
check("negative output clamped", est.output_tokens == 0)

print("\n=== C14: Singleton reset ===")
reset_cost_tracker()
t1 = get_cost_tracker()
reset_cost_tracker()
t2 = get_cost_tracker()
check("singleton reset", t1 is not t2)

print("\n=== estimate_cost accuracy ===")
ct4 = CostTracker(default_gpu_type="A100-80GB")
est = ct4.estimate_cost(1000, 500, "llama-70b")
# A100-80GB: $1.80/hr, ~300 tok/s for 70b (2000*0.15), 1500 tokens total
# GPU seconds = 1500 / 300 = 5s, cost = (5/3600) * 1.80 = $0.0025
check("cost > 0", est.estimated_cost_usd > 0)
check("cost reasonable", 0.001 < est.estimated_cost_usd < 0.01, f"${est.estimated_cost_usd:.6f}")
check("cloud api matched", est.cloud_api_name == "llama-3.1-70b")
check("cloud cost > 0", est.cloud_total_cost > 0)

print("\n=== get_cost_summary tenant ===")
ct5 = CostTracker()
ct5.record_request("tenant-a", 100, 50, 100)
ct5.record_request("tenant-b", 200, 100, 200)
s1 = ct5.get_cost_summary("tenant-a")
s2 = ct5.get_cost_summary("tenant-b")
check("tenant-a summary", s1["tenant_id"] == "tenant-a")
check("tenant-b summary", s2["tenant_id"] == "tenant-b")
check("tenant costs differ", s1["cost_last_hour"] != s2["cost_last_hour"])


# ── streaming_cost.py ────────────────────────────────────────────────────────

from distllm.core.streaming_cost import (
    StreamingCostTracker, StreamingCostState,
    get_streaming_cost_tracker, reset_streaming_cost_tracker,
)

print("\n=== C5+E6: Separate input/output cost rates ===")
sct = StreamingCostTracker()
state = sct.start_tracking(
    "req-1", 100, "llama-70b", "A100-80GB",
    cost_per_token=0.000002,
    cloud_cost_per_token=0.000001,
    input_cost_per_token=0.0000015,
    cloud_input_cost_per_token=0.0000005,
)
state.record_input(100)
state.record_output_token()
state.record_output_token()
state.record_output_token()

check("input cost rate", state.input_cost_per_token == 0.0000015)
check("output cost rate", state.output_cost_per_token == 0.000002)
check("input cost > 0", state.cumulative_input_cost > 0)
check("output cost > 0", state.cumulative_output_cost > 0)
check("total = input + output", abs(state.cumulative_cost - (state.cumulative_input_cost + state.cumulative_output_cost)) < 1e-15)

# Verify separate cloud rates
check("cloud input rate", state.cloud_input_cost_per_token == 0.0000005)
check("cloud output rate", state.cloud_output_cost_per_token == 0.000001)

print("\n=== StreamingCostTracker lifecycle ===")
sct2 = StreamingCostTracker()
state2 = sct2.start_tracking("req-2", 50, "model", "gpu", cost_per_token=0.001)
state2.record_input(50)
for _ in range(10):
    state2.record_output_token()
event = state2.to_token_event()
check("token event has cost", "cost" in event)
check("token event has savings", "savings" in event)
check("token event has timing", "timing" in event)

summary = sct2.finish_tracking("req-2")
check("finish returns summary", summary is not None)
check("summary has cost_usd", "cost_usd" in summary)
check("summary has savings_usd", "savings_usd" in summary)

# Stats
stats = sct2.get_stats()
check("stats has active_streams", "active_streams" in stats)
check("stats has total_cost_tracked", "total_cost_tracked" in stats)

print("\n=== StreamingCostTracker singleton ===")
reset_streaming_cost_tracker()
st1 = get_streaming_cost_tracker()
reset_streaming_cost_tracker()
st2 = get_streaming_cost_tracker()
check("singleton reset", st1 is not st2)


# ── cost_middleware.py ────────────────────────────────────────────────────────

from distllm.api.cost_middleware import _estimate_tokens

print("\n=== C3: Token estimation ===")
# tiktoken should give accurate results
text_hello = "Hello, how are you?"
est_hello = _estimate_tokens(text_hello)
check("hello tokens reasonable", 3 <= est_hello <= 8, f"got {est_hello}")

text_long = "This is a longer piece of text with multiple sentences. It should have more tokens."
est_long = _estimate_tokens(text_long)
check("long text tokens > hello", est_long > est_hello, f"hello={est_hello}, long={est_long}")

text_code = "def foo():\n    return 42\n\ndef bar():\n    return 'hello world'"
est_code = _estimate_tokens(text_code)
check("code tokens > 0", est_code > 0, f"got {est_code}")


# ── usage_meter.py ───────────────────────────────────────────────────────────

print("\n=== C11: SQLite WAL mode ===")
import tempfile
import sqlite3
with tempfile.TemporaryDirectory() as tmpdir:
    from distllm.core.usage_meter import UsageMeter
    db_path = f"{tmpdir}/test.db"
    meter = UsageMeter(storage_path=db_path, use_sqlite=True)
    # Check WAL mode was set
    result = meter._conn.execute("PRAGMA journal_mode").fetchone()
    check("WAL mode", result[0] == "wal", f"got {result[0]}")
    check("busy_timeout", meter._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000)
    meter._conn.close()


# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} errors")
    for e in errors:
        print(f"  {e}")
    raise AssertionError(f"{len(errors)} tests failed")
else:
    print("ALL TESTS PASSED")
