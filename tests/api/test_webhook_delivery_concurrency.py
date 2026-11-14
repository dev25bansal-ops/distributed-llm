"""PERF-03 regression — webhook fan-out is bounded-concurrent, not sequential.

Before the fix, ``WebhookManager.dispatch`` ran each matched hook's blocking
``httpx.post`` inline in a for-loop, so dispatch wall time was
Σ(per-hook latency) — 10 slow hooks at the 30s timeout ≈ 300s, stalling
every later event.  The fix fans deliveries out on a bounded
``ThreadPoolExecutor`` (default 10 in flight) so wall time is ~max(latency)
instead of the sum, and adds ``dispatch_async`` that off-loads that fan-out
off the event loop.

Network is mocked (``httpx.post`` side-effect sleeps), so these tests prove
parallelism without DNS/servers, and pin the concurrency bound.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from distllm.api.webhooks.delivery import WebhookManager

# httpx.Response is only used for its status_code/text attributes.
class _Resp:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("DISTLLM_WEBHOOK_ALLOWLIST", raising=False)


def _hooks(mgr: WebhookManager, n: int) -> list[str]:
    return [
        mgr.register(f"https://hooks.example.com/{i}", {"job.completed"}, "sekret")
        for i in range(n)
    ]


class TestSequentialRegression:
    def test_n_slow_hooks_no_longer_sum(self):
        """5 hooks × 0.5s each must finish well under the 2.5s sequential sum.

        Sequential dispatch would take ≥2.5s; concurrent fan-out on the
        default pool (10 workers) takes ~0.5s.  The 2.0s ceiling gives CI
        headroom while still failing a sequential implementation.
        """
        mgr = WebhookManager()
        try:
            _hooks(mgr, 5)

            def slow_post(*args, **kwargs):
                time.sleep(0.5)
                return _Resp(200)

            with patch(
                "distllm.api.webhooks.delivery.httpx.post",
                side_effect=slow_post,
            ) as mock_post:
                start = time.monotonic()
                matched = mgr.dispatch("job.completed", {"job_id": "abc"})
                elapsed = time.monotonic() - start

            assert len(matched) == 5
            assert mock_post.call_count == 5
            assert elapsed < 2.0, f"fan-out took {elapsed:.2f}s — looks sequential"
            log = mgr.get_delivery_log()
            assert len(log) == 5
            assert all(d.status == "success" for d in log)
        finally:
            mgr.close()

    def test_failure_in_one_hook_does_not_block_others(self):
        """A failing hook is recorded, but concurrent peers still deliver."""
        mgr = WebhookManager()
        try:
            _hooks(mgr, 3)

            def flaky_post(url, **kwargs):
                if "/1" in str(url):
                    return _Resp(500, "boom")
                return _Resp(200)

            with patch(
                "distllm.api.webhooks.delivery.httpx.post",
                side_effect=flaky_post,
            ):
                mgr.dispatch("job.completed", {"job_id": "abc"})

            log = mgr.get_delivery_log()
            assert len(log) == 3
            statuses = sorted(d.status for d in log)
            assert statuses == ["failed", "success", "success"]
        finally:
            mgr.close()

    def test_concurrency_is_bounded(self):
        """In-flight POSTs never exceed the configured bound.

        20 hooks with a bound of 4: peak concurrency must be ≤4 while still
        completing all 20 deliveries.
        """
        mgr = WebhookManager(max_concurrent_deliveries=4)
        try:
            _hooks(mgr, 20)

            import threading

            lock = threading.Lock()
            state = {"in_flight": 0, "peak": 0}

            def counting_post(*args, **kwargs):
                with lock:
                    state["in_flight"] += 1
                    state["peak"] = max(state["peak"], state["in_flight"])
                time.sleep(0.1)
                with lock:
                    state["in_flight"] -= 1
                return _Resp(200)

            with patch(
                "distllm.api.webhooks.delivery.httpx.post",
                side_effect=counting_post,
            ) as mock_post:
                mgr.dispatch("job.completed", {"job_id": "abc"})

            assert mock_post.call_count == 20
            assert state["peak"] <= 4, f"peak concurrency {state['peak']} > bound 4"
            assert state["peak"] >= 2, "deliveries did not actually overlap"
            assert len(mgr.get_delivery_log()) == 20
        finally:
            mgr.close()

    def test_max_concurrent_must_be_positive(self):
        with pytest.raises(ValueError):
            WebhookManager(max_concurrent_deliveries=0)

    def test_close_is_idempotent(self):
        mgr = WebhookManager()
        _hooks(mgr, 2)
        with patch(
            "distllm.api.webhooks.delivery.httpx.post",
            return_value=_Resp(200),
        ):
            mgr.dispatch("job.completed", {"job_id": "abc"})
        mgr.close()
        mgr.close()  # second call must not raise
        # Manager still usable after close (pool re-created on demand).
        mgr2_close_needed = mgr._executor is None
        assert mgr2_close_needed


class TestDispatchAsync:
    @pytest.mark.asyncio
    async def test_async_fan_out_is_concurrent_and_nonblocking(self):
        """dispatch_async overlaps slow hooks off the event loop."""
        mgr = WebhookManager()
        try:
            _hooks(mgr, 5)

            def slow_post(*args, **kwargs):
                time.sleep(0.4)
                return _Resp(200)

            with patch(
                "distllm.api.webhooks.delivery.httpx.post",
                side_effect=slow_post,
            ):
                start = time.monotonic()
                matched = await mgr.dispatch_async(
                    "job.completed", {"job_id": "abc"}
                )
                elapsed = time.monotonic() - start

            assert len(matched) == 5
            # Sequential would be ≥2.0s; concurrent ~0.4s.
            assert elapsed < 1.6, f"async fan-out took {elapsed:.2f}s — sequential?"
            assert len(mgr.get_delivery_log()) == 5
        finally:
            mgr.close()

    @pytest.mark.asyncio
    async def test_async_no_matches_returns_empty(self):
        mgr = WebhookManager()
        try:
            _hooks(mgr, 2)
            with patch(
                "distllm.api.webhooks.delivery.httpx.post"
            ) as mock_post:
                matched = await mgr.dispatch_async("model.loaded", {})
            assert matched == []
            mock_post.assert_not_called()
        finally:
            mgr.close()
