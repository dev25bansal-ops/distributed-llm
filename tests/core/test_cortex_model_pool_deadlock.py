"""Regression tests for the ModelPool LRU-eviction deadlock (audit C8).

Root cause: ``ModelPool._evict_lru`` is always called while holding
``self._lock``, but it read memory usage through the ``total_memory_bytes``
property, which re-acquires that same non-reentrant ``threading.Lock``.
Any eviction check -- including the very first ``load()`` on a fresh pool --
therefore hung forever.

The fix tracks the running total locally inside ``_evict_lru`` instead of
calling the lock-acquiring property.  These tests run every eviction entry
point in a watchdog thread and fail if it does not complete quickly.
"""

from __future__ import annotations

import threading

import pytest

from distllm.core.cortex_multimodel import ModelPool


TIMEOUT = 10.0


def _run_with_watchdog(operation, timeout: float = TIMEOUT):
    """Run *operation* in a thread; fail the test if it exceeds *timeout*."""
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = operation()
        except BaseException as exc:  # noqa: BLE001 - re-raised in main thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        pytest.fail(
            f"operation did not finish within {timeout}s "
            "(lock re-acquisition deadlock)"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class TestModelPoolEvictionDeadlock:
    """Every path that enters _evict_lru under _lock must terminate."""

    def test_first_load_triggers_eviction_check_without_hanging(self):
        """Fresh pool + oversized model exercises the exact deadlock repro.

        Before the fix this hung forever on the first property read inside
        ``_evict_lru``.
        """
        pool = ModelPool(max_memory_bytes=1024)
        model_id = _run_with_watchdog(
            lambda: pool.load("meta-llama/Llama-3-7B", memory_bytes=2048)
        )
        assert model_id
        # Budget exceeded but pool was empty -> nothing to evict; load succeeds.
        assert len(pool.list_loaded()) == 1

    def test_load_forces_real_eviction_without_hanging(self):
        """Second oversized load must evict the LRU resident and complete."""
        pool = ModelPool(max_memory_bytes=2048)
        first_id = pool.load("model-a", memory_bytes=1500)
        second_id = _run_with_watchdog(
            lambda: pool.load("model-b", memory_bytes=1500)
        )

        handles = {h.model_id: h.model_name for h in pool.list_loaded()}
        # model-a had to be evicted to fit model-b.
        assert first_id not in handles
        assert handles.get(second_id) == "model-b"

    def test_direct_evict_lru_under_lock_terminates(self):
        """Direct call to _evict_lru while holding the documented lock.

        This mirrors the internal contract ('must be called while holding
        _lock') that used to self-deadlock via the property getter.
        """
        pool = ModelPool(max_memory_bytes=200)
        pool.load("m1", memory_bytes=80)
        pool.load("m2", memory_bytes=80)

        def evict_under_lock() -> None:
            with pool._lock:
                # 160 resident + 50 incoming > 200 -> must evict LRU (m1).
                pool._evict_lru(50)

        _run_with_watchdog(evict_under_lock)
        assert [h.model_name for h in pool.list_loaded()] == ["m2"]
        assert pool.total_memory_bytes == 80

    def test_max_memory_setter_eviction_path_without_hanging(self):
        """Shrinking max_memory_bytes triggers _evict_lru(0) under the lock."""
        pool = ModelPool(max_memory_bytes=4096)
        pool.load("big-model", memory_bytes=3000)
        pool.load("small-model", memory_bytes=500)

        def shrink() -> None:
            pool.max_memory_bytes = 1000

        _run_with_watchdog(shrink)
        remaining = [h.model_name for h in pool.list_loaded()]
        assert "small-model" in remaining
        assert sum(h.memory_bytes for h in pool.list_loaded()) <= 1000

    def test_public_total_memory_property_still_works(self):
        """The property itself remains usable from external threads."""
        pool = ModelPool(max_memory_bytes=4096)
        pool.load("a", memory_bytes=111)
        pool.load("b", memory_bytes=222)
        assert _run_with_watchdog(lambda: pool.total_memory_bytes) == 333

    def test_repeated_load_unload_cycles_stay_live(self):
        """Hammer the pool so a latent reentrancy bug would resurface."""
        pool = ModelPool(max_memory_bytes=512)

        def churn() -> None:
            ids = [
                pool.load(f"m{i}", memory_bytes=300) for i in range(20)
            ]
            # Only the most recent model fits the budget; earlier ids were
            # evicted, so unload just those still resident.
            resident = {h.model_id for h in pool.list_loaded()}
            for mid in ids:
                if mid in resident:
                    pool.unload(mid)

        _run_with_watchdog(churn)
        assert len(pool.list_loaded()) <= 2  # 512-byte budget, 300-byte models
        assert pool.total_memory_bytes == sum(
            h.memory_bytes for h in pool.list_loaded()
        )
