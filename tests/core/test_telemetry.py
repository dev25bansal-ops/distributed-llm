"""Regression tests for the telemetry collector.

Covers B20: ``TelemetryCollector._add_event`` held a non-reentrant
``threading.Lock`` and called ``flush()`` at the ``BATCH_SIZE`` boundary,
while ``flush()`` re-acquired the same lock on the same thread — a
guaranteed deadlock whenever telemetry was enabled and enough events
accumulated.
"""

from __future__ import annotations

import threading
from pathlib import Path

from distllm.core.telemetry import TelemetryCollector


def _make_collector(tmp_path: Path) -> TelemetryCollector:
    return TelemetryCollector(enabled=True, data_dir=str(tmp_path))


def _record_requests(collector: TelemetryCollector, count: int) -> None:
    for _ in range(count):
        collector.record_request(model_size_b=1, tokens=10, latency_ms=5)


def test_no_deadlock_past_batch_size(tmp_path):
    """Recording > BATCH_SIZE events must not deadlock (B20 regression)."""
    collector = _make_collector(tmp_path)
    started = threading.Event()

    def record() -> None:
        started.set()
        _record_requests(collector, collector.BATCH_SIZE * 3)

    thread = threading.Thread(target=record, name="telemetry-recorder")
    thread.start()
    assert started.wait(timeout=5)
    thread.join(timeout=10)

    assert not thread.is_alive(), (
        "recorder thread deadlocked after BATCH_SIZE events: flush() "
        "re-acquired the non-reentrant lock held by _add_event"
    )
    # Auto-flush at each BATCH_SIZE boundary consumes every event.
    assert collector._events == []


def test_concurrent_record_no_deadlock(tmp_path):
    """Concurrent recording threads must not deadlock or lose events."""
    collector = _make_collector(tmp_path)
    num_threads = 4
    per_thread = collector.BATCH_SIZE * 2
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            _record_requests(collector, per_thread)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, name=f"telemetry-worker-{i}")
        for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert all(not t.is_alive() for t in threads), "a worker thread deadlocked"
    assert not errors

    collector.flush()
    total = sum(
        len(f.read_text().splitlines())
        for f in tmp_path.glob("events_*.jsonl")
    )
    assert total == num_threads * per_thread


def test_flush_writes_events_to_disk(tmp_path):
    """flush() persists pending events as a JSONL file."""
    collector = _make_collector(tmp_path)
    collector.record_request(model_size_b=1, tokens=10, latency_ms=5)
    collector.record_feature("kv_cache")
    collector.flush()

    files = list(tmp_path.glob("events_*.jsonl"))
    assert len(files) == 1
    lines = [ln for ln in files[0].read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        assert '"instance"' in line
        assert '"type"' in line

    # Events are drained after a flush.
    assert collector._events == []


def test_flush_with_no_pending_events(tmp_path):
    """flush() with an empty queue is a no-op and creates no file."""
    collector = _make_collector(tmp_path)
    collector.flush()
    assert list(tmp_path.glob("events_*.jsonl")) == []
