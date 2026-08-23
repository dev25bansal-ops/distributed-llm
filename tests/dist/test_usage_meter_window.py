"""Regression: UsageMeter time-window queries must not under-count after the
in-memory cap is hit (F-024).

Once ``_max_records`` is full, new records spill to SQLite only; ``get_usage``
must query the DB window so billing/quota aggregation stays correct.
"""

from __future__ import annotations

import time

import pytest

from distllm.dist.daas.usage_meter import UsageMeter


def test_window_query_counts_records_after_cap(tmp_path):
    meter = UsageMeter(db_path=tmp_path / "usage.db")
    meter._max_records = 5  # small cap to force spill quickly

    # Fill the in-memory list to the cap.
    for _ in range(5):
        meter.record_usage("t1", prompt_tokens=1, completion_tokens=1, duration_ms=1.0)

    # These exceed the cap → DB only.
    meter.record_usage("t1", prompt_tokens=10, completion_tokens=20, duration_ms=30.0)

    # Window query must count BOTH the in-memory and the DB-spilled records.
    agg = meter.get_usage("t1", since_timestamp=time.time() - 3600)
    assert agg is not None
    assert agg.request_count == 6, "records that spilled to DB must still be counted"
    assert agg.prompt_tokens == 5 + 10
    assert agg.completion_tokens == 5 + 20


def test_window_query_no_spill_still_works(tmp_path):
    meter = UsageMeter(db_path=tmp_path / "usage.db")
    meter._max_records = 100_000
    meter.record_usage("t1", prompt_tokens=7, completion_tokens=3, duration_ms=5.0)

    agg = meter.get_usage("t1", since_timestamp=time.time() - 3600)
    assert agg is not None
    assert agg.request_count == 1
    assert agg.prompt_tokens == 7


def test_lifetime_usage_aggregates_db_at_load(tmp_path):
    meter = UsageMeter(db_path=tmp_path / "usage.db")
    meter.record_usage("t1", prompt_tokens=2, completion_tokens=3, duration_ms=4.0)
    meter.close()

    # Re-open from the DB: lifetime aggregate must reflect persisted records.
    meter2 = UsageMeter(db_path=tmp_path / "usage.db")
    agg = meter2.get_usage("t1")
    assert agg is not None
    assert agg.request_count == 1
    assert agg.prompt_tokens == 2
