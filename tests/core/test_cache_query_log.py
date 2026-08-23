"""Tests for CacheQueryLogger (structured JSONL cache audit trail)."""

import json
import tempfile
from pathlib import Path

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/cache_query_log.py")
CacheQueryLogger = _mod.CacheQueryLogger
CacheQueryLog = _mod.CacheQueryLog


class TestCacheQueryLog:
    def test_defaults(self):
        entry = CacheQueryLog(request_id="req-1")
        assert entry.request_id == "req-1"
        assert entry.operation == "lookup"
        assert entry.tokens_prefix == ""
        assert entry.hit is False


class TestCacheQueryLogger:
    def test_init_no_path(self):
        log = CacheQueryLogger()
        assert log._file is None
        assert log._entries == []

    def test_log_lookup(self):
        log = CacheQueryLogger()
        log.log_lookup("req-1", [1, 2, 3], hit=True, match_length=2, latency_ms=1.5)
        assert len(log._entries) == 1
        entry = log._entries[0]
        assert entry.request_id == "req-1"
        assert entry.hit is True
        assert entry.match_length == 2
        assert entry.total_latency_ms == 1.5
        assert entry.operation == "lookup"

    def test_log_lookup_miss(self):
        log = CacheQueryLogger()
        log.log_lookup("req-2", [10, 20], hit=False)
        entry = log._entries[0]
        assert entry.hit is False
        assert entry.match_length == 0

    def test_log_store(self):
        log = CacheQueryLogger()
        log.log_store("req-3", [1, 2, 3], tier="gpu")
        entry = log._entries[0]
        assert entry.operation == "store"
        assert entry.hit is True
        assert entry.match_length == 3
        assert entry.tier_hits == ["gpu"]

    def test_log_evict(self):
        log = CacheQueryLogger()
        log.log_evict("req-4", [5, 6, 7], tier="ssd", reason="lru")
        entry = log._entries[0]
        assert entry.operation == "evict"
        assert entry.hit is False
        assert entry.metadata["reason"] == "lru"

    def test_get_entries_filter_by_operation(self):
        log = CacheQueryLogger()
        log.log_lookup("r1", [1], hit=True)
        log.log_store("r2", [2])
        log.log_evict("r3", [3])
        lookups = log.get_entries(operation="lookup")
        assert len(lookups) == 1
        assert lookups[0].request_id == "r1"

    def test_get_entries_hit_only(self):
        log = CacheQueryLogger()
        log.log_lookup("r1", [1], hit=True)
        log.log_lookup("r2", [2], hit=False)
        hits = log.get_entries(hit_only=True)
        assert len(hits) == 1
        assert hits[0].request_id == "r1"

    def test_get_entries_limit(self):
        log = CacheQueryLogger()
        for i in range(10):
            log.log_lookup(f"r{i}", [i], hit=True)
        entries = log.get_entries(limit=3)
        assert len(entries) == 3

    def test_get_stats_empty(self):
        log = CacheQueryLogger()
        stats = log.get_stats()
        assert stats["total_operations"] == 0

    def test_get_stats_with_data(self):
        log = CacheQueryLogger()
        log.log_lookup("r1", [1], hit=True, latency_ms=2.0, match_length=5)
        log.log_lookup("r2", [2], hit=False, latency_ms=3.0)
        log.log_store("r3", [3])
        stats = log.get_stats()
        assert stats["total_operations"] == 3
        assert stats["lookups"] == 2
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["avg_latency_ms"] == 2.5

    def test_context_manager(self):
        with CacheQueryLogger() as log:
            log.log_lookup("r1", [1], hit=True)
        assert len(log._entries) == 1

    def test_max_entries_trim(self):
        log = CacheQueryLogger(max_entries=3)
        for i in range(5):
            log.log_lookup(f"r{i}", [i], hit=True)
        assert len(log._entries) == 3
        assert log._entries[-1].request_id == "r4"

    def test_file_logging(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            log_path = tmp.name

        log = CacheQueryLogger(log_path=log_path)
        log.log_lookup("r1", [1, 2], hit=True)
        log.close()

        with open(log_path) as f:
            line = f.readline()
            data = json.loads(line)
        assert data["request_id"] == "r1"
        assert data["hit"] is True
        assert data["operation"] == "lookup"
        Path(log_path).unlink(missing_ok=True)
