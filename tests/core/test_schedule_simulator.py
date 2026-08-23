"""Tests for schedule simulator: TraceEntry, SimulationResult, simulate, load/save.

Uses the import-helper pattern to avoid circular imports.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_sim_mod = load_module("distllm/core/schedule_simulator.py")
TraceEntry = _sim_mod.TraceEntry
SimulationResult = _sim_mod.SimulationResult
load_trace = _sim_mod.load_trace
save_trace = _sim_mod.save_trace
simulate = _sim_mod.simulate


class TestTraceEntry:
    def test_defaults(self):
        entry = TraceEntry(request_id="r1", arrival_time=0.0, prompt_tokens=100, max_new_tokens=128)
        assert entry.priority == 2
        assert entry.max_latency_ms is None

    def test_from_dict(self):
        d = {"request_id": "r1", "arrival_time": 1.5, "prompt_tokens": 200, "max_new_tokens": 64, "priority": 0}
        entry = TraceEntry.from_dict(d)
        assert entry.request_id == "r1"
        assert entry.arrival_time == 1.5
        assert entry.prompt_tokens == 200
        assert entry.max_new_tokens == 64
        assert entry.priority == 0

    def test_from_dict_with_max_latency(self):
        d = {"request_id": "r1", "arrival_time": 0.0, "prompt_tokens": 100, "max_new_tokens": 50, "max_latency_ms": 500.0}
        entry = TraceEntry.from_dict(d)
        assert entry.max_latency_ms == 500.0

    def test_from_dict_missing_fields(self):
        entry = TraceEntry.from_dict({"request_id": "r1"})
        assert entry.arrival_time == 0
        assert entry.prompt_tokens == 100


class TestSimulationResult:
    def test_defaults(self):
        r = SimulationResult()
        assert r.total_requests == 0
        assert r.completed_requests == 0
        assert r.preempted_count == 0
        assert r.avg_wait_time_ms == 0.0
        assert r.config_used == {}

    def test_summary_includes_fields(self):
        r = SimulationResult(
            total_requests=10, completed_requests=8,
            preempted_count=1, avg_wait_time_ms=50.0,
            throughput_tokens_per_sec=1200.0,
        )
        summary = r.summary()
        assert "Total requests:" in summary
        assert "10" in summary
        assert "Completed:" in summary
        assert "8" in summary
        assert "Throughput:" in summary
        assert "1200" in summary


class TestSaveAndLoadTrace:
    def test_save_trace(self):
        entries = [
            TraceEntry("r1", 0.0, 100, 128),
            TraceEntry("r2", 1.0, 50, 64, priority=1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            save_trace(entries, str(path))
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data["requests"]) == 2
            assert data["requests"][0]["request_id"] == "r1"

    def test_load_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            data = {
                "requests": [
                    {"request_id": "r1", "arrival_time": 0.0, "prompt_tokens": 100, "max_new_tokens": 128},
                ]
            }
            path.write_text(json.dumps(data))
            entries = load_trace(str(path))
            assert len(entries) == 1
            assert entries[0].request_id == "r1"

    def test_load_trace_plain_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            data = [
                {"request_id": "r1", "arrival_time": 0.0, "prompt_tokens": 50, "max_new_tokens": 100},
            ]
            path.write_text(json.dumps(data))
            entries = load_trace(str(path))
            assert len(entries) == 1

    def test_load_trace_sort_by_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            data = {
                "requests": [
                    {"request_id": "r2", "arrival_time": 5.0, "prompt_tokens": 50, "max_new_tokens": 100},
                    {"request_id": "r1", "arrival_time": 1.0, "prompt_tokens": 50, "max_new_tokens": 100},
                ]
            }
            path.write_text(json.dumps(data))
            entries = load_trace(str(path))
            assert entries[0].request_id == "r1"
            assert entries[1].request_id == "r2"


class TestSimulate:
    def test_empty_trace(self):
        result = simulate([], max_batch_size=32)
        assert result.total_requests == 0
        assert result.completed_requests == 0
        assert result.total_iterations == 0

    def test_single_request(self):
        trace = [TraceEntry("r1", 0.0, prompt_tokens=10, max_new_tokens=5)]
        result = simulate(trace, max_batch_size=32)
        assert result.total_requests == 1
        assert result.completed_requests == 1
        assert result.total_iterations > 0

    def test_two_requests(self):
        trace = [
            TraceEntry("r1", 0.0, prompt_tokens=10, max_new_tokens=5),
            TraceEntry("r2", 0.1, prompt_tokens=10, max_new_tokens=5),
        ]
        result = simulate(trace, max_batch_size=32)
        assert result.total_requests == 2
        assert result.completed_requests == 2

    def test_config_passed_through(self):
        trace = [TraceEntry("r1", 0.0, prompt_tokens=5, max_new_tokens=5)]
        result = simulate(trace, max_batch_size=16, max_tokens_per_batch=16384)
        assert result.config_used["max_batch_size"] == 16
        assert result.config_used["max_tokens_per_batch"] == 16384

    def test_aging_disabled(self):
        trace = [TraceEntry("r1", 0.0, prompt_tokens=10, max_new_tokens=5)]
        result = simulate(trace, aging_enabled=False)
        assert result.config_used["aging_enabled"] is False

    def test_chunked_prefill_disabled(self):
        trace = [TraceEntry("r1", 0.0, prompt_tokens=10, max_new_tokens=5)]
        result = simulate(trace, enable_chunked_prefill=False)
        assert result.config_used["enable_chunked_prefill"] is False
