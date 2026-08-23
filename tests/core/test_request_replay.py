"""Tests for RequestReplayBuffer and DeterministicMode.

Covers:
- Construction with valid/invalid max_requests
- store and get operations
- LRU eviction
- list_recent ordering
- replay with handler function
- export and import_requests
- size and clear
- DeterministicMode enable/disable, context manager, seed application
- Singleton module-level helpers
"""

from __future__ import annotations

import time
import threading
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/request_replay.py")
RequestReplayBuffer = _mod.RequestReplayBuffer
DeterministicMode = _mod.DeterministicMode
StoredRequest = _mod.StoredRequest
get_replay_buffer = _mod.get_replay_buffer
get_deterministic_mode = _mod.get_deterministic_mode


# ---------------------------------------------------------------------------
# StoredRequest dataclass
# ---------------------------------------------------------------------------


class TestStoredRequest:
    """Dataclass StoredRequest."""

    def test_defaults(self) -> None:
        req = StoredRequest(request_id="r1", prompt="hello", params={"temp": 0.7})
        assert req.request_id == "r1"
        assert req.prompt == "hello"
        assert req.params == {"temp": 0.7}
        assert req.response is None
        assert req.error is None
        assert req.duration_ms == 0.0
        assert req.replay_count == 0
        assert req.model == ""

    def test_full_construction(self) -> None:
        req = StoredRequest(
            request_id="r1",
            prompt="hi",
            params={"top_p": 0.9},
            response="hello world",
            error=None,
            duration_ms=12.5,
            logprobs=[{"token": "hello", "logprob": -0.5}],
            generated_tokens=[101, 102],
            model="gpt-4",
        )
        assert req.response == "hello world"
        assert req.duration_ms == 12.5
        assert req.logprobs == [{"token": "hello", "logprob": -0.5}]
        assert req.generated_tokens == [101, 102]
        assert req.model == "gpt-4"


# ---------------------------------------------------------------------------
# RequestReplayBuffer construction
# ---------------------------------------------------------------------------


class TestRequestReplayBufferConstruction:
    """Construction with valid and invalid parameters."""

    def test_default_construction(self) -> None:
        buf = RequestReplayBuffer()
        assert buf._max == 100
        assert buf.size() == 0

    def test_custom_max_requests(self) -> None:
        buf = RequestReplayBuffer(max_requests=5)
        assert buf._max == 5

    def test_invalid_max_requests_raises(self) -> None:
        with pytest.raises(ValueError, match="max_requests must be >= 1"):
            RequestReplayBuffer(max_requests=0)
        with pytest.raises(ValueError, match="max_requests must be >= 1"):
            RequestReplayBuffer(max_requests=-1)


# ---------------------------------------------------------------------------
# RequestReplayBuffer store / get
# ---------------------------------------------------------------------------


class TestRequestReplayBufferStoreAndGet:
    """Store and retrieve requests."""

    def test_store_and_get(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "hello", {"temp": 0.7}, response="world")
        entry = buf.get("r1")
        assert entry is not None
        assert entry.request_id == "r1"
        assert entry.prompt == "hello"
        assert entry.params == {"temp": 0.7}
        assert entry.response == "world"

    def test_get_nonexistent(self) -> None:
        buf = RequestReplayBuffer()
        assert buf.get("nonexistent") is None

    def test_store_updates_existing(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "hello", {}, response="first")
        buf.store("r1", "hello", {}, response="second", duration_ms=5.0)
        entry = buf.get("r1")
        assert entry is not None
        assert entry.response == "second"
        assert entry.duration_ms == 5.0

    def test_store_advances_lru_position(self) -> None:
        buf = RequestReplayBuffer(max_requests=2)
        buf.store("r1", "a", {})
        buf.store("r2", "b", {})
        buf.store("r1", "a", {}, response="updated")  # refresh MRU
        buf.store("r3", "c", {})  # should evict r2 (LRU), not r1
        assert buf.get("r1") is not None
        assert buf.get("r2") is None
        assert buf.get("r3") is not None


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestRequestReplayBufferEviction:
    """LRU eviction behavior."""

    def test_evicts_oldest_when_full(self) -> None:
        buf = RequestReplayBuffer(max_requests=3)
        buf.store("r1", "a", {})
        buf.store("r2", "b", {})
        buf.store("r3", "c", {})
        buf.store("r4", "d", {})
        assert buf.get("r1") is None  # evicted
        assert buf.get("r2") is not None
        assert buf.get("r3") is not None
        assert buf.get("r4") is not None
        assert buf.size() == 3

    def test_get_refreshes_position(self) -> None:
        buf = RequestReplayBuffer(max_requests=3)
        buf.store("r1", "a", {})
        buf.store("r2", "b", {})
        buf.store("r3", "c", {})
        buf.get("r1")  # refresh r1
        buf.store("r4", "d", {})  # evicts r2, not r1
        assert buf.get("r1") is not None
        assert buf.get("r2") is None
        assert buf.get("r4") is not None

    def test_evicts_in_insertion_order_by_default(self) -> None:
        buf = RequestReplayBuffer(max_requests=2)
        buf.store("r1", "a", {})
        buf.store("r2", "b", {})
        buf.store("r3", "c", {})
        assert buf.get("r1") is None
        assert buf.get("r2") is not None
        assert buf.get("r3") is not None


# ---------------------------------------------------------------------------
# list_recent
# ---------------------------------------------------------------------------


class TestRequestReplayBufferListRecent:
    """list_recent ordering."""

    def test_list_recent_returns_n_most_recent(self) -> None:
        buf = RequestReplayBuffer()
        for i in range(5):
            buf.store(f"r{i}", str(i), {})
        recent = buf.list_recent(3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].request_id == "r4"
        assert recent[1].request_id == "r3"
        assert recent[2].request_id == "r2"

    def test_list_recent_n_gt_size(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "a", {})
        recent = buf.list_recent(10)
        assert len(recent) == 1
        assert recent[0].request_id == "r1"

    def test_list_recent_zero_returns_empty(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "a", {})
        assert buf.list_recent(0) == []

    def test_list_recent_empty_buffer(self) -> None:
        buf = RequestReplayBuffer()
        assert buf.list_recent(5) == []


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


class TestRequestReplayBufferReplay:
    """Replay behavior."""

    def test_replay_with_handler(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "hello", {"temp": 0.7})
        result = buf.replay("r1", handler=lambda prompt, params: prompt.upper())
        assert result == "HELLO"
        entry = buf.get("r1")
        assert entry is not None
        assert entry.replay_count == 1

    def test_replay_nonexistent(self) -> None:
        buf = RequestReplayBuffer()
        result = buf.replay("nonexistent", handler=lambda p, pa: p)
        assert result is None

    def test_replay_handler_exception_returns_none(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "hello", {})

        def failing(_p: str, _pa: dict[str, Any]) -> str:
            msg = "handler failed"
            raise RuntimeError(msg)

        result = buf.replay("r1", handler=failing)
        assert result is None

    def test_replay_increments_count(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "hello", {})
        buf.replay("r1", handler=lambda p, pa: p)
        buf.replay("r1", handler=lambda p, pa: p)
        entry = buf.get("r1")
        assert entry is not None
        assert entry.replay_count == 2


# ---------------------------------------------------------------------------
# export / import
# ---------------------------------------------------------------------------


class TestRequestReplayBufferExportImport:
    """Export and import requests."""

    def test_export_all(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "hello", {"temp": 0.7}, response="world", duration_ms=5.0, model="gpt-4")
        exported = buf.export()
        assert len(exported) == 1
        assert exported[0]["request_id"] == "r1"
        assert exported[0]["prompt"] == "hello"
        assert exported[0]["params"] == {"temp": 0.7}
        assert exported[0]["response"] == "world"
        assert exported[0]["model"] == "gpt-4"

    def test_export_specific_ids(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "a", {})
        buf.store("r2", "b", {})
        buf.store("r3", "c", {})
        exported = buf.export(request_ids=["r1", "r3"])
        assert len(exported) == 2
        exported_ids = {e["request_id"] for e in exported}
        assert exported_ids == {"r1", "r3"}

    def test_export_nonexistent_ids_skipped(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "a", {})
        exported = buf.export(request_ids=["r1", "nonexistent"])
        assert len(exported) == 1

    def test_import_requests(self) -> None:
        buf = RequestReplayBuffer(max_requests=10)
        data = [
            {"prompt": "hello", "params": {"temp": 0.7}, "response": "world", "model": "gpt-4"},
            {"prompt": "foo", "params": {}, "response": "bar", "duration_ms": 10.0},
        ]
        count = buf.import_requests(data)
        assert count == 2
        assert buf.size() == 2

    def test_import_adds_request_id_when_missing(self) -> None:
        buf = RequestReplayBuffer()
        data = [{"prompt": "test", "params": {}}]
        buf.import_requests(data)
        all_entries = buf.list_recent(10)
        assert len(all_entries) == 1
        assert all_entries[0].request_id != ""


# ---------------------------------------------------------------------------
# size / clear
# ---------------------------------------------------------------------------


class TestRequestReplayBufferSizeClear:
    """Size and clear operations."""

    def test_size(self) -> None:
        buf = RequestReplayBuffer()
        assert buf.size() == 0
        buf.store("r1", "a", {})
        assert buf.size() == 1
        buf.store("r2", "b", {})
        assert buf.size() == 2

    def test_clear(self) -> None:
        buf = RequestReplayBuffer()
        buf.store("r1", "a", {})
        buf.store("r2", "b", {})
        buf.clear()
        assert buf.size() == 0
        assert buf.get("r1") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestRequestReplayBufferThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_store(self) -> None:
        buf = RequestReplayBuffer(max_requests=1000)
        errors: list[Exception] = []

        def store_range(start: int, count: int) -> None:
            try:
                for i in range(count):
                    buf.store(f"r{start + i}", str(start + i), {})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=store_range, args=(0, 50)),
            threading.Thread(target=store_range, args=(50, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert buf.size() == 100

    def test_concurrent_get_does_not_raise(self) -> None:
        buf = RequestReplayBuffer(max_requests=100)
        buf.store("r1", "hello", {})

        def access() -> None:
            for _ in range(100):
                buf.get("r1")
                buf.list_recent(5)

        threads = [threading.Thread(target=access) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert buf.size() == 1


# ---------------------------------------------------------------------------
# DeterministicMode
# ---------------------------------------------------------------------------


class TestDeterministicMode:
    """DeterministicMode for reproducible debugging."""

    def test_default_disabled(self) -> None:
        dm = DeterministicMode(seed=42)
        assert dm.is_enabled is False

    def test_enable(self) -> None:
        dm = DeterministicMode(seed=42)
        dm.enable()
        assert dm.is_enabled is True

    def test_disable(self) -> None:
        dm = DeterministicMode(seed=42, enabled=True)
        dm.disable()
        assert dm.is_enabled is False

    def test_enable_with_custom_seed(self) -> None:
        dm = DeterministicMode(seed=42)
        dm.enable(seed=99)
        assert dm._seed == 99

    def test_enable_without_seed_keeps_current(self) -> None:
        dm = DeterministicMode(seed=42)
        dm.enable()
        assert dm._seed == 42

    def test_context_manager_enabled(self) -> None:
        dm = DeterministicMode(seed=42, enabled=True)
        before = dm.is_enabled
        with dm:
            inside = dm.is_enabled
        assert before is True
        assert inside is True

    def test_context_manager_disabled(self) -> None:
        dm = DeterministicMode(seed=42, enabled=False)
        with dm:
            assert dm.is_enabled is False

    def test_original_state_tracked(self) -> None:
        dm = DeterministicMode(seed=123)
        assert dm._original_state == {}
        assert dm._seed == 123


# ---------------------------------------------------------------------------
# Singleton module-level helpers
# ---------------------------------------------------------------------------


class TestSingletonHelpers:
    """Module-level get_replay_buffer and get_deterministic_mode."""

    def test_get_replay_buffer_singleton(self) -> None:
        # Clean up global state from previous tests
        _mod._replay_buffer = None
        buf1 = get_replay_buffer(max_requests=200)
        assert buf1._max == 200
        buf2 = get_replay_buffer(max_requests=300)  # second call returns existing
        assert buf2 is buf1
        assert buf2._max == 200  # not updated

    def test_get_deterministic_mode_singleton(self) -> None:
        _mod._deterministic_mode = None
        dm1 = get_deterministic_mode(seed=42, enabled=True)
        assert dm1.is_enabled is True
        dm2 = get_deterministic_mode(seed=99)
        assert dm2 is dm1
