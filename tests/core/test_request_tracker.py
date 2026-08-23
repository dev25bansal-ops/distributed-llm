"""Tests for RequestTracker -- async request result tracking.

Covers:
- Construction and initial state
- register_request and has_request
- set_result and wait_for_result
- set_error propagates as RuntimeError
- set_logprobs and get_logprobs
- cancel with pending request
- pending_count
- complete_batch_requests

No MagicMock -- real threading.Event and dict state.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/request_tracker.py")
RequestTracker = _mod.RequestTracker


class _StubTokenizer:
    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return "decoded:" + "".join(chr(t) if 32 <= t < 127 else "?" for t in token_ids)


class _StubSequence:
    def __init__(self, tokens: list[int] | None = None) -> None:
        self.generated_tokens = tokens or []
        self.request_id = "seq-1"


class TestRequestTrackerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        tracker = RequestTracker()
        assert tracker._results == {}
        assert tracker._events == {}
        assert tracker._logprobs == {}
        assert tracker._errors == {}

    def test_pending_count_starts_at_zero(self) -> None:
        tracker = RequestTracker()
        assert tracker.pending_count() == 0


class TestRequestTrackerRegister:
    """Registration and query."""

    def test_register_request_creates_event(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        assert tracker.has_request("req-1") is True
        assert "req-1" in tracker._events
        assert isinstance(tracker._events["req-1"], threading.Event)

    def test_has_request_returns_false_for_unknown(self) -> None:
        tracker = RequestTracker()
        assert tracker.has_request("nonexistent") is False

    def test_pending_count_after_register(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.register_request("req-2")
        assert tracker.pending_count() == 2


class TestRequestTrackerSetResult:
    """Set and wait for results."""

    def test_set_result_and_wait(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_result("req-1", "Hello, world!")
        result = tracker.wait_for_result("req-1", timeout=1.0)
        assert result == "Hello, world!"

    def test_wait_for_unknown_request_raises(self) -> None:
        tracker = RequestTracker()
        with pytest.raises(ValueError):
            tracker.wait_for_result("nonexistent")

    def test_wait_for_result_timeout_raises(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        with pytest.raises(TimeoutError):
            tracker.wait_for_result("req-1", timeout=0.01)

    def test_wait_for_result_cleans_up(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_result("req-1", "done")
        tracker.wait_for_result("req-1", timeout=1.0)
        # After wait, the event should be cleaned up
        assert tracker.pending_count() == 0


class TestRequestTrackerSetError:
    """Error handling."""

    def test_set_error_propagates_on_wait(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_error("req-1", ValueError("model crashed"))
        with pytest.raises(RuntimeError, match="model crashed"):
            tracker.wait_for_result("req-1", timeout=1.0)

    def test_set_error_wakes_waiters(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_error("req-1", RuntimeError("timeout"))
        result = tracker._errors.pop("req-1", None)
        assert result is not None


class TestRequestTrackerLogprobs:
    """Logprobs storage."""

    def test_set_and_get_logprobs(self) -> None:
        tracker = RequestTracker()
        tracker.set_logprobs("req-1", {"token_0": {"logprob": -0.5}})
        lp = tracker.get_logprobs("req-1")
        assert lp is not None
        assert lp["token_0"]["logprob"] == -0.5

    def test_get_logprobs_nonexistent(self) -> None:
        tracker = RequestTracker()
        assert tracker.get_logprobs("nonexistent") is None


class TestRequestTrackerCancel:
    """Cancel method."""

    def test_cancel_returns_true_for_existing(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        assert tracker.cancel("req-1") is True

    def test_cancel_returns_false_for_nonexistent(self) -> None:
        tracker = RequestTracker()
        assert tracker.cancel("nonexistent") is False

    def test_cancel_removes_request(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.cancel("req-1")
        assert tracker.has_request("req-1") is False

    def test_cancel_wakes_waiters(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.cancel("req-1")
        result = tracker._results.get("req-1")
        assert result is not None


class TestRequestTrackerCompleteBatch:
    """complete_batch_requests method."""

    def test_complete_active_sequence_with_tokens(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("seq-1")
        seq = _StubSequence(tokens=[72, 101, 108, 108, 111])
        active_seqs = {"seq-1": seq}
        tracker.complete_batch_requests(
            active_seqs=active_seqs,
            pending_seqs=[],
            tokenizer=_StubTokenizer(),
        )
        result = tracker._results.get("seq-1")
        assert result is not None
        assert "decoded" in result

    def test_complete_active_sequence_no_tokens(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("seq-1")
        seq = _StubSequence(tokens=[])
        tracker.complete_batch_requests(
            active_seqs={"seq-1": seq},
            pending_seqs=[],
            tokenizer=_StubTokenizer(),
        )
        result = tracker._results.get("seq-1")
        assert result == "[Error: Sequence completed without output]"

    def test_complete_pending_sequence_times_out(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("pending-1")
        pending = [_StubSequence(tokens=[])]
        pending[0].request_id = "pending-1"
        tracker.complete_batch_requests(
            active_seqs={},
            pending_seqs=pending,
            tokenizer=None,
        )
        result = tracker._results.get("pending-1")
        assert result is not None
        assert "timed out" in result

    def test_complete_sets_events(self) -> None:
        tracker = RequestTracker()
        tracker.register_request("seq-1")
        seq = _StubSequence(tokens=[65])
        tracker.complete_batch_requests(
            active_seqs={"seq-1": seq},
            pending_seqs=[],
            tokenizer=_StubTokenizer(),
        )
        # The event must STAY (set, not popped): wait_for_result needs it to
        # retrieve the result.  Popping here made completed requests raise
        # "Unknown request_id" when generate_batch finished before the waiter.
        assert tracker._events["seq-1"].is_set()
        assert tracker.wait_for_result("seq-1", timeout=0.1) == "decoded:A"
