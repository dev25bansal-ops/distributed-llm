"""Tests for coordinator lifecycle -- RequestTracker and ServerLifecycle.

Covers:
- RequestTracker: register, set_result, wait_for_result, timeout, error
- RequestTracker: cancellation, clear, logprobs, shutdown flag
- RequestTracker: complete_batch_requests during shutdown
- ServerLifecycle: start, stop, is_running, shutdown coordination
"""

from __future__ import annotations

import threading
import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_lifecycle = load_module("distllm/core/coordinator_lifecycle.py")
RequestTracker = _lifecycle.RequestTracker
ServerLifecycle = _lifecycle.ServerLifecycle


# ======================================================================
# RequestTracker
# ======================================================================


class TestRequestTrackerRegister:
    """Registering requests."""

    def test_register_returns_event(self):
        tracker = RequestTracker()
        event = tracker.register_request("req-1")
        assert isinstance(event, threading.Event)
        assert event.is_set() is False

    def test_pending_count_after_register(self):
        tracker = RequestTracker()
        assert tracker.pending_count == 0
        tracker.register_request("req-1")
        assert tracker.pending_count == 1

    def test_multiple_registrations(self):
        tracker = RequestTracker()
        for i in range(5):
            tracker.register_request(f"req-{i}")
        assert tracker.pending_count == 5


class TestRequestTrackerSetResult:
    """Setting results and waiting for them."""

    def test_set_result_wakes_waiter(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_result("req-1", "hello world")
        result = tracker.wait_for_result("req-1", timeout=2.0)
        assert result == "hello world"

    def test_set_result_pending_decreases(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_result("req-1", "done")
        # wait_for_result pops the entry
        tracker.wait_for_result("req-1", timeout=2.0)
        assert tracker.pending_count == 0

    def test_set_result_unknown_request_no_error(self):
        tracker = RequestTracker()
        tracker.set_result("nonexistent", "value")  # should not raise

    def test_wait_for_result_unknown_request_raises(self):
        tracker = RequestTracker()
        with pytest.raises(ValueError, match="Unknown request_id"):
            tracker.wait_for_result("does-not-exist", timeout=0.1)


class TestRequestTrackerTimeout:
    """Timeout behavior."""

    def test_wait_timeout_raises(self):
        tracker = RequestTracker()
        tracker.register_request("req-timeout")
        with pytest.raises(TimeoutError, match="timed out"):
            tracker.wait_for_result("req-timeout", timeout=0.01)


class TestRequestTrackerError:
    """Error handling."""

    def test_set_error_raises_on_wait(self):
        tracker = RequestTracker()
        tracker.register_request("req-err")
        tracker.set_error("req-err", RuntimeError("something broke"))
        with pytest.raises(RuntimeError, match="something broke"):
            tracker.wait_for_result("req-err", timeout=2.0)

    def test_set_error_unknown_request_no_error(self):
        tracker = RequestTracker()
        tracker.set_error("nonexistent", RuntimeError("x"))  # should not raise


class TestRequestTrackerCancel:
    """Cancellation."""

    def test_cancel_pending_request(self):
        tracker = RequestTracker()
        tracker.register_request("req-cancel")
        cancelled = tracker.cancel("req-cancel")
        assert cancelled is True
        result = tracker.wait_for_result("req-cancel", timeout=2.0)
        assert result == "[Error: Request cancelled]"

    def test_cancel_nonexistent(self):
        tracker = RequestTracker()
        assert tracker.cancel("does-not-exist") is False

    def test_cancel_reduces_pending(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.cancel("req-1")
        assert tracker.pending_count == 0


class TestRequestTrackerLogprobs:
    """Logprobs storage and retrieval."""

    def test_set_and_get_logprobs(self):
        tracker = RequestTracker()
        tracker.set_logprobs("req-1", {"token_0": 0.5})
        lp = tracker.get_logprobs("req-1")
        assert lp == {"token_0": 0.5}

    def test_get_logprobs_nonexistent(self):
        tracker = RequestTracker()
        assert tracker.get_logprobs("nonexistent") is None

    def test_logprobs_removed_after_wait(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_logprobs("req-1", {"tok": 0.9})
        tracker.set_result("req-1", "done")
        tracker.wait_for_result("req-1", timeout=2.0)
        assert tracker.get_logprobs("req-1") is None


class TestRequestTrackerShutdownFlag:
    """Shutdown flag."""

    def test_initial_shutdown_false(self):
        tracker = RequestTracker()
        assert tracker.shutting_down is False

    def test_set_shutdown_true(self):
        tracker = RequestTracker()
        tracker.shutting_down = True
        assert tracker.shutting_down is True

    def test_clear_resets_shutdown(self):
        tracker = RequestTracker()
        tracker.shutting_down = True
        tracker.clear()
        assert tracker.shutting_down is False


class TestRequestTrackerClear:
    """Clear resets all state."""

    def test_clear_empties_events(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.register_request("req-2")
        tracker.clear()
        # clear() leaves SET events + cancellation results so blocked waiters
        # get an answer; entries are reclaimed when each waiter consumes it.
        assert "cancelled" in tracker.wait_for_result("req-1", timeout=1.0).lower()
        assert "cancelled" in tracker.wait_for_result("req-2", timeout=1.0).lower()
        assert tracker.pending_count == 0

    def test_clear_signals_waiters(self):
        tracker = RequestTracker()
        tracker.register_request("req-blocked")
        tracker.clear()
        # waiter should unblock immediately with a cancellation message
        result = tracker.wait_for_result("req-blocked", timeout=1.0)
        assert "cancelled" in result.lower()

    def test_clear_removes_logprobs(self):
        tracker = RequestTracker()
        tracker.register_request("req-1")
        tracker.set_logprobs("req-1", {"a": 1})
        tracker.clear()
        assert tracker.get_logprobs("req-1") is None


class TestRequestTrackerCompleteBatch:
    """complete_batch_requests during shutdown."""

    def test_complete_batch_active_seqs(self):
        tracker = RequestTracker()
        tracker.register_request("seq-1")

        class FakeSeq:
            request_id = "seq-1"
            generated_tokens = [10, 20, 30]

        tracker.complete_batch_requests(
            active_seqs={"seq-1": FakeSeq()},
            pending_seqs=[],
            tokenizer=None,
        )
        result = tracker.wait_for_result("seq-1", timeout=2.0)
        assert "[10, 20, 30]" in result or "Error" in result

    def test_complete_batch_active_seqs_with_tokenizer(self):
        tracker = RequestTracker()
        tracker.register_request("seq-1")

        class FakeSeq:
            request_id = "seq-1"
            generated_tokens = [101, 102]

        fake_tokenizer = type("FakeTok", (), {})()
        fake_tokenizer.decode = lambda tokens, **kw: f"decoded-{tokens}"

        tracker.complete_batch_requests(
            active_seqs={"seq-1": FakeSeq()},
            pending_seqs=[],
            tokenizer=fake_tokenizer,
        )
        result = tracker.wait_for_result("seq-1", timeout=2.0)
        assert "decoded" in result

    def test_complete_batch_pending_seqs(self):
        tracker = RequestTracker()
        tracker.register_request("pending-1")

        class FakeSeq:
            request_id = "pending-1"

        tracker.complete_batch_requests(
            active_seqs={},
            pending_seqs=[FakeSeq()],
            tokenizer=None,
        )
        result = tracker.wait_for_result("pending-1", timeout=2.0)
        assert "timed out" in result or "Error" in result

    def test_complete_batch_active_seq_decode_error(self):
        tracker = RequestTracker()
        tracker.register_request("bad-seq")

        class BadSeq:
            request_id = "bad-seq"
            generated_tokens = [1]

        def broken_decode(tokens, **kw):
            raise ValueError("decode failed")

        fake_tokenizer = type("FakeTok", (), {})()
        fake_tokenizer.decode = broken_decode

        tracker.complete_batch_requests(
            active_seqs={"bad-seq": BadSeq()},
            pending_seqs=[],
            tokenizer=fake_tokenizer,
        )
        result = tracker.wait_for_result("bad-seq", timeout=2.0)
        assert "Error decoding" in result

    def test_complete_batch_active_seqs_empty_dict(self):
        """Empty active_seqs dict should not error."""
        tracker = RequestTracker()
        tracker.complete_batch_requests(
            active_seqs={},
            pending_seqs=[],
            tokenizer=None,
        )
        # no registered requests, should be a no-op
        assert tracker.pending_count == 0


# ======================================================================
# ServerLifecycle
# ======================================================================


class TestServerLifecycleInit:
    def test_initial_not_running(self):
        ls = ServerLifecycle()
        assert ls.is_running is False

    def test_has_shutdown_event(self):
        ls = ServerLifecycle()
        assert isinstance(ls.shutdown_event, threading.Event)
        assert ls.shutdown_event.is_set() is False


class TestServerLifecycleStartStop:
    def test_start_sets_running(self):
        ls = ServerLifecycle()
        ls.start()
        assert ls.is_running is True

    def test_stop_clears_running(self):
        ls = ServerLifecycle()
        ls.start()
        ls.stop()
        assert ls.is_running is False

    def test_start_clears_shutdown_event(self):
        ls = ServerLifecycle()
        ls.initiate_shutdown()
        ls.start()
        assert ls.shutdown_event.is_set() is False

    def test_stop_sets_shutdown_event(self):
        ls = ServerLifecycle()
        ls.start()
        ls.stop()
        assert ls.shutdown_event.is_set() is True


class TestServerLifecycleInitiateShutdown:
    def test_initiate_shutdown_clears_running(self):
        ls = ServerLifecycle()
        ls.start()
        ls.initiate_shutdown(timeout=5.0)
        assert ls.is_running is False

    def test_initiate_shutdown_signals_event(self):
        ls = ServerLifecycle()
        ls.start()
        ls.initiate_shutdown()
        assert ls.shutdown_event.is_set() is True


class TestServerLifecycleWaitForShutdown:
    def test_wait_already_stopped(self):
        ls = ServerLifecycle()
        ls.stop()
        ls.wait_for_shutdown(timeout=1.0)  # should return immediately
        # no exception means pass

    def test_wait_blocks_until_shutdown(self):
        ls = ServerLifecycle()
        ls.start()

        results = []

        def waiter():
            ls.wait_for_shutdown(timeout=5.0)
            results.append("done")

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        assert len(results) == 0  # waiter blocked
        ls.stop()
        t.join(timeout=2.0)
        assert len(results) == 1
        assert results[0] == "done"
