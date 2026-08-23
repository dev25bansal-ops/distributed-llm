"""Coordinator lifecycle management: request tracking and server lifecycle.

Provides:
- RequestTracker: thread-safe async request result tracking with cancellation
- ServerLifecycle: manages server start/stop and graceful shutdown
"""

from __future__ import annotations

import threading
import time
from typing import Any


class RequestTracker:
    """Thread-safe tracker for async generation results.

    Tracks pending requests, stores results, and supports cancellation
    and graceful shutdown.

    Usage::

        tracker = RequestTracker()
        event = tracker.register_request("req-1")
        # ... in batch scheduler:
        tracker.set_result("req-1", "generated text")
        # ... in API layer:
        result = tracker.wait_for_result("req-1", timeout=120.0)
    """

    def __init__(self) -> None:
        self._results: dict[str, str] = {}
        self._events: dict[str, threading.Event] = {}
        self._logprobs: dict[str, dict] = {}
        self._errors: dict[str, Exception] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    def register_request(self, request_id: str) -> threading.Event:
        """Register a new request and return its completion event.

        Args:
            request_id: Unique request identifier.

        Returns:
            A threading.Event that will be set when the request completes.
        """
        event = threading.Event()
        with self._lock:
            self._events[request_id] = event
        return event

    def set_result(self, request_id: str, result: str) -> None:
        """Set the result for a completed request and wake waiters."""
        with self._lock:
            self._results[request_id] = result
            event = self._events.get(request_id)
            if event:
                event.set()

    def set_error(self, request_id: str, error: Exception) -> None:
        """Set an error for a failed request and wake waiters."""
        with self._lock:
            self._errors[request_id] = error
            event = self._events.get(request_id)
            if event:
                event.set()

    def set_logprobs(self, request_id: str, logprobs: dict) -> None:
        """Store logprobs for a request."""
        with self._lock:
            self._logprobs[request_id] = logprobs

    def get_logprobs(self, request_id: str) -> dict | None:
        """Retrieve logprobs for a request."""
        with self._lock:
            return self._logprobs.get(request_id)

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        """Wait for a request to complete and return its result.

        Args:
            request_id: The request to wait on.
            timeout: Maximum seconds to wait.

        Returns:
            The generated text.

        Raises:
            ValueError: If request_id is not registered.
            TimeoutError: If the request does not complete within timeout.
            RuntimeError: If the request failed with an error.
        """
        event = self._events.get(request_id)
        if event is None:
            # Completed before wait: surface the ready result/error.
            with self._lock:
                result = self._results.pop(request_id, None)
                error = self._errors.pop(request_id, None)
                self._logprobs.pop(request_id, None)
            if error is not None:
                raise RuntimeError(f"Request {request_id} failed: {error}") from error
            if result is not None:
                return result
            raise ValueError(f"Unknown request_id: {request_id}")

        event.wait(timeout=timeout)

        with self._lock:
            result = self._results.pop(request_id, None)
            error = self._errors.pop(request_id, None)
            self._events.pop(request_id, None)
            self._logprobs.pop(request_id, None)

        if error is not None:
            raise RuntimeError(f"Request {request_id} failed: {error}") from error

        if result is None:
            raise TimeoutError(f"Request {request_id} timed out after {timeout}s")

        return result

    @property
    def pending_count(self) -> int:
        """Return the number of registered (pending) requests."""
        with self._lock:
            return len(self._events)

    @property
    def shutting_down(self) -> bool:
        """Return True if the tracker is in shutdown mode."""
        with self._lock:
            return self._shutting_down

    @shutting_down.setter
    def shutting_down(self, value: bool) -> None:
        """Set the shutdown flag."""
        with self._lock:
            self._shutting_down = value

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request.

        Returns True if the request was found and cancelled.
        """
        with self._lock:
            if request_id not in self._events:
                return False
            self._results[request_id] = "[Error: Request cancelled]"
            event = self._events.pop(request_id, None)
            if event:
                event.set()
            self._logprobs.pop(request_id, None)
            return True

    def clear(self) -> None:
        """Reset all state, unblocking waiters with a cancellation result.

        Each registered request gets a "[Error: Request cancelled]" result
        and a SET event which the waiter consumes via wait_for_result
        (that call also reclaims the entry).
        """
        with self._lock:
            for rid in list(self._events.keys()):
                self._results.setdefault(rid, "[Error: Request cancelled]")
                self._events[rid].set()
            self._logprobs.clear()
            self._errors.clear()
            self._shutting_down = False

    def complete_batch_requests(
        self,
        active_seqs: dict[str, Any],
        pending_seqs: list[Any],
        tokenizer: Any,
    ) -> None:
        """Complete all active/pending requests during shutdown.

        Args:
            active_seqs: Dict of request_id -> Sequence for active requests.
            pending_seqs: List of Sequence objects still pending.
            tokenizer: Tokenizer to decode generated tokens.
        """
        with self._lock:
            for seq_id, seq in (active_seqs.items() if isinstance(active_seqs, dict) else []):
                try:
                    if hasattr(seq, "generated_tokens") and seq.generated_tokens:
                        if tokenizer is not None:
                            result = tokenizer.decode(
                                seq.generated_tokens, skip_special_tokens=True
                            )
                        else:
                            result = str(seq.generated_tokens)
                        self._results[seq_id] = result
                    else:
                        self._results[seq_id] = "[Error: Sequence completed without output]"
                except Exception as e:
                    self._results[seq_id] = f"[Error decoding output: {e}]"
                # SET, never pop: wait_for_result consumes the event.
                event = self._events.get(seq_id)
                if event:
                    event.set()
                self._logprobs.pop(seq_id, None)

            for seq in pending_seqs:
                sid = getattr(seq, "request_id", str(seq))
                self._results[sid] = "[Error: Request timed out waiting in scheduler queue]"
                event = self._events.get(sid)
                if event:
                    event.set()


class ServerLifecycle:
    """Manages server start/stop and graceful shutdown.

    Tracks server state and provides a mechanism for coordinating
    shutdown across multiple components.

    Usage::

        lifecycle = ServerLifecycle()
        lifecycle.start()
        # ... serve requests ...
        lifecycle.initiate_shutdown(timeout=30.0)
        lifecycle.wait_for_shutdown()
    """

    def __init__(self) -> None:
        self._running = False
        self._start_time: float | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Mark the server as running."""
        with self._lock:
            self._running = True
            self._start_time = time.monotonic()
            self._shutdown_event.clear()

    def stop(self) -> None:
        """Mark the server as stopped and signal shutdown."""
        with self._lock:
            self._running = False
            self._shutdown_event.set()

    @property
    def is_running(self) -> bool:
        """Return True if the server is running."""
        with self._lock:
            return self._running

    def initiate_shutdown(self, timeout: float = 30.0) -> None:
        """Begin graceful shutdown.

        Args:
            timeout: Maximum seconds to wait for in-flight requests.
        """
        self.stop()

    def wait_for_shutdown(self, timeout: float | None = None) -> None:
        """Block until shutdown is complete.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.
        """
        self._shutdown_event.wait(timeout=timeout)

    @property
    def shutdown_event(self) -> threading.Event:
        """Return the shutdown event for external coordination."""
        return self._shutdown_event
