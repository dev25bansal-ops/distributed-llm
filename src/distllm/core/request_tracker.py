"""Tracks async request results and completion events.

Provides a thread-safe mechanism for the batch scheduler to signal
when a request has completed, and for the API layer to wait on
the result.
"""

from __future__ import annotations

import threading
from typing import Any


class RequestTracker:
    """Thread-safe tracker for async generation results.

    Usage::

        tracker = RequestTracker()
        tracker.register_request(request_id)

        # In batch scheduler callback:
        tracker.set_result(request_id, generated_text)

        # In API layer:
        result = tracker.wait_for_result(request_id, timeout=120.0)
    """

    def __init__(self) -> None:
        self._results: dict[str, str] = {}
        self._events: dict[str, threading.Event] = {}
        self._logprobs: dict[str, dict] = {}
        self._errors: dict[str, Exception] = {}
        self._lock = threading.Lock()

    def register_request(self, request_id: str) -> None:
        """Register a new request for tracking."""
        with self._lock:
            self._events[request_id] = threading.Event()

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

    def complete_batch_requests(
        self,
        active_seqs: dict[str, Any],
        pending_seqs: list[Any],
        tokenizer: Any,
    ) -> None:
        """Complete all active/pending requests in the batch.

        Called when the batch scheduler is shutting down or timing out.
        Finishes any active sequences by decoding their generated tokens,
        and marks pending sequences with an error.

        Args:
            active_seqs: Dict of request_id -> Sequence for active requests.
            pending_seqs: List of Sequence objects still pending.
            tokenizer: Tokenizer to decode generated tokens.
        """
        with self._lock:
            # Complete active sequences that have generated tokens
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
                event = self._events.pop(seq_id, None)
                if event:
                    event.set()
                self._logprobs.pop(seq_id, None)

            # Mark pending sequences as timed out
            for seq in pending_seqs:
                sid = getattr(seq, "request_id", str(seq))
                self._results[sid] = "[Error: Request timed out waiting in scheduler queue]"
                event = self._events.pop(sid, None)
                if event:
                    event.set()

    def has_request(self, request_id: str) -> bool:
        """Check if a request is registered."""
        with self._lock:
            return request_id in self._events

    def pending_count(self) -> int:
        """Return the number of pending requests."""
        with self._lock:
            return len(self._events)

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
