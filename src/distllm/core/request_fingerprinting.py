"""Request Fingerprinting: hash requests for deduplication (avoid re-processing identical prompts).

Generates content-based fingerprints for requests to:
  - Deduplicate identical concurrent requests (same prompt + params)
  - Cache fingerprint-to-response mappings
  - Rate-limit by fingerprint (avoid abuse via repeated identical prompts)
  - Track prompt popularity

Uses SHA-256 hashing with canonical parameter serialization.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FingerprintEntry:
    """A cached fingerprint entry."""
    fingerprint: str
    prompt: str
    params_hash: str
    request_id: str
    created_at: float
    response: str | None = None
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class RequestFingerprinter:
    """Generates and tracks content-based fingerprints for requests.

    Two-tier:
      - Fingerprint in-flight: detect duplicate requests being processed
      - Fingerprint cache: return cached response for identical requests

    Usage:
        fp = RequestFingerprinter()
        fprint = fp.fingerprint(prompt="hello", params={"max_tokens": 50})

        if fp.is_in_flight(fprint):
            wait_for_result()

        if cached := fp.lookup(fprint):
            return cached.response

        fp.mark_in_flight(fprint, request_id="req-1")
        response = generate(prompt)
        fp.store(fprint, request_id="req-1", response=response)
        fp.clear_in_flight(fprint)
    """

    def __init__(
        self,
        cache_size: int = 10000,
        cache_ttl_s: float = 3600.0,
        enable_dedup: bool = True,
    ):
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl_s
        self._enable_dedup = enable_dedup

        # In-flight tracking (for concurrent dedup)
        self._in_flight: dict[str, set[str]] = {}  # fingerprint -> set of request_ids
        self._in_flight_results: dict[str, str | None] = {}  # fingerprint -> response
        self._wait_events: dict[str, set[threading.Event]] = {}  # fingerprint -> waiting threads

        # Response cache (fingerprint -> response)
        self._cache: OrderedDict[str, FingerprintEntry] = OrderedDict()
        self._lock = threading.Lock()

    def fingerprint(
        self,
        prompt: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Generate a deterministic content-based fingerprint for a request."""
        param_str = json.dumps(params or {}, sort_keys=True, default=str)
        raw = f"{prompt}|{param_str}"
        # Note: blake2b(size=16) is 2x faster than SHA-256 for dedup
        return hashlib.sha256(raw.encode()).hexdigest()

    def mark_in_flight(self, fingerprint: str, request_id: str) -> None:
        """Mark a request as being processed (for concurrent dedup)."""
        if not self._enable_dedup:
            return
        with self._lock:
            if fingerprint not in self._in_flight:
                self._in_flight[fingerprint] = set()
            self._in_flight[fingerprint].add(request_id)
            self._in_flight_results.pop(fingerprint, None)

    def clear_in_flight(self, fingerprint: str, request_id: str) -> None:
        """Remove a request from in-flight tracking."""
        with self._lock:
            ids = self._in_flight.get(fingerprint)
            if ids:
                ids.discard(request_id)
                if not ids:
                    self._in_flight.pop(fingerprint, None)
                    self._in_flight_results.pop(fingerprint, None)

    def is_in_flight(self, fingerprint: str) -> bool:
        """Check if an identical request is currently being processed."""
        with self._lock:
            ids = self._in_flight.get(fingerprint)
            return ids is not None and len(ids) > 0

    def _signal_waiting(self, fingerprint: str) -> None:
        """Signal all threads waiting on a fingerprint result."""
        with self._lock:
            waiters = self._wait_events.pop(fingerprint, set())
        for evt in waiters:
            evt.set()

    def store(
        self,
        fingerprint: str,
        request_id: str,
        response: str,
        prompt: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        """Store a fingerprint -> response mapping in cache."""
        now = time.time()
        param_str = json.dumps(params or {}, sort_keys=True, default=str)
        params_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]

        entry = FingerprintEntry(
            fingerprint=fingerprint,
            prompt=prompt,
            params_hash=params_hash,
            request_id=request_id,
            created_at=now,
            response=response,
        )

        with self._lock:
            self._cache[fingerprint] = entry
            self._cache.move_to_end(fingerprint)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        self._signal_waiting(fingerprint)

    def lookup(
        self,
        fingerprint: str,
        ttl_override: float | None = None,
    ) -> FingerprintEntry | None:
        """Look up a cached response by fingerprint. Returns None on miss."""
        ttl = ttl_override or self._cache_ttl
        with self._lock:
            entry = self._cache.get(fingerprint)
            if entry is None:
                return None
            if time.time() - entry.created_at > ttl:
                self._cache.pop(fingerprint, None)
                return None
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(fingerprint)
            return entry

    def wait_for_result(
        self,
        fingerprint: str,
        poll_interval_s: float = 0.05,
        timeout_s: float = 30.0,
    ) -> str | None:
        """Wait for an in-flight request to complete and return its result.

        Uses event-based notification rather than busy-polling.
        """
        wait_event = threading.Event()
        with self._lock:
            # Check if result already available
            result = self._in_flight_results.get(fingerprint)
            if result is not None:
                return result
            # Register for notification
            if fingerprint in self._in_flight:
                self._wait_events.setdefault(fingerprint, set()).add(wait_event)

        wait_event.wait(timeout=timeout_s)

        # Clean up event registration
        with self._lock:
            events = self._wait_events.get(fingerprint)
            if events:
                events.discard(wait_event)
                if not events:
                    self._wait_events.pop(fingerprint, None)

        with self._lock:
            result = self._in_flight_results.get(fingerprint)
            return result

    def popularity(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the top N most popular fingerprints by hit count."""
        with self._lock:
            sorted_entries = sorted(
                self._cache.values(), key=lambda e: e.hit_count, reverse=True
            )
            return [(e.fingerprint[:16], e.hit_count) for e in sorted_entries[:top_n]]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._cache)
            total_hits = sum(e.hit_count for e in self._cache.values())
            in_flight_count = sum(len(v) for v in self._in_flight.values())
            return {
                "cache_entries": total,
                "cache_max": self._cache_size,
                "total_hits": total_hits,
                "in_flight_requests": in_flight_count,
                "dedup_enabled": self._enable_dedup,
            }
