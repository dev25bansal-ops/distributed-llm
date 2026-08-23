"""Verified-equivalence speculative decoding.

Speculative decoding is normally a *trust-us* optimization: it accelerates
generation while claiming bit-identical output to greedy/reference decoding.
This module makes that claim **provable** by sampling a fraction of requests,
running the reference (greedy target) decode alongside, and asserting the
speculative output is token-identical.  Any divergence increments the
``spec_divergence_total`` counter — exported both through the project's
in-memory :class:`MetricsManager` (Prometheus-compatible dict) and, when the
optional ``prometheus_client`` package is installed, a real Prometheus
``Counter``.

Usage::

    checker = SpecEquivalenceChecker(sample_rate=0.05)  # verify 5% of requests
    for request in requests:
        ran_spec = checker.should_sample() and checker.start(request.id)
        output = decode(request, verify=ran_spec)
        if ran_spec:
            reference = greedy_target_decode(request)
            if not checker.check(output.tokens, reference.tokens):
                logger.error("spec divergence on %s", request.id)
                # output is still returned to the user; we just flag it.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Sequence

from loguru import logger

from distllm.core.coordinator_metrics import MetricsManager


# Optional real Prometheus client.  The project does not hard-depend on it,
# so we degrade gracefully to the in-memory MetricsManager when absent.
try:  # pragma: no cover - exercised only when prometheus_client is present
    from prometheus_client import Counter as _PromCounter

    _PROM_DIVERGENCE = _PromCounter(
        "spec_divergence_total",
        "Number of speculative-decoding requests whose output diverged "
        "from the reference (greedy target) decode.",
    )
    _HAVE_PROM = True
except Exception:  # ImportError or any construction error
    _PROM_DIVERGENCE = None
    _HAVE_PROM = False


def _tokens_equal(a: Sequence[int], b: Sequence[int]) -> bool:
    """Token-identical comparison (order + values, same length)."""
    return list(a) == list(b)


class SpecEquivalenceChecker:
    """Samples requests for equivalence verification and tracks divergence.

    Args:
        sample_rate: Fraction of requests to verify (0.0 = disabled,
            1.0 = verify every request).  Clamped to ``[0, 1]``.
        metrics: Optional :class:`MetricsManager` to record the divergence
            counter into (so it surfaces in the existing ``/metrics`` export).
        rng: Optional ``callable() -> float in [0, 1)`` for deterministic
            sampling in tests; defaults to :func:`random.random`.
    """

    def __init__(
        self,
        sample_rate: float = 0.0,
        metrics: MetricsManager | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self._sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self._metrics = metrics
        self._rng = rng
        self._lock = threading.Lock()
        self._checked = 0
        self._diverged = 0

    @property
    def enabled(self) -> bool:
        return self._sample_rate > 0.0

    @property
    def checked(self) -> int:
        return self._checked

    @property
    def diverged(self) -> int:
        return self._diverged

    def should_sample(self) -> bool:
        """Decide whether the current request should be verified."""
        if not self.enabled:
            return False
        if self._sample_rate >= 1.0:
            return True
        if self._rng is not None:
            return self._rng() < self._sample_rate
        import random

        return random.random() < self._sample_rate

    def check(
        self,
        speculative_tokens: Sequence[int],
        reference_tokens: Sequence[int],
        request_id: str = "",
    ) -> bool:
        """Compare a speculative output against the reference decode.

        Increments ``spec_divergence_total`` (in-memory + Prometheus when
        available) on any mismatch.  Returns ``True`` when identical.

        Note: the speculative output is NOT discarded on divergence — the
        user still receives it.  Verification is observational: it proves
        (or disproves) the equivalence guarantee without changing behavior.
        """
        with self._lock:
            self._checked += 1
        equal = _tokens_equal(speculative_tokens, reference_tokens)
        if not equal:
            with self._lock:
                self._diverged += 1
            logger.warning(
                "spec_equivalence divergence request=%s spec=%s ref=%s",
                request_id,
                list(speculative_tokens)[:16],
                list(reference_tokens)[:16],
            )
            self._record_divergence()
        return equal

    def _record_divergence(self) -> None:
        if self._metrics is not None:
            self._metrics.increment("spec_divergence_total")
        if _HAVE_PROM and _PROM_DIVERGENCE is not None:  # pragma: no cover
            _PROM_DIVERGENCE.inc()
