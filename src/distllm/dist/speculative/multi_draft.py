"""Multi-draft verification — parallel speculative decoding from multiple draft models.

Sends the same prompt to several draft models concurrently, collects all
responses, and selects the best result using a configurable strategy:

- ``confidence`` — picks the draft with the highest confidence score.
- ``latency`` — picks the fastest-responding draft.
- ``consensus`` — picks the most common prediction across drafts.

When every draft fails (timeout, error, empty response) the verifier
returns ``None``, letting the caller fall back to the target model alone.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass(frozen=True)
class DraftResult:
    """Result from a single draft model."""

    draft_id: str
    tokens: list[int]
    logprobs: list[float] | None = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None


SelectionStrategy = Callable[[list[DraftResult]], DraftResult | None]


def _pick_by_confidence(results: list[DraftResult]) -> DraftResult | None:
    """Return the result with the highest confidence score."""
    best: DraftResult | None = None
    best_conf = -1.0
    for r in results:
        if r.success and r.confidence > best_conf:
            best_conf = r.confidence
            best = r
    return best


def _pick_by_latency(results: list[DraftResult]) -> DraftResult | None:
    """Return the fastest successful result."""
    best: DraftResult | None = None
    best_lat = float("inf")
    for r in results:
        if r.success and r.latency_ms < best_lat:
            best_lat = r.latency_ms
            best = r
    return best


def _pick_by_consensus(results: list[DraftResult]) -> DraftResult | None:
    """Return the result whose tokens match the majority vote.

    For each position the most common token across all drafts wins.
    If no draft agrees at position 0, the highest-confidence draft
    is returned as a tiebreaker.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return None
    if len(successful) == 1:
        return successful[0]

    min_len = min(len(r.tokens) for r in successful)
    if min_len == 0:
        return max(successful, key=lambda r: r.confidence)

    consensus_tokens: list[int] = []
    for pos in range(min_len):
        counts: dict[int, int] = {}
        for r in successful:
            token = r.tokens[pos]
            counts[token] = counts.get(token, 0) + 1
        majority_token = max(counts, key=counts.get)
        consensus_tokens.append(majority_token)

    # Find the draft whose prefix best matches the consensus.
    best_match: DraftResult | None = None
    best_match_count = -1
    for r in successful:
        match_count = sum(
            1 for i in range(min(len(r.tokens), min_len))
            if r.tokens[i] == consensus_tokens[i]
        )
        if match_count > best_match_count:
            best_match_count = match_count
            best_match = r
    return best_match or successful[0]


# Mapping from strategy name to callable.
_STRATEGIES: dict[str, SelectionStrategy] = {
    "confidence": _pick_by_confidence,
    "latency": _pick_by_latency,
    "consensus": _pick_by_consensus,
}


@dataclass
class MultiDraftConfig:
    """Configuration for multi-draft verification.

    Attributes:
        strategy: Selection strategy — ``"confidence"``, ``"latency"``,
            or ``"consensus"``.
        timeout_s: Per-draft request timeout in seconds.
        max_concurrent: Maximum number of drafts to query simultaneously.
    """
    strategy: str = "confidence"
    timeout_s: float = 5.0
    max_concurrent: int = 8


class MultiDraftVerifier:
    """Sends a prompt to multiple draft models and selects the best result.

    Usage::

        verifier = MultiDraftVerifier(config=MultiDraftConfig(strategy="consensus"))

        # Provide async callables that return DraftResult.
        drafts = [
            ("draft-a", partial(draft_a_forward, prompt)),
            ("draft-b", partial(draft_b_forward, prompt)),
        ]
        best = await verifier.verify(prompt, drafts=drafts)
        if best is not None:
            target_model.speculative_step(best.tokens)
    """

    def __init__(self, config: MultiDraftConfig | None = None):
        self._config = config or MultiDraftConfig()
        self._strategy_fn: SelectionStrategy = _STRATEGIES.get(
            self._config.strategy, _pick_by_confidence,
        )
        self._total_calls = 0
        self._total_successes = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        prompt: Any,
        drafts: list[tuple[str, Callable[[], Any]]],
    ) -> DraftResult | None:
        """Run the prompt through multiple draft models in parallel.

        Args:
            prompt: The prompt to send to each draft.  Passed to the
                callable as its sole positional argument.
            drafts: List of ``(draft_id, async_callable)`` tuples.

        Returns:
            The best ``DraftResult`` according to the configured selection
            strategy, or ``None`` if all drafts fail.
        """
        if not drafts:
            logger.warning("MultiDraftVerifier.verify called with empty draft list")
            return None

        self._total_calls += 1
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def _run_draft(
            draft_id: str,
            callable_fn: Callable[[Any], Any],
        ) -> DraftResult:
            """Execute a single draft with timeout and error handling."""
            t0 = time.monotonic()
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(callable_fn):
                        result = await callable_fn(prompt)
                    else:
                        result = await asyncio.to_thread(callable_fn, prompt)

                    latency_ms = (time.monotonic() - t0) * 1000

                    if isinstance(result, DraftResult):
                        return DraftResult(
                            draft_id=draft_id,
                            tokens=result.tokens,
                            logprobs=result.logprobs,
                            confidence=result.confidence,
                            latency_ms=latency_ms,
                            success=result.success,
                            error=result.error,
                        )
                    if isinstance(result, dict):
                        return DraftResult(
                            draft_id=draft_id,
                            tokens=result.get("tokens", []),
                            logprobs=result.get("logprobs"),
                            confidence=result.get("confidence", 0.0),
                            latency_ms=latency_ms,
                            success=True,
                        )
                    if isinstance(result, (list, tuple)):
                        return DraftResult(
                            draft_id=draft_id,
                            tokens=list(result),
                            latency_ms=latency_ms,
                            success=True,
                        )
                    return DraftResult(
                        draft_id=draft_id,
                        tokens=[],
                        latency_ms=latency_ms,
                        success=False,
                        error=f"Unexpected result type: {type(result).__name__}",
                    )
                except asyncio.TimeoutError:
                    latency_ms = (time.monotonic() - t0) * 1000
                    return DraftResult(
                        draft_id=draft_id,
                        tokens=[],
                        latency_ms=latency_ms,
                        success=False,
                        error="Timeout",
                    )
                except Exception as exc:
                    latency_ms = (time.monotonic() - t0) * 1000
                    return DraftResult(
                        draft_id=draft_id,
                        tokens=[],
                        latency_ms=latency_ms,
                        success=False,
                        error=str(exc),
                    )

        tasks = [
            asyncio.create_task(
                asyncio.wait_for(
                    _run_draft(did, fn),
                    timeout=self._config.timeout_s,
                ),
            )
            for did, fn in drafts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any uncaught exceptions.
        processed: list[DraftResult] = []
        for i, r in enumerate(results):
            if isinstance(r, DraftResult):
                processed.append(r)
            elif isinstance(r, BaseException):
                processed.append(DraftResult(
                    draft_id=drafts[i][0],
                    tokens=[],
                    success=False,
                    error=f"Unhandled exception: {r}",
                ))
            tasks[i] = None  # allow GC

        successes = [r for r in processed if r.success]
        self._total_successes += len(successes)

        if not successes:
            logger.warning(
                "All {} drafts failed for this request", len(processed),
            )
            return None

        best = self._strategy_fn(successes)
        logger.debug(
            "Multi-draft: selected {!r} from {} drafts "
            "(strategy={}, latency={:.1f}ms)",
            best.draft_id if best else None,
            len(successes),
            self._config.strategy,
            best.latency_ms if best else 0.0,
        )
        return best

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative statistics."""
        return {
            "strategy": self._config.strategy,
            "total_calls": self._total_calls,
            "total_successes": self._total_successes,
            "success_rate": (
                round(self._total_successes / max(self._total_calls, 1), 4)
            ),
        }
