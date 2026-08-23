"""Shared helpers for CI integration modules.

Provides the retry utility, data types, and constants used by both
:mod:`distllm.integrations.ci.gitlab` and :mod:`distllm.integrations.ci.jenkins`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger("distllm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 120.0  # seconds
_MAX_RETRIES = 3
_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _BASE_DELAY,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential backoff on transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            # Do not retry 4xx client errors (except 429 rate-limit).
            status = exc.response.status_code
            if 400 <= status < 500 and status != 429:
                raise
            last_exc = exc
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
        if attempt < max_retries - 1:
            delay = min(base_delay * (2**attempt), 30.0)
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs …",
                attempt + 1,
                max_retries,
                last_exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Outcome of a single evaluation run."""

    pipeline_id: int
    project: str
    ref: str
    model: str
    status: str  # e.g. "success", "failed", "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BuildInfo:
    """Jenkins build status and metadata."""

    build_number: int
    job: str
    status: str  # e.g. "SUCCESS", "FAILURE", "UNSTABLE", "RUNNING"
    url: str
    duration_ms: int = 0
    estimated_duration_ms: int = 0
