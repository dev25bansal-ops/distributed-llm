"""Compatibility shim — re-exports from distllm.dist.latency.

.. deprecated::
    Import from ``distllm.dist.latency`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.latency_tracker is deprecated. "
    "Import from distllm.dist.latency instead.",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.dist.latency import LatencyTracker

__all__ = ["LatencyTracker"]
