"""Re-export PredictiveCacheManager from distributed module for test compatibility.

.. deprecated::
    Import from ``distllm.dist.predictive_cache`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.predictive_cache is deprecated. "
    "Import from distllm.dist.predictive_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.dist.predictive_cache import (
    PatternLearner,
    PredictiveCacheManager,
    PrefixPattern,
    CachePrediction,
    CacheTier,
)

__all__ = [
    "PatternLearner",
    "PredictiveCacheManager",
    "PrefixPattern",
    "CachePrediction",
    "CacheTier",
]
