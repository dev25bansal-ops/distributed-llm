"""Re-export PredictiveCacheManager from distributed module for test compatibility."""

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
