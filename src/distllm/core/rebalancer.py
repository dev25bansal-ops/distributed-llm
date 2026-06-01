"""Compatibility shim — re-exports from distllm.dist.rebalancer.

.. deprecated::
    Import from ``distllm.dist.rebalancer`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.rebalancer is deprecated. "
    "Import from distllm.dist.rebalancer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.dist.rebalancer import (
    PartitionRecommendation,
    Rebalancer,
    StragglerAction,
)

__all__ = [
    "Rebalancer",
    "PartitionRecommendation",
    "StragglerAction",
]
