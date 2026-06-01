"""Compatibility shim — re-exports from distllm.dist.recovery.

.. deprecated::
    Import from ``distllm.dist.recovery`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.node_recovery is deprecated. "
    "Import from distllm.dist.recovery instead.",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.dist.recovery import (
    LayerRedistribution,
    NodeRecoveryManager,
    NodeRecoveryPlan,
    SequenceCheckpoint,
)

__all__ = [
    "NodeRecoveryManager",
    "SequenceCheckpoint",
    "LayerRedistribution",
    "NodeRecoveryPlan",
]
