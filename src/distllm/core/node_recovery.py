"""Compatibility shim — re-exports from distllm.dist.recovery.

Tests import from distllm.core.node_recovery but the actual module
lives at distllm.dist.recovery.  This file provides the re-export
so existing tests pass without modification.
"""

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
