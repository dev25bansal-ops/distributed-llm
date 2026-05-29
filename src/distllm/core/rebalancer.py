"""Compatibility shim — re-exports from distllm.dist.rebalancer."""

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
