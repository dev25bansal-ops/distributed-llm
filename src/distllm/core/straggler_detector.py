"""Compatibility shim — re-exports from distllm.dist.straggler."""

from distllm.dist.straggler import (
    DetectionMethod,
    NodeTiming,
    StragglerDetector,
    StragglerReport,
    StragglerSeverity,
)

__all__ = [
    "StragglerDetector",
    "DetectionMethod",
    "StragglerSeverity",
    "StragglerReport",
    "NodeTiming",
]
