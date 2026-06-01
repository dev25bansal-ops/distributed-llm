"""Compatibility shim — re-exports from distllm.dist.straggler.

.. deprecated::
    Import from ``distllm.dist.straggler`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.straggler_detector is deprecated. "
    "Import from distllm.dist.straggler instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
