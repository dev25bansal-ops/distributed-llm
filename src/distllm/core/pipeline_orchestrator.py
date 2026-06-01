"""Compatibility shim — re-exports from distllm.dist.pipeline.

.. deprecated::
    Import from ``distllm.dist.pipeline`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.pipeline_orchestrator is deprecated. "
    "Import from distllm.dist.pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.dist.pipeline import PipelineOrchestrator

__all__ = ["PipelineOrchestrator"]
