"""Registry of models available for hot-swap on this node.

Thin re-export: the implementation lives in
:mod:`distllm.core.multi_model_serving` (single source of truth).
"""

from distllm.core.multi_model_serving import ModelEntry, ModelRegistry

__all__ = ["ModelEntry", "ModelRegistry"]
