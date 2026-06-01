"""vLLM backend adapter — re-exported from ``distllm.backends``.

.. deprecated::
    This module is kept for backward compatibility. Import from
    ``distllm.backends`` instead::

        from distllm.backends import VLLMNodeAdapter
"""

import warnings

warnings.warn(
    "distllm.core.vllm_backend is deprecated. "
    "Import from distllm.backends instead: from distllm.backends import VLLMNodeAdapter",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.backends.vllm_backend import VLLMNodeAdapter

__all__ = ["VLLMNodeAdapter"]
