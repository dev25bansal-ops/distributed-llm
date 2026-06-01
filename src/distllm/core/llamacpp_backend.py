"""Re-exports LlamacppNodeAdapter from the backends package.

This module exists for backward compatibility — import directly from
``distllm.backends`` instead.

.. deprecated::
    Import from ``distllm.backends`` instead::

        from distllm.backends import LlamacppNodeAdapter
"""

import warnings

warnings.warn(
    "distllm.core.llamacpp_backend is deprecated. "
    "Import from distllm.backends instead: from distllm.backends import LlamacppNodeAdapter",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.backends.llamacpp_backend import LlamacppNodeAdapter

__all__ = ["LlamacppNodeAdapter"]
