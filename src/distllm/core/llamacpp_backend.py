"""Re-exports LlamacppNodeAdapter from the backends package.

This module exists for backward compatibility — import directly from
``distllm.backends`` instead.
"""

from distllm.backends.llamacpp_backend import LlamacppNodeAdapter

__all__ = ["LlamacppNodeAdapter"]
