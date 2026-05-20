"""Lightweight llama.cpp backend adapter for CPU/GPU inference.

Re-exports LlamacppNodeAdapter from the core module for a clean
``from distllm.backends import LlamacppNodeAdapter`` path.

See ``distllm.core.llamacpp_backend`` for full implementation.
"""

from distllm.core.llamacpp_backend import LlamacppNodeAdapter

__all__ = ["LlamacppNodeAdapter"]
