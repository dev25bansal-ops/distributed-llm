"""Re-export PrefixCache from distributed module for test compatibility."""

from distllm.dist.prefix_cache import PrefixCache

__all__ = ["PrefixCache"]
