"""Re-export PrefixCache from distributed module for test compatibility.

.. deprecated::
    Import from ``distllm.dist.prefix_cache`` instead.
"""

import warnings
warnings.warn(
    "distllm.core.prefix_cache is deprecated. "
    "Import from distllm.dist.prefix_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)

from distllm.dist.prefix_cache import PrefixCache

__all__ = ["PrefixCache"]
