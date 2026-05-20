"""vLLM backend adapter — re-exported from ``distllm.backends``.

.. deprecated::
    This module is kept for backward compatibility. Import from
    ``distllm.backends`` instead::

        from distllm.backends import VLLMNodeAdapter
"""

from distllm.backends.vllm_backend import VLLMNodeAdapter

__all__ = ["VLLMNodeAdapter"]
