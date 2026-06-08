"""DistLLM Haystack integration — Generator and Text Embedder components."""

__version__ = "0.1.0"

from distllm_haystack.generator import DistLLMGenerator
from distllm_haystack.embedder import DistLLMTextEmbedder

__all__ = ["__version__", "DistLLMGenerator", "DistLLMTextEmbedder"]
