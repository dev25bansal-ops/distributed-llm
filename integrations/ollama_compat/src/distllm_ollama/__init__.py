"""DistLLM Ollama compatibility layer — proxy that translates Ollama API to DistLLM."""

__version__ = "0.1.0"

from distllm_ollama.server import create_app

__all__ = ["__version__", "create_app"]
