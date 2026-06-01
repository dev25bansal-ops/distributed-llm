"""Jupyter Notebook magic commands for DistLLM.

Usage in Jupyter:

    %load_ext distllm.jupyter
    %%distllm chat
    What is the meaning of life?

    %%distllm complete
    Once upon a time

    %distllm models
    %distllm status
    %distllm benchmark --model llama-3-8b
"""

from __future__ import annotations

import json
import time
from typing import Any

from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring


@magics_class
class DistLLMMagics(Magics):
    """DistLLM magic commands for Jupyter notebooks."""

    def __init__(self, shell: Any = None) -> None:
        super().__init__(shell)
        self._client = None
        self._base_url = "http://localhost:8000/v1"
        self._api_key = ""
        self._default_model = ""

    def _get_client(self):
        """Get or create the DistLLM client."""
        if self._client is None:
            from distllm_sdk.compat import openai_compat
            self._client = openai_compat.OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    @line_magic
    def distllm_connect(self, line: str) -> None:
        """Connect to a DistLLM cluster.

        Usage: %distllm_connect http://localhost:8000/v1 [api_key]
        """
        parts = line.strip().split()
        if not parts:
            print(f"Usage: %distllm_connect <base_url> [api_key]")
            print(f"Current: {self._base_url}")
            return

        self._base_url = parts[0]
        self._api_key = parts[1] if len(parts) > 1 else ""
        self._client = None  # Reset client
        print(f"Connected to {self._base_url}")

    @line_magic
    def distllm_model(self, line: str) -> None:
        """Set the default model.

        Usage: %distllm_model llama-3-70b
        """
        model = line.strip()
        if not model:
            print(f"Current model: {self._default_model or '(not set)'}")
            return
        self._default_model = model
        print(f"Default model set to: {model}")

    @line_magic
    def distllm_models(self, line: str) -> None:
        """List available models.

        Usage: %distllm_models
        """
        try:
            client = self._get_client()
            models = client.models.list()
            print("Available models:")
            for m in models.data:
                print(f"  - {m.get('id', 'unknown')}")
        except Exception as e:
            print(f"Error listing models: {e}")

    @line_magic
    def distllm_status(self, line: str) -> None:
        """Show cluster status.

        Usage: %distllm_status
        """
        try:
            import httpx
            resp = httpx.get(f"{self._base_url}/health", timeout=5)
            data = resp.json()
            print(f"Status: {data.get('status', 'unknown')}")
            print(f"Nodes: {data.get('num_nodes', 0)}")
            print(f"Total layers: {data.get('total_layers', 0)}")
        except Exception as e:
            print(f"Error: {e}")

    @cell_magic
    def distllm(self, line: str, cell: str) -> None:
        """Chat with DistLLM.

        Usage:
            %%distllm chat
            What is Python?

            %%distllm chat --model llama-3-70b --temperature 0.5
            Explain quantum computing
        """
        parts = line.strip().split()
        mode = parts[0] if parts else "chat"
        model = self._default_model or "default"

        # Parse simple flags
        temperature = 0.7
        max_tokens = 256
        for i, part in enumerate(parts):
            if part == "--model" and i + 1 < len(parts):
                model = parts[i + 1]
            elif part == "--temperature" and i + 1 < len(parts):
                temperature = float(parts[i + 1])
            elif part == "--max-tokens" and i + 1 < len(parts):
                max_tokens = int(parts[i + 1])

        prompt = cell.strip()
        if not prompt:
            print("No prompt provided")
            return

        try:
            client = self._get_client()
            t0 = time.time()

            if mode == "chat":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                tokens = response.usage.completion_tokens
            elif mode == "complete":
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0]["text"]
                tokens = response.usage.completion_tokens
            else:
                print(f"Unknown mode: {mode}. Use 'chat' or 'complete'.")
                return

            elapsed = time.time() - t0
            print(content)
            print(f"\n--- {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s) ---")

        except Exception as e:
            print(f"Error: {e}")


def load_ipython_extension(ipython: Any) -> None:
    """Load the DistLLM magic commands."""
    ipython.register_magic_class(DistLLMMagics)
    print("DistLLM magic commands loaded. Use %distllm_connect to connect.")


def unload_ipython_extension(ipython: Any) -> None:
    """Unload the DistLLM magic commands."""
    pass
