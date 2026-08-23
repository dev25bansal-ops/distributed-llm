"""Ollama BackendAdapter - connects DistLLM to Ollama servers."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from distllm.backends.protocol import BackendAdapter


@dataclass
class OllamaBackendConfig:
    base_url: str = "http://localhost:11434"
    timeout_s: float = 120.0
    keep_alive: str = "5m"

class OllamaBackendAdapter(BackendAdapter):
    def __init__(self, config=None):
        self.config = config or OllamaBackendConfig()
    @property
    def display_name(self) -> str:
        return "Ollama"
    @property
    def version(self) -> str:
        return "0.1.0"
    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError("Ollama adapter does not support layer-level forward")
    async def generate(self, prompt, max_new_tokens=256, temperature=0.7, **kwargs):
        payload = {"model": kwargs.get("model", "llama3.2"), "prompt": prompt, "stream": False,
                   "options": {"num_predict": max_new_tokens, "temperature": temperature},
                   "keep_alive": self.config.keep_alive}
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            resp = await client.post(f"{self.config.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {"text": data.get("response", ""), "tokens": data.get("eval_count", 0)}
    async def generate_stream(self, prompt, max_new_tokens=256, temperature=0.7, **kwargs):
        payload = {"model": kwargs.get("model", "llama3.2"), "prompt": prompt, "stream": True,
                   "options": {"num_predict": max_new_tokens, "temperature": temperature}}
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            async with client.stream("POST", f"{self.config.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
