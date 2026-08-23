"""Chunked prefill for predictable latency with long prompts.

Splits very long prompts into manageable chunks and tracks progress
so the KV cache can be built incrementally across pipeline iterations.
"""


from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

# Default max context — overridden by model config when available
DEFAULT_MAX_CONTEXT_TOKENS: int = 131072

# Known model context lengths (matched by substring in model name)
_MODEL_CONTEXT_LENGTHS: dict[str, int] = {
    "llama-3": 131072,
    "llama3": 131072,
    "qwen2.5": 131072,
    "qwen-2.5": 131072,
    "mistral": 32768,
    "mixtral": 32768,
    "gemma": 8192,
    "phi-3": 131072,
    "phi3": 131072,
    "command-r": 128000,
    "deepseek": 131072,
    "yi": 200000,
    "internlm": 32768,
    "codellama": 16384,
}


def get_max_context_tokens(
    model_name: str = "",
    model_info: dict | None = None,
) -> int:
    """Derive max context tokens from model name or config.


    Checks (in order):
    1. model_info["max_position_embeddings"] if present
    2. model_info["max_seq_len"] if present
    3. Known model name substring match
    4. DEFAULT_MAX_CONTEXT_TOKENS (131072)
    """

    if model_info:
        mpe = model_info.get("max_position_embeddings")
        if mpe and mpe > 0:
            return int(mpe)
        msl = model_info.get("max_seq_len")
        if msl and msl > 0:
            return int(msl)

    name_lower = model_name.lower()
    for pattern, ctx_len in _MODEL_CONTEXT_LENGTHS.items():
        if pattern in name_lower:
            return ctx_len

    return DEFAULT_MAX_CONTEXT_TOKENS


@dataclass
class ChunkState:
    prompt_tokens: List[int]
    chunk_size: int
    current_offset: int = 0

    @property
    def remaining(self) -> List[int]:
        return self.prompt_tokens[self.current_offset:]

    @property
    def is_done(self) -> bool:
        return self.current_offset >= len(self.prompt_tokens)

    def next_chunk(self) -> List[int]:
        end = min(self.current_offset + self.chunk_size, len(self.prompt_tokens))
        chunk = self.prompt_tokens[self.current_offset:end]
        self.current_offset = end
        return chunk

    @property
    def position_offset(self) -> int:
        return self.current_offset

    @property
    def chunks_total(self) -> int:
        return math.ceil(len(self.prompt_tokens) / self.chunk_size)

    @property
    def chunks_done(self) -> int:
        pos = self.current_offset
        return min(pos // self.chunk_size + (1 if pos % self.chunk_size > 0 else 0), self.chunks_total)


def maybe_chunk(
    token_ids: List[int],
    chunk_size: int,
    enabled: bool = True,
    max_context_tokens: int | None = None,
    model_name: str = "",
    model_info: dict | None = None,
) -> Optional[ChunkState]:
    """Optionally split a prompt into chunks for incremental prefill.


    Args:
        token_ids: Full prompt token IDs.
        chunk_size: Maximum tokens per chunk.
        enabled: If False, always returns None (no chunking).
        max_context_tokens: Override for max context length. If None,
            derived from model_name/model_info via get_max_context_tokens().
        model_name: Model name for auto-detecting context length.
        model_info: Model config dict for auto-detecting context length.

    Returns:
        ChunkState if chunking is needed, None if prompt fits in one chunk.

    Raises:
        ValueError: If prompt exceeds max context length.
    """

    if max_context_tokens is None:
        max_context_tokens = get_max_context_tokens(model_name, model_info)

    if not enabled or len(token_ids) <= chunk_size:
        return None
    if len(token_ids) > max_context_tokens:
        raise ValueError(
            f"Prompt length ({len(token_ids)} tokens) exceeds maximum "
            f"context length ({max_context_tokens} tokens)"
        )
    return ChunkState(prompt_tokens=token_ids, chunk_size=chunk_size)
