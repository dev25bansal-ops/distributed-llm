"""Chunked prefill for predictable latency with long prompts.

Splits very long prompts into manageable chunks and tracks progress
so the KV cache can be built incrementally across pipeline iterations.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

# Maximum supported context length — prompts exceeding this will be rejected
# rather than silently producing OOM errors downstream.
MAX_CONTEXT_TOKENS: int = 131072


@dataclass
class ChunkState:
    """Tracks chunking progress for a single request's prompt.

    When a prompt is longer than chunk_size, it is split into chunks.
    Each chunk is processed sequentially through the pipeline, building
    up the KV cache incrementally.

    Important: ``chunk_size`` here is a hard upper bound per chunk.
    The batch scheduler's ``ChunkedPrefillInfo`` also tracks progress
    independently; the two systems are kept in sync via per-sequence
    metadata in the scheduler.
    """
    prompt_tokens: List[int]
    chunk_size: int
    current_offset: int = 0

    @property
    def remaining(self) -> List[int]:
        """Tokens not yet processed."""
        return self.prompt_tokens[self.current_offset:]

    @property
    def is_done(self) -> bool:
        return self.current_offset >= len(self.prompt_tokens)

    def next_chunk(self) -> List[int]:
        """Get the next chunk of tokens and advance the offset.

        Respects chunk_size boundary, returning at most chunk_size tokens.
        The caller must advance ``current_offset`` by the returned length.
        """
        end = min(self.current_offset + self.chunk_size, len(self.prompt_tokens))
        chunk = self.prompt_tokens[self.current_offset:end]
        self.current_offset = end
        return chunk

    @property
    def position_offset(self) -> int:
        """Number of tokens already processed (used for positional embeddings)."""
        return self.current_offset

    @property
    def chunks_total(self) -> int:
        """Total number of chunks."""
        return math.ceil(len(self.prompt_tokens) / self.chunk_size)

    @property
    def chunks_done(self) -> int:
        """Number of chunks already processed."""
        pos = self.current_offset
        return min(pos // self.chunk_size + (1 if pos % self.chunk_size > 0 else 0), self.chunks_total)


def maybe_chunk(
    token_ids: List[int],
    chunk_size: int,
    enabled: bool = True,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> Optional[ChunkState]:
    """Create a ChunkState if the prompt is longer than chunk_size.

    Args:
        token_ids: The full prompt token IDs.
        chunk_size: Maximum tokens per chunk.
        enabled: Whether chunking is enabled at all.
        max_context_tokens: Hard upper bound on prompt length.

    Returns:
        ChunkState if chunking is needed, None if prompt fits in one chunk.

    Raises:
        ValueError: If the prompt exceeds max_context_tokens.
    """
    if not enabled or len(token_ids) <= chunk_size:
        return None
    if len(token_ids) > max_context_tokens:
        raise ValueError(
            f"Prompt length ({len(token_ids)} tokens) exceeds maximum "
            f"context length ({max_context_tokens} tokens)"
        )
    return ChunkState(prompt_tokens=token_ids, chunk_size=chunk_size)
