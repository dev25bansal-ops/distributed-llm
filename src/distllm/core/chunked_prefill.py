"""Chunked prefill for predictable latency with long prompts."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkState:
    """Tracks chunking progress for a single request's prompt.

    When a prompt is longer than chunk_size, it is split into chunks.
    Each chunk is processed sequentially through the pipeline, building
    up the KV cache incrementally.
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
        """Get the next chunk of tokens and advance the offset."""
        chunk = self.prompt_tokens[self.current_offset:self.current_offset + self.chunk_size]
        self.current_offset += self.chunk_size
        return chunk

    @property
    def position_offset(self) -> int:
        """Number of tokens already processed (used for positional embeddings)."""
        return self.current_offset

    @property
    def chunks_total(self) -> int:
        """Total number of chunks."""
        import math
        return math.ceil(len(self.prompt_tokens) / self.chunk_size)

    @property
    def chunks_done(self) -> int:
        """Number of chunks already processed."""
        import math
        return min(self.current_offset // self.chunk_size + (1 if self.current_offset % self.chunk_size > 0 else 0), self.chunks_total)


def maybe_chunk(
    token_ids: List[int],
    chunk_size: int,
    enabled: bool = True,
) -> Optional[ChunkState]:
    """Create a ChunkState if the prompt is longer than chunk_size.

    Returns None if chunking is disabled or the prompt fits in one chunk.
    """
    if not enabled or len(token_ids) <= chunk_size:
        return None
    return ChunkState(prompt_tokens=token_ids, chunk_size=chunk_size)
