"""Chunked prefill info for long prompt handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distllm.core.chunked_prefill import ChunkState


@dataclass
class ChunkedPrefillInfo:
    """Tracks chunked prefill state for a sequence.

    Thin adapter around chunked_prefill.ChunkState that provides
    the interface the scheduler needs.
    """

    seq_id: str
    total_prompt_tokens: int
    tokens_processed: int = 0
    chunk_size: int = 0
    chunks_remaining: int = 0

    @property
    def is_complete(self) -> bool:
        return self.tokens_processed >= self.total_prompt_tokens

    @property
    def remaining(self) -> int:
        return self.total_prompt_tokens - self.tokens_processed

    @classmethod
    def from_chunk_state(cls, seq_id: str, cs: ChunkState) -> ChunkedPrefillInfo:
        """Create from a chunked_prefill.ChunkState instance."""
        return cls(
            seq_id=seq_id,
            total_prompt_tokens=len(cs.prompt_tokens),
            tokens_processed=cs.current_offset,
            chunk_size=cs.chunk_size,
            chunks_remaining=cs.chunks_total - cs.chunks_done,
        )

    def advance(self, tokens_processed: int) -> None:
        """Update tokens_processed after a chunk is consumed."""
        self.tokens_processed += tokens_processed
        if self.chunk_size > 0:
            self.chunks_remaining = max(
                0,
                (self.total_prompt_tokens - self.tokens_processed + self.chunk_size - 1) // self.chunk_size,
            )
