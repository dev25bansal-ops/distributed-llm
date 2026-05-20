"""Tests for chunked prefill: ChunkState dataclass and maybe_chunk function.

Tests: ChunkState properties (remaining, is_done, position_offset,
chunks_total, chunks_done), next_chunk, and maybe_chunk function.

Run: pytest tests/core/test_chunked_prefill.py -v
"""

import math

import pytest

from distllm.core.chunked_prefill import ChunkState, maybe_chunk


# --- ChunkState tests ---


class TestChunkStateInit:
    """Tests for ChunkState initialization."""

    def test_basic_init(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        assert state.prompt_tokens == [1, 2, 3, 4, 5]
        assert state.chunk_size == 2
        assert state.current_offset == 0

    def test_empty_prompt(self):
        state = ChunkState(prompt_tokens=[], chunk_size=4)
        assert state.is_done is True
        assert state.remaining == []


class TestChunkStateRemaining:
    """Tests for the remaining property."""

    def test_all_remaining_at_start(self):
        state = ChunkState(prompt_tokens=[10, 20, 30], chunk_size=2)
        assert state.remaining == [10, 20, 30]

    def test_partial_remaining(self):
        state = ChunkState(prompt_tokens=[10, 20, 30, 40], chunk_size=2)
        state.current_offset = 2
        assert state.remaining == [30, 40]

    def test_no_remaining_when_done(self):
        state = ChunkState(prompt_tokens=[1, 2], chunk_size=2)
        state.current_offset = 2
        assert state.remaining == []


class TestChunkStateIsDone:
    """Tests for the is_done property."""

    def test_not_done_initially(self):
        state = ChunkState(prompt_tokens=[1, 2, 3], chunk_size=2)
        assert state.is_done is False

    def test_done_when_offset_equals_length(self):
        state = ChunkState(prompt_tokens=[1, 2], chunk_size=2)
        state.current_offset = 2
        assert state.is_done is True

    def test_done_when_offset_exceeds_length(self):
        state = ChunkState(prompt_tokens=[1, 2], chunk_size=2)
        state.current_offset = 3
        assert state.is_done is True


class TestChunkStateNextChunk:
    """Tests for the next_chunk method."""

    def test_first_chunk(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        chunk = state.next_chunk()
        assert chunk == [1, 2]
        assert state.current_offset == 2

    def test_second_chunk(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        chunk = state.next_chunk()
        assert chunk == [3, 4]
        assert state.current_offset == 4

    def test_last_chunk_partial(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        state.next_chunk()
        chunk = state.next_chunk()
        assert chunk == [5]
        # offset advances by chunk_size even past end
        assert state.current_offset == 6

    def test_empty_chunk_when_done(self):
        state = ChunkState(prompt_tokens=[1, 2], chunk_size=2)
        state.next_chunk()
        chunk = state.next_chunk()
        assert chunk == []


class TestChunkStatePositionOffset:
    """Tests for the position_offset property."""

    def test_zero_initially(self):
        state = ChunkState(prompt_tokens=[1, 2, 3], chunk_size=2)
        assert state.position_offset == 0

    def test_after_chunks(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        assert state.position_offset == 2
        state.next_chunk()
        assert state.position_offset == 4


class TestChunkStateChunksTotal:
    """Tests for the chunks_total property."""

    def test_exact_division(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4], chunk_size=2)
        assert state.chunks_total == 2

    def test_partial_last_chunk(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        assert state.chunks_total == 3

    def test_single_chunk(self):
        state = ChunkState(prompt_tokens=[1], chunk_size=4)
        assert state.chunks_total == 1

    def test_empty_prompt(self):
        state = ChunkState(prompt_tokens=[], chunk_size=4)
        assert state.chunks_total == 0


class TestChunkStateChunksDone:
    """Tests for the chunks_done property."""

    def test_zero_initially(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4], chunk_size=2)
        assert state.chunks_done == 0

    def test_after_first_chunk(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4], chunk_size=2)
        state.next_chunk()
        assert state.chunks_done == 1

    def test_after_all_chunks(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4], chunk_size=2)
        state.next_chunk()
        state.next_chunk()
        assert state.chunks_done == 2

    def test_partial_chunk_counts(self):
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        assert state.chunks_done == 1


# --- maybe_chunk tests ---


class TestMaybeChunk:
    """Tests for the maybe_chunk function."""

    def test_returns_none_when_disabled(self):
        result = maybe_chunk([1, 2, 3, 4, 5], chunk_size=2, enabled=False)
        assert result is None

    def test_returns_none_when_prompt_fits(self):
        result = maybe_chunk([1, 2, 3], chunk_size=4, enabled=True)
        assert result is None

    def test_returns_none_when_equal_size(self):
        result = maybe_chunk([1, 2, 3, 4], chunk_size=4, enabled=True)
        assert result is None

    def test_returns_chunk_state_when_prompt_too_long(self):
        tokens = list(range(10))
        result = maybe_chunk(tokens, chunk_size=3, enabled=True)
        assert result is not None
        assert isinstance(result, ChunkState)
        assert result.prompt_tokens == tokens
        assert result.chunk_size == 3
        assert result.current_offset == 0

    def test_chunk_state_can_process_all_tokens(self):
        tokens = [1, 2, 3, 4, 5, 6, 7]
        state = maybe_chunk(tokens, chunk_size=3)
        assert state is not None

        chunks = []
        while not state.is_done:
            chunks.append(state.next_chunk())

        # Flatten and compare
        flattened = [t for chunk in chunks for t in chunk]
        assert flattened == tokens
