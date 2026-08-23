"""Tests for chunked_prefill.py — real objects only, no mocks."""

from __future__ import annotations

import math
import pytest

from distllm.dist.chunked_prefill import (
    ChunkState,
    get_max_context_tokens,
    maybe_chunk,
    DEFAULT_MAX_CONTEXT_TOKENS,
    _MODEL_CONTEXT_LENGTHS,
)


# ---------------------------------------------------------------------------
# get_max_context_tokens
# ---------------------------------------------------------------------------


class TestGetMaxContextTokens:
    """Tests for get_max_context_tokens()."""

    def test_model_info_max_position_embeddings_wins(self) -> None:
        assert (
            get_max_context_tokens(
                model_name="llama-3",
                model_info={"max_position_embeddings": 4096, "max_seq_len": 8192},
            )
            == 4096
        )

    def test_model_info_max_seq_len_fallback(self) -> None:
        assert (
            get_max_context_tokens(
                model_name="llama-3",
                model_info={"max_seq_len": 8192},
            )
            == 8192
        )

    def test_model_info_zero_values_ignored(self) -> None:
        # max_position_embeddings is 0 (falsy) -> fall through
        assert (
            get_max_context_tokens(
                model_name="llama-3",
                model_info={"max_position_embeddings": 0, "max_seq_len": 16384},
            )
            == 16384
        )

    def test_model_info_none_max_seq_len(self) -> None:
        # max_position_embeddings is None -> skip; no max_seq_len -> name lookup
        assert (
            get_max_context_tokens(
                model_name="llama-3",
                model_info={"max_position_embeddings": None},
            )
            == 131072
        )

    def test_known_model_substring_llama3(self) -> None:
        assert get_max_context_tokens(model_name="meta-llama-3-70b") == 131072

    def test_known_model_substring_mistral(self) -> None:
        assert get_max_context_tokens(model_name="mistral-small") == 32768

    def test_known_model_substring_gemma(self) -> None:
        assert get_max_context_tokens(model_name="gemma-2-9b") == 8192

    def test_known_model_substring_phi3(self) -> None:
        assert get_max_context_tokens(model_name="phi-3-mini") == 131072

    def test_known_model_substring_yi(self) -> None:
        assert get_max_context_tokens(model_name="yi-34b") == 200000

    def test_known_model_substring_deepseek(self) -> None:
        assert get_max_context_tokens(model_name="deepseek-coder") == 131072

    def test_known_model_substring_internlm(self) -> None:
        assert get_max_context_tokens(model_name="internlm2-20b") == 32768

    def test_known_model_substring_codellama(self) -> None:
        # Use "CodeLlama" (no version suffix) so it doesn't match "llama-3" earlier
        assert get_max_context_tokens(model_name="CodeLlama") == 16384

    def test_known_model_substring_command_r(self) -> None:
        assert get_max_context_tokens(model_name="command-r-plus") == 128000

    def test_unknown_model_returns_default(self) -> None:
        assert get_max_context_tokens(model_name="gpt-4") == DEFAULT_MAX_CONTEXT_TOKENS

    def test_empty_model_name_returns_default(self) -> None:
        assert get_max_context_tokens(model_name="") == DEFAULT_MAX_CONTEXT_TOKENS

    def test_all_none_args_returns_default(self) -> None:
        assert get_max_context_tokens() == DEFAULT_MAX_CONTEXT_TOKENS

    def test_case_insensitive_matching(self) -> None:
        assert get_max_context_tokens(model_name="LLaMA-3-Turbo") == 131072

    def test_qwen25_variants(self) -> None:
        assert get_max_context_tokens(model_name="Qwen2.5-72B") == 131072
        assert get_max_context_tokens(model_name="qwen-2.5-72b") == 131072

    def test_known_model_substring_prefers_name_over_default(self) -> None:
        """When model_info is empty but name matches a known pattern, use that."""
        assert (
            get_max_context_tokens(model_name="yi-vision", model_info={})
            == 200000
        )


# ---------------------------------------------------------------------------
# ChunkState
# ---------------------------------------------------------------------------


class TestChunkStateInit:
    """Basic instantiation and invariants."""

    def test_initial_state(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        assert state.prompt_tokens == [1, 2, 3, 4, 5]
        assert state.chunk_size == 2
        assert state.current_offset == 0

    @pytest.mark.parametrize(
        "tokens, chunk_size",
        [
            ([], 1),
            ([42], 1),
            ([1, 2, 3], 100),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
        ],
    )
    def test_various_sizes(self, tokens: list[int], chunk_size: int) -> None:
        state = ChunkState(prompt_tokens=tokens, chunk_size=chunk_size)
        assert state.prompt_tokens == tokens
        assert state.chunk_size == chunk_size


class TestChunkStateRemaining:
    @pytest.mark.parametrize(
        "tokens, offset, expected",
        [
            ([1, 2, 3, 4, 5], 0, [1, 2, 3, 4, 5]),
            ([1, 2, 3, 4, 5], 2, [3, 4, 5]),
            ([1, 2, 3, 4, 5], 5, []),
            ([], 0, []),
        ],
    )
    def test_remaining(
        self, tokens: list[int], offset: int, expected: list[int]
    ) -> None:
        state = ChunkState(prompt_tokens=tokens, chunk_size=2, current_offset=offset)
        assert state.remaining == expected


class TestChunkStateIsDone:
    @pytest.mark.parametrize(
        "tokens, offset, expected",
        [
            ([1, 2, 3], 0, False),
            ([1, 2, 3], 2, False),
            ([1, 2, 3], 3, True),
            ([1, 2, 3], 10, True),
            ([], 0, True),
        ],
    )
    def test_is_done(self, tokens: list[int], offset: int, expected: bool) -> None:
        state = ChunkState(prompt_tokens=tokens, chunk_size=2, current_offset=offset)
        assert state.is_done is expected


class TestChunkStateNextChunk:
    def test_returns_full_chunks_then_partial(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        assert state.next_chunk() == [1, 2]
        assert state.current_offset == 2
        assert state.next_chunk() == [3, 4]
        assert state.current_offset == 4
        assert state.next_chunk() == [5]
        assert state.current_offset == 5
        assert state.is_done

    def test_exact_fit(self) -> None:
        state = ChunkState(prompt_tokens=[10, 20, 30, 40], chunk_size=4)
        assert state.next_chunk() == [10, 20, 30, 40]
        assert state.is_done

    def test_single_token_chunks(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3], chunk_size=1)
        assert state.next_chunk() == [1]
        assert state.next_chunk() == [2]
        assert state.next_chunk() == [3]
        assert state.is_done

    def test_chunk_larger_than_prompt(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3], chunk_size=100)
        assert state.next_chunk() == [1, 2, 3]
        assert state.is_done

    def test_empty_prompt_next_chunk_returns_empty(self) -> None:
        state = ChunkState(prompt_tokens=[], chunk_size=4)
        assert state.next_chunk() == []
        assert state.is_done

    def test_next_chunk_on_done_state(self) -> None:
        """Calling next_chunk after completion returns empty list."""
        state = ChunkState(prompt_tokens=[1, 2], chunk_size=2)
        state.next_chunk()
        assert state.is_done
        assert state.next_chunk() == []
        assert state.is_done  # stays done


class TestChunkStatePositionOffset:
    def test_initial_offset(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3], chunk_size=2)
        assert state.position_offset == 0

    def test_after_one_chunk(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        assert state.position_offset == 2

    def test_after_all_chunks(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3], chunk_size=1)
        for _ in range(3):
            state.next_chunk()
        assert state.position_offset == 3


class TestChunkStateChunksTotal:
    @pytest.mark.parametrize(
        "tokens, chunk_size, expected",
        [
            ([1, 2, 3, 4, 5], 2, 3),
            ([1, 2, 3, 4], 2, 2),
            ([1], 2, 1),
            ([], 2, 0),
            ([1, 2, 3, 4, 5], 5, 1),
            ([1, 2, 3], 1, 3),
            ([1, 2, 3], 100, 1),
        ],
    )
    def test_chunks_total(
        self, tokens: list[int], chunk_size: int, expected: int
    ) -> None:
        state = ChunkState(prompt_tokens=tokens, chunk_size=chunk_size)
        assert state.chunks_total == expected


class TestChunkStateChunksDone:
    def test_initial_zero(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        assert state.chunks_done == 0

    def test_after_first_chunk(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        assert state.chunks_done == 1

    def test_after_second_chunk(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        state.next_chunk()
        state.next_chunk()
        assert state.chunks_done == 2

    def test_after_final_partial_chunk(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2)
        for _ in range(3):
            state.next_chunk()
        assert state.chunks_done == 3

    def test_exact_fit_one_chunk(self) -> None:
        state = ChunkState(prompt_tokens=[1, 2, 3, 4], chunk_size=4)
        assert state.chunks_done == 0
        state.next_chunk()
        assert state.chunks_done == 1

    @pytest.mark.parametrize(
        "offset, expected",
        [
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
        ],
    )
    def test_various_offsets(self, offset: int, expected: int) -> None:
        """chunks_done from manual offset with chunk_size=2, prompt_len=5."""
        state = ChunkState(
            prompt_tokens=[1, 2, 3, 4, 5], chunk_size=2, current_offset=offset
        )
        assert state.chunks_done == expected

    def test_empty_prompt_chunks_done_zero(self) -> None:
        state = ChunkState(prompt_tokens=[], chunk_size=4)
        assert state.chunks_done == 0


# ---------------------------------------------------------------------------
# maybe_chunk
# ---------------------------------------------------------------------------


class TestMaybeChunk:
    """Tests for maybe_chunk()."""

    def test_disabled_returns_none(self) -> None:
        assert maybe_chunk([1, 2, 3], chunk_size=2, enabled=False) is None

    def test_small_prompt_returns_none(self) -> None:
        assert maybe_chunk([1, 2], chunk_size=10, enabled=True) is None

    def test_prompt_equal_to_chunk_size_returns_none(self) -> None:
        assert maybe_chunk([1, 2, 3, 4, 5], chunk_size=5, enabled=True) is None

    def test_large_prompt_returns_chunk_state(self) -> None:
        state = maybe_chunk([1, 2, 3, 4, 5], chunk_size=2, enabled=True)
        assert state is not None
        assert isinstance(state, ChunkState)
        assert state.chunk_size == 2
        assert state.prompt_tokens == [1, 2, 3, 4, 5]

    def test_large_prompt_chunking_works(self) -> None:
        state = maybe_chunk([1, 2, 3, 4, 5, 6, 7], chunk_size=3, enabled=True)
        assert state is not None
        assert state.next_chunk() == [1, 2, 3]
        assert state.next_chunk() == [4, 5, 6]
        assert state.next_chunk() == [7]
        assert state.is_done

    def test_prompt_exceeds_max_context_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum context length"):
            maybe_chunk(
                token_ids=[1] * 5000,
                chunk_size=100,
                max_context_tokens=4096,
            )

    def test_prompt_within_max_context_returns_state(self) -> None:
        state = maybe_chunk(
            token_ids=[1] * 5000,
            chunk_size=100,
            max_context_tokens=10000,
        )
        assert state is not None

    def test_model_name_auto_detection_tight_bound(self) -> None:
        """Prompt exactly at model limit should return state, not raise."""
        state = maybe_chunk(
            token_ids=[1] * 131072,
            chunk_size=1024,
            model_name="llama-3",
        )
        assert state is not None

    def test_model_name_auto_detection_exceeded(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum context length"):
            maybe_chunk(
                token_ids=[1] * 131073,
                chunk_size=1024,
                model_name="llama-3",
            )

    def test_model_info_used_for_max_context(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum context length"):
            maybe_chunk(
                token_ids=[1] * 3000,
                chunk_size=100,
                model_info={"max_position_embeddings": 2048},
            )

    def test_max_context_tokens_override_takes_precedence(self) -> None:
        """When explicitly provided, max_context_tokens win over model_name lookup."""
        with pytest.raises(ValueError, match="exceeds maximum context length"):
            maybe_chunk(
                token_ids=[1] * 100,
                chunk_size=10,
                max_context_tokens=50,
                model_name="yi",
            )

    def test_one_token_over_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            maybe_chunk(
                token_ids=[1] * 8193,
                chunk_size=1024,
                max_context_tokens=8192,
            )

    def test_empty_token_ids_disabled(self) -> None:
        assert maybe_chunk([], chunk_size=10, enabled=False) is None

    def test_empty_token_ids_enabled(self) -> None:
        # len([])=0 <= chunk_size -> returns None
        assert maybe_chunk([], chunk_size=10, enabled=True) is None

    def test_returns_chunk_state_when_chunk_size_exceeds_prompt_but_not_context(
        self,
    ) -> None:
        """If prompt fits in chunk, no chunking needed even if context allows it."""
        assert (
            maybe_chunk(
                [1, 2, 3],
                chunk_size=10,
                max_context_tokens=2048,
            )
            is None
        )


# ---------------------------------------------------------------------------
# Module constants sanity
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_default_max_context_positive(self) -> None:
        assert DEFAULT_MAX_CONTEXT_TOKENS > 0

    def test_model_context_lengths_all_positive(self) -> None:
        for name, ctx in _MODEL_CONTEXT_LENGTHS.items():
            assert ctx > 0, f"Non-positive context for {name}"
