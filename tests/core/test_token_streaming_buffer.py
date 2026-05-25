"""Tests for TokenStreamingBuffer: batching, flushing, SSE delivery.

Tests: TokenBatch, TokenStreamingBuffer init, add_token, flush, finish,
max_batch_size trigger, interval trigger, special char trigger,
flush_handler callback, stats, threading safety, edge cases.

Run: pytest tests/core/test_token_streaming_buffer.py -v
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.token_streaming_buffer import TokenBatch, TokenStreamingBuffer


class TestTokenBatch:
    """Tests for TokenBatch dataclass."""

    def test_defaults(self):
        batch = TokenBatch()
        assert batch.tokens == []
        assert batch.token_ids == []
        assert batch.logprobs == []
        assert batch.is_final is False
        assert batch.finish_reason is None
        assert isinstance(batch.created_at, float)

    def test_token_count(self):
        batch = TokenBatch(tokens=["a", "b", "c"])
        assert batch.token_count == 3

    def test_text(self):
        batch = TokenBatch(tokens=["Hello", " ", "world"])
        assert batch.text == "Hello world"

    def test_empty_text(self):
        batch = TokenBatch()
        assert batch.text == ""

    def test_custom_values(self):
        batch = TokenBatch(
            tokens=["x"],
            token_ids=[1],
            logprobs=[-0.5],
            is_final=True,
            finish_reason="stop",
        )
        assert batch.tokens == ["x"]
        assert batch.token_ids == [1]
        assert batch.logprobs == [-0.5]
        assert batch.is_final is True
        assert batch.finish_reason == "stop"


class TestTokenStreamingBufferInit:
    """Tests for TokenStreamingBuffer initialization."""

    def test_defaults(self):
        buffer = TokenStreamingBuffer()
        assert buffer._max_batch_size == 5
        assert buffer._flush_interval == 0.05
        assert buffer._compress_threshold == 4096
        assert buffer._flush_handler is None
        assert buffer._flush_on_special == {"\n", ".", "!"}
        assert buffer._batch.token_count == 0
        assert buffer._total_tokens == 0
        assert buffer._total_batches == 0

    def test_custom_values(self):
        handler = MagicMock()
        buffer = TokenStreamingBuffer(
            flush_handler=handler,
            max_batch_size=10,
            flush_interval_ms=100.0,
            compress_threshold_bytes=2048,
            flush_on_special={"\n\n"},
        )
        assert buffer._max_batch_size == 10
        assert buffer._flush_interval == 0.10
        assert buffer._compress_threshold == 2048
        assert buffer._flush_handler is handler
        assert buffer._flush_on_special == {"\n\n"}


class TestTokenStreamingBufferAddToken:
    """Tests for add_token() behavior."""

    def test_add_token_returns_none_on_partial(self):
        buffer = TokenStreamingBuffer(max_batch_size=5)
        result = buffer.add_token("Hello")
        assert result is None
        assert buffer._batch.token_count == 1

    def test_add_token_flushes_on_max_batch_size(self):
        buffer = TokenStreamingBuffer(max_batch_size=3)
        buffer.add_token("a")
        buffer.add_token("b")
        batch = buffer.add_token("c")
        assert batch is not None
        assert batch.tokens == ["a", "b", "c"]
        assert batch.token_count == 3
        assert batch.is_final is False
        assert buffer._batch.token_count == 0

    def test_add_token_with_ids_and_logprobs(self):
        buffer = TokenStreamingBuffer(max_batch_size=2)
        buffer.add_token("a", token_id=1, logprob=-0.1)
        batch = buffer.add_token("b", token_id=2, logprob=-0.2)
        assert batch.token_ids == [1, 2]
        assert batch.logprobs == [-0.1, -0.2]

    def test_add_token_triggers_flush_handler(self):
        handler = MagicMock()
        buffer = TokenStreamingBuffer(flush_handler=handler, max_batch_size=2)
        buffer.add_token("a")
        assert handler.call_count == 0
        buffer.add_token("b")
        assert handler.call_count == 1
        called_batch = handler.call_args[0][0]
        assert called_batch.tokens == ["a", "b"]

    def test_add_token_flushes_on_special_char(self):
        buffer = TokenStreamingBuffer(max_batch_size=10)
        buffer.add_token("hello")
        buffer.add_token(" ")
        batch = buffer.add_token("world\n")
        assert batch is not None
        assert batch.tokens == ["hello", " ", "world\n"]

    def test_add_token_special_chars_configured(self):
        buffer = TokenStreamingBuffer(max_batch_size=10, flush_on_special={"?"})
        buffer.add_token("how")
        buffer.add_token(" are")
        batch = buffer.add_token(" you?")
        assert batch is not None
        assert batch.text == "how are you?"

    def test_add_token_special_char_in_middle(self):
        buffer = TokenStreamingBuffer(max_batch_size=10)
        buffer.add_token("line1")
        batch = buffer.add_token("\n")
        assert batch is not None
        assert batch.tokens == ["line1", "\n"]

    def test_multiple_special_char_triggers(self):
        buffer = TokenStreamingBuffer(max_batch_size=10)
        buffer.add_token("a.\n")
        assert buffer._batch.token_count == 0


class TestTokenStreamingBufferFlush:
    """Tests for manual flush()."""

    def test_flush_returns_batch_and_resets(self):
        buffer = TokenStreamingBuffer()
        buffer.add_token("a")
        buffer.add_token("b")
        batch = buffer.flush()
        assert batch.tokens == ["a", "b"]
        assert buffer._batch.token_count == 0

    def test_flush_when_empty(self):
        buffer = TokenStreamingBuffer()
        result = buffer.flush()
        assert result is None

    def test_flush_invokes_handler(self):
        handler = MagicMock()
        buffer = TokenStreamingBuffer(flush_handler=handler)
        buffer.add_token("test")
        buffer.flush()
        handler.assert_called_once()
        assert handler.call_args[0][0].tokens == ["test"]

    def test_flush_updates_stats(self):
        buffer = TokenStreamingBuffer(max_batch_size=3)
        buffer.add_token("a")
        buffer.add_token("b")
        buffer.add_token("c")
        assert buffer._total_batches == 1
        assert buffer._total_tokens == 3
        buffer.add_token("d")
        buffer.add_token("e")
        buffer.add_token("f")
        assert buffer._total_batches == 2
        assert buffer._total_tokens == 6


class TestTokenStreamingBufferFinish:
    """Tests for finish() method."""

    def test_finish_marks_final_and_flushes(self):
        buffer = TokenStreamingBuffer()
        buffer.add_token("last")
        batch = buffer.finish()
        assert batch is not None
        assert batch.tokens == ["last"]
        assert batch.is_final is True
        assert batch.finish_reason == "stop"

    def test_finish_with_custom_reason(self):
        buffer = TokenStreamingBuffer()
        buffer.add_token("x")
        batch = buffer.finish(reason="length")
        assert batch.finish_reason == "length"

    def test_finish_empty_buffer(self):
        buffer = TokenStreamingBuffer()
        batch = buffer.finish()
        assert batch is None

    def test_finish_invokes_handler(self):
        handler = MagicMock()
        buffer = TokenStreamingBuffer(flush_handler=handler)
        buffer.add_token("done")
        buffer.finish()
        handler.assert_called_once()
        assert handler.call_args[0][0].is_final is True

    def test_finish_no_double_flush(self):
        handler = MagicMock()
        buffer = TokenStreamingBuffer(flush_handler=handler)
        buffer.add_token("data")
        buffer.flush()
        assert handler.call_count == 1
        buffer.finish()
        assert handler.call_count == 1


class TestTokenStreamingBufferInterval:
    """Tests for time-based flush interval."""

    @patch("time.time")
    def test_flush_on_interval_exact(self, mock_time):
        mock_time.return_value = 100.0
        buffer = TokenStreamingBuffer(max_batch_size=10, flush_interval_ms=100.0)
        buffer.add_token("a")
        assert buffer._batch.token_count == 1
        mock_time.return_value = 100.2
        batch = buffer.add_token("b")
        assert batch is not None
        assert batch.tokens == ["a", "b"]

    @patch("time.time")
    def test_no_flush_before_interval(self, mock_time):
        mock_time.return_value = 100.0
        buffer = TokenStreamingBuffer(max_batch_size=10, flush_interval_ms=100.0)
        buffer.add_token("a")
        mock_time.return_value = 100.05
        result = buffer.add_token("b")
        assert result is None

    @patch("time.time")
    def test_flush_interval_after_batch(self, mock_time):
        mock_time.return_value = 100.0
        buffer = TokenStreamingBuffer(max_batch_size=2, flush_interval_ms=100.0)
        buffer.add_token("a")
        batch = buffer.add_token("b")
        assert batch is not None
        assert batch.tokens == ["a", "b"]
        mock_time.return_value = 100.05
        buffer.add_token("c")
        assert buffer._batch.token_count == 1
        mock_time.return_value = 100.2
        batch = buffer.add_token("d")
        assert batch is not None
        assert batch.tokens == ["c", "d"]


class TestTokenStreamingBufferStats:
    """Tests for stats() method."""

    def test_stats_initial(self):
        buffer = TokenStreamingBuffer()
        stats = buffer.stats()
        assert stats["total_tokens"] == 0
        assert stats["total_batches"] == 0
        assert stats["current_buffered"] == 0
        assert stats["batch_compression_ratio"] == 0.0

    def test_stats_after_tokens(self):
        buffer = TokenStreamingBuffer(max_batch_size=3)
        buffer.add_token("a")
        buffer.add_token("b")
        buffer.add_token("c")
        buffer.add_token("d")
        stats = buffer.stats()
        assert stats["total_tokens"] == 4
        assert stats["total_batches"] == 1
        assert stats["current_buffered"] == 1
        assert stats["batch_compression_ratio"] == 4.0

    def test_stats_thread_safe(self):
        buffer = TokenStreamingBuffer(max_batch_size=3)
        results = []

        def worker():
            for i in range(5):
                buffer.add_token(str(i))
            results.append(buffer.stats())

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = buffer.stats()
        assert stats["total_tokens"] == 15
        assert stats["total_batches"] == 5


class TestTokenStreamingBufferEdgeCases:
    """Tests for edge cases and thread safety."""

    def test_single_token(self):
        buffer = TokenStreamingBuffer(max_batch_size=1)
        batch = buffer.add_token("only")
        assert batch is not None
        assert batch.tokens == ["only"]

    def test_zero_max_batch_size(self):
        buffer = TokenStreamingBuffer(max_batch_size=0)
        batch = buffer.add_token("x")
        assert batch is not None
        assert batch.tokens == ["x"]

    def test_concurrent_add_token(self):
        buffer = TokenStreamingBuffer(max_batch_size=10)
        errors = []

        def worker(n):
            try:
                for i in range(100):
                    buffer.add_token(f"t{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = buffer.stats()
        assert stats["total_tokens"] == 400
        assert stats["total_batches"] == 40

    def test_concurrent_flush_and_add(self):
        buffer = TokenStreamingBuffer(max_batch_size=50)
        errors = []

        def adder():
            try:
                for i in range(100):
                    buffer.add_token(f"a-{i}")
            except Exception as e:
                errors.append(e)

        def flusher():
            try:
                for _ in range(5):
                    buffer.flush()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=flusher),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_empty_token_string(self):
        buffer = TokenStreamingBuffer(max_batch_size=2)
        buffer.add_token("")
        batch = buffer.add_token("b")
        assert batch.tokens == ["", "b"]

    def test_unicode_tokens(self):
        buffer = TokenStreamingBuffer(max_batch_size=2)
        buffer.add_token("你好")
        batch = buffer.add_token("世界")
        assert batch.text == "你好世界"
