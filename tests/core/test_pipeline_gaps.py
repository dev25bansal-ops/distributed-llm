"""Gap tests: TokenGenerator sampling, stop conditions, streaming buffer, backpressure."""

import time

import pytest
import torch

from distllm.core.token_generator import TokenGenerator
from distllm.core.token_streaming_buffer import TokenStreamingBuffer, TokenBatch
from distllm.core.streaming_generator import StreamingGenerator, StreamChunk, StreamingConfig
from distllm.core.param_update_channel import ParamUpdateChannel, GenerationParams


class TestTokenGeneratorSampling:
    def test_sample_returns_tokens(self):
        gen = TokenGenerator()
        logits = torch.randn(1, 100)
        tokens, _ = gen.sample(logits, temperature=1.0)
        assert tokens.shape == (1,)

    def test_top_k_filters_correctly(self):
        gen = TokenGenerator()
        logits = torch.randn(1, 1, 100)
        filtered = gen._top_k_top_p_filtering(logits, top_k=10)
        kept = (filtered > float('-inf')).sum().item()
        assert kept <= 10

    def test_top_p_nucleus(self):
        logits = torch.randn(1, 50)
        result = TokenGenerator._top_k_top_p_filtering(logits, top_k=50, top_p=0.9, min_tokens_to_keep=1)
        assert result.shape[-1] == 50

    def test_sample_temperature_zero_is_greedy(self):
        gen = TokenGenerator()
        logits = torch.randn(1, 100)
        logits[0, 42] = 100.0
        tokens, _ = gen.sample(logits, temperature=0.0, top_p=1.0, top_k=0)
        assert tokens[0].item() == 42

    def test_compute_logprobs_valid(self):
        logits = torch.randn(1, 20)
        lp = TokenGenerator._compute_logprobs(logits, torch.tensor([5]), top_logprobs=2)
        assert isinstance(lp, dict)
        assert "logprob" in lp


class TestParamUpdateChannel:
    def test_register_and_get(self):
        ch = ParamUpdateChannel()
        ch.register("req-1", GenerationParams(temperature=0.5))
        params = ch.get("req-1")
        assert params.temperature == 0.5

    def test_update_modifies_params(self):
        ch = ParamUpdateChannel()
        ch.register("req-1")
        ch.update("req-1", temperature=0.3)
        params = ch.get("req-1")
        assert params.temperature == 0.3

    def test_unregister_removes(self):
        ch = ParamUpdateChannel()
        ch.register("req-1")
        ch.unregister("req-1")
        assert ch.get("req-1") is None

    def test_list_requests(self):
        ch = ParamUpdateChannel()
        ch.register("req-1")
        ch.register("req-2")
        assert "req-1" in ch.list_requests()
        assert "req-2" in ch.list_requests()


class TestTokenStreamingBuffer:
    def test_add_token_returns_batch_on_flush(self):
        buf = TokenStreamingBuffer(max_batch_size=3)
        result1 = buf.add_token("hello")
        assert result1 is None
        result2 = buf.add_token(" ")
        assert result2 is None
        result3 = buf.add_token("world")
        assert result3 is not None
        assert result3.token_count == 3

    def test_finish_flushes_remaining(self):
        buf = TokenStreamingBuffer(max_batch_size=10)
        buf.add_token("hello")
        batch = buf.finish(reason="stop")
        assert batch is not None
        assert batch.is_final
        assert batch.finish_reason == "stop"
        assert batch.text == "hello"

    def test_flush_on_special_character(self):
        buf = TokenStreamingBuffer(max_batch_size=10, flush_on_special={".", "\n"})
        buf.add_token("hello")
        batch = buf.add_token(".\n")
        assert batch is not None
        assert batch.text == "hello.\n"

    def test_time_based_flush(self):
        buf = TokenStreamingBuffer(max_batch_size=100, flush_interval_ms=10)
        buf.add_token("a")
        time.sleep(0.02)
        batch = buf.add_token("b")
        assert batch is not None

    def test_stats_returns_dict(self):
        buf = TokenStreamingBuffer()
        buf.add_token("test")
        buf.finish()
        s = buf.stats()
        assert "total_tokens" in s
        assert "total_batches" in s
        assert s["total_tokens"] >= 1

    def test_token_batch_properties(self):
        batch = TokenBatch(tokens=["a", "b"], is_final=True, finish_reason="stop")
        assert batch.token_count == 2
        assert batch.text == "ab"


class TestStreamingGeneratorSSE:
    def test_stream_chunk_to_sse(self):
        chunk = StreamChunk(id="test-1", model="test")
        sse = chunk.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

    def test_data_done_format(self):
        done = StreamChunk.data_done()
        assert done == "data: [DONE]\n\n"

    def test_streaming_config_defaults(self):
        cfg = StreamingConfig()
        assert cfg.max_tokens == 1024
        assert cfg.stream_chunk_size == 1
        assert cfg.include_usage is False
        assert cfg.backpressure_sleep_s == 0.01
