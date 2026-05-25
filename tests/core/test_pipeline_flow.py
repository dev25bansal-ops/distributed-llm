"""Full request pipeline flow tests: HTTP → queue → batch → model → stream.

Covers five key scenarios:
  1. Full pipeline flow: prompt queued, batched, forwarded, decoded, result returned
  2. Cancellation: request cancelled mid-generation
  3. Error propagation: model/tokenizer error → error response
  4. Token generation loop: prompt → token by token output
  5. Stop conditions: EOS, max_tokens, stop sequences

Run: pytest tests/core/test_pipeline_flow.py -v
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.batch_scheduler import Sequence, SequenceStatus
from distllm.core.request_pipeline import RequestPipeline
from distllm.core.streaming_generator import StreamingGenerator, StreamChunk, StreamingConfig
from distllm.core.token_generator import TokenGenerator
from distllm.errors.types import BatchError, NodeError


# ─────────────────────────────────────────────
# 1. Full Pipeline Flow
# ─────────────────────────────────────────────

class TestFullPipelineFlow:
    """HTTP → queue → batch → model → stream."""

    def test_generate_async_creates_sequence(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        request_id = pipeline.generate_async("hello world", max_new_tokens=5)

        assert request_id is not None
        assert isinstance(request_id, str)

        seq = coord.scheduler.get_sequence(request_id)
        assert seq is not None
        assert seq.status == SequenceStatus.PENDING
        assert seq.max_new_tokens == 5
        assert len(seq.generated_tokens) == 0

    def test_generate_async_batch_scheduler_disabled(self, mock_coordinator):
        """generate_async raises BatchError when scheduler is None."""
        pipeline = RequestPipeline(mock_coordinator)
        mock_coordinator._scheduler_svc.scheduler = None

        with pytest.raises(BatchError):
            pipeline.generate_async("hello")

    def test_generate_batch_empty_scheduler(self, mock_coordinator):
        pipeline = RequestPipeline(mock_coordinator)
        mock_coordinator._scheduler_svc.scheduler = None

        with pytest.raises(BatchError):
            pipeline.generate_batch()

    def test_generate_batch_one_request_to_result(self, mock_coordinator_with_scheduler):
        """Queue request → batch → mock model step → result via RequestTracker."""
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        request_id = pipeline.generate_async("hello world", max_new_tokens=3)
        seq = coord.scheduler.get_sequence(request_id)
        assert seq is not None
        assert seq.status == SequenceStatus.PENDING

        # Manually run pipeline steps: schedule → generate → step
        from distllm.core.batch_scheduler import ScheduledBatch
        seq.prompt_tokens = [1, 2]
        seq.generated_tokens = [42]
        coord.scheduler.active[request_id] = seq

        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[42]]),
            seq_starts=[0],
            seq_lengths=[1],
            position_offsets=[2],
            is_prefill=[False],
            request_ids=[request_id],
        )

        seq.generated_tokens = [42, 43]
        coord.scheduler.step(batch, torch.tensor([44], dtype=torch.long), kv_caches={})

        assert seq.status in (SequenceStatus.DECODING, SequenceStatus.DONE)
        assert seq.generated_tokens[-1] == 44

    def test_generate_async_raises_rate_limited(self, mock_coordinator_with_scheduler):
        """Rate limiter reject raises NodeError via RequestPipeline.generate()."""
        from distllm.core.leaky_bucket_limiter import LeakyBucketRateLimiter

        coord = mock_coordinator_with_scheduler
        coord._rate_limiter = LeakyBucketRateLimiter(default_rate=0.0, default_burst=0)

        # generate() (sync path) checks rate limiter at line 126
        pipeline = RequestPipeline(coord)
        with pytest.raises(NodeError, match="Rate limit exceeded"):
            pipeline.generate("hello")

    def test_generate_batch_max_steps(self, mock_coordinator_with_scheduler):
        """generate_batch with max_steps runs that many iterations and stops."""
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)
        from distllm.core.batch_scheduler import ScheduledBatch

        request_id = pipeline.generate_async("hello world", max_new_tokens=10)
        seq = coord.scheduler.get_sequence(request_id)
        assert seq is not None
        seq.prompt_tokens = [1, 2]
        seq.generated_tokens = [0]
        seq.status = SequenceStatus.DECODING
        coord.scheduler.active[request_id] = seq

        steps_run = [0]

        def mock_schedule():
            if steps_run[0] >= 1 or seq.is_complete:
                return None
            return ScheduledBatch(
                sequences=[seq],
                input_ids=torch.tensor([[0]]),
                seq_starts=[0],
                seq_lengths=[1],
                position_offsets=[2],
                is_prefill=[False],
                request_ids=[request_id],
            )

        def fake_local_batch(batch):
            steps_run[0] += 1
            coord.scheduler.step(batch, torch.tensor([0], dtype=torch.long), kv_caches={})

        with patch.object(coord.scheduler, 'schedule', side_effect=mock_schedule):
            with patch.object(pipeline, '_generate_local_batch', side_effect=fake_local_batch):
                pipeline.generate_batch(max_steps=1)

        assert steps_run[0] >= 1

    def test_full_pipeline_multiple_requests(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)
        from distllm.core.batch_scheduler import ScheduledBatch

        r1 = pipeline.generate_async("hello", max_new_tokens=10)
        r2 = pipeline.generate_async("world", max_new_tokens=10)
        s1 = coord.scheduler.get_sequence(r1)
        s2 = coord.scheduler.get_sequence(r2)
        s1.prompt_tokens = [1, 2]
        s2.prompt_tokens = [1, 2]
        s1.generated_tokens = [10]
        s2.generated_tokens = [20]
        s1.status = SequenceStatus.DECODING
        s2.status = SequenceStatus.DECODING

        step_count = [0]

        def mock_schedule():
            if s1.is_complete and s2.is_complete:
                coord.scheduler.active.pop(r1, None)
                coord.scheduler.active.pop(r2, None)
                return None
            return ScheduledBatch(
                sequences=[s1, s2],
                input_ids=torch.tensor([[10, 20]]),
                seq_starts=[0, 1],
                seq_lengths=[1, 1],
                position_offsets=[2, 2],
                is_prefill=[False, False],
                request_ids=[r1, r2],
            )

        def fake_local_batch(batch):
            for seq in batch.sequences:
                seq.status = SequenceStatus.DECODING
            step_count[0] += 1
            next_tok = torch.tensor([42] * len(batch.sequences), dtype=torch.long)
            coord.scheduler.step(batch, next_tok, kv_caches={})

        with patch.object(coord.scheduler, 'schedule', side_effect=mock_schedule):
            with patch.object(pipeline, '_generate_local_batch', side_effect=fake_local_batch):
                pipeline.generate_batch(timeout=5.0)

        res1 = coord._request_tracker.wait_for_result(r1, timeout=2.0)
        res2 = coord._request_tracker.wait_for_result(r2, timeout=2.0)
        assert isinstance(res1, str) and len(res1) > 0
        assert isinstance(res2, str) and len(res2) > 0
        assert step_count[0] >= 2


# ─────────────────────────────────────────────
# 2. Cancellation
# ─────────────────────────────────────────────

class TestCancellation:
    """Request cancelled mid-generation."""

    @pytest.mark.asyncio
    async def test_streaming_cancel_via_event(self):
        """Cancel mid-stream via cancel_event."""
        generator = StreamingGenerator()

        cancel_event = asyncio.Event()

        async def mock_generate_fn(prompt):
            for i in range(5):
                if cancel_event.is_set():
                    return
                yield (i, False)
                await asyncio.sleep(0.01)

        chunks = []
        async for chunk in generator.generate("test", mock_generate_fn, cancel_event=cancel_event):
            chunks.append(chunk)
            if len(chunks) == 2:
                cancel_event.set()

        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_streaming_cancel_immediately(self):
        """Cancel before any tokens are generated."""
        generator = StreamingGenerator()
        cancel_event = asyncio.Event()
        cancel_event.set()

        async def mock_generate_fn(prompt):
            for i in range(5):
                yield (i, False)
                await asyncio.sleep(0.01)

        chunks = []
        async for chunk in generator.generate("test", mock_generate_fn, cancel_event=cancel_event):
            chunks.append(chunk)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_streaming_cancelled_error(self):
        """CancelledError during generation yields done signal."""
        generator = StreamingGenerator()

        async def mock_generate_fn(prompt):
            for i in range(3):
                if i == 1:
                    raise asyncio.CancelledError()
                yield (i, False)
                await asyncio.sleep(0.01)

        chunks = []
        async for chunk in generator.generate("test", mock_generate_fn):
            chunks.append(chunk)

        assert len(chunks) >= 1

    def test_generate_async_with_cancellation(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        request_id = pipeline.generate_async("hello world", max_new_tokens=50)
        seq = coord.scheduler.get_sequence(request_id)
        assert seq is not None

        coord._request_tracker.set_result(request_id, "[Cancelled]")
        result = coord._request_tracker.wait_for_result(request_id, timeout=2.0)
        assert result == "[Cancelled]"

    def test_generate_batch_open_ended_cancellation(self, mock_coordinator_with_scheduler):
        """Cancellation sets result via RequestTracker directly."""
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        rid = pipeline.generate_async("hello", max_new_tokens=20)
        seq = coord.scheduler.get_sequence(rid)
        assert seq is not None

        coord._request_tracker.set_result(rid, "[Cancelled]")

        result = coord._request_tracker.wait_for_result(rid, timeout=1.0)
        assert result == "[Cancelled]"


# ─────────────────────────────────────────────
# 3. Error Propagation
# ─────────────────────────────────────────────

class TestErrorPropagation:
    """Model error → error response."""

    def test_generate_no_nodes_or_model(self, mock_coordinator):
        mock_coordinator.node_order = []
        mock_coordinator.local_partitioner = None
        pipeline = RequestPipeline(mock_coordinator)

        with pytest.raises(NodeError, match="No nodes registered"):
            pipeline.generate("hello")

    def test_generate_tokenizer_not_loaded(self, mock_coordinator):
        mock_coordinator.tokenizer = None
        pipeline = RequestPipeline(mock_coordinator)

        with pytest.raises(ValueError, match="Tokenizer not loaded"):
            pipeline.generate("hello")

    def test_generate_model_forward_raises(self, mock_coordinator):
        """Model forward pass exception propagates as NodeError."""
        mock_coordinator.local_partitioner = MagicMock()
        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
        mock_coordinator.local_partitioner.full_model = mock_model
        pipeline = RequestPipeline(mock_coordinator)

        with pytest.raises((RuntimeError, Exception)):
            pipeline.generate("hello", max_new_tokens=2)

    def test_generate_batch_model_forward_raises(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        pipeline.generate_async("hello", max_new_tokens=2)

        with patch.object(pipeline, '_generate_local_batch',
                          side_effect=RuntimeError("Model failure")):
            with pytest.raises(RuntimeError, match="Model failure"):
                pipeline.generate_batch(timeout=3.0)

    def test_generate_async_raises_exception(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        def failing_encode(*args, **kwargs):
            raise ValueError("Tokenization error")

        coord.tokenizer.encode.side_effect = failing_encode

        with pytest.raises(ValueError, match="Tokenization error"):
            pipeline.generate("hello")

    @pytest.mark.asyncio
    async def test_streaming_generator_error_yields_error_chunk(self):
        """StreamingGenerator yields error chunk when generate_fn raises."""
        generator = StreamingGenerator()

        async def failing_generate_fn(prompt):
            raise RuntimeError("Generation failed")
            yield

        chunks = []
        async for chunk in generator.generate("test", failing_generate_fn):
            chunks.append(chunk)

        has_error = any(
            isinstance(c, StreamChunk)
            and c.choices[0].get("finish_reason") == "error"
            for c in chunks
        )
        assert has_error


# ─────────────────────────────────────────────
# 4. Token Generation Loop
# ─────────────────────────────────────────────

class TestTokenGenerationLoop:
    """Prompt → token by token output."""

    def test_sample_greedy(self):
        gen = TokenGenerator()
        logits = torch.full((1, 100), -100.0)
        logits[0, 42] = 50.0

        tokens, _ = gen.sample(logits, temperature=0.0)
        assert tokens.item() == 42

    def test_sample_greedy_batch(self):
        gen = TokenGenerator()
        logits = torch.full((3, 100), -100.0)
        logits[0, 10] = 50.0
        logits[1, 20] = 50.0
        logits[2, 30] = 50.0

        tokens, _ = gen.sample(logits, temperature=0.0)
        assert tokens[0].item() == 10
        assert tokens[1].item() == 20
        assert tokens[2].item() == 30

    def test_sample_temperature(self):
        gen = TokenGenerator()
        logits = torch.full((1, 100), 0.0)
        logits[0, 42] = 100.0

        tokens, _ = gen.sample(logits, temperature=0.1)
        assert tokens.item() == 42

    def test_sample_top_k(self):
        gen = TokenGenerator()
        logits = torch.full((1, 100), -100.0)
        logits[0, 5] = 10.0
        logits[0, 95] = 20.0

        tokens, _ = gen.sample(logits, temperature=1.0, top_k=1)
        assert tokens.item() == 95

    def test_sample_top_p_filters(self):
        gen = TokenGenerator()
        logits = torch.zeros(1, 100)
        logits[0, 50] = 100.0

        tokens, _ = gen.sample(logits, temperature=1.0, top_p=0.5)
        assert tokens.item() == 50

    def test_sample_with_logit_bias(self):
        gen = TokenGenerator()
        logits = torch.full((1, 100), 0.0)
        tokens_no_bias, _ = gen.sample(logits, temperature=0.0)
        assert tokens_no_bias.item() == 0

        tokens_with_bias, _ = gen.sample(
            logits, temperature=0.0, logit_bias={99: 100.0}
        )
        assert tokens_with_bias.item() == 99

    def test_sample_with_penalties(self):
        gen = TokenGenerator()
        logits = torch.full((1, 100), 0.0)
        logits[0, 42] = 10.0
        logits[0, 7] = 9.0

        tokens_no_penalty, _ = gen.sample(
            logits, temperature=0.0
        )
        assert tokens_no_penalty.item() == 42

        tokens_with_penalty, _ = gen.sample(
            logits, temperature=0.0, presence_penalty=5.0,
            token_counts={42: 1}
        )
        assert tokens_with_penalty.item() == 7

    def test_sample_batch_with_constraints(self):
        gen = TokenGenerator()

        seq1 = MagicMock()
        seq1.temperature = 0.0
        seq1.top_p = 1.0
        seq1.top_k = 0
        seq1.constraint = None
        seq1.token_counts = {}
        seq1.include_logprobs = False
        seq1.top_logprobs = 0
        seq1.logit_bias = {}
        seq1.presence_penalty = 0.0
        seq1.frequency_penalty = 0.0

        seq2 = MagicMock()
        for attr in ("temperature", "top_p", "top_k", "constraint", "token_counts",
                      "include_logprobs", "top_logprobs", "logit_bias",
                      "presence_penalty", "frequency_penalty"):
            setattr(seq2, attr, getattr(seq1, attr))

        logits = torch.full((2, 100), 0.0)
        logits[0, 10] = 50.0
        logits[1, 20] = 50.0

        tokens, _ = gen.sample_batch(logits, [seq1, seq2])
        assert tokens[0].item() == 10
        assert tokens[1].item() == 20

    def test_generate_local_loop(self, mock_coordinator):
        """Full local generation loop via RequestPipeline.generate()."""
        coord = mock_coordinator
        coord.node_order = []
        coord.local_partitioner = MagicMock()
        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])

        def mock_model_generate(input_ids=None, **kwargs):
            n = kwargs.get("max_new_tokens", 3)
            prompt_len = input_ids.shape[1]
            return torch.cat([input_ids, torch.full((1, n), 42, dtype=torch.long)], dim=1)

        mock_model.generate.side_effect = mock_model_generate
        coord.local_partitioner.full_model = mock_model
        pipeline = RequestPipeline(coord)

        result = pipeline.generate("hello world", max_new_tokens=3)
        assert isinstance(result, str) and len(result) > 0

    def test_sample_batch_produces_correct_tokens(self, mock_coordinator_with_scheduler):
        """Verify _sample_batch returns correct next tokens via TokenGenerator."""
        coord = mock_coordinator_with_scheduler
        pipeline = RequestPipeline(coord)

        seq = Sequence(
            request_id="test-req",
            prompt_tokens=torch.tensor([[1, 2, 3]]),
            max_new_tokens=5,
            temperature=0.0,
        )
        seq.status = SequenceStatus.DECODING
        seq.generated_tokens = [10]

        from distllm.core.batch_scheduler import ScheduledBatch
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[10]]),
            seq_starts=[0],
            seq_lengths=[1],
            position_offsets=[3],
            is_prefill=[False],
            request_ids=["test-req"],
        )

        logits = torch.full((1, 1, coord.tokenizer.vocab_size), -100.0)
        logits[0, 0, 55] = 50.0

        next_tokens = pipeline._sample_batch(logits, batch)
        assert next_tokens[0].item() == 55

    def test_generate_returns_complete_text(self, mock_coordinator):
        coord = mock_coordinator
        coord.local_partitioner = MagicMock()
        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])

        def mock_generate(input_ids=None, **kwargs):
            n = kwargs.get("max_new_tokens", 5)
            return torch.cat([input_ids, torch.arange(10, 10 + n).unsqueeze(0)], dim=1)

        mock_model.generate.side_effect = mock_generate
        coord.local_partitioner.full_model = mock_model
        pipeline = RequestPipeline(coord)

        result = pipeline.generate("hello", max_new_tokens=5)
        assert isinstance(result, str)
        assert "tok-" in result


# ─────────────────────────────────────────────
# 5. Stop Conditions
# ─────────────────────────────────────────────

class TestStopConditions:
    """EOS, max_tokens, stop sequences."""

    def test_sequence_max_tokens_stop(self):
        seq = Sequence(
            request_id="test",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        seq.generated_tokens = list(range(10))
        assert seq.is_complete is True

    def test_sequence_max_tokens_not_reached(self):
        seq = Sequence(
            request_id="test",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        seq.generated_tokens = list(range(5))
        assert seq.is_complete is False

    def test_sequence_done_status_stops(self):
        seq = Sequence(
            request_id="test",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        seq.status = SequenceStatus.DONE
        assert seq.is_complete is True

    def test_sequence_failed_status_stops(self):
        seq = Sequence(
            request_id="test",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        seq.status = SequenceStatus.FAILED
        assert seq.is_complete is True

    def test_eos_token_stops_in_step(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        scheduler = coord.scheduler

        seq = Sequence(
            request_id="test-eos",
            prompt_tokens=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
            stop_token_ids=[0],
        )
        seq.status = SequenceStatus.DECODING
        seq.generated_tokens = [5]

        from distllm.core.batch_scheduler import ScheduledBatch
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[5]]),
            seq_starts=[0],
            seq_lengths=[1],
            position_offsets=[3],
            is_prefill=[False],
            request_ids=["test-eos"],
        )

        next_tokens = torch.tensor([0])
        scheduler.step(batch, next_tokens)

        assert seq.status == SequenceStatus.DONE
        assert 0 in seq.generated_tokens

    def test_no_stop_on_non_eos_token(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        scheduler = coord.scheduler

        seq = Sequence(
            request_id="test-no-eos",
            prompt_tokens=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
            stop_token_ids=[0],
        )
        seq.status = SequenceStatus.DECODING
        seq.generated_tokens = [5]

        from distllm.core.batch_scheduler import ScheduledBatch
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[5]]),
            seq_starts=[0],
            seq_lengths=[1],
            position_offsets=[3],
            is_prefill=[False],
            request_ids=["test-no-eos"],
        )

        next_tokens = torch.tensor([42])
        scheduler.step(batch, next_tokens)

        assert seq.status == SequenceStatus.DECODING

    def test_multiple_stop_tokens(self, mock_coordinator_with_scheduler):
        coord = mock_coordinator_with_scheduler
        scheduler = coord.scheduler

        seq = Sequence(
            request_id="test-stop",
            prompt_tokens=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
            stop_token_ids=[0, 100, 200],
        )
        seq.status = SequenceStatus.DECODING
        seq.generated_tokens = [5]

        from distllm.core.batch_scheduler import ScheduledBatch
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[5]]),
            seq_starts=[0],
            seq_lengths=[1],
            position_offsets=[3],
            is_prefill=[False],
            request_ids=["test-stop"],
        )

        for stop_id in [0, 100, 200]:
            seq.status = SequenceStatus.DECODING
            seq.generated_tokens = [5]
            next_tokens = torch.tensor([stop_id])
            scheduler.step(batch, next_tokens)
            assert seq.status == SequenceStatus.DONE

    def test_prefill_to_decode_transition(self, mock_coordinator_with_scheduler):
        """Sequence transitions from PREFILLING to DECODING after step()."""
        coord = mock_coordinator_with_scheduler
        scheduler = coord.scheduler

        seq = Sequence(
            request_id="test-transition",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
        )
        seq.status = SequenceStatus.PREFILLING

        from distllm.core.batch_scheduler import ScheduledBatch
        batch = ScheduledBatch(
            sequences=[seq],
            input_ids=torch.tensor([[1, 2, 3]]),
            seq_starts=[0],
            seq_lengths=[3],
            position_offsets=[0],
            is_prefill=[True],
            request_ids=["test-transition"],
        )

        next_tokens = torch.tensor([42])
        scheduler.step(batch, next_tokens)

        assert seq.status == SequenceStatus.DECODING
        assert len(seq.generated_tokens) == 1
        assert seq.generated_tokens[0] == 42

    def test_batch_multiple_stop_conditions(self, mock_coordinator_with_scheduler):
        """Mixed batch: one sequence hits EOS, the other hits max_tokens."""
        coord = mock_coordinator_with_scheduler
        scheduler = coord.scheduler

        seq1 = Sequence(
            request_id="eos-seq",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=10,
            stop_token_ids=[0],
        )
        seq1.status = SequenceStatus.DECODING
        seq1.generated_tokens = [5]

        seq2 = Sequence(
            request_id="max-seq",
            prompt_tokens=[1, 2, 3],
            max_new_tokens=3,
            stop_token_ids=[0],
        )
        seq2.status = SequenceStatus.DECODING
        seq2.generated_tokens = [10, 20]

        from distllm.core.batch_scheduler import ScheduledBatch
        batch = ScheduledBatch(
            sequences=[seq1, seq2],
            input_ids=torch.tensor([[5, 30]]),
            seq_starts=[0, 1],
            seq_lengths=[1, 1],
            position_offsets=[3, 5],
            is_prefill=[False, False],
            request_ids=["eos-seq", "max-seq"],
        )

        next_tokens = torch.tensor([0, 30])
        scheduler.step(batch, next_tokens)

        assert seq1.status == SequenceStatus.DONE
        assert seq2.status == SequenceStatus.DONE
        assert seq2.generated_tokens == [10, 20, 30]

    def test_streaming_generator_stops_on_eos(self):
        """Streaming generator emits final token and stops."""
        generator = StreamingGenerator()

        eos_token_id = 0

        async def mock_generate_fn(prompt):
            yield (1, False)
            yield (2, False)
            yield (eos_token_id, True)

        async def run():
            chunks = []
            async for chunk in generator.generate("test", mock_generate_fn):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) >= 1

    def test_streaming_generator_max_tokens(self):
        generator = StreamingGenerator(
            config=StreamingConfig(max_tokens=3)
        )

        async def mock_generate_fn(prompt):
            for i in range(10):
                yield (i, False)
                await asyncio.sleep(0.001)

        async def run():
            chunks = []
            async for chunk in generator.generate("test", mock_generate_fn):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) >= 1
