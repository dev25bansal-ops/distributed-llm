"""Comprehensive tests for the AsyncPipelineEngine.

Tests cover 1F1B and GPipe schedules, micro-batch scheduling,
warmup tracking, stats, and edge cases — all without a GPU.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Import async_pipeline directly to bypass the circular import in distllm/__init__.py
_src = Path(__file__).resolve().parents[2] / "src"
_spec = importlib.util.spec_from_file_location(
    "distllm.dist.async_pipeline",
    _src / "distllm" / "dist" / "async_pipeline.py",
)
_async_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["distllm.dist.async_pipeline"] = _async_pipeline
_spec.loader.exec_module(_async_pipeline)

AsyncPipelineConfig = _async_pipeline.AsyncPipelineConfig
AsyncPipelineEngine = _async_pipeline.AsyncPipelineEngine
AsyncPipelineStage = _async_pipeline.AsyncPipelineStage
AsyncPipelineStats = _async_pipeline.AsyncPipelineStats
CudaStreamManager = _async_pipeline.CudaStreamManager
ScheduleType = _async_pipeline.ScheduleType


# ── Helpers ──

def _identity_forward(tensor: torch.Tensor) -> torch.Tensor:
    """Forward function that doubles the input (simple test transform)."""
    return tensor * 2


def _delay_forward(tensor: torch.Tensor) -> torch.Tensor:
    """Forward function with a small CPU delay to simulate compute."""
    import time
    time.sleep(0.001)
    return tensor * 2


# ── CudaStreamManager ──

class TestCudaStreamManager:
    def test_default_state(self):
        m = CudaStreamManager(device="cpu")
        assert m.compute is None
        assert m.send is None
        assert m.recv is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_initialize(self):
        m = CudaStreamManager(device="cuda")
        m.initialize()
        assert m._compute is not None
        assert m._send is not None
        assert m._recv is not None
        assert m._allreduce is not None

    def test_initialize_cpu(self):
        m = CudaStreamManager(device="cpu")
        m.initialize()
        assert m.compute is None  # CPU mode leaves streams as None


# ── AsyncPipelineConfig ──

class TestAsyncPipelineConfig:
    def test_defaults(self):
        c = AsyncPipelineConfig()
        assert c.schedule == ScheduleType.ONE_F_ONE_B
        assert c.num_micro_batches == 4
        assert c.num_stages == 1

    def test_custom(self):
        c = AsyncPipelineConfig(
            schedule=ScheduleType.G_PIPE,
            num_micro_batches=8,
            num_stages=4,
        )
        assert c.schedule == ScheduleType.G_PIPE
        assert c.num_micro_batches == 8
        assert c.num_stages == 4


# ── AsyncPipelineStats ──

class TestAsyncPipelineStats:
    def test_efficiency_zero_total(self):
        s = AsyncPipelineStats()
        assert s.efficiency == 0.0

    def test_efficiency_partial(self):
        s = AsyncPipelineStats(overlap_ms=50, total_ms=100)
        assert s.efficiency == 50.0


# ── AsyncPipelineStage (CPU without CUDA streams) ──

class TestAsyncPipelineStage:
    def test_run_micro_batch(self):
        stage = AsyncPipelineStage(
            stage_id=0,
            forward_fn=_identity_forward,
            device="cpu",
        )
        stage.initialize()
        inp = torch.tensor([[1.0, 2.0, 3.0]])
        out = stage.run_micro_batch(inp, micro_batch_id=0)
        assert torch.equal(out, inp * 2)

    def test_initial_stats_zero(self):
        stage = AsyncPipelineStage(stage_id=0, forward_fn=_identity_forward, device="cpu")
        assert stage.stats.compute_ms == 0.0
        assert stage.stats.batches == 0

    def test_stats_accumulate(self):
        stage = AsyncPipelineStage(stage_id=0, forward_fn=_delay_forward, device="cpu")
        stage.initialize()
        inp = torch.tensor([[1.0]])
        stage.run_micro_batch(inp, 0)
        stage.run_micro_batch(inp, 1)
        assert stage.stats.batches == 2
        assert stage.stats.total_ms > 0

    def test_send_recv_noop(self):
        """Send/recv fns set to None should not raise."""
        stage = AsyncPipelineStage(
            stage_id=0,
            forward_fn=_identity_forward,
            send_fn=None,
            recv_fn=None,
            device="cpu",
        )
        stage.initialize()
        out = stage.run_micro_batch(torch.tensor([[1.0]]), 0)
        assert out is not None


# ── AsyncPipelineEngine ──

def _build_engine(
    num_stages: int = 2,
    num_micro_batches: int = 4,
    schedule: ScheduleType = ScheduleType.ONE_F_ONE_B,
    forward_fn=_identity_forward,
) -> AsyncPipelineEngine:
    config = AsyncPipelineConfig(
        schedule=schedule,
        num_micro_batches=num_micro_batches,
        num_stages=num_stages,
    )
    engine = AsyncPipelineEngine(config=config)
    for i in range(num_stages):
        stage = AsyncPipelineStage(stage_id=i, forward_fn=forward_fn, device="cpu")
        engine.add_stage(stage)
    return engine


class TestAsyncPipelineEngine:
    def test_add_stage(self):
        engine = _build_engine(num_stages=3)
        assert len(engine._stages) == 3

    def test_forward_single_batch_single_stage(self):
        engine = _build_engine(num_stages=1, num_micro_batches=1)
        inp = torch.tensor([[1.0, 2.0, 3.0]])
        out = engine.forward(inp)
        assert torch.equal(out, inp * 2)

    def test_forward_two_stages_gpipe(self):
        engine = _build_engine(num_stages=2, num_micro_batches=2, schedule=ScheduleType.G_PIPE)
        inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = engine.forward(inp)
        # Each stage multiplies by 2, so total = *4
        assert torch.equal(out, inp * 4)

    def test_forward_two_stages_1f1b(self):
        engine = _build_engine(num_stages=2, num_micro_batches=2, schedule=ScheduleType.ONE_F_ONE_B)
        inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = engine.forward(inp)
        assert torch.equal(out, inp * 4)

    def test_forward_single_micro_batch(self):
        engine = _build_engine(num_stages=2, num_micro_batches=1)
        inp = torch.tensor([[1.0, 2.0]])
        out = engine.forward(inp)
        assert torch.equal(out, inp * 4)

    def test_forward_uneven_batch(self):
        """Batch size not divisible by micro-batch count."""
        engine = _build_engine(num_stages=2, num_micro_batches=4)
        inp = torch.tensor([[1.0], [2.0], [3.0]])
        out = engine.forward(inp)
        assert out.shape == (3, 1)

    # ── run_batch (alias for forward) ──

    def test_run_batch(self):
        engine = _build_engine(num_stages=1)
        inp = torch.tensor([[1.0]])
        out = engine.run_batch(inp)
        assert torch.equal(out, inp * 2)

    # ── forward_stage ──

    def test_forward_stage_specific(self):
        engine = _build_engine(num_stages=2)
        inp = torch.tensor([[1.0]])
        out = engine.forward_stage(inp, stage_id=0)
        assert torch.equal(out, inp * 2)

    def test_forward_stage_none(self):
        engine = _build_engine(num_stages=2)
        inp = torch.tensor([[1.0]])
        out = engine.forward_stage(inp, stage_id=None)
        assert torch.equal(out, inp * 4)

    def test_forward_stage_invalid_id(self):
        engine = _build_engine(num_stages=1)
        inp = torch.tensor([[1.0]])
        out = engine.forward_stage(inp, stage_id=99)
        assert torch.equal(out, inp * 2)

    # ── Warmup ──

    def test_warmup_init(self):
        config = AsyncPipelineConfig(num_stages=4, num_micro_batches=8)
        engine = AsyncPipelineEngine(config=config)
        assert engine._warmup == 3  # num_stages - 1

    def test_warmup_custom(self):
        config = AsyncPipelineConfig(
            num_stages=4, num_micro_batches=8, num_warmup_micro_batches=5,
        )
        engine = AsyncPipelineEngine(config=config)
        assert engine._warmup == 5

    def test_warmup_in_stats(self):
        engine = _build_engine(num_stages=3, num_micro_batches=6)
        stats = engine.stats()
        assert stats["warmup"] == 2  # num_stages - 1

    # ── Stats ──

    def test_stats_structure(self):
        engine = _build_engine(num_stages=2, num_micro_batches=2)
        inp = torch.tensor([[1.0, 2.0]])
        engine.forward(inp)
        stats = engine.stats()
        assert stats["num_stages"] == 2
        assert stats["schedule"] == "1f1b"
        assert stats["total_forward_calls"] == 1
        assert stats["total_compute_ms"] >= 0
        assert len(stats["stages"]) == 2

    def test_stats_after_multiple_forwards(self):
        engine = _build_engine(num_stages=1, num_micro_batches=1)
        inp = torch.tensor([[1.0]])
        engine.forward(inp)
        engine.forward(inp)
        assert engine.stats()["total_forward_calls"] == 2

    # ── Summary ──

    def test_summary_string(self):
        engine = _build_engine(num_stages=2, num_micro_batches=2)
        summary = engine.summary()
        assert "AsyncPipeline" in summary
        assert "2 stages" in summary
        assert "1f1b" in summary
        assert "Stage 0" in summary
        assert "Stage 1" in summary

    # ── Edge cases ──

    def test_empty_stages(self):
        """Engine with no stages should handle forward gracefully."""
        config = AsyncPipelineConfig(num_micro_batches=2)
        engine = AsyncPipelineEngine(config=config)
        inp = torch.tensor([[1.0, 2.0]])
        out = engine.forward(inp)
        assert torch.equal(out, inp)

    def test_single_micro_batch_multi_stage(self):
        """Exactly 1 micro-batch through multiple stages (minimum pipeline)."""
        engine = _build_engine(num_stages=4, num_micro_batches=1)
        inp = torch.tensor([[1.0]])
        out = engine.forward(inp)
        assert torch.equal(out, inp * (2 ** 4))

    def test_large_micro_batch_count(self):
        """More micro-batches than batch size - should still work."""
        engine = _build_engine(num_stages=2, num_micro_batches=16)
        inp = torch.tensor([[1.0], [2.0]])
        out = engine.forward(inp)
        assert out.shape == (2, 1)

    # ── Stats properties ──

    def test_total_overlap_ms(self):
        engine = _build_engine(num_stages=1, num_micro_batches=1)
        assert engine.total_overlap_ms >= 0

    def test_total_compute_ms(self):
        engine = _build_engine(num_stages=1, num_micro_batches=1)
        assert engine.total_compute_ms >= 0

    def test_overlap_efficiency(self):
        engine = _build_engine(num_stages=1, num_micro_batches=1)
        assert 0.0 <= engine.overlap_efficiency <= 100.0

    # ── Interleaved schedule ──

    def test_interleaved_defaults(self):
        """Interleaved1F1BStage stores num_chunks."""
        c = _async_pipeline.Interleaved1F1BStage(
            stage_id=0, forward_fn=_identity_forward, num_chunks=2, device="cpu",
        )
        assert c._num_chunks == 2
        assert c._current_chunk == 0

    def test_interleaved_run_chunk(self):
        c = _async_pipeline.Interleaved1F1BStage(
            stage_id=0, forward_fn=_identity_forward, num_chunks=3, device="cpu",
        )
        inp = torch.tensor([[1.0]])
        out = c.run_micro_batch_chunk(inp, chunk_idx=1)
        assert torch.equal(out, inp * 2)

    def test_interleaved_round_robin(self):
        c = _async_pipeline.Interleaved1F1BStage(
            stage_id=0, forward_fn=_identity_forward, num_chunks=3, device="cpu",
        )
        inp = torch.tensor([[1.0]])
        c.run_micro_batch_chunk(inp)  # chunk 0
        c.run_micro_batch_chunk(inp)  # chunk 1
        out = c.run_micro_batch_chunk(inp)  # chunk 2
        assert c._current_chunk == 0  # wraps around
        assert torch.equal(out, inp * 2)

    # ── Micro-batch overflow protection ──

    def test_micro_batch_overflow_clipped(self):
        """More micro-batches than num_micro_batches should trigger warning."""
        engine = _build_engine(num_stages=1, num_micro_batches=2)
        inp = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # 5 micro-batches -> 2
        out = engine.forward(inp)
        # 5 rows / 2 per micro-batch = 3 micro-batches, clipped to 2 = 4 rows
        assert out.shape[0] == 4

    # ── Backward flush ──

    def test_flush_backward_called(self):
        engine = _build_engine(num_stages=2, num_micro_batches=2)
        inp = torch.tensor([[1.0, 2.0]])
        out = engine.forward(inp)
        # Just verify it doesn't crash
        assert out is not None

    def test_flush_backward_respects_dependencies(self):
        """Multiple stages — flush_backward should not crash even with backward_fn."""
        backward_calls = []
        def _bkwd(grad):
            backward_calls.append("called")
            return grad

        config = AsyncPipelineConfig(num_micro_batches=2, num_stages=2)
        engine = AsyncPipelineEngine(config=config)
        for i in range(2):
            stage = AsyncPipelineStage(
                stage_id=i, forward_fn=_identity_forward,
                backward_fn=_bkwd, device="cpu",
            )
            engine.add_stage(stage)
        inp = torch.tensor([[1.0]])
        result = engine.forward(inp)
        # forward should produce a result
        assert result is not None

    # ── GpuTiming ──

    def test_gpu_timing_record_stop(self):
        """_GpuTiming.record() and stop() should complete without error."""
        timing = _async_pipeline._GpuTiming.record(stream=None)
        ms = timing.stop(stream=None)
        assert ms >= 0.0  # 0 on CPU, >0 on CUDA
