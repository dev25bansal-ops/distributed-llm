"""Tests for all scheduler subpackage modules.

Covers:
    budget.py         -- IterationBudget
    budget_computer.py -- BudgetComputer
    chunked_prefill.py -- ChunkedPrefillInfo
    kv_cache_manager.py -- KVCacheManager
    preemption_manager.py -- PreemptionManager
    pressure.py        -- DecodePressureTracker
    sequence.py        -- Sequence, SequenceStatus, GenerationConfig

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load all scheduler submodules via load_module.
# Modules with no distllm-level imports first, so they are already cached
# when __init__.py fires from the dependent modules.
_seq_mod = load_module("distllm/core/scheduler/sequence.py")
_budget_mod = load_module("distllm/core/scheduler/budget.py")
_pressure_mod = load_module("distllm/core/scheduler/pressure.py")
_chunked_mod = load_module("distllm/core/scheduler/chunked_prefill.py")
_kv_mod = load_module("distllm/core/scheduler/kv_cache_manager.py")

# These modules have ``from distllm.core.scheduler.… import …`` at module
# level, which triggers the scheduler __init__.py.  By now all submodule
# modules are already in sys.modules so the import chain succeeds.
_budget_comp_mod = load_module("distllm/core/scheduler/budget_computer.py")
_preemption_mod = load_module("distllm/core/scheduler/preemption_manager.py")

# Re-export symbols for test readability
SequenceStatus = _seq_mod.SequenceStatus
GenerationConfig = _seq_mod.GenerationConfig
OpenAICompliance = _seq_mod.OpenAICompliance
SchedulingHints = _seq_mod.SchedulingHints
Sequence = _seq_mod.Sequence
ScheduledBatch = _seq_mod.ScheduledBatch
IterationBudget = _budget_mod.IterationBudget
DecodePressureTracker = _pressure_mod.DecodePressureTracker
ChunkedPrefillInfo = _chunked_mod.ChunkedPrefillInfo
KVCacheManager = _kv_mod.KVCacheManager
BudgetComputer = _budget_comp_mod.BudgetComputer
PreemptionManager = _preemption_mod.PreemptionManager


# ===================================================================
# Helpers
# ===================================================================

def make_seq(
    request_id: str = "req-1",
    prompt_len: int = 10,
    max_new: int = 20,
    priority: int = 2,
) -> Sequence:
    """Factory helper: deterministic Sequence for tests."""
    return Sequence(
        request_id=request_id,
        prompt_tokens=list(range(prompt_len)),
        max_new_tokens=max_new,
        priority=priority,
    )


class _StubPagedAttention:
    """Minimal stub with the attributes BudgetComputer/KVCacheManager query.

    Provides ``pool``, ``block_size``, ``free_count``, and basic
    alloc/free/swap methods without touching GPU or real PagedAttention.
    """

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def free_count(self) -> int:
        return self._pool_free_count

    @property
    def pool_utilization(self) -> float:
        return self._pool_utilization

    class _Pool:
        """Nested pool object so ``paged_attention_mgr.pool.free_count`` works."""
        def __init__(self, free_count: int, block_size: int):
            self.free_count = free_count
            self.block_size = block_size

        def restore_block(self, block_id: int) -> None:
            pass  # stub -- no-op

    def __init__(
        self,
        pool_free_count: int = 256,
        block_size: int = 16,
        pool_utilization: float = 0.5,
    ):
        self._block_size = block_size
        self._pool_free_count = pool_free_count
        self._pool_utilization = pool_utilization
        self._allocated: dict[str, list[int]] = {}
        self._swapped_cpu: dict[str, list[int]] = {}
        self.pool = self._Pool(pool_free_count, block_size)

    def allocate_sequence(self, request_id: str, num_tokens: int) -> list[int]:
        block_count = (num_tokens + self._block_size - 1) // self._block_size
        if self._pool_free_count < block_count:
            raise RuntimeError("Not enough free blocks")
        self._pool_free_count -= block_count
        blocks = list(range(block_count))
        self._allocated[request_id] = blocks
        return blocks

    def free_sequence(self, request_id: str) -> None:
        blocks = self._allocated.pop(request_id, [])
        self._pool_free_count += len(blocks)

    def swap_blocks_to_cpu(self, request_id: str) -> int:
        blocks = self._allocated.pop(request_id, [])
        self._swapped_cpu[request_id] = blocks
        return len(blocks)

    def swap_blocks_to_gpu(self, request_id: str) -> int:
        blocks = self._swapped_cpu.pop(request_id, [])
        self._allocated[request_id] = blocks
        return len(blocks)

    def swap_out_sequence(self, request_id: str) -> None:
        self.swap_blocks_to_cpu(request_id)

    def get_physical_blocks(self, request_id: str) -> list[int]:
        return self._allocated.get(request_id, [])

    def copy_on_write(self, source_id: str, dest_id: str) -> None:
        pass  # stub -- no-op


class _StubPreemptionPolicy:
    """Minimal stub for the PreemptionPolicy interface."""

    def __init__(self, should_preempt: bool = False):
        self._should_preempt = should_preempt

    def should_preempt(self, pending_count: int, min_priority: int) -> bool:
        return self._should_preempt


class _StubWanPolicy:
    """Minimal stub for the WAN policy interface."""

    def __init__(self, is_wan_active: bool = False, disable_pressure: bool = False):
        self.is_wan_active = is_wan_active
        self._disable_pressure = disable_pressure

    def should_disable_pressure_adaptation(self) -> bool:
        return self._disable_pressure

    def adjust_budget_for_wan(
        self,
        base_prefill_tokens: int,
        base_batch_size: int,
        base_total_tokens: int,
    ) -> tuple[int, int, int]:
        # WAN: enlarge batch, shrink prefill tokens
        adj_prefill = base_prefill_tokens // 2
        adj_batch = base_batch_size * 2
        adj_total = base_total_tokens
        return adj_prefill, adj_batch, adj_total


class _StubHetBudget:
    """Minimal stub for heterogeneous budget computer."""

    def compute_budget(
        self,
        base_prefill_tokens: int,
        base_decode_tokens: int,
        base_batch_size: int,
        base_total_tokens: int,
    ) -> IterationBudget:
        return IterationBudget(
            max_prefill_tokens=base_prefill_tokens // 2,
            max_decode_tokens=base_decode_tokens,
            max_batch_size=base_batch_size // 2,
            max_total_tokens=base_total_tokens // 2,
        )


class _StubEnergyScheduler:
    """Minimal stub for energy scheduler."""

    def __init__(self, reduce: bool = False):
        self._reduce = reduce

    def adjust_for_energy(
        self,
        base_batch_size: int,
        base_prefill_tokens: int,
    ) -> tuple[int, int]:
        if self._reduce:
            return base_batch_size // 2, base_prefill_tokens // 2
        return base_batch_size, base_prefill_tokens


# ===================================================================
# SEQUENCE TESTS
# ===================================================================

class TestSequence:
    """Sequence dataclass -- construction, defaults, lifecycle."""

    def test_default_construction(self) -> None:
        """A minimal Sequence should get reasonable defaults."""
        seq = Sequence(request_id="req-1")
        assert seq.request_id == "req-1"
        assert seq.prompt_tokens == []
        assert seq.generated_tokens == []
        assert seq.status == SequenceStatus.PENDING
        assert seq.priority == 2
        assert seq.max_new_tokens == 256
        assert seq.temperature == 0.7
        # Nested config objects populated by __post_init__
        assert seq.generation is not None
        assert seq.generation.temperature == 0.7
        assert seq.scheduling is not None
        assert seq.scheduling.priority == 2
        assert seq.openai is not None

    def test_post_init_populates_nested_configs(self) -> None:
        """Nested configs should fall back to top-level fields when not provided."""
        seq = Sequence(
            request_id="req-2",
            temperature=0.3,
            top_p=0.95,
            max_new_tokens=512,
            priority=1,
        )
        assert seq.generation.temperature == 0.3
        assert seq.generation.top_p == 0.95
        assert seq.generation.max_new_tokens == 512
        assert seq.scheduling.priority == 1

    def test_is_complete_by_status_done(self) -> None:
        """is_complete should be True when status is DONE."""
        seq = Sequence(request_id="req-1", max_new_tokens=256)
        seq.status = SequenceStatus.DONE
        assert seq.is_complete

    def test_is_complete_by_status_failed(self) -> None:
        """is_complete should be True when status is FAILED."""
        seq = Sequence(request_id="req-1", max_new_tokens=256)
        seq.status = SequenceStatus.FAILED
        assert seq.is_complete

    def test_is_complete_by_max_new_tokens(self) -> None:
        """is_complete should be True when generated_tokens >= max_new_tokens."""
        seq = Sequence(request_id="req-1", max_new_tokens=3)
        seq.generated_tokens = [10, 11, 12]
        assert seq.is_complete

    def test_is_not_complete_when_still_generating(self) -> None:
        """is_complete should be False when still under max_new_tokens."""
        seq = Sequence(request_id="req-1", max_new_tokens=256)
        seq.status = SequenceStatus.DECODING
        seq.generated_tokens = [10, 11]
        assert not seq.is_complete

    def test_total_len(self) -> None:
        """total_len should be sum of prompt and generated tokens."""
        seq = Sequence(request_id="req-1", prompt_tokens=[1, 2, 3])
        seq.generated_tokens = [10, 11]
        assert seq.total_len == 5

    def test_decode_input_token(self) -> None:
        """decode_input_token should return the last generated token."""
        seq = Sequence(request_id="req-1")
        seq.generated_tokens = [100, 200, 300]
        assert seq.decode_input_token == 300

    def test_decode_input_token_single(self) -> None:
        """decode_input_token should work with one generated token."""
        seq = Sequence(request_id="req-1")
        seq.generated_tokens = [42]
        assert seq.decode_input_token == 42

    def test_negative_priority_clamped_to_zero(self) -> None:
        """Negative priority should be clamped to 0 (critical)."""
        seq = Sequence(request_id="req-neg", priority=-1)
        assert seq.priority == 0

    def test_high_priority_accepted(self) -> None:
        """Priority > 3 should be accepted with a debug log (no clamp)."""
        seq = Sequence(request_id="req-high", priority=10)
        assert seq.priority == 10

    def test_priority_bounds_clamping(self) -> None:
        """Priority < 0 should be clamped, but 0-3 should pass through."""
        seq_low = Sequence(request_id="r1", priority=0)
        seq_mid = Sequence(request_id="r2", priority=2)
        seq_high = Sequence(request_id="r3", priority=3)
        assert seq_low.priority == 0
        assert seq_mid.priority == 2
        assert seq_high.priority == 3

    def test_explicit_nested_configs_are_used(self) -> None:
        """When generation/scheduling/openai are passed, they should be used."""
        gen = GenerationConfig(temperature=0.1, max_new_tokens=64)
        sched = SchedulingHints(priority=0, max_latency_ms=500.0)
        seq = Sequence(
            request_id="req-nested",
            priority=1,
            generation=gen,
            scheduling=sched,
        )
        # The priority from Sequence field still applies, but scheduling
        # hint object is the one we passed.
        assert seq.generation.temperature == 0.1
        assert seq.generation.max_new_tokens == 64
        assert seq.scheduling.max_latency_ms == 500.0


class TestSequenceStatus:
    """SequenceStatus enum values and membership."""

    def test_all_status_values(self) -> None:
        assert SequenceStatus.PENDING.value == "pending"
        assert SequenceStatus.PREFILLING.value == "prefilling"
        assert SequenceStatus.DECODING.value == "decoding"
        assert SequenceStatus.DONE.value == "done"
        assert SequenceStatus.FAILED.value == "failed"
        assert SequenceStatus.PREEMPTED.value == "preempted"

    def test_status_ordering_independence(self) -> None:
        """Status transitions are not enforced by the dataclass itself."""
        seq = Sequence(request_id="req-1")
        # Any transition is valid at the Sequence level
        seq.status = SequenceStatus.DECODING
        seq.status = SequenceStatus.PREFILLING
        seq.status = SequenceStatus.FAILED
        assert seq.status == SequenceStatus.FAILED


class TestGenerationConfig:
    """GenerationConfig defaults and construction."""

    def test_defaults(self) -> None:
        gc = GenerationConfig()
        assert gc.temperature == 0.7
        assert gc.top_p == 0.9
        assert gc.top_k == 0
        assert gc.max_new_tokens == 256
        assert gc.stop_token_ids == []

    def test_custom_values(self) -> None:
        gc = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            max_new_tokens=128,
            stop_token_ids=[2, 3],
        )
        assert gc.temperature == 0.0
        assert gc.top_k == 50
        assert gc.stop_token_ids == [2, 3]

    def test_stop_token_ids_mutability(self) -> None:
        """Default factory for stop_token_ids should give independent lists."""
        gc1 = GenerationConfig()
        gc2 = GenerationConfig()
        gc1.stop_token_ids.append(42)
        assert 42 not in gc2.stop_token_ids


class TestScheduledBatch:
    """ScheduledBatch dataclass -- properties."""

    def test_default_construction(self) -> None:
        batch = ScheduledBatch(sequences=[], input_ids=[])
        assert batch.batch_size == 0
        assert batch.max_seq_len == 0
        assert batch.total_tokens == 0

    def test_batch_size_property(self) -> None:
        seqs = [
            Sequence(request_id="a"),
            Sequence(request_id="b"),
            Sequence(request_id="c"),
        ]
        batch = ScheduledBatch(sequences=seqs, input_ids=[])
        assert batch.batch_size == 3

    def test_max_seq_len_property(self) -> None:
        seqs = [
            Sequence(request_id="a", prompt_tokens=[1, 2, 3]),
            Sequence(request_id="b", prompt_tokens=[1, 2, 3, 4, 5]),
        ]
        batch = ScheduledBatch(
            sequences=seqs,
            input_ids=[],
            seq_lengths=[3, 5],
        )
        assert batch.max_seq_len == 5

    def test_total_tokens_property(self) -> None:
        batch = ScheduledBatch(
            sequences=[],
            input_ids=[],
            seq_lengths=[3, 5, 2],
        )
        assert batch.total_tokens == 10

    def test_max_seq_len_empty(self) -> None:
        batch = ScheduledBatch(sequences=[], input_ids=[])
        assert batch.max_seq_len == 0

    def test_total_tokens_empty(self) -> None:
        batch = ScheduledBatch(sequences=[], input_ids=[])
        assert batch.total_tokens == 0


# ===================================================================
# BUDGET TESTS
# ===================================================================

class TestIterationBudget:
    """IterationBudget dataclass -- defaults and decode_slots property."""

    def test_defaults(self) -> None:
        b = IterationBudget()
        assert b.max_prefill_tokens == 4096
        assert b.max_decode_tokens == 512
        assert b.max_batch_size == 32
        assert b.max_total_tokens == 32768
        assert b.enable_chunked_prefill is True
        assert b.prefill_slack_ratio == 0.3

    def test_decode_slots_normal(self) -> None:
        """decode_slots should be min(max_batch_size, max_decode_tokens)."""
        b = IterationBudget(max_batch_size=32, max_decode_tokens=512)
        assert b.decode_slots == 32

    def test_decode_slots_bounded_by_decode_tokens(self) -> None:
        """When max_decode_tokens < max_batch_size, decode_slots is smaller."""
        b = IterationBudget(max_batch_size=32, max_decode_tokens=10)
        assert b.decode_slots == 10

    def test_decode_slots_with_small_batch(self) -> None:
        b = IterationBudget(max_batch_size=2, max_decode_tokens=512)
        assert b.decode_slots == 2

    def test_custom_values(self) -> None:
        b = IterationBudget(
            max_prefill_tokens=8192,
            max_decode_tokens=1024,
            max_batch_size=64,
            max_total_tokens=65536,
            enable_chunked_prefill=False,
            prefill_slack_ratio=0.5,
        )
        assert b.max_prefill_tokens == 8192
        assert b.enable_chunked_prefill is False
        assert b.prefill_slack_ratio == 0.5


# ===================================================================
# PRESSURE TRACKER TESTS
# ===================================================================

class TestDecodePressureTracker:
    """DecodePressureTracker -- EMA-based pressure and timing."""

    def test_default_construction(self) -> None:
        dpt = DecodePressureTracker()
        assert dpt.pressure == 0.0
        assert dpt.avg_ms_per_token == 0.0

    def test_custom_alpha_and_target(self) -> None:
        dpt = DecodePressureTracker(alpha=0.5, target_ms_per_token=10.0)
        assert dpt.pressure == 0.0

    def test_single_record_establishes_ema(self) -> None:
        dpt = DecodePressureTracker(alpha=0.5, target_ms_per_token=10.0)
        dpt.record_decode_step(batch_decode_count=4, elapsed_ms=40.0)
        # per_token = 40 / 4 = 10 ms -> ema = 10 (first sample takes full value)
        assert dpt.avg_ms_per_token == 10.0
        # pressure = 10 / 10 = 1.0
        assert dpt.pressure == 1.0

    def test_pressure_above_one_is_capped(self) -> None:
        dpt = DecodePressureTracker(alpha=0.5, target_ms_per_token=10.0)
        dpt.record_decode_step(batch_decode_count=1, elapsed_ms=50.0)
        # per_token = 50 ms, ema = 50, pressure = min(1.0, 50/10) = 1.0
        assert dpt.pressure == 1.0

    def test_low_pressure(self) -> None:
        dpt = DecodePressureTracker(alpha=0.5, target_ms_per_token=10.0)
        dpt.record_decode_step(batch_decode_count=10, elapsed_ms=10.0)
        # per_token = 1 ms, ema = 1, pressure = 0.1
        assert dpt.avg_ms_per_token == 1.0
        assert dpt.pressure == 0.1

    def test_ema_smoothing(self) -> None:
        dpt = DecodePressureTracker(alpha=0.3, target_ms_per_token=10.0)
        dpt.record_decode_step(batch_decode_count=1, elapsed_ms=20.0)
        # ema = 20 (first sample)
        dpt.record_decode_step(batch_decode_count=1, elapsed_ms=0.0)
        # ema = 0.3*0 + 0.7*20 = 14
        assert dpt.avg_ms_per_token == pytest.approx(14.0)

    def test_zero_batch_decode_count_uses_one(self) -> None:
        """record_decode_step should not divide by zero."""
        dpt = DecodePressureTracker(alpha=0.5, target_ms_per_token=10.0)
        dpt.record_decode_step(batch_decode_count=0, elapsed_ms=0.0)
        assert dpt.avg_ms_per_token == 0.0
        assert dpt.pressure == 0.0

    def test_no_samples_returns_zero(self) -> None:
        dpt = DecodePressureTracker()
        assert dpt.pressure == 0.0
        assert dpt.avg_ms_per_token == 0.0

    def test_target_ms_cannot_be_zero(self) -> None:
        """pressure should avoid division by zero by using max(target_ms, 0.1)."""
        dpt = DecodePressureTracker(alpha=0.5, target_ms_per_token=0.0)
        dpt.record_decode_step(batch_decode_count=1, elapsed_ms=100.0)
        # ema = 100, target maxed to 0.1, pressure = min(1.0, 100/0.1) = 1.0
        assert dpt.pressure == 1.0


# ===================================================================
# CHUNKED PREFILL TESTS
# ===================================================================

class TestChunkedPrefillInfo:
    """ChunkedPrefillInfo dataclass -- progress tracking."""

    def test_default_construction(self) -> None:
        cpi = ChunkedPrefillInfo(seq_id="seq-1", total_prompt_tokens=100)
        assert cpi.seq_id == "seq-1"
        assert cpi.total_prompt_tokens == 100
        assert cpi.tokens_processed == 0
        assert cpi.chunk_size == 0
        assert cpi.chunks_remaining == 0

    def test_is_complete_initially_false(self) -> None:
        cpi = ChunkedPrefillInfo(seq_id="s1", total_prompt_tokens=100)
        assert not cpi.is_complete

    def test_is_complete_when_fully_processed(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=100, tokens_processed=100
        )
        assert cpi.is_complete

    def test_remaining_property(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=100, tokens_processed=30
        )
        assert cpi.remaining == 70

    def test_advance_increases_tokens_processed(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=100, tokens_processed=20,
            chunk_size=32,
        )
        cpi.advance(32)
        assert cpi.tokens_processed == 52

    def test_advance_updates_chunks_remaining(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=100, tokens_processed=0,
            chunk_size=32,
        )
        # Initially chunks_remaining defaults to 0 (set during construction).
        # After advance(), the value is computed from remaining tokens.
        assert cpi.chunks_remaining == 0
        cpi.advance(32)
        # Processed: 32, remaining tokens: 68, chunks: ceil(68/32) = 3
        assert cpi.chunks_remaining == 3

    def test_advance_beyond_total(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=50, tokens_processed=40,
            chunk_size=32,
        )
        cpi.advance(20)  # would go to 60, but total is 50
        assert cpi.tokens_processed == 60
        assert cpi.chunks_remaining == 0
        assert cpi.is_complete

    def test_advance_no_chunk_size(self) -> None:
        """advance with chunk_size=0 should not update chunks_remaining."""
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=100, tokens_processed=0,
            chunk_size=0,
        )
        cpi.advance(50)
        assert cpi.tokens_processed == 50
        assert cpi.chunks_remaining == 0

    def test_remaining_zero_when_done(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=50, tokens_processed=50,
        )
        assert cpi.remaining == 0

    def test_remaining_never_negative(self) -> None:
        cpi = ChunkedPrefillInfo(
            seq_id="s1", total_prompt_tokens=50, tokens_processed=60,
        )
        assert cpi.remaining == -10  # property is a simple subtraction

    def test_from_chunk_state(self) -> None:
        """from_chunk_state should adapt a ChunkState into ChunkedPrefillInfo."""
        # We need a real ChunkState object.  Since it lives in
        # distllm.dist.chunked_prefill which is faked, we import via
        # the real chunked_prefill under distllm.dist.
        # Use load_module to get it.
        dist_chunked_mod = load_module(
            "distllm/dist/chunked_prefill.py",
            package_override="distllm.dist.chunked_prefill",
        )
        ChunkState = dist_chunked_mod.ChunkState

        cs = ChunkState(prompt_tokens=[10, 20, 30, 40, 50], chunk_size=16)
        cpi = ChunkedPrefillInfo.from_chunk_state(seq_id="seq-1", cs=cs)
        assert cpi.seq_id == "seq-1"
        assert cpi.total_prompt_tokens == 5
        assert cpi.tokens_processed == 0
        assert cpi.chunk_size == 16
        assert cpi.chunks_remaining == 1  # ceil(5/16) = 1


# ===================================================================
# KV CACHE MANAGER TESTS
# ===================================================================

class TestKVCacheManager:
    """KVCacheManager -- block management, swap, compression."""

    def test_default_construction(self) -> None:
        """Without paged_attention_mgr, all operations should no-op."""
        mgr = KVCacheManager()
        assert mgr is not None

    def test_set_paged_attention(self) -> None:
        mgr = KVCacheManager()
        stub = _StubPagedAttention()
        mgr.set_paged_attention(stub)
        # No assertion needed -- just verifying no crash

    def test_allocate_paged_blocks_no_mgr(self) -> None:
        mgr = KVCacheManager()
        seq = make_seq()
        result = mgr.allocate_paged_blocks(seq)
        assert result is None

    def test_allocate_paged_blocks_with_mgr(self) -> None:
        stub = _StubPagedAttention(pool_free_count=64)
        mgr = KVCacheManager(paged_attention_mgr=stub)
        seq = make_seq(prompt_len=32, max_new=64)
        blocks = mgr.allocate_paged_blocks(seq)
        # num_tokens = 32 + 64 = 96, block_size=16 => ceil(96/16)=6 blocks
        assert blocks is not None
        assert len(blocks) == 6

    def test_allocate_paged_blocks_runtime_error(self) -> None:
        """allocate should return None on RuntimeError (not enough blocks)."""
        stub = _StubPagedAttention(pool_free_count=1)
        mgr = KVCacheManager(paged_attention_mgr=stub)
        seq = make_seq(prompt_len=1000, max_new=1000)
        blocks = mgr.allocate_paged_blocks(seq)
        assert blocks is None  # caught by the try/except

    def test_free_paged_blocks_no_mgr(self) -> None:
        mgr = KVCacheManager()
        mgr.free_paged_blocks("req-1")  # should not raise

    def test_free_paged_blocks_with_mgr(self) -> None:
        stub = _StubPagedAttention(pool_free_count=64)
        mgr = KVCacheManager(paged_attention_mgr=stub)
        seq = make_seq(prompt_len=10, max_new=20)
        mgr.allocate_paged_blocks(seq)
        mgr.free_paged_blocks("req-1")

    def test_paged_kv_block_count_no_mgr(self) -> None:
        mgr = KVCacheManager()
        count = mgr.paged_kv_block_count(100)
        assert count == math.ceil(100 / 16)  # default block_size=16

    def test_paged_kv_block_count_with_mgr(self) -> None:
        stub = _StubPagedAttention(block_size=32)
        mgr = KVCacheManager(paged_attention_mgr=stub)
        count = mgr.paged_kv_block_count(100)
        assert count == math.ceil(100 / 32)

    def test_swap_evict_to_cpu_no_mgr(self) -> None:
        mgr = KVCacheManager()
        active = {"a": make_seq(request_id="a", priority=3)}
        freed = mgr.swap_evict_to_cpu(active, threading.Lock(), min_blocks=1)
        assert freed == 0

    def test_swap_evict_to_cpu_selects_lowest_priority(self) -> None:
        stub = _StubPagedAttention(pool_free_count=64)
        mgr = KVCacheManager(paged_attention_mgr=stub)
        seq_a = make_seq(request_id="a", priority=3)  # low
        seq_b = make_seq(request_id="b", priority=0)  # critical
        seq_c = make_seq(request_id="c", priority=2)  # normal
        active = {"a": seq_a, "b": seq_b, "c": seq_c}
        # Allocate blocks for them
        mgr.allocate_paged_blocks(seq_a)
        mgr.allocate_paged_blocks(seq_b)
        mgr.allocate_paged_blocks(seq_c)

        freed = mgr.swap_evict_to_cpu(active, threading.Lock(), min_blocks=1)
        assert freed > 0
        # The code sorts ascending by (priority, -created_at), so it evicts
        # priority 0 (most important / critical) first, not priority 3.
        assert "b" not in stub._allocated
        assert "b" in stub._swapped_cpu

    def test_restore_from_cpu_no_mgr(self) -> None:
        mgr = KVCacheManager()
        result = mgr.restore_from_cpu("req-1")
        assert result == 0

    def test_restore_from_cpu_with_mgr(self) -> None:
        stub = _StubPagedAttention(pool_free_count=64)
        mgr = KVCacheManager(paged_attention_mgr=stub)
        seq = make_seq(request_id="req-r", prompt_len=10, max_new=20)
        mgr.allocate_paged_blocks(seq)
        mgr.free_paged_blocks("req-r")  # free memory to allow re-alloc
        # Now simulate an allocation, swap to CPU, then restore
        mgr.allocate_paged_blocks(seq)
        stub.swap_blocks_to_cpu("req-r")
        count = mgr.restore_from_cpu("req-r")
        assert count > 0
        assert "req-r" in stub._allocated

    def test_copy_on_write_no_mgr(self) -> None:
        mgr = KVCacheManager()
        mgr.copy_on_write("src", "dst")  # should not raise

    def test_copy_on_write_with_mgr(self) -> None:
        stub = _StubPagedAttention()
        mgr = KVCacheManager(paged_attention_mgr=stub)
        mgr.copy_on_write("src", "dst")  # should not raise

    def test_save_and_restore_kv_state(self) -> None:
        mgr = KVCacheManager()
        preempted_kv: dict[str, dict] = {}
        kv_state = {"req-1": {"key": [1.0, 2.0], "value": [3.0, 4.0]}}

        mgr.save_kv_state("req-1", preempted_kv, kv_state)
        assert "req-1" in preempted_kv

        result = mgr.restore_kv_state("req-1", preempted_kv, kv_state)
        assert result is True
        assert "req-1" in kv_state
        assert "req-1" not in preempted_kv  # popped

    def test_restore_kv_state_not_found(self) -> None:
        mgr = KVCacheManager()
        result = mgr.restore_kv_state("nonexistent", {}, {})
        assert result is False

    def test_save_kv_state_no_kv_cache_state(self) -> None:
        """save_kv_state should do nothing if kv_cache_state is None."""
        mgr = KVCacheManager()
        preempted_kv: dict[str, dict] = {}
        mgr.save_kv_state("req-1", preempted_kv, kv_cache_state=None)
        assert "req-1" not in preempted_kv

    def test_compress_and_decompress_round_trip(self) -> None:
        """int4 compression/decompression round-trip with known tensor."""
        import torch

        mgr = KVCacheManager()
        orig = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16)
        compressed = mgr._compress_kv_for_preemption(orig, method="int4")
        assert compressed["_compressed"] is True
        assert compressed["method"] == "int4"

        decompressed = mgr.decompress_preempted_kv(compressed)
        assert isinstance(decompressed, torch.Tensor)
        assert decompressed.shape == orig.shape

    def test_compress_method_none_passthrough(self) -> None:
        mgr = KVCacheManager()
        result = mgr._compress_kv_for_preemption({"key": "value"}, method="none")
        assert result == {"key": "value"}

    def test_compress_none_value(self) -> None:
        mgr = KVCacheManager()
        result = mgr._compress_kv_for_preemption(None, method="int4")
        assert result is None

    def test_compress_non_tensor_data(self) -> None:
        """Non-tensor data (strings, ints) should pass through unchanged."""
        mgr = KVCacheManager()
        result = mgr._compress_kv_for_preemption("plain string", method="int4")
        assert result == "plain string"

    def test_compress_list_of_tensors(self) -> None:
        import torch

        mgr = KVCacheManager()
        tensors = [torch.tensor([1.0, 2.0], dtype=torch.float16)]
        result = mgr._compress_kv_for_preemption(tensors, method="int4")
        assert isinstance(result, list)
        assert result[0]["_compressed"] is True

    def test_decompress_uncompressed_data(self) -> None:
        mgr = KVCacheManager()
        result = mgr.decompress_preempted_kv({"data": 42, "_compressed": False})
        assert result["data"] == 42

    def test_decompress_plain_data(self) -> None:
        mgr = KVCacheManager()
        result = mgr.decompress_preempted_kv({"key": "value"})
        assert result == {"key": "value"}

    def test_decompress_list_of_dicts(self) -> None:
        mgr = KVCacheManager()
        result = mgr.decompress_preempted_kv([{"a": 1}, {"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]


# ===================================================================
# BUDGET COMPUTER TESTS
# ===================================================================

class TestBudgetComputer:
    """BudgetComputer -- Sarathi-Serve adaptation, dynamic scaling, policy chain."""

    def test_default_construction(self) -> None:
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker()
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        assert comp is not None

    def test_adapt_disabled(self) -> None:
        """When adapt_prefill_budget=False, apply_budget_policy returns base."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker()
        comp = BudgetComputer(
            kv_cache_mgr=kv_mgr,
            pressure_tracker=pressure,
            adapt_prefill_budget=False,
        )
        base = IterationBudget()
        result = comp.apply_budget_policy(base, None, {}, threading.Lock())
        assert result is base  # same object, passthrough

    def test_sarathi_budget_no_pressure(self) -> None:
        """With zero pressure, Sarathi budget should relax decode slots."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker()
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        base = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
            max_total_tokens=32768,
        )
        active: dict[str, Sequence] = {}
        with threading.Lock():
            result = comp.compute_sarathi_budget(
                base, active, threading.Lock(),
            )
        # pressure=0 < 0.3 => relax decode slots: int(32*0.6)=19
        assert result.max_decode_tokens == 19
        # prefill_scale = 1.0 (since pressure <= 0.5)
        assert result.max_prefill_tokens == 4096

    def test_sarathi_budget_medium_pressure(self) -> None:
        """Pressure between 0.3 and 0.7 should keep base decode slots."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker(alpha=1.0, target_ms_per_token=10.0)
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        base = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
        )
        # elapsed_ms=6.0 / 10.0 gives pressure=0.6 (medium, > 0.5)
        pressure.record_decode_step(batch_decode_count=1, elapsed_ms=6.0)
        assert pressure.pressure == 0.6

        with threading.Lock():
            result = comp.compute_sarathi_budget(
                base, {}, threading.Lock(),
            )
        # medium pressure => base_decode_slots = 32, no adjustment
        assert result.max_decode_tokens == 32
        # pressure > 0.5 => prefill_scale = 0.75
        assert result.max_prefill_tokens == int(4096 * 0.75)

    def test_sarathi_budget_high_pressure(self) -> None:
        """Pressure > 0.7 should saturate decode slots."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker(alpha=1.0, target_ms_per_token=10.0)
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        base = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
        )
        # elapsed_ms=8.5 / 10.0 gives pressure=0.85 (high, > 0.7 and > 0.8)
        pressure.record_decode_step(batch_decode_count=1, elapsed_ms=8.5)
        assert pressure.pressure == 0.85

        with threading.Lock():
            result = comp.compute_sarathi_budget(
                base, {}, threading.Lock(),
            )
        # decode = min(32, int(32 * 1.85)) = 32
        assert result.max_decode_tokens == 32
        # pressure > 0.8 => prefill_scale = max(0.25, 1.0 - 0.85) = 0.25
        assert result.max_prefill_tokens == int(4096 * 0.25)

    def test_sarathi_budget_severe_pressure(self) -> None:
        """Pressure > 0.9 should also limit batch size."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker(alpha=1.0, target_ms_per_token=10.0)
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        base = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
        )
        pressure.record_decode_step(batch_decode_count=1, elapsed_ms=9.5)
        # pressure = 9.5/10 = 0.95 > 0.9

        with threading.Lock():
            result = comp.compute_sarathi_budget(
                base, {}, threading.Lock(),
            )
        # adjusted_batch = max(32, int(32 * 0.5)) = 32 -- but decode needed at
        # least 32 anyway so batch stays at 32.
        assert result.max_batch_size == 32
        assert result.max_prefill_tokens == int(4096 * 0.25)

    def test_sarathi_budget_with_active_decodes(self) -> None:
        """Active decodes should guarantee decode slots."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker(alpha=1.0, target_ms_per_token=10.0)
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        base = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=8,
        )
        # 3 active decodes, 1 pending prefill with partial generation
        active = {
            "a": make_seq(request_id="a", priority=0, prompt_len=5),
            "b": make_seq(request_id="b", priority=0, prompt_len=5),
            "c": make_seq(request_id="c", priority=0, prompt_len=5),
            "d": make_seq(request_id="d", priority=0, prompt_len=10),
        }
        active["a"].status = SequenceStatus.DECODING
        active["a"].generated_tokens = [100]
        active["b"].status = SequenceStatus.DECODING
        active["b"].generated_tokens = [101]
        active["c"].status = SequenceStatus.DECODING
        active["c"].generated_tokens = [102]
        active["d"].status = SequenceStatus.PREFILLING
        active["d"].generated_tokens = [200]  # has generated tokens => pending decode

        # pressure < 0.3 => relax decode slots
        with threading.Lock():
            result = comp.compute_sarathi_budget(
                base, active, threading.Lock(),
            )
        # total_decode_demand = 3 + 1 = 4
        # guaranteed = max(8, 4) = 8
        # Since pressure < 0.3: adjusted_decode = max(1, int(8*0.6)) = 4
        assert result.max_decode_tokens == 4

    def test_sarathi_budget_wan_skip(self) -> None:
        """WAN mode should skip pressure adaptation, return budget unchanged."""
        kv_mgr = KVCacheManager()
        pressure = DecodePressureTracker(alpha=1.0, target_ms_per_token=10.0)
        comp = BudgetComputer(kv_cache_mgr=kv_mgr, pressure_tracker=pressure)
        base = IterationBudget()
        wan = _StubWanPolicy(is_wan_active=True, disable_pressure=True)
        pressure.record_decode_step(batch_decode_count=1, elapsed_ms=100.0)

        with threading.Lock():
            result = comp.compute_sarathi_budget(
                base, {}, threading.Lock(), wan_policy=wan,
            )
        # Should return the same IterationBudget (identical fields)
        assert result.max_prefill_tokens == base.max_prefill_tokens
        assert result.max_decode_tokens == base.max_decode_tokens

    def test_compute_dynamic_budget_without_paged_attention(self) -> None:
        """Without PagedAttention, budget should be capped at 128K."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        budget = comp.compute_dynamic_budget(base_budget=200000, paged_attention_mgr=None)
        assert budget == 131072

    def test_compute_dynamic_budget_with_pool(self) -> None:
        """With a PagedAttention pool, budget is min(cap, block_token_budget)."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        stub = _StubPagedAttention(pool_free_count=100, block_size=16)
        budget = comp.compute_dynamic_budget(
            base_budget=200000, paged_attention_mgr=stub,
        )
        # block_token_budget = 100 * 16 * 0.9 = 1440
        assert budget == 1440

    def test_compute_dynamic_budget_high_utilization(self) -> None:
        """With high pool utilization (no pool attr), budget should scale down."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )

        class _MgrWithUtil:
            pool_utilization = 0.9

        budget = comp.compute_dynamic_budget(
            base_budget=100000, paged_attention_mgr=_MgrWithUtil(),
        )
        # utilization > 0.85 => budget * 0.75
        assert budget == int(100000 * 0.75)

    def test_adjust_budget_delegates(self) -> None:
        """adjust_budget should delegate to compute_dynamic_budget."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        result = comp.adjust_budget(50000, None)
        assert result == 50000  # capped to 128K, still under

    def test_get_iteration_budget_base(self) -> None:
        """Basic iteration budget with chunked prefill enabled."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        base = IterationBudget(
            max_prefill_tokens=4096,
            max_decode_tokens=512,
            max_batch_size=32,
            max_total_tokens=32768,
            enable_chunked_prefill=True,
        )
        result = comp.get_iteration_budget(
            base_budget=base,
            enable_chunked_prefill=True,
            max_tokens_per_batch=32768,
            max_batch_size=32,
        )
        assert result.max_prefill_tokens == 4096
        assert result.max_decode_tokens == 512

    def test_get_iteration_budget_chunked_disabled(self) -> None:
        """When chunked prefill is disabled, budget uses max_tokens_per_batch."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        base = IterationBudget(enable_chunked_prefill=True)
        result = comp.get_iteration_budget(
            base_budget=base,
            enable_chunked_prefill=False,
            max_tokens_per_batch=8192,
            max_batch_size=16,
        )
        # Chunked disabled => all fields set to max_tokens_per_batch
        assert result.max_prefill_tokens == 8192
        assert result.max_decode_tokens == 8192
        assert result.max_batch_size == 16
        assert result.max_total_tokens == 8192
        assert result.enable_chunked_prefill is False

    def test_get_iteration_budget_with_het(self) -> None:
        """Heterogeneous budget should override the base."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        base = IterationBudget(max_prefill_tokens=4096, max_batch_size=32)
        het = _StubHetBudget()
        result = comp.get_iteration_budget(
            base_budget=base,
            enable_chunked_prefill=True,
            max_tokens_per_batch=32768,
            max_batch_size=32,
            het_budget=het,
        )
        assert result.max_prefill_tokens == 2048  # halved
        assert result.max_batch_size == 16  # halved

    def test_get_iteration_budget_with_wan(self) -> None:
        """WAN adjustment should double batch and halve prefill."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        base = IterationBudget(max_prefill_tokens=4096, max_batch_size=8)
        wan = _StubWanPolicy(is_wan_active=True)
        result = comp.get_iteration_budget(
            base_budget=base,
            enable_chunked_prefill=True,
            max_tokens_per_batch=32768,
            max_batch_size=8,
            wan_policy=wan,
        )
        assert result.max_prefill_tokens == 2048  # halved
        assert result.max_batch_size == 16  # doubled

    def test_get_iteration_budget_with_energy(self) -> None:
        """Energy adjustment should reduce batch/prefill."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )
        base = IterationBudget(max_prefill_tokens=4096, max_batch_size=8)
        energy = _StubEnergyScheduler(reduce=True)
        result = comp.get_iteration_budget(
            base_budget=base,
            enable_chunked_prefill=True,
            max_tokens_per_batch=32768,
            max_batch_size=8,
            energy_scheduler=energy,
        )
        assert result.max_batch_size == 4
        assert result.max_prefill_tokens == 2048

    def test_apply_budget_policy_with_pluggable_policy(self) -> None:
        """A scheduling policy should take precedence over Sarathi-Serve."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
        )

        class _PluggablePolicy:
            def compute_budget(self, budget: IterationBudget) -> IterationBudget:
                return IterationBudget(max_prefill_tokens=999)

        result = comp.apply_budget_policy(
            IterationBudget(), _PluggablePolicy(), {}, threading.Lock(),
        )
        assert result.max_prefill_tokens == 999

    def test_apply_budget_policy_without_policy_uses_sarathi(self) -> None:
        """Without a scheduling policy, Sarathi-Serve adaptation should run."""
        comp = BudgetComputer(
            kv_cache_mgr=KVCacheManager(),
            pressure_tracker=DecodePressureTracker(),
            adapt_prefill_budget=True,
        )
        base = IterationBudget(max_prefill_tokens=4096, max_decode_tokens=512)
        result = comp.apply_budget_policy(
            base, None, {}, threading.Lock(),
        )
        # Sarathi ran: pressure=0 => relax decode slots
        assert result.max_decode_tokens == 19


# ===================================================================
# PREEMPTION MANAGER TESTS
# ===================================================================

class TestPreemptionManager:
    """PreemptionManager -- preempt, restore, policy integration."""

    def test_default_construction(self) -> None:
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        assert mgr.get_preempted_count() == 0
        assert mgr.preempted == {}

    def test_set_max_preempted(self) -> None:
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager(), max_preempted=4)
        assert mgr.get_preempted_count() == 0
        mgr.set_max_preempted(10)
        assert mgr.get_preempted_count() == 0

    def test_set_max_preempted_clamps_negative(self) -> None:
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager(), max_preempted=4)
        mgr.set_max_preempted(-5)
        # max(0, -5) = 0
        assert mgr._max_preempted == 0

    def test_preempt_if_needed_no_policy(self) -> None:
        """Without a policy, preempt_if_needed should return None."""
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        result = mgr.preempt_if_needed(pending_count=100)
        assert result is None

    def test_preempt_if_needed_policy_triggers(self) -> None:
        """With policy saying yes, preempt_if_needed should call preempt_fn."""
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        mgr.set_preemption_policy(_StubPreemptionPolicy(should_preempt=True))

        def fake_preempt(min_priority: int) -> Sequence:
            return make_seq(request_id="victim")

        result = mgr.preempt_if_needed(
            pending_count=50, preempt_fn=fake_preempt,
        )
        assert result is not None
        assert result.request_id == "victim"

    def test_preempt_if_needed_policy_does_not_trigger(self) -> None:
        """With policy saying no, preempt_fn should not be called."""
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        mgr.set_preemption_policy(_StubPreemptionPolicy(should_preempt=False))
        called = False

        def fake_preempt(min_priority: int) -> Sequence:
            nonlocal called
            called = True
            return make_seq(request_id="victim")

        result = mgr.preempt_if_needed(
            pending_count=50, preempt_fn=fake_preempt,
        )
        assert result is None
        assert called is False

    def test_preempt_if_needed_no_preempt_fn(self) -> None:
        """When preempt_fn is None, policy trigger does not crash."""
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        mgr.set_preemption_policy(_StubPreemptionPolicy(should_preempt=True))
        result = mgr.preempt_if_needed(pending_count=50, preempt_fn=None)
        assert result is None

    def test_preempt_lowest_at_capacity(self) -> None:
        """When at max preempted, preempt_lowest should return None."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=1)
        # Fill the preempted slot
        mgr._preempted["existing"] = make_seq(request_id="existing")
        assert mgr.get_preempted_count() == 1

        result = mgr.preempt_lowest(
            active={"a": make_seq(request_id="a", priority=3)},
            total_tokens_ref=[100],
            pending_heap=[],
            counter_ref=[0],
            paged_attention_mgr=None,
        )
        assert result is None

    def test_preempt_lowest_no_eligible_candidate(self) -> None:
        """When no sequence meets min_priority, return None."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        active = {"a": make_seq(request_id="a", priority=0)}  # critical, below min_priority=3
        result = mgr.preempt_lowest(
            active=active,
            total_tokens_ref=[100],
            pending_heap=[],
            counter_ref=[0],
            paged_attention_mgr=None,
            min_priority=3,
        )
        assert result is None

    def test_preempt_lowest_selects_highest_priority_value(self) -> None:
        """preempt_lowest should select the sequence with highest numeric priority."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        active = {
            "a": make_seq(request_id="a", priority=0, prompt_len=3),  # critical
            "b": make_seq(request_id="b", priority=2, prompt_len=3),  # normal
            "c": make_seq(request_id="c", priority=3, prompt_len=3),  # low
        }
        for seq in active.values():
            seq.status = SequenceStatus.DECODING

        victim = mgr.preempt_lowest(
            active=active,
            total_tokens_ref=total_ref,
            pending_heap=pending,
            counter_ref=counter,
            paged_attention_mgr=None,
        )
        assert victim is not None
        assert victim.request_id == "c"  # priority 3 = lowest importance
        assert "c" not in active  # removed from active
        assert victim.status == SequenceStatus.PENDING
        assert len(pending) == 1
        assert pending[0][2].request_id == "c"

    def test_preempt_lowest_adjusts_total_tokens(self) -> None:
        """total_tokens_ref should decrease by victim's total_len."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        active = {
            "a": make_seq(request_id="a", priority=3, prompt_len=5),
        }
        active["a"].generated_tokens = [1, 2]  # total_len = 7

        mgr.preempt_lowest(active, total_ref, pending, counter, None)
        assert total_ref[0] == 93  # 100 - 7

    def test_preempt_lowest_increments_counter(self) -> None:
        """counter_ref should be incremented for heap ordering."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [5]
        total_ref = [100]
        active = {"a": make_seq(request_id="a", priority=3, prompt_len=3)}

        mgr.preempt_lowest(active, total_ref, pending, counter, None)
        assert counter[0] == 6

    def test_restore_preempted_empty(self) -> None:
        """restore_preempted with no preempted sequences should return []."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr)
        result = mgr.restore_preempted(
            active={},
            total_tokens_ref=[0],
            pending_heap=[],
            paged_attention_mgr=None,
        )
        assert result == []

    def test_restore_preempted_restores_sequences(self) -> None:
        """restore_preempted should return preempted sequences to active."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        active = {"a": make_seq(request_id="a", priority=3, prompt_len=5)}

        # Preempt it
        victim = mgr.preempt_lowest(active, total_ref, pending, counter, None)
        assert victim is not None
        assert "a" not in active
        assert len(mgr._preempted) == 1

        # Restore it
        restored = mgr.restore_preempted(active, total_ref, pending, None)
        assert len(restored) == 1
        assert restored[0].request_id == "a"
        assert "a" in active
        assert active["a"].status == SequenceStatus.DECODING
        assert mgr.get_preempted_count() == 0
        # total_tokens restored
        assert total_ref[0] == 100  # back to original

    def test_restore_preempted_clears_pending_heap(self) -> None:
        """After restore, the pending heap should no longer contain the request."""
        kv_mgr = KVCacheManager()
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        active = {"a": make_seq(request_id="a", priority=3, prompt_len=5)}

        mgr.preempt_lowest(active, total_ref, pending, counter, None)
        assert len(pending) == 1

        mgr.restore_preempted(active, total_ref, pending, None)
        assert len(pending) == 0

    def test_preempt_lowest_saves_kv_state(self) -> None:
        """KV state should be saved during preemption."""
        kv_mgr = KVCacheManager()
        kv_cache: dict[str, Any] = {
            "a": {"key_tensor": "mock", "value_tensor": "mock"},
        }
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        active = {"a": make_seq(request_id="a", priority=3, prompt_len=5)}

        mgr.preempt_lowest(active, total_ref, pending, counter, None,
                           kv_cache_state=kv_cache)
        assert "a" in mgr._preempted_kv_state

    def test_preempt_lowest_with_paged_attention(self) -> None:
        """preempt_lowest should call swap_out_sequence on PagedAttention."""
        stub = _StubPagedAttention(pool_free_count=64)
        kv_mgr = KVCacheManager(paged_attention_mgr=stub)
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        seq = make_seq(request_id="a", priority=3, prompt_len=5)
        kv_mgr.allocate_paged_blocks(seq)
        active = {"a": seq}

        victim = mgr.preempt_lowest(active, total_ref, pending, counter, stub)
        assert victim is not None
        assert "a" in stub._swapped_cpu

    def test_restore_preempted_with_paged_attention(self) -> None:
        """restore_preempted should call pool.restore_block."""
        stub = _StubPagedAttention(pool_free_count=64)
        kv_mgr = KVCacheManager(paged_attention_mgr=stub)
        mgr = PreemptionManager(kv_cache_mgr=kv_mgr, max_preempted=4)
        pending: list = []
        counter = [0]
        total_ref = [100]
        seq = make_seq(request_id="a", priority=3, prompt_len=5)
        kv_mgr.allocate_paged_blocks(seq)
        active = {"a": seq}

        mgr.preempt_lowest(active, total_ref, pending, counter, stub)
        # stub already has pool with restore_block
        restored = mgr.restore_preempted(active, total_ref, pending, stub)
        assert len(restored) == 1

    def test_set_preemption_policy(self) -> None:
        """set_preemption_policy should connect the policy."""
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        policy = _StubPreemptionPolicy()
        mgr.set_preemption_policy(policy)
        assert mgr._preemption_policy is policy

    def test_set_preemption_policy_none(self) -> None:
        """set_preemption_policy(None) should not crash."""
        mgr = PreemptionManager(kv_cache_mgr=KVCacheManager())
        mgr.set_preemption_policy(None)
        assert mgr._preemption_policy is None
