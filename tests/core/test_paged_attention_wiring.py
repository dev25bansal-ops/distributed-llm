"""C7 regression: PagedAttention + defrag must actually be wired.

Root cause: ``PagedAttentionManager`` was only instantiated inside an
orphaned ``HybridKVCache``; nothing called ``BatchScheduler.set_paged_attention()``
in production; and ``Coordinator._get_paged_backends()`` read
``engine._paged_mgr`` / ``engine.backends`` attributes that InferenceEngine
never defined — so paged allocation always returned None, CPU-swap/preemption
no-op'd, and the coordinator defrag loop iterated zero backends forever.

Fix: ``InferenceEngine.load_local_model()`` constructs a manager sized from
the model config; ``Coordinator.start()`` wires it into the batch scheduler
via ``_wire_paged_attention()``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import distllm.core.inference_engine as ie_mod
from distllm.backends.paged_attention import PagedAttentionManager
from distllm.core.coordinator import Coordinator
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.inference_engine import InferenceEngine
from distllm.core.scheduler.sequence import Sequence


MODEL_INFO = {
    "model_type": "gpt2",
    "num_layers": 4,
    "hidden_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "vocab_size": 256,
    "max_position_embeddings": 1024,
}


@pytest.fixture()
def engine_with_loaded_model(monkeypatch) -> InferenceEngine:
    """InferenceEngine whose load_local_model() runs without HF/network."""
    class _FakePartitioner:
        def __init__(self, *a, **k):
            pass

        def load_full_model(self):
            pass

    monkeypatch.setattr(ie_mod, "ModelPartitioner", _FakePartitioner)
    monkeypatch.setattr(
        ie_mod, "get_model_info", lambda *a, **k: dict(MODEL_INFO)
    )
    monkeypatch.setattr(
        ie_mod, "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: object()),
    )

    engine = InferenceEngine(model_name="fake/tiny-model")
    assert engine._paged_mgr is None  # not constructed before load
    engine.load_local_model()
    return engine


class TestEngineConstructsPagedManager:
    """load_local_model() must build a manager sized from the model config."""

    def test_manager_created_on_load(self, engine_with_loaded_model):
        mgr = engine_with_loaded_model._paged_mgr
        assert isinstance(mgr, PagedAttentionManager)

    def test_manager_sized_from_model_config(self, engine_with_loaded_model):
        mgr = engine_with_loaded_model._paged_mgr
        assert mgr._num_layers == MODEL_INFO["num_layers"]
        assert mgr._num_heads == MODEL_INFO["num_key_value_heads"]
        assert mgr._head_dim == MODEL_INFO["head_dim"]

    def test_manager_allocates_blocks(self, engine_with_loaded_model):
        """allocate_sequence works end-to-end (lazy per-block storage)."""
        block_ids = engine_with_loaded_model._paged_mgr.allocate_sequence("req-1", 48)
        assert block_ids, "expected non-empty block allocation"


@pytest.fixture()
def coord() -> Coordinator:
    """Coordinator with a (CPU) PagedAttention manager as if a model loaded."""
    c = Coordinator(config=CoordinatorConfig(model_name="test"))
    # Simulate load_local_model() having run (avoids HF download here;
    # construction itself is covered by TestEngineConstructsPagedManager).
    c._inference_engine._paged_mgr = PagedAttentionManager(
        num_blocks=64, block_size=16,
        num_layers=2, num_heads=2, head_dim=8, device="cpu",
    )
    return c


class TestCoordinatorWiring:
    """The coordinator must find the manager and wire it into the scheduler."""

    def test_get_paged_backends_finds_engine_manager(self, coord):
        backends = coord._get_paged_backends()
        assert backends == [coord._inference_engine._paged_mgr]

    def test_wire_connects_scheduler_to_manager(self, coord):
        coord._wire_paged_attention()
        assert (
            coord._batch_scheduler._kv_cache_mgr._paged_attention_mgr
            is coord._inference_engine._paged_mgr
        )

    def test_scheduler_allocate_paged_blocks_no_longer_none(self, coord):
        """Before the fix allocate_paged_blocks always returned None."""
        coord._wire_paged_attention()
        seq = Sequence(request_id="req-9", prompt_tokens=list(range(32)),
                       max_new_tokens=16)
        block_ids = coord._batch_scheduler.allocate_paged_blocks(seq)
        assert block_ids is not None and len(block_ids) > 0

    def test_no_crash_when_no_manager(self):
        """Wiring without a loaded model is a no-op, not a crash."""
        coord = Coordinator(config=CoordinatorConfig(model_name="test"))
        assert coord._get_paged_backends() == []
        coord._wire_paged_attention()  # must not raise
        assert coord._batch_scheduler._kv_cache_mgr._paged_attention_mgr is None


class TestDefragSeesBackend:
    """The defrag loop/status must sample at least one real backend."""

    def test_defrag_status_reports_sampled_fragmentation(self, coord):
        from distllm.config._cache import DefragmentationSettings

        coord.init_defragmentation(
            settings=DefragmentationSettings(enabled=True)
        )
        mgr = coord._inference_engine._paged_mgr
        # Give the pool some allocation state to sample.
        mgr.allocate_sequence("seq-a", 64)
        mgr.allocate_sequence("seq-b", 64)

        status = coord.defrag_status()
        assert status["enabled"] is True
        assert "fragmentation_ratio" in status
        assert status["fragmentation_ratio"] >= 0.0

    def test_run_now_executes_against_backend(self, coord):
        from distllm.config._cache import DefragmentationSettings

        coord.init_defragmentation(
            settings=DefragmentationSettings(enabled=True)
        )
        mgr = coord._inference_engine._paged_mgr
        mgr.allocate_sequence("seq-a", 128)

        results = coord.defrag_run_now()
        assert results, "expected at least one backend defragmented"
        assert "backend_0" in results
