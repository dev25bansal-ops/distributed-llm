"""Concurrency tests for ClusterManager -- register/unregister/list races.

Regression coverage for W3-T6: ``get_node_gpu_summary`` used to iterate the
pipeline's live node mapping outside the orchestrator lock and crashed with
``RuntimeError: dictionary changed size during iteration`` whenever nodes
were registered/unregistered concurrently (batch registration uses a thread
pool; the admin API unregisters nodes from request handlers).

Covers:
    - stress: N writer threads (register+manual_register+unregister) vs M
      reader threads (gpu summary, weight-source lookup, node_count)
    - final-state consistency: node_order <-> nodes agreement
    - stub-pipeline fallback path (no internal lock)

Uses the REAL PipelineOrchestrator (its locking is part of the contract);
only NodeRegistrar is stubbed (avoids HuggingFace).
No network, no GPU, no sleeps.  Deterministically bounded work per thread.
"""

from __future__ import annotations

import sys
import threading

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_cluster_mod = load_module("distllm/core/cluster_manager.py")
ClusterManager = _cluster_mod.ClusterManager
PipelineOrchestrator = _cluster_mod.PipelineOrchestrator


class _StubNodeRegistrar:
    """Minimal NodeRegistrar substitute without HuggingFace dependency."""

    def __init__(self, pipeline=None, model_name="", trust_remote_code=None):
        self.pipeline = pipeline
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code

    def manual_register(self, *args, **kwargs) -> None:
        pass  # real topology mutation happens via pipeline.register_node


def _make_manager() -> tuple[ClusterManager, PipelineOrchestrator]:
    """Build a ClusterManager over a REAL orchestrator with a stub registrar."""
    pipeline = PipelineOrchestrator()
    mgr = ClusterManager(pipeline=pipeline, model_name="stress-model")
    mgr._node_registrar = _StubNodeRegistrar(
        pipeline=pipeline, model_name="stress-model",
    )
    mgr.tokenizer = object()          # prevent HF AutoTokenizer call
    mgr.model_info = {"num_layers": 64}
    return mgr, pipeline


@pytest.fixture(autouse=True)
def _aggressive_scheduling():
    """Shrink the GIL switch interval to expose interleavings, then restore."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    yield
    sys.setswitchinterval(old)


# Number of distinct node ids churning at any time.  Small enough that the
# dict stays hot (high mutation rate over the same slots), large enough that
# iteration takes many bytecodes -- the window the original race lived in.
_NODE_POOL = 24
_WRITER_THREADS = 6
_READER_THREADS = 3
_WRITER_CYCLES = 120
_READER_CYCLES = 3000


class TestRegisterUnregisterStress:
    """Writers churn nodes while readers take snapshots -- no exceptions."""

    def test_stress_register_unregister_list(self) -> None:
        mgr, pipeline = _make_manager()
        errors: list[str] = []
        barrier = threading.Barrier(_WRITER_THREADS + _READER_THREADS)

        def writer(wid: int) -> None:
            try:
                barrier.wait(timeout=10)
                for i in range(_WRITER_CYCLES):
                    nid = f"w{wid}-n{i % _NODE_POOL}"
                    start = i % 32
                    pipeline.register_node(nid, "10.0.0.1", 50051, start, start)
                    mgr.manual_register(nid, "10.0.0.1", 50051, start, start,
                                        total_layers=64)
                    # Readers exercise the fixed snapshot path mid-churn.
                    mgr.get_node_gpu_summary()
                    mgr._get_weight_source("stress-model", start, start)
                    mgr.node_count
                    pipeline.unregister_node(nid)
            except Exception as exc:  # noqa: BLE001 - collected below
                errors.append(f"writer{wid}: {exc!r}")

        def reader(rid: int) -> None:
            try:
                barrier.wait(timeout=10)
                for _ in range(_READER_CYCLES):
                    summary = mgr.get_node_gpu_summary()
                    for nid, info in summary.items():
                        assert isinstance(nid, str)
                        assert set(info) == {
                            "gpu_name", "memory_total_gb", "memory_free_gb",
                        }
                    mgr._get_weight_source("stress-model", 0, 0)
                    mgr.node_count
            except Exception as exc:  # noqa: BLE001 - collected below
                errors.append(f"reader{rid}: {exc!r}")

        threads = (
            [threading.Thread(target=writer, args=(w,)) for w in range(_WRITER_THREADS)]
            + [threading.Thread(target=reader, args=(r,)) for r in range(_READER_THREADS)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not any(t.is_alive() for t in threads), "threads hung"
        assert errors == [], f"concurrency failures:\n" + "\n".join(errors)

    def test_final_state_consistent_after_churn(self) -> None:
        """After all writers join: node_order mirrors nodes exactly."""
        mgr, pipeline = _make_manager()

        def writer(wid: int) -> None:
            for i in range(_WRITER_CYCLES):
                nid = f"w{wid}-n{i % _NODE_POOL}"
                start = i % 32
                pipeline.register_node(nid, "10.0.0.1", 50051, start, start)
                pipeline.unregister_node(nid)
                if i % 7 == 0:  # leave some residue behind
                    pipeline.register_node(nid, "10.0.0.1", 50051, start, start)

        threads = [threading.Thread(target=writer, args=(w,))
                   for w in range(_WRITER_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        nodes = pipeline.nodes
        order = pipeline.node_order
        # Structural agreement, checked twice for stability.
        for _ in range(2):
            assert set(order) == set(nodes.keys()), (
                f"node_order/nodes diverged: {sorted(set(order) ^ set(nodes))}"
            )
            assert mgr.node_count == len(nodes)
        # Order is sorted by start_layer (register_node's contract).
        layers = [nodes[nid].start_layer for nid in order]
        assert layers == sorted(layers)


class TestWeightSourceRegistryConcurrency:
    """_model_registry mutations vs lookups under the registry lock."""

    def test_concurrent_weight_source_round_trip(self) -> None:
        mgr, pipeline = _make_manager()
        errors: list[str] = []
        n_threads = 8
        barrier = threading.Barrier(n_threads)

        def worker(wid: int) -> None:
            try:
                barrier.wait(timeout=10)
                for i in range(200):
                    start, end = i % 16, i % 16
                    nid = f"w{wid}-{i % 10}"
                    pipeline.register_node(nid, "10.0.0.2", 50052, start, end)
                    mgr._register_weight_source(nid, "stress-model", start, end)
                    got = mgr._get_weight_source("stress-model", start, end)
                    # Either absent (never registered for this exact key by
                    # THIS worker yet raced away is impossible -- keys are
                    # overwritten, never deleted) or a valid (host, port).
                    assert got is None or (
                        isinstance(got, tuple) and len(got) == 2
                    ), f"corrupt registry entry: {got!r}"
            except Exception as exc:  # noqa: BLE001 - collected below
                errors.append(f"worker{wid}: {exc!r}")

        threads = [threading.Thread(target=worker, args=(w,))
                   for w in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert errors == [], "registry concurrency failures:\n" + "\n".join(errors)


class TestSnapshotFallbackPath:
    """Pipelines without an internal lock fall back to a plain copy."""

    def test_stub_pipeline_without_lock_attribute(self) -> None:
        class _LocklessPipeline:
            def __init__(self):
                self._nodes = {"n1": {"host": "h", "port": 1}}

            @property
            def nodes(self):
                return self._nodes

        mgr = ClusterManager(pipeline=_LocklessPipeline(), model_name="m")
        summary = mgr.get_node_gpu_summary()
        assert set(summary) == {"n1"}
        assert summary["n1"]["memory_total_gb"] == 0.0
