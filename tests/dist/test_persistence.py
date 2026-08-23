"""Real tests for partition/persistence — PartitionStore, PartitionRun, RunComparison.

Zero mocks — all tests use real SQLite instances and deterministic logic.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from distllm.dist.partition.persistence import PartitionRun, PartitionStore, RunComparison


@pytest.fixture
def store() -> PartitionStore:
    path = os.path.join(tempfile.gettempdir(), "test_partition_store.db")
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    s = PartitionStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class TestPartitionRun:
    """Test the PartitionRun dataclass."""

    def test_init_basic(self):
        run = PartitionRun(
            run_id=1,
            model_name="meta-llama/Llama-3-70B",
            created_at=1000.0,
            config={"hidden_size": 8192, "num_layers": 80},
            solution={"max_node_time_ms": 50.0, "total_memory_gb": 140.0},
            gpu_profiles=[{"name": "A100", "memory_gb": 80}],
        )
        assert run.run_id == 1
        assert run.model_name == "meta-llama/Llama-3-70B"
        assert run.created_at == 1000.0
        assert run.config == {"hidden_size": 8192, "num_layers": 80}
        assert run.solution == {"max_node_time_ms": 50.0, "total_memory_gb": 140.0}
        assert run.gpu_profiles == [{"name": "A100", "memory_gb": 80}]

    def test_default_values(self):
        run = PartitionRun(
            run_id=10,
            model_name="test-model",
            created_at=500.0,
            config={},
            solution={},
            gpu_profiles=[],
        )
        assert run.metrics == {}
        assert run.tags == []
        assert run.is_good is True

    def test_custom_metrics_tags_and_is_good(self):
        run = PartitionRun(
            run_id=20,
            model_name="test-model",
            created_at=500.0,
            config={},
            solution={},
            gpu_profiles=[],
            metrics={"actual_latency_ms": 45.2, "actual_throughput_tok_s": 120.0},
            tags=["production", "v2"],
            is_good=False,
        )
        assert run.metrics["actual_latency_ms"] == 45.2
        assert run.metrics["actual_throughput_tok_s"] == 120.0
        assert run.tags == ["production", "v2"]
        assert run.is_good is False


class TestRunComparison:
    """Test the RunComparison dataclass."""

    def test_init(self):
        run_a = PartitionRun(1, "m", 0.0, {}, {}, [])
        run_b = PartitionRun(2, "m", 0.0, {}, {}, [])
        comp = RunComparison(
            run_a=run_a,
            run_b=run_b,
            latency_diff_ms=-10.5,
            latency_diff_pct=-15.2,
            throughput_diff_tok_s=30.0,
            throughput_diff_pct=25.0,
            memory_diff_gb=-3.5,
            winner="B",
            summary="B is better",
        )
        assert comp.run_a is run_a
        assert comp.run_b is run_b
        assert comp.latency_diff_ms == -10.5
        assert comp.latency_diff_pct == -15.2
        assert comp.throughput_diff_tok_s == 30.0
        assert comp.throughput_diff_pct == 25.0
        assert comp.memory_diff_gb == -3.5
        assert comp.winner == "B"
        assert comp.summary == "B is better"


class TestPartitionStore:
    """Test the PartitionStore class with real SQLite backing."""

    # -- lifecycle -----------------------------------------------------------

    def test_init_and_close(self, store: PartitionStore):
        assert store is not None
        store.close()
        # close is idempotent
        store.close()

    # -- save_run / get_run round-trip ----------------------------------------

    def test_save_and_get_run(self, store: PartitionStore):
        run_id = store.save_run(
            model_name="meta-llama/Llama-3-70B",
            solution={"max_node_time_ms": 50.0, "total_memory_gb": 140.0},
            config={"hidden_size": 8192, "num_layers": 80},
            gpu_profiles=[{"name": "A100", "memory_gb": 80, "count": 2}],
            tags=["v1", "production"],
        )
        assert isinstance(run_id, int)
        assert run_id >= 1

        run = store.get_run(run_id)
        assert run is not None
        assert run.model_name == "meta-llama/Llama-3-70B"
        assert run.config == {"hidden_size": 8192, "num_layers": 80}
        assert run.solution == {"max_node_time_ms": 50.0, "total_memory_gb": 140.0}
        assert run.gpu_profiles == [{"name": "A100", "memory_gb": 80, "count": 2}]
        assert run.tags == ["v1", "production"]
        assert run.is_good is True
        assert run.created_at > 0

    def test_save_run_with_none_gpu_profiles_and_tags(self, store: PartitionStore):
        run_id = store.save_run(
            model_name="test",
            solution={},
            config={},
            gpu_profiles=None,
            tags=None,
        )
        run = store.get_run(run_id)
        assert run is not None
        assert run.gpu_profiles == []
        assert run.tags == []

    def test_save_run_with_dict_solution(self, store: PartitionStore):
        run_id = store.save_run("m", {"key": "value"}, {})
        run = store.get_run(run_id)
        assert run is not None
        assert run.solution == {"key": "value"}

    def test_save_run_with_object_solution(self, store: PartitionStore):
        class _Solution:
            def __init__(self) -> None:
                self.max_node_time_ms = 75.0
                self.total_memory_gb = 160.0
                self._internal = "skipped"

        sol = _Solution()
        run_id = store.save_run("model", solution=sol, config={"layers": 40})
        run = store.get_run(run_id)
        assert run is not None
        assert run.solution["max_node_time_ms"] == 75.0
        assert run.solution["total_memory_gb"] == 160.0
        # Private attributes (starting with _) are excluded
        assert "_internal" not in run.solution

    def test_save_run_with_nested_object_solution(self, store: PartitionStore):
        class _Nested:
            def __init__(self) -> None:
                self.value = 42

        class _Solution:
            def __init__(self) -> None:
                self.inner = _Nested()
                self.name = "test"

        sol = _Solution()
        run_id = store.save_run("model", solution=sol, config={})
        run = store.get_run(run_id)
        assert run is not None
        assert run.solution["inner"]["value"] == 42
        assert run.solution["name"] == "test"

    def test_save_run_with_raw_solution(self, store: PartitionStore):
        run_id = store.save_run("model", solution="some_raw_string", config={})
        run = store.get_run(run_id)
        assert run is not None
        assert run.solution == {"raw": "some_raw_string"}

    def test_save_run_with_list_solution_objects(self, store: PartitionStore):
        class _Item:
            def __init__(self) -> None:
                self.x = 10

        class _Solution:
            def __init__(self) -> None:
                self.items = [_Item(), _Item()]

        sol = _Solution()
        run_id = store.save_run("model", solution=sol, config={})
        run = store.get_run(run_id)
        assert run is not None
        assert len(run.solution["items"]) == 2
        assert run.solution["items"][0]["x"] == 10
        assert run.solution["items"][1]["x"] == 10

    # -- get_run edge cases --------------------------------------------------

    def test_get_run_nonexistent(self, store: PartitionStore):
        assert store.get_run(9999) is None

    # -- record_metric / get_best_run ----------------------------------------

    def test_record_and_get_metrics(self, store: PartitionStore):
        run_id = store.save_run("model", {}, {})
        store.record_metric(run_id, "actual_latency_ms", 45.2)
        store.record_metric(run_id, "actual_throughput_tok_s", 120.0)
        store.record_metric(run_id, "actual_memory_gb", 70.5)

        run = store.get_run(run_id)
        assert run is not None
        assert run.metrics["actual_latency_ms"] == 45.2
        assert run.metrics["actual_throughput_tok_s"] == 120.0
        assert run.metrics["actual_memory_gb"] == 70.5

    def test_get_best_run_lower_latency_wins(self, store: PartitionStore):
        id_slow = store.save_run("model", {}, {})
        id_fast = store.save_run("model", {}, {})
        store.record_metric(id_slow, "actual_latency_ms", 100.0)
        store.record_metric(id_fast, "actual_latency_ms", 40.0)
        best = store.get_best_run("model")
        assert best is not None
        assert best.run_id == id_fast

    def test_get_best_run_no_data(self, store: PartitionStore):
        assert store.get_best_run("nonexistent") is None

    def test_get_best_run_no_metrics(self, store: PartitionStore):
        store.save_run("model", {}, {})
        assert store.get_best_run("model") is None

    def test_get_best_run_custom_metric(self, store: PartitionStore):
        id_low = store.save_run("model", {}, {})
        id_high = store.save_run("model", {}, {})
        store.record_metric(id_low, "memory_usage_gb", 10.0)
        store.record_metric(id_high, "memory_usage_gb", 20.0)
        best = store.get_best_run("model", metric="memory_usage_gb")
        assert best is not None
        assert best.run_id == id_low

    # -- get_runs ------------------------------------------------------------

    def test_get_runs_empty(self, store: PartitionStore):
        assert store.get_runs() == []

    def test_get_runs_all(self, store: PartitionStore):
        for _ in range(5):
            store.save_run("model", {}, {})
        runs = store.get_runs()
        assert len(runs) == 5

    def test_get_runs_filter_by_model(self, store: PartitionStore):
        store.save_run("model-a", {}, {})
        store.save_run("model-a", {}, {})
        store.save_run("model-b", {}, {})
        runs_a = store.get_runs(model_name="model-a")
        assert len(runs_a) == 2
        runs_b = store.get_runs(model_name="model-b")
        assert len(runs_b) == 1

    def test_get_runs_limit(self, store: PartitionStore):
        for _ in range(10):
            store.save_run("model", {}, {})
        runs = store.get_runs(limit=3)
        assert len(runs) == 3

    def test_get_runs_filter_by_tags(self, store: PartitionStore):
        store.save_run("model", {}, {}, tags=["stable"])
        store.save_run("model", {}, {}, tags=["experimental"])
        store.save_run("model", {}, {}, tags=["stable", "production"])
        runs = store.get_runs(tags=["stable"])
        # Two runs have "stable" tag
        assert len(runs) == 2
        runs = store.get_runs(tags=["experimental"])
        assert len(runs) == 1

    def test_get_runs_tags_empty_list_no_filter(self, store: PartitionStore):
        store.save_run("model", {}, {}, tags=["a"])
        store.save_run("model", {}, {}, tags=["b"])
        # Empty tags list is falsy, so no tag filtering is applied
        runs = store.get_runs(tags=[])
        assert len(runs) == 2

    def test_get_runs_ordered_by_created_at_desc(self, store: PartitionStore):
        id1 = store.save_run("model", {}, {})
        id2 = store.save_run("model", {}, {})
        id3 = store.save_run("model", {}, {})
        runs = store.get_runs()
        assert runs[0].run_id == id3
        assert runs[1].run_id == id2
        assert runs[2].run_id == id1

    # -- mark_run_quality / get_last_known_good -------------------------------

    def test_mark_run_quality_and_get_last_known_good(self, store: PartitionStore):
        id_bad = store.save_run("model", {}, {}, tags=["bad"])
        id_good = store.save_run("model", {}, {}, tags=["good"])
        store.mark_run_quality(id_bad, False)
        lkg = store.get_last_known_good("model")
        assert lkg is not None
        assert lkg.run_id == id_good

    def test_get_last_known_good_most_recent_good(self, store: PartitionStore):
        id1 = store.save_run("model", {}, {})
        id2 = store.save_run("model", {}, {})
        id3 = store.save_run("model", {}, {})
        store.mark_run_quality(id1, False)
        store.mark_run_quality(id3, False)
        # id2 is the only good run
        lkg = store.get_last_known_good("model")
        assert lkg is not None
        assert lkg.run_id == id2

    def test_get_last_known_good_all_bad(self, store: PartitionStore):
        id1 = store.save_run("model", {}, {})
        store.mark_run_quality(id1, False)
        assert store.get_last_known_good("model") is None

    def test_get_last_known_good_no_runs(self, store: PartitionStore):
        assert store.get_last_known_good("nonexistent") is None

    # -- compare_runs --------------------------------------------------------

    def test_compare_runs_winner_b(self, store: PartitionStore):
        """B has lower latency and higher throughput -> winner B."""
        id_a = store.save_run("model", {}, {})
        id_b = store.save_run("model", {}, {})
        store.record_metric(id_a, "actual_latency_ms", 100.0)
        store.record_metric(id_a, "actual_throughput_tok_s", 50.0)
        store.record_metric(id_b, "actual_latency_ms", 80.0)
        store.record_metric(id_b, "actual_throughput_tok_s", 70.0)
        comp = store.compare_runs(id_a, id_b)
        assert comp is not None
        assert comp.winner == "B"

    def test_compare_runs_winner_a(self, store: PartitionStore):
        """A has lower latency and higher throughput -> winner A."""
        # A is first arg, B is second arg; A (good) vs B (bad) -> A wins
        id_good = store.save_run("model", {}, {})
        id_bad = store.save_run("model", {}, {})
        store.record_metric(id_good, "actual_latency_ms", 80.0)
        store.record_metric(id_good, "actual_throughput_tok_s", 70.0)
        store.record_metric(id_bad, "actual_latency_ms", 100.0)
        store.record_metric(id_bad, "actual_throughput_tok_s", 50.0)
        comp = store.compare_runs(id_good, id_bad)
        assert comp is not None
        assert comp.winner == "A"

    def test_compare_runs_tie(self, store: PartitionStore):
        """When both metrics change by < 2% it's a tie."""
        id_a = store.save_run("model", {}, {})
        id_b = store.save_run("model", {}, {})
        store.record_metric(id_a, "actual_latency_ms", 100.0)
        store.record_metric(id_a, "actual_throughput_tok_s", 50.0)
        store.record_metric(id_b, "actual_latency_ms", 101.0)
        store.record_metric(id_b, "actual_throughput_tok_s", 50.5)
        comp = store.compare_runs(id_a, id_b)
        assert comp is not None
        assert comp.winner == "tie"

    def test_compare_runs_nonexistent(self, store: PartitionStore):
        id_real = store.save_run("model", {}, {})
        assert store.compare_runs(id_real, 9999) is None
        assert store.compare_runs(9999, id_real) is None

    def test_compare_runs_fallback_to_solution_estimates(self, store: PartitionStore):
        """When no recorded metrics, falls back to solution estimates."""
        id_a = store.save_run(
            "model",
            solution={
                "max_node_time_ms": 100.0,
                "estimated_throughput_tok_s": 50.0,
                "total_memory_gb": 80.0,
            },
            config={},
        )
        id_b = store.save_run(
            "model",
            solution={
                "max_node_time_ms": 200.0,
                "estimated_throughput_tok_s": 30.0,
                "total_memory_gb": 70.0,
            },
            config={},
        )
        comp = store.compare_runs(id_a, id_b)
        assert comp is not None
        # B has higher latency -> positive diff
        assert comp.latency_diff_ms > 0
        # B has lower throughput -> negative diff
        assert comp.throughput_diff_tok_s < 0
        # B has less memory -> negative diff
        assert comp.memory_diff_gb < 0

    def test_compare_runs_summary(self, store: PartitionStore):
        id_a = store.save_run("model", {}, {})
        id_b = store.save_run("model", {}, {})
        store.record_metric(id_a, "actual_latency_ms", 100.0)
        store.record_metric(id_a, "actual_throughput_tok_s", 50.0)
        store.record_metric(id_b, "actual_latency_ms", 80.0)
        store.record_metric(id_b, "actual_throughput_tok_s", 70.0)
        comp = store.compare_runs(id_a, id_b)
        assert comp is not None
        assert comp.summary.startswith("Run")
        assert "winner:" in comp.summary

    # -- get_accuracy_report -------------------------------------------------

    def test_accuracy_report_with_data(self, store: PartitionStore):
        id1 = store.save_run("model", {"max_node_time_ms": 100.0}, {})
        id2 = store.save_run("model", {"max_node_time_ms": 200.0}, {})
        store.record_metric(id1, "actual_latency_ms", 90.0)
        store.record_metric(id2, "actual_latency_ms", 180.0)
        report = store.get_accuracy_report("model")
        assert report["num_samples"] == 2
        assert report["mae_ms"] > 0
        assert report["mape_pct"] > 0
        assert report["max_error_ms"] > 0
        assert len(report["entries"]) == 2

    def test_accuracy_report_no_runs(self, store: PartitionStore):
        report = store.get_accuracy_report("nonexistent")
        assert report["num_samples"] == 0
        assert report["mae_ms"] == 0
        assert report["mape_pct"] == 0
        assert report["entries"] == []

    def test_accuracy_report_no_metrics(self, store: PartitionStore):
        store.save_run("model", {"max_node_time_ms": 100.0}, {})
        report = store.get_accuracy_report("model")
        assert report["num_samples"] == 0

    def test_accuracy_report_skips_zero_prediction(self, store: PartitionStore):
        """Runs with zero predicted latency are skipped."""
        id_ok = store.save_run("model", {"max_node_time_ms": 100.0}, {})
        id_bad = store.save_run("model", {"max_node_time_ms": 0.0}, {})
        store.record_metric(id_ok, "actual_latency_ms", 90.0)
        store.record_metric(id_bad, "actual_latency_ms", 50.0)
        report = store.get_accuracy_report("model")
        assert report["num_samples"] == 1
        assert report["entries"][0]["run_id"] == id_ok

    def test_accuracy_report_skips_missing_actual(self, store: PartitionStore):
        """Runs without an actual metric are skipped."""
        id_ok = store.save_run("model", {"max_node_time_ms": 100.0}, {})
        id_no_metric = store.save_run("model", {"max_node_time_ms": 100.0}, {})
        store.record_metric(id_ok, "actual_latency_ms", 90.0)
        report = store.get_accuracy_report("model")
        assert report["num_samples"] == 1
        assert report["entries"][0]["run_id"] == id_ok

    # -- delete_run ----------------------------------------------------------

    def test_delete_run_removes_row_and_metrics(self, store: PartitionStore):
        id1 = store.save_run("model", {}, {})
        id2 = store.save_run("model", {}, {})
        store.record_metric(id1, "actual_latency_ms", 10.0)
        store.delete_run(id1)
        assert store.get_run(id1) is None
        # id2 should still exist
        assert store.get_run(id2) is not None

    def test_delete_run_nonexistent(self, store: PartitionStore):
        # Should not raise
        store.delete_run(9999)

    # -- integration: full workflow ------------------------------------------

    def test_full_save_compare_accuracy_workflow(self, store: PartitionStore):
        """End-to-end: save runs, record metrics, compare, get report."""
        # Save two runs
        id_a = store.save_run(
            "gpt-4",
            solution={"max_node_time_ms": 120.0, "estimated_throughput_tok_s": 80.0, "total_memory_gb": 100.0},
            config={"hidden_size": 8192},
            gpu_profiles=[{"name": "A100"}],
            tags=["baseline"],
        )
        id_b = store.save_run(
            "gpt-4",
            solution={"max_node_time_ms": 90.0, "estimated_throughput_tok_s": 110.0, "total_memory_gb": 95.0},
            config={"hidden_size": 8192},
            gpu_profiles=[{"name": "A100"}],
            tags=["optimized"],
        )

        # Record actual metrics
        store.record_metric(id_a, "actual_latency_ms", 115.0)
        store.record_metric(id_a, "actual_throughput_tok_s", 82.0)
        store.record_metric(id_b, "actual_latency_ms", 88.0)
        store.record_metric(id_b, "actual_throughput_tok_s", 112.0)

        # Compare
        comp = store.compare_runs(id_a, id_b)
        assert comp is not None
        assert comp.winner == "B"

        # Accuracy report
        report = store.get_accuracy_report("gpt-4")
        assert report["num_samples"] == 2

        # Mark old run as bad and verify LKG
        store.mark_run_quality(id_a, False)
        lkg = store.get_last_known_good("gpt-4")
        assert lkg is not None
        assert lkg.run_id == id_b

        # Filter by tags
        runs = store.get_runs(tags=["optimized"])
        assert len(runs) == 1
        assert runs[0].run_id == id_b
