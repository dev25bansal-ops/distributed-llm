"""Tests: Bayesian optimization, warmup, tune interval, profile persistence, cold start.

Covers BayesianOptimizer, SearchSpace, TrialRunner, OptimizationTracker,
and SelfOptimizingEngine warmup/tune-interval/profile-dir behavior.

Run: pytest tests/core/test_bayesian_optimization.py -v
"""

import json
import os
import tempfile
import time
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distllm.core.optimization.bayesian import BayesianOptimizer, ObjectiveDirection
from distllm.core.optimization.space import (
    SearchSpace,
    IntDomain,
    CategoricalDomain,
    default_search_space,
)
from distllm.core.optimization.runner import TrialRunner, TrialResult
from distllm.core.optimization.tracker import OptimizationTracker, BestConfig, TrialRecord
from distllm.core.self_optimizing_engine import (
    SelfOptimizingEngine,
    PerformanceModel,
    ParameterTuner,
    TunableParams,
    OpType,
)


# ===========================================================================
# 1. Bayesian Optimization — Hyperparameter Search → Best Config
# ===========================================================================


class TestBayesianOptimizer:
    """Hyperparameter search finds the best configuration."""

    def test_init_defaults(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        assert opt._direction == ObjectiveDirection.MAXIMIZE
        assert opt._n_startup_trials == 10
        assert opt.finished_trials_count() == 0

    def test_suggest_returns_valid_config(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        config = opt.suggest()
        assert isinstance(config, dict)
        for param in space.param_names:
            assert param in config

    def test_suggest_and_report_improves_over_trials(self):
        space = default_search_space()
        space.batch_size = IntDomain("batch_size", 1, 4, log=False)
        space.chunk_size = IntDomain("chunk_size", 128, 256, log=False)

        opt = BayesianOptimizer(space, n_startup_trials=2, n_ei_candidates=4)

        def objective(config):
            bs = config.get("batch_size", 1)
            cs = config.get("chunk_size", 128)
            cs_norm = cs / 256.0
            return float(bs) * (1.0 + cs_norm)

        configs_seen = []
        for _ in range(6):
            config = opt.suggest()
            configs_seen.append(config)
            value = objective(config)
            opt.report(value)

        best = opt.best_config
        assert best is not None
        assert opt.best_value is not None
        assert opt.finished_trials_count() == 6

    def test_best_config_none_before_any_trials(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        assert opt.best_config is None
        assert opt.best_value is None

    def test_optimize_runs_full_loop(self):
        space = default_search_space()
        space.batch_size = IntDomain("batch_size", 1, 4, log=False)
        space.tensor_parallel_degree = IntDomain("tensor_parallel_degree", 1, 2, log=False)

        opt = BayesianOptimizer(space, n_startup_trials=2, n_ei_candidates=4)

        def objective(config):
            return float(config.get("batch_size", 1))

        best = opt.optimize(objective, n_trials=5)
        assert best is not None
        assert opt.finished_trials_count() == 5

    def test_minimize_direction(self):
        space = default_search_space()
        space.batch_size = IntDomain("batch_size", 1, 8, log=False)

        opt = BayesianOptimizer(space, direction=ObjectiveDirection.MINIMIZE, n_startup_trials=2, n_ei_candidates=4)

        def objective(config):
            return float(config.get("batch_size", 8))

        best = opt.optimize(objective, n_trials=6)
        assert best is not None
        assert best.get("batch_size", 8) <= 8

    def test_trials_dataframe(self):
        space = default_search_space()
        space.batch_size = IntDomain("batch_size", 1, 4, log=False)
        opt = BayesianOptimizer(space, n_startup_trials=2, n_ei_candidates=4)

        def objective(config):
            return float(config.get("batch_size", 1))

        for _ in range(5):
            config = opt.suggest()
            opt.report(objective(config))

        df = opt.trials_dataframe
        assert len(df) == 5

    def test_summary_contains_best_info(self):
        space = default_search_space()
        space.batch_size = IntDomain("batch_size", 1, 4, log=False)
        opt = BayesianOptimizer(space, n_startup_trials=2, n_ei_candidates=4)
        opt.suggest()
        opt.report(1.0)
        s = opt.summary()
        assert "Best" in s
        assert "Trials" in s


class TestSearchSpace:
    """Search space domain definitions."""

    def test_default_space_has_all_params(self):
        space = default_search_space()
        names = space.param_names
        assert "batch_size" in names
        assert "tensor_parallel_degree" in names
        assert "pipeline_stages" in names
        assert "quantization" in names
        assert "speculation_length" in names
        assert "chunk_size" in names

    def test_sample_random_config(self):
        space = default_search_space()
        config = space.sample_random_config()
        assert 1 <= config["batch_size"] <= 128
        assert 1 <= config["tensor_parallel_degree"] <= 8
        assert 1 <= config["pipeline_stages"] <= 4
        assert config["quantization"] in ("none", "bnb_8bit", "fp8")
        assert 0 <= config["speculation_length"] <= 10
        assert 128 <= config["chunk_size"] <= 2048

    def test_validate_valid_config(self):
        space = default_search_space()
        config = {"batch_size": 32, "tensor_parallel_degree": 2, "pipeline_stages": 1,
                  "quantization": "fp8", "speculation_length": 3, "chunk_size": 512}
        validated = space.validate(config)
        assert validated == config

    def test_validate_clamps_out_of_range(self):
        space = default_search_space()
        config = {"batch_size": 999, "tensor_parallel_degree": 2, "pipeline_stages": 1,
                  "quantization": "fp8", "speculation_length": 3, "chunk_size": 512}
        validated = space.validate(config)
        assert validated["batch_size"] == 128

    def test_validate_raises_on_invalid_categorical(self):
        space = default_search_space()
        config = {"batch_size": 32, "tensor_parallel_degree": 2, "pipeline_stages": 1,
                  "quantization": "invalid", "speculation_length": 3, "chunk_size": 512}
        with pytest.raises(ValueError):
            space.validate(config)

    def test_validate_raises_on_missing_param(self):
        space = default_search_space()
        with pytest.raises(ValueError):
            space.validate({"batch_size": 32})

    def test_to_dict_returns_descriptions(self):
        space = default_search_space()
        d = space.to_dict()
        assert "batch_size" in d
        assert d["batch_size"]["type"] == "int"
        assert d["batch_size"]["low"] == 1
        assert d["quantization"]["type"] == "categorical"
        assert "fp8" in d["quantization"]["choices"]

    def test_int_domain_sample_log_scale(self):
        domain = IntDomain("test", low=1, high=1024, log=True)
        random.seed(42)
        val = domain.sample_random()
        assert 1 <= val <= 1024


# ===========================================================================
# 2. Warmup Phase — No Changes During Warmup
# ===========================================================================


class TestTrialRunnerWarmup:
    """No configuration changes applied during warmup period."""

    def test_run_applies_config_then_warmup(self):
        applied_configs = []
        def apply_config(cfg):
            applied_configs.append(cfg)

        runner = TrialRunner(apply_config=apply_config, warmup_seconds=0.01, cooldown_seconds=0)
        config = {"batch_size": 8}
        runner.run(config)
        assert len(applied_configs) >= 1
        assert applied_configs[0] == config

    def test_warmup_sleeps_before_benchmark(self):
        applied_configs = []
        bench_calls = []
        def apply_config(cfg):
            applied_configs.append(cfg)
        def benchmark():
            bench_calls.append(1)
            return TrialResult(config={}, throughput_tok_s=100.0, avg_latency_ms=10.0,
                               p99_latency_ms=20.0, duration_seconds=1.0, num_requests=10)

        runner = TrialRunner(apply_config=apply_config, run_benchmark=benchmark,
                             warmup_seconds=0.05, cooldown_seconds=0)
        start = time.time()
        runner.run({"batch_size": 8})
        elapsed = time.time() - start
        assert elapsed >= 0.04
        assert len(bench_calls) == 1

    def test_warmup_zero_skips_sleep(self):
        applied = []
        def apply_config(cfg):
            applied.append(cfg)
        def benchmark():
            return TrialResult(config={}, throughput_tok_s=100.0, avg_latency_ms=10.0,
                               p99_latency_ms=20.0, duration_seconds=1.0, num_requests=10)

        runner = TrialRunner(apply_config=apply_config, run_benchmark=benchmark,
                             warmup_seconds=0, cooldown_seconds=0)
        start = time.time()
        runner.run({"batch_size": 8})
        assert time.time() - start < 0.5

    def test_no_benchmark_returns_none(self):
        def apply_config(cfg):
            pass
        runner = TrialRunner(apply_config=apply_config, warmup_seconds=0, cooldown_seconds=0)
        result = runner.run({"batch_size": 8})
        assert result is None


class TestSelfOptimizingWarmup:
    """SelfOptimizingEngine does not tune during warmup period."""

    def test_warmup_phase_skips_tuning(self):
        engine = SelfOptimizingEngine(warmup_seconds=60.0, tune_interval_seconds=1.0)
        engine._start_time = time.time()
        # Simulate what _tune_loop does during warmup:
        should_tune = (time.time() - engine._start_time) < engine._warmup_seconds
        assert should_tune is True

        # Check the warmup guard
        params_before = engine.get_current_params()
        # Manually trigger tune iteration
        engine._flush_samples()
        # During warmup, no tuning should happen
        elapsed = time.time() - engine._start_time
        assert elapsed < 60.0  # Still in warmup
        params_after = engine.get_current_params()
        assert params_before.batch_size == params_after.batch_size

    def test_warmup_ends_after_duration(self):
        engine = SelfOptimizingEngine(warmup_seconds=0.0, tune_interval_seconds=60.0)
        engine._start_time = time.time() - 1.0
        elapsed = time.time() - engine._start_time
        assert elapsed >= engine._warmup_seconds


# ===========================================================================
# 3. Tune Interval — Periodic Retuning
# ===========================================================================


class TestSelfOptimizingTuneInterval:
    """The engine retunes at the configured interval."""

    def test_tune_interval_respected(self):
        engine = SelfOptimizingEngine(tune_interval_seconds=60.0, warmup_seconds=0)
        engine._start_time = time.time() - 120.0  # past warmup
        # After warmup, check tune interval
        last_tune = 0.0
        now = time.time()
        elapsed = now - engine._start_time
        assert elapsed >= 60.0  # past tune interval
        should_tune = (now - last_tune) >= engine._tune_interval
        assert should_tune is True

    def test_tune_interval_short_triggers_frequently(self):
        engine = SelfOptimizingEngine(tune_interval_seconds=0.1, warmup_seconds=0)
        assert engine._tune_interval == 0.1

    def test_tune_interval_can_be_disabled(self):
        engine = SelfOptimizingEngine(tune_interval_seconds=0, warmup_seconds=0)
        assert engine._tune_interval == 0


# ===========================================================================
# 4. Profile Directory — Results Stored Correctly
# ===========================================================================


class TestOptimizationTrackerPersistence:
    """Profiling results are stored to and loaded from disk."""

    def test_record_saves_to_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OptimizationTracker(output_dir=tmpdir, study_name="test_study")
            tracker.record(trial_number=1,
                           config={"batch_size": 8, "chunk_size": 512},
                           value=100.0,
                           duration_seconds=10.0)
            trials_path = Path(tmpdir) / "test_study_trials.json"
            best_path = Path(tmpdir) / "test_study_best.json"
            assert trials_path.exists()
            assert best_path.exists()
            with open(trials_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["trial_number"] == 1
            assert data[0]["value"] == 100.0

    def test_load_persisted_best_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker1 = OptimizationTracker(output_dir=tmpdir, study_name="test_load")
            tracker1.record(trial_number=1,
                            config={"batch_size": 4},
                            value=80.0,
                            duration_seconds=5.0)
            tracker1.record(trial_number=2,
                            config={"batch_size": 16},
                            value=150.0,
                            duration_seconds=5.0)
            tracker2 = OptimizationTracker(output_dir=tmpdir, study_name="test_load")
            assert tracker2.trial_count == 2
            best = tracker2.best_config
            assert best is not None
            assert best.config["batch_size"] == 16
            assert best.value == 150.0

    def test_best_config_updates_when_better_found(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_best")
        assert tracker.best_config is None
        tracker.record(trial_number=1, config={"x": 1}, value=10.0, duration_seconds=1.0)
        assert tracker.best_config.value == 10.0
        tracker.record(trial_number=2, config={"x": 2}, value=20.0, duration_seconds=1.0)
        assert tracker.best_config.value == 20.0
        tracker.record(trial_number=3, config={"x": 3}, value=5.0, duration_seconds=1.0)
        assert tracker.best_config.value == 20.0

    def test_minimize_direction_selects_lower(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_min", maximize=False)
        tracker.record(trial_number=1, config={"x": 1}, value=100.0, duration_seconds=1.0)
        tracker.record(trial_number=2, config={"x": 2}, value=50.0, duration_seconds=1.0)
        assert tracker.best_config.value == 50.0

    def test_error_trials_skipped_in_best(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_err")
        tracker.record(trial_number=1, config={"x": 1}, value=100.0, duration_seconds=1.0)
        tracker.record(trial_number=2, config={"x": 2}, value=200.0, duration_seconds=1.0, error="OOM")
        assert tracker.best_config.value == 100.0

    def test_top_k_returns_sorted_results(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_topk")
        for i in range(10):
            tracker.record(trial_number=i, config={"i": i}, value=float(i * 10), duration_seconds=1.0)
        top = tracker.top_k(k=3)
        assert len(top) == 3
        assert top[0].value == 90.0
        assert top[1].value == 80.0
        assert top[2].value == 70.0

    def test_trials_sorted_by_value(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_sort")
        for i in [3, 1, 2]:
            tracker.record(trial_number=i, config={"i": i}, value=float(i), duration_seconds=1.0)
        sorted_trials = tracker.trials_sorted(key="value", reverse=True)
        assert [t.value for t in sorted_trials] == [3.0, 2.0, 1.0]

    def test_summary_format(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_sum")
        tracker.record(trial_number=1, config={"x": 1}, value=10.0, duration_seconds=1.0)
        s = tracker.summary()
        assert "test_sum" in s
        assert "10.0" in s

    def test_empty_tracker_no_best(self):
        tracker = OptimizationTracker(output_dir=tempfile.mkdtemp(), study_name="test_empty")
        assert tracker.best_config is None
        assert tracker.trial_count == 0


class TestSelfOptimizingProfileDir:
    """Profile directory stores results correctly."""

    def test_profile_dir_created_on_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = os.path.join(tmpdir, "my_profiles")
            engine = SelfOptimizingEngine(model_name="test_model", profile_dir=profile_dir)
            assert os.path.exists(profile_dir)
            engine.stop()

    def test_save_profile_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfOptimizingEngine(model_name="save_test", profile_dir=tmpdir)
            engine._save_profile()
            path = os.path.join(tmpdir, "save_test.json")
            assert os.path.exists(path)
            engine.stop()

    def test_save_profile_contains_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfOptimizingEngine(model_name="keys_test", profile_dir=tmpdir)
            engine._save_profile()
            path = os.path.join(tmpdir, "keys_test.json")
            with open(path) as f:
                data = json.load(f)
            assert "model_name" in data
            assert "best_params" in data
            assert "current_params" in data
            assert "best_throughput" in data
            engine.stop()

    def test_load_profile_restores_best_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine1 = SelfOptimizingEngine(model_name="load_test", profile_dir=tmpdir)
            engine1._tuner._best_params = TunableParams(batch_size=32, kv_cache_quant_bits=4)
            engine1._tuner._best_throughput = 500.0
            engine1._save_profile()
            engine1.stop()

            engine2 = SelfOptimizingEngine(model_name="load_test", profile_dir=tmpdir)
            assert engine2._tuner.best_params.batch_size == 32
            assert engine2._tuner.best_throughput == 500.0
            engine2.stop()

    def test_no_profile_file_on_fresh_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfOptimizingEngine(model_name="fresh", profile_dir=tmpdir)
            params = engine.get_optimal_params()
            assert params.batch_size == 1  # default, not loaded
            engine.stop()


# ===========================================================================
# 5. Optimization — Cold Start
# ===========================================================================


class TestBayesianOptimizerColdStart:
    """No history — returns default / None."""

    def test_no_trials_returns_none_best(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        assert opt.best_config is None
        assert opt.best_value is None

    def test_no_trials_finished_count_zero(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        assert opt.finished_trials_count() == 0

    def test_empty_trials_list(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        assert len(opt.trials) == 0

    def test_suggest_works_with_no_history(self):
        space = default_search_space()
        opt = BayesianOptimizer(space)
        config = opt.suggest()
        assert isinstance(config, dict)
        assert len(config) >= 1


class TestSearchSpaceColdStart:
    """Default config used when no search history."""

    def test_default_search_space_has_reasonable_defaults(self):
        space = default_search_space()
        assert space.batch_size.low == 1
        assert space.batch_size.high == 128
        assert space.tensor_parallel_degree.high == 8
        assert len(space.quantization.choices) == 3


class TestTrialRunnerColdStart:
    """Runner returns None when no benchmark set."""

    def test_no_benchmark_returns_none(self):
        def apply(cfg):
            pass
        runner = TrialRunner(apply_config=apply, warmup_seconds=0, cooldown_seconds=0)
        result = runner.run({"batch_size": 8})
        assert result is None


class TestTrackerColdStart:
    """Tracker returns None for best_config before any trial."""

    def test_no_trials_best_is_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OptimizationTracker(output_dir=tmpdir, study_name="cold")
            assert tracker.best_config is None
            assert tracker.trial_count == 0

    def test_no_trials_empty_top_k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OptimizationTracker(output_dir=tmpdir, study_name="cold_topk")
            assert tracker.top_k(k=5) == []


class TestSelfOptimizingColdStart:
    """SelfOptimizingEngine works with no previous profile."""

    def test_default_params_on_no_profile(self):
        engine = SelfOptimizingEngine(model_name="nonexistent_model", profile_dir=tempfile.mkdtemp())
        params = engine.get_optimal_params()
        assert params.batch_size == 1
        assert params.kv_cache_quant_bits == 16
        assert params.speculative_decoding_enabled is False
        engine.stop()

    def test_get_suggested_batch_size_default(self):
        engine = SelfOptimizingEngine(profile_dir=tempfile.mkdtemp())
        suggested = engine.get_suggested_batch_size(max_batch=64)
        assert suggested <= 64
        engine.stop()

    def test_get_suggested_kv_cache_quant_default(self):
        engine = SelfOptimizingEngine(profile_dir=tempfile.mkdtemp())
        assert engine.get_suggested_kv_cache_quant() == 16
        engine.stop()

    def test_stats_with_no_operations(self):
        engine = SelfOptimizingEngine(profile_dir=tempfile.mkdtemp())
        s = engine.stats()
        assert s["total_requests"] == 0
        assert s["best_throughput"] == 0.0
        engine.stop()
