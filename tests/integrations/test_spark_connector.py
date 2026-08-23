"""Tests for the DistLLM Spark connector module.

Tests cover:
- Graceful degradation when PySpark is not installed
- Module-level constants and data classes
- Constructor configuration
- Batch-size candidate logic
- Retry and backoff logic
- Auto-tuning helpers
- SparkMLflowIntegration no-op behaviour
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


# ---------------------------------------------------------------------------
# Graceful degradation (no PySpark)
# ---------------------------------------------------------------------------


class TestModuleImportsWithoutPySpark:
    """The module can be imported and used without PySpark installed."""

    def test_module_imports(self) -> None:
        """Module imports without error when PySpark is absent."""
        import distllm.integrations.spark_connector as sc

        assert hasattr(sc, "DistLLMSparkTransformer")
        assert hasattr(sc, "SparkMLflowIntegration")
        assert hasattr(sc, "BatchInferenceResult")
        assert hasattr(sc, "BatchMetrics")
        assert hasattr(sc, "_SPARK_AVAILABLE")

    def test_dummy_types_available(self) -> None:
        """Dummy type stubs exist when PySpark is not installed."""
        import distllm.integrations.spark_connector as sc

        # If PySpark happens to be installed, skip these assertions.
        if not sc._SPARK_AVAILABLE:
            # All four dummy types should be referenceable
            assert sc.SparkDataFrame is not None
            assert sc.F is not None
            assert sc.StructType is not None
            assert sc.StructField is not None
            assert callable(sc.pandas_udf)


class TestGracefulDegradation:
    """Verify graceful degradation when PySpark is not available.

    These tests patch the module-level _SPARK_AVAILABLE flag to simulate
    a missing PySpark installation regardless of the test environment.
    """

    @pytest.fixture(autouse=True)
    def _patch_spark_flag(self) -> None:
        """Force _SPARK_AVAILABLE to False for the duration of each test."""
        with patch("distllm.integrations.spark_connector._SPARK_AVAILABLE", False):
            yield

    def test_transform_raises_without_pyspark(self) -> None:
        """transform() raises ImportError when PySpark is not installed."""
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        with pytest.raises(ImportError, match="PySpark"):
            transformer.transform(None, "input_col")

    def test_transform_batch_raises_without_pyspark(self) -> None:
        """transform_batch() raises ImportError when PySpark is not installed."""
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        with pytest.raises(ImportError, match="PySpark"):
            transformer.transform_batch(None, "input_col")

    def test_transform_stream_raises_without_pyspark(self) -> None:
        """transform_stream() raises ImportError when PySpark is not installed."""
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        with pytest.raises(ImportError, match="PySpark"):
            transformer.transform_stream(None, "input_col")

    def test_spark_mlflow_noop_without_pyspark(self) -> None:
        """SparkMLflowIntegration is a no-op when PySpark and MLflow are missing."""
        from distllm.integrations.spark_connector import (
            SparkMLflowIntegration,
        )

        with patch("distllm.integrations.spark_connector._MLFLOW_AVAILABLE", False):
            integration = SparkMLflowIntegration(tracking_uri="http://test:5000")
            assert integration._mlflow is None
            assert integration.log_batch_run() is None


class TestNoSDKClient:
    """DistLLMSparkTransformer raises ImportError when the SDK is unavailable."""

    def test_constructor_raises_without_sdk(self) -> None:
        """Constructor raises ImportError when distllm SDK is not available."""
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        with patch(
            "distllm.integrations.spark_connector._CLIENT_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="DistLLM SDK"):
                DistLLMSparkTransformer(api_url="http://test:8000")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants have expected values."""

    def test_default_batch_size(self) -> None:
        from distllm.integrations.spark_connector import _DEFAULT_BATCH_SIZE

        assert _DEFAULT_BATCH_SIZE == 32

    def test_min_batch_size(self) -> None:
        from distllm.integrations.spark_connector import _MIN_BATCH_SIZE

        assert _MIN_BATCH_SIZE == 1

    def test_max_batch_size(self) -> None:
        from distllm.integrations.spark_connector import _MAX_BATCH_SIZE

        assert _MAX_BATCH_SIZE == 512

    def test_max_retries(self) -> None:
        from distllm.integrations.spark_connector import _MAX_RETRIES

        assert _MAX_RETRIES == 3

    def test_tune_thresholds(self) -> None:
        from distllm.integrations.spark_connector import (
            _TUNE_LATENCY_THRESHOLD_HIGH,
            _TUNE_LATENCY_THRESHOLD_LOW,
        )

        assert _TUNE_LATENCY_THRESHOLD_HIGH == 2000.0
        assert _TUNE_LATENCY_THRESHOLD_LOW == 200.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestBatchMetrics:
    """Verify BatchMetrics behaviour."""

    def test_to_dict_returns_all_fields(self) -> None:
        from distllm.integrations.spark_connector import BatchMetrics

        metrics = BatchMetrics(
            num_rows=100,
            num_batches=10,
            total_latency_ms=5000.0,
            avg_latency_ms=50.0,
            p50_latency_ms=45.0,
            p95_latency_ms=90.0,
            p99_latency_ms=120.0,
            throughput_rows_per_sec=20.0,
            throughput_tokens_per_sec=1000.0,
            total_tokens=5000,
            avg_tokens_per_row=50.0,
            batch_size_used=32,
            num_retries=2,
            num_errors=0,
        )

        d = metrics.to_dict()
        assert d["num_rows"] == 100
        assert d["num_batches"] == 10
        assert d["batch_size_used"] == 32
        assert d["num_retries"] == 2
        assert d["num_errors"] == 0
        assert len(d) == 14  # all fields present


# ---------------------------------------------------------------------------
# Constructor and configuration
# ---------------------------------------------------------------------------


class TestConstructor:
    """Verify DistLLMSparkTransformer construction and default values."""

    def test_defaults(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        assert transformer._api_url == "http://test:8000"
        assert transformer._model == "distributed-llm"
        assert transformer._batch_size is None  # triggers auto-tune
        assert transformer._max_retries == 3
        assert transformer._temperature == 0.7
        assert transformer._max_tokens == 256
        assert transformer._actual_batch_size == 32  # _DEFAULT_BATCH_SIZE
        assert transformer._tuned is False
        assert transformer._total_retries == 0
        assert transformer._total_errors == 0
        assert transformer._latency_samples == []
        assert transformer._client is None

    def test_custom_values(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(
            api_url="http://cluster:9000",
            model="custom-model",
            batch_size=16,
            max_retries=5,
            temperature=0.3,
            max_tokens=512,
            client_kwargs={"timeout": 30},
        )
        assert transformer._api_url == "http://cluster:9000"
        assert transformer._model == "custom-model"
        assert transformer._batch_size == 16
        assert transformer._max_retries == 5
        assert transformer._temperature == 0.3
        assert transformer._max_tokens == 512
        assert transformer._client_kwargs == {"timeout": 30}
        # When batch_size is explicitly set, _tuned is True
        assert transformer._tuned is True
        assert transformer._actual_batch_size == 16

    def test_properties(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        assert transformer.actual_batch_size == 32
        assert transformer.total_retries == 0
        assert transformer.total_errors == 0

    def test_reset_stats(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        transformer._total_retries = 10
        transformer._total_errors = 3
        transformer.reset_stats()
        assert transformer._total_retries == 0
        assert transformer._total_errors == 0


# ---------------------------------------------------------------------------
# Batch-size candidate logic
# ---------------------------------------------------------------------------


class TestBatchSizeCandidates:
    """Verify the _batch_size_candidates static method."""

    def test_all_sizes_when_enough_rows(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        candidates = DistLLMSparkTransformer._batch_size_candidates(1000)
        assert candidates == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    def test_filtered_when_few_rows(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        candidates = DistLLMSparkTransformer._batch_size_candidates(10)
        assert candidates == [1, 2, 4, 8]

    def test_single_row(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        candidates = DistLLMSparkTransformer._batch_size_candidates(1)
        assert candidates == [1]

    def test_no_rows(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        candidates = DistLLMSparkTransformer._batch_size_candidates(0)
        assert candidates == []


# ---------------------------------------------------------------------------
# Auto-tuning logic
# ---------------------------------------------------------------------------


class TestAutoTune:
    """Verify the auto-tune batch-size adjustment logic."""

    def test_latency_window_returns_recent(self) -> None:
        from distllm.integrations.spark_connector import (
            _BATCH_SIZE_TUNE_SAMPLES,
            DistLLMSparkTransformer,
        )

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        # Populate with more samples than the window
        transformer._latency_samples = list(range(200))
        window = transformer._latency_window()
        assert len(window) == _BATCH_SIZE_TUNE_SAMPLES  # 50
        assert window[0] == 150  # last 50 from 200 samples

    def test_maybe_tune_does_nothing_with_few_samples(self) -> None:
        """Re-tuning is skipped when there are fewer than _TUNE_MIN_SAMPLES."""
        from distllm.integrations.spark_connector import (
            _DEFAULT_BATCH_SIZE,
            DistLLMSparkTransformer,
        )

        transformer = DistLLMSparkTransformer(api_url="http://test:8000")
        transformer._latency_samples = [100.0] * 5  # less than 10
        transformer._maybe_tune_batch_size()
        assert transformer._actual_batch_size == _DEFAULT_BATCH_SIZE

    def test_tune_reduces_batch_on_high_latency(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(
            api_url="http://test:8000",
            batch_size=64,
        )
        # Set recent latencies to be very high (> 2000ms)
        transformer._latency_samples = [3000.0] * 50
        transformer._last_tune_check = 0.0  # force re-tuning check
        transformer._maybe_tune_batch_size()
        assert transformer._actual_batch_size == 32  # 64 // 2

    def test_tune_increases_batch_on_low_latency(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(
            api_url="http://test:8000",
            batch_size=16,
        )
        # Set recent latencies to be very low (< 200ms)
        transformer._latency_samples = [50.0] * 50
        transformer._last_tune_check = 0.0  # force re-tuning check
        transformer._maybe_tune_batch_size()
        assert transformer._actual_batch_size == 32  # 16 * 2

    def test_tune_does_not_exceed_max(self) -> None:
        from distllm.integrations.spark_connector import (
            _MAX_BATCH_SIZE,
            DistLLMSparkTransformer,
        )

        transformer = DistLLMSparkTransformer(
            api_url="http://test:8000",
            batch_size=_MAX_BATCH_SIZE,
        )
        transformer._latency_samples = [50.0] * 50
        transformer._last_tune_check = 0.0
        transformer._maybe_tune_batch_size()
        assert transformer._actual_batch_size == _MAX_BATCH_SIZE  # no change

    def test_tune_does_not_go_below_min(self) -> None:
        from distllm.integrations.spark_connector import (
            _MIN_BATCH_SIZE,
            DistLLMSparkTransformer,
        )

        transformer = DistLLMSparkTransformer(
            api_url="http://test:8000",
            batch_size=_MIN_BATCH_SIZE,
        )
        transformer._latency_samples = [5000.0] * 50
        transformer._last_tune_check = 0.0
        transformer._maybe_tune_batch_size()
        assert transformer._actual_batch_size == _MIN_BATCH_SIZE  # no change

    def test_tune_skips_if_interval_not_elapsed(self) -> None:
        from distllm.integrations.spark_connector import DistLLMSparkTransformer

        transformer = DistLLMSparkTransformer(
            api_url="http://test:8000",
            batch_size=64,
        )
        transformer._latency_samples = [5000.0] * 50
        # _last_tune_check is set to current time in __init__, so the
        # interval has not elapsed yet
        transformer._maybe_tune_batch_size()
        assert transformer._actual_batch_size == 64  # unchanged


# ---------------------------------------------------------------------------
# Retry / backoff logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Verify retry delay computation."""

    def test_backoff_delay_increases(self) -> None:
        from distllm.integrations.spark_connector import _backoff_delay

        d0 = _backoff_delay(0)
        d1 = _backoff_delay(1)
        d2 = _backoff_delay(2)
        assert d1 >= d0 or abs(d1 - d0) < 0.5  # jitter can decrease slightly
        assert d2 >= d1 or abs(d2 - d1) < 0.5

    def test_backoff_delay_capped(self) -> None:
        from distllm.integrations.spark_connector import _backoff_delay

        d10 = _backoff_delay(10)
        # max_delay is 30.0, so with jitter it should be <= 30.0
        assert d10 <= 30.0

    def test_backoff_positive(self) -> None:
        from distllm.integrations.spark_connector import _backoff_delay

        for attempt in range(5):
            assert _backoff_delay(attempt) > 0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class TestPercentiles:
    """Verify the _percentiles helper."""

    def test_empty(self) -> None:
        from distllm.integrations.spark_connector import _percentiles

        assert _percentiles([]) == (0.0, 0.0, 0.0)

    def test_single_value(self) -> None:
        from distllm.integrations.spark_connector import _percentiles

        assert _percentiles([42.0]) == (42.0, 42.0, 42.0)

    def test_sorted_values(self) -> None:
        from distllm.integrations.spark_connector import _percentiles

        values = list(range(1, 101))  # 1..100
        p50, p95, p99 = _percentiles(values)
        assert p50 == 50
        assert p95 == 95
        assert p99 == 99


class TestSchemaSummary:
    """Verify the _schema_summary helper."""

    def test_none(self) -> None:
        from distllm.integrations.spark_connector import _schema_summary

        assert _schema_summary(None) == ""

    def test_with_fields(self) -> None:
        from distllm.integrations.spark_connector import _schema_summary

        # Create a mock schema with StructField-like objects
        mock_field1 = MagicMock()
        mock_field1.name = "col1"
        mock_field1.dataType = "StringType"
        mock_field2 = MagicMock()
        mock_field2.name = "col2"
        mock_field2.dataType = "IntegerType"

        mock_schema = MagicMock()
        mock_schema.fields = [mock_field1, mock_field2]

        result = _schema_summary(mock_schema)
        assert "col1:StringType" in result
        assert "col2:IntegerType" in result


# ---------------------------------------------------------------------------
# SparkMLflowIntegration
# ---------------------------------------------------------------------------


class TestSparkMLflowIntegration:
    """Verify SparkMLflowIntegration construction and no-op behaviour."""

    def test_create_without_mlflow(self) -> None:
        """Construction works when MLflow is not available (no-op)."""
        from distllm.integrations.spark_connector import (
            SparkMLflowIntegration,
        )

        with patch(
            "distllm.integrations.spark_connector._MLFLOW_AVAILABLE", False
        ):
            integration = SparkMLflowIntegration(
                tracking_uri="http://test:5000",
                experiment_name="my-experiment",
            )
            assert integration._experiment_name == "my-experiment"
            assert integration._mlflow is None

    def test_log_batch_run_noop(self) -> None:
        """log_batch_run returns None when MLflow is not available."""
        from distllm.integrations.spark_connector import (
            SparkMLflowIntegration,
        )

        with patch(
            "distllm.integrations.spark_connector._MLFLOW_AVAILABLE", False
        ):
            integration = SparkMLflowIntegration()
            assert integration.log_batch_run() is None

    def test_log_streaming_query_noop(self) -> None:
        """log_streaming_query returns None when MLflow is not available."""
        from distllm.integrations.spark_connector import (
            SparkMLflowIntegration,
        )

        with patch(
            "distllm.integrations.spark_connector._MLFLOW_AVAILABLE", False
        ):
            integration = SparkMLflowIntegration()
            assert integration.log_streaming_query(None) is None


# ---------------------------------------------------------------------------
# Integration-style: _process_one_batch (with mocked client)
# ---------------------------------------------------------------------------


class TestProcessOneBatch:
    """Verify _process_one_batch retry and fallback behaviour."""

    @pytest.fixture
    def transformer(self) -> Any:
        from distllm.integrations.spark_connector import (
            DistLLMSparkTransformer,
        )

        return DistLLMSparkTransformer(api_url="http://test:8000")

    def test_successful_batch(self, transformer: Any) -> None:
        """A successful batch returns output texts."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.outputs = ["hello", "world"]
        mock_result.total_tokens = 10
        mock_result.latencies_ms = [50.0, 75.0]

        with patch.object(transformer, "_call_batch", return_value=mock_result):
            texts, retries, tokens, latencies = transformer._process_one_batch(
                mock_client, ["hi", "there"], "model"
            )
            assert texts == ["hello", "world"]
            assert retries == 0
            assert tokens == 10
            assert latencies == [50.0, 75.0]

    def test_batch_all_retries_exhausted(self, transformer: Any) -> None:
        """When all retries are exhausted, fallback to empty strings."""
        mock_client = MagicMock()

        with patch.object(
            transformer, "_call_batch", side_effect=RuntimeError("API down")
        ):
            texts, retries, tokens, latencies = transformer._process_one_batch(
                mock_client, ["hi"], "model"
            )
            assert texts == [""]
            assert retries == 3  # _MAX_RETRIES
            assert tokens is None
            assert latencies == []

    def test_batch_retry_then_succeeds(self, transformer: Any) -> None:
        """After a transient failure, the retry succeeds."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.outputs = ["ok"]
        mock_result.total_tokens = 5
        mock_result.latencies_ms = [100.0]

        call_count = [0]

        def _side_effect(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Transient error")
            return mock_result

        with patch.object(transformer, "_call_batch", side_effect=_side_effect):
            texts, retries, tokens, latencies = transformer._process_one_batch(
                mock_client, ["hi"], "model"
            )
            assert texts == ["ok"]
            assert retries == 1  # one retry occurred
            assert tokens == 5
            assert latencies == [100.0]


# ---------------------------------------------------------------------------
# Spark-specific tests (gated)
# ---------------------------------------------------------------------------


class TestWithPySpark:
    """Tests that require actual PySpark installation."""

    def test_import_pyspark(self) -> None:
        """PySpark is available for these tests."""
        pyspark = pytest.importorskip("pyspark")
        assert pyspark is not None

    def test_spark_dataframe_type(self) -> None:
        """When PySpark IS available, SparkDataFrame is the real type."""
        pytest.importorskip("pyspark")
        from distllm.integrations.spark_connector import _SPARK_AVAILABLE

        assert _SPARK_AVAILABLE is True

    def test_spark_mlflow_log_batch_run_with_pyspark(self) -> None:
        """log_batch_run accepts a real DataFrame reference (does not call MLflow)."""
        pytest.importorskip("pyspark")
        from distllm.integrations.spark_connector import (
            SparkMLflowIntegration,
        )

        with patch(
            "distllm.integrations.spark_connector._MLFLOW_AVAILABLE", False
        ):
            integration = SparkMLflowIntegration()
            # Without MLflow, this is a no-op regardless of PySpark
            result = integration.log_batch_run(df=MagicMock())
            assert result is None
