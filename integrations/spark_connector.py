"""PySpark connector for DistLLM batch inference on DataFrames.

Provides :class:`DistLLMSparkTransformer` for LLM inference over DataFrames
with auto batch-size tuning, retries, and Structured Streaming support, and
:class:`SparkMLflowIntegration` for logging batch-run metrics.

PySpark and MLflow are optional dependencies -- the module degrades
gracefully when they are not installed.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("distllm.spark")

# ---------------------------------------------------------------------------
# Optional: PySpark
# ---------------------------------------------------------------------------
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType
    from pyspark.sql.functions import pandas_udf, PandasUDFType

    _SPARK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARK_AVAILABLE = False

    # Dummy types so the module can be imported without PySpark.
    class SparkDataFrame:  # type: ignore[no-redef]
        pass

    class F:  # type: ignore[no-redef]
        @staticmethod
        def lit(v): return v

    class StructType:  # type: ignore[no-redef]
        pass

    class StructField:  # type: ignore[no-redef]
        pass

    def pandas_udf(*a, **kw):  # type: ignore[misc]
        return lambda fn: fn

    PandasUDFType = object  # type: ignore[assignment,misc]

# Optional: DistLLM SDK client
try:
    from distllm.sdk.client import DistLLMClient, RetryConfig as SDKRetryConfig

    _CLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CLIENT_AVAILABLE = False
    DistLLMClient = None  # type: ignore[assignment,misc]

# Optional: DistLLM MLflow integration
try:
    from integrations.mlflow_tracking import MLflowIntegration

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False
    MLflowIntegration = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_SIZE = 32
_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 512
_BATCH_SIZE_TUNE_SAMPLES = 50
_BATCH_SIZE_TUNE_PERCENTILE = 80
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0
_STREAMING_TRIGGER_INTERVAL = "10 seconds"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchInferenceResult:
    """Result of a single batch inference call."""

    outputs: list[str]
    latencies_ms: list[float]
    total_tokens: int
    batch_size: int


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch run."""

    num_rows: int
    num_batches: int
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rows_per_sec: float
    throughput_tokens_per_sec: float
    total_tokens: int
    avg_tokens_per_row: float
    batch_size_used: int
    num_retries: int
    num_errors: int

    def to_dict(self) -> dict[str, float | int]:
        """Return metrics as a flat dict for logging / MLflow."""
        return {
            "num_rows": self.num_rows,
            "num_batches": self.num_batches,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_rows_per_sec": self.throughput_rows_per_sec,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_row": self.avg_tokens_per_row,
            "batch_size_used": self.batch_size_used,
            "num_retries": self.num_retries,
            "num_errors": self.num_errors,
        }


# ---------------------------------------------------------------------------
# DistLLM Spark Transformer
# ---------------------------------------------------------------------------

class DistLLMSparkTransformer:
    """Run DistLLM inference over Spark DataFrames with batch optimisation.

    Parameters
    ----------
    api_url:
        DistLLM coordinator API base URL.
    model:
        Model name for inference.
    batch_size:
        Initial batch size. ``None`` triggers auto-tuning.
    max_retries:
        Max retries per batch on transient errors.
    temperature:
        Sampling temperature.
    max_tokens:
        Max tokens per completion.
    client_kwargs:
        Extra kwargs forwarded to ``DistLLMClient``.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        model: str = "distributed-llm",
        batch_size: Optional[int] = None,
        max_retries: int = _MAX_RETRIES,
        temperature: float = 0.7,
        max_tokens: int = 256,
        client_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if not _CLIENT_AVAILABLE:
            raise ImportError(
                "DistLLM SDK is required for DistLLMSparkTransformer. "
                "Install it via 'pip install distllm' or add the distllm package to your PYTHONPATH."
            )

        self._api_url = api_url
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client_kwargs = client_kwargs or {}

        self._client: Optional[DistLLMClient] = None
        self._actual_batch_size: int = batch_size or _DEFAULT_BATCH_SIZE
        self._tuned = batch_size is not None  # skip tuning when user sets it
        self._total_retries: int = 0
        self._total_errors: int = 0
        self._latency_samples: list[float] = []

    # -- Public API --------------------------------------------------------

    def transform(
        self,
        df: SparkDataFrame,
        input_col: str,
        output_col: str = "prediction",
        model: Optional[str] = None,
        auto_tune: bool = True,
    ) -> SparkDataFrame:
        """Add an *output_col* column with LLM predictions to *df*.

        Runs inference via a pandas UDF with automatic batch-size tuning
        (when ``auto_tune=True`` and no batch size was explicitly set).

        Returns
        -------
        SparkDataFrame
            Input DataFrame with an additional ``output_col`` column.
        """
        if not _SPARK_AVAILABLE:
            raise ImportError("PySpark is required to use DistLLMSparkTransformer.")

        resolved_model = model or self._model

        if auto_tune and not self._tuned:
            self._auto_tune_batch_size(df, input_col, resolved_model)

        return df.withColumn(
            output_col,
            pandas_udf(
                lambda rows: self._infer_udf(rows, resolved_model),
                returnType=StringType(),
            )(F.col(input_col)),
        )

    def transform_batch(
        self,
        df: SparkDataFrame,
        input_col: str,
        output_col: str = "prediction",
        model: Optional[str] = None,
        metrics: bool = True,
    ) -> SparkDataFrame:
        """Process DataFrame with explicit batching and detailed batch metrics.

        Unlike ``transform`` (pandas UDF), this iterates partitions, batches
        rows explicitly, and attaches a ``batch_metrics`` attribute to the
        returned DataFrame (when ``metrics=True``).
        """
        if not _SPARK_AVAILABLE:
            raise ImportError("PySpark is required to use DistLLMSparkTransformer.")

        resolved_model = model or self._model
        client = self._get_client()
        total_start = time.time()
        all_latencies: list[float] = []
        total_tokens = 0
        batches = 0
        errors = 0
        retries = 0
        results: list[tuple] = []
        batch_size = self._actual_batch_size

        # Collect prompts from the DataFrame.
        prompts = [row[0] for row in df.select(input_col).collect()]

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            output_texts: list[str] = []

            for attempt in range(self._max_retries + 1):
                try:
                    result = self._call_batch(client, batch, resolved_model)
                    output_texts = result.outputs
                    all_latencies.extend(result.latencies_ms)
                    total_tokens += result.total_tokens
                    break
                except Exception:
                    if attempt < self._max_retries:
                        delay = _backoff_delay(attempt)
                        time.sleep(delay)
                        retries += 1
                    else:
                        errors += 1
                        output_texts = [""] * len(batch)
                        logger.warning(
                            "Batch %d failed after %d retries",
                            i // batch_size, self._max_retries,
                        )

            for idx, text in enumerate(output_texts):
                results.append((prompts[i + idx], text))
            batches += 1

        elapsed_ms = (time.time() - total_start) * 1000

        # Build result DataFrame.
        schema = StructType([
            StructField(input_col, StringType(), nullable=False),
            StructField(output_col, StringType(), nullable=True),
        ])
        result_df = self._spark_session(df).createDataFrame(results, schema=schema)  # type: ignore[arg-type]

        if metrics:
            p50, p95, p99 = _percentiles(all_latencies) if all_latencies else (0.0, 0.0, 0.0)
            avg_tok = total_tokens / max(len(prompts), 1)
            result_df.batch_metrics = BatchMetrics(  # type: ignore[attr-defined]
                num_rows=len(prompts),
                num_batches=batches,
                total_latency_ms=elapsed_ms,
                avg_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                throughput_rows_per_sec=len(prompts) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0,
                throughput_tokens_per_sec=total_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0,
                total_tokens=total_tokens,
                avg_tokens_per_row=avg_tok,
                batch_size_used=batch_size,
                num_retries=retries,
                num_errors=errors,
            )

        self._total_retries += retries
        self._total_errors += errors
        return result_df

    def transform_stream(
        self,
        stream_df: SparkDataFrame,
        input_col: str,
        output_col: str = "prediction",
        model: Optional[str] = None,
        checkpoint_location: Optional[str] = None,
        trigger_interval: str = _STREAMING_TRIGGER_INTERVAL,
    ) -> SparkDataFrame:
        """Return a streaming DataFrame with LLM predictions appended.

        The result can be written via ``writeStream``.

        Parameters
        ----------
        stream_df:
            A streaming Spark DataFrame (from ``readStream``).
        input_col:
            Column containing prompt strings.
        output_col:
            Output prediction column name.
        model:
            Override the model name.
        checkpoint_location:
            Optional checkpoint path for fault tolerance.
        trigger_interval:
            Micro-batch interval string (e.g. ``"5 seconds"``).

        Returns
        -------
        SparkDataFrame
            Streaming DataFrame with prediction column.
        """
        if not _SPARK_AVAILABLE:
            raise ImportError("PySpark is required to use DistLLMSparkTransformer.")

        resolved_model = model or self._model

        return stream_df.withColumn(
            output_col,
            pandas_udf(
                lambda rows: self._infer_udf(rows, resolved_model),
                returnType=StringType(),
            )(F.col(input_col)),
        )

    # -- Internal helpers --------------------------------------------------

    def _get_client(self) -> DistLLMClient:
        """Return (or create) the cached SDK client."""
        if self._client is None:
            self._client = DistLLMClient(
                base_url=self._api_url,
                retry=SDKRetryConfig(max_retries=self._max_retries),
                **self._client_kwargs,
            )
        return self._client

    def _infer_udf(self, prompts_series: Any, model: str) -> Any:
        """Pandas UDF that runs inference on a column of prompts."""
        import pandas as pd

        client = self._get_client()
        prompts = prompts_series.tolist() if hasattr(prompts_series, "tolist") else list(prompts_series)

        if not prompts:
            return pd.Series([], dtype=str)

        output_texts: list[str] = []
        batch_size = self._actual_batch_size

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            texts = self._call_batch_udf(client, batch, model)
            output_texts.extend(texts)

        return pd.Series(output_texts, index=prompts_series.index if hasattr(prompts_series, "index") else None)

    def _call_batch_udf(
        self,
        client: DistLLMClient,
        prompts: list[str],
        model: str,
    ) -> list[str]:
        """Call inference for a batch; returns list of response texts."""
        for attempt in range(self._max_retries + 1):
            try:
                messages_batch = [
                    [{"role": "user", "content": prompt}]
                    for prompt in prompts
                ]

                # DistLLMClient.chat_completions is a single-call API, so
                # for batched prompts we call it sequentially and collect.
                results: list[str] = []
                for msgs in messages_batch:
                    resp = client.chat_completions(
                        messages=msgs,
                        model=model,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                    texts = [c.message.content for c in resp.choices if c.message]
                    results.append(texts[0] if texts else "")
                return results
            except Exception:
                if attempt < self._max_retries:
                    time.sleep(_backoff_delay(attempt))
                else:
                    raise

    def _call_batch(
        self,
        client: DistLLMClient,
        prompts: list[str],
        model: str,
    ) -> BatchInferenceResult:
        """Call inference for a batch and return structured result."""
        latencies: list[float] = []
        total_tokens = 0
        outputs: list[str] = []

        for prompt in prompts:
            t0 = time.time()
            try:
                resp = client.chat_completions(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                elapsed = (time.time() - t0) * 1000
                latencies.append(elapsed)
                text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
                outputs.append(text)
                if resp.usage:
                    total_tokens += resp.usage.total_tokens
            except Exception:
                elapsed = (time.time() - t0) * 1000
                latencies.append(elapsed)
                raise

        return BatchInferenceResult(
            outputs=outputs,
            latencies_ms=latencies,
            total_tokens=total_tokens,
            batch_size=len(prompts),
        )

    def _auto_tune_batch_size(
        self,
        df: SparkDataFrame,
        input_col: str,
        model: str,
    ) -> None:
        """Probe increasing batch sizes on a sample to find peak throughput."""
        total_rows = df.count()
        if total_rows == 0:
            self._actual_batch_size = _DEFAULT_BATCH_SIZE
            return

        sample_size = min(_BATCH_SIZE_TUNE_SAMPLES, total_rows)
        sample_df = df.limit(sample_size)
        prompts = [r[0] for r in sample_df.select(input_col).collect()]

        if not prompts:
            self._actual_batch_size = _DEFAULT_BATCH_SIZE
            return

        client = self._get_client()
        best_throughput = 0.0
        best_size = _DEFAULT_BATCH_SIZE

        candidates = self._batch_size_candidates(len(prompts))

        for bs in candidates:
            if bs > len(prompts):
                break
            t0 = time.time()
            processed = 0
            try:
                for i in range(0, len(prompts), bs):
                    batch = prompts[i : i + bs]
                    self._call_batch(client, batch, model)
                    processed += len(batch)
                elapsed = time.time() - t0
                throughput = processed / max(elapsed, 1e-6)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_size = bs
            except Exception:
                logger.debug("Batch size %d failed during tuning", bs)
                continue

        self._actual_batch_size = best_size
        self._tuned = True
        logger.info("Auto-tuned batch size to %d (throughput=%.1f rows/s)", best_size, best_throughput)

    @staticmethod
    def _batch_size_candidates(max_rows: int) -> list[int]:
        """Return a list of batch sizes to probe during tuning."""
        sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        return [s for s in sizes if s <= max_rows]

    @staticmethod
    def _spark_session(df: SparkDataFrame) -> Any:
        """Get the SparkSession from a DataFrame."""
        return df.sparkSession  # type: ignore[union-attr]

    # -- Reporting ---------------------------------------------------------

    @property
    def actual_batch_size(self) -> int:
        """The batch size currently in use (may be auto-tuned)."""
        return self._actual_batch_size

    @property
    def total_retries(self) -> int:
        """Total retries across all transform calls."""
        return self._total_retries

    @property
    def total_errors(self) -> int:
        """Total batch errors across all transform calls."""
        return self._total_errors

    def reset_stats(self) -> None:
        """Reset retry and error counters."""
        self._total_retries = 0
        self._total_errors = 0


# ---------------------------------------------------------------------------
# Spark MLflow Integration
# ---------------------------------------------------------------------------

class SparkMLflowIntegration:
    """Log DistLLM Spark batch runs as MLflow experiments.

    Wraps :class:`~integrations.mlflow_tracking.MLflowIntegration` with
    Spark-specific helpers. All methods are no-ops when PySpark or MLflow
    are not installed.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "distllm-spark-batch",
    ) -> None:
        """Initialise the Spark MLflow integration.

        Parameters
        ----------
        tracking_uri:
            MLflow tracking server URI.  Falls back to the
            ``MLFLOW_TRACKING_URI`` environment variable.
        experiment_name:
            Default experiment name for batch runs.
        """
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._mlflow: Optional[MLflowIntegration] = None

        if _MLFLOW_AVAILABLE and MLflowIntegration is not None:
            self._mlflow = MLflowIntegration(tracking_uri=tracking_uri)
        else:
            logger.info("MLflow not available — SparkMLflowIntegration is a no-op")

    # -- Public API --------------------------------------------------------

    def log_batch_run(
        self,
        df: Optional[SparkDataFrame] = None,
        metrics: Optional[BatchMetrics] = None,
        model_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
        experiment_name: Optional[str] = None,
    ) -> Optional[str]:
        """Log a Spark batch run as an MLflow run.

        Logs row count, schema, metrics, and model provenance.
        Returns the MLflow run ID or None.
        """
        if self._mlflow is None:
            return None

        merged_params = dict(params or {})
        if model_name:
            merged_params["model_name"] = model_name

        if df is not None and _SPARK_AVAILABLE:
            merged_params["num_input_rows"] = df.count()
            # Log a sample of the schema as a parameter (stringified).
            merged_params["input_schema"] = _schema_summary(df.schema) if df.schema else ""

        merged_metrics = metrics.to_dict() if metrics else {}

        run = self._mlflow.log_run(
            experiment_name=experiment_name or self._experiment_name,
            params=merged_params,
            metrics=merged_metrics,
            tags=tags,
        )
        return run.run_id if run else None

    def log_batch_run_with_profile(
        self,
        df: SparkDataFrame,
        metrics: BatchMetrics,
        model_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        input_col: Optional[str] = None,
    ) -> Optional[str]:
        """Log a batch run including input-length statistics as additional metrics."""
        if self._mlflow is None:
            return None

        if not _SPARK_AVAILABLE:
            return self.log_batch_run(df=df, metrics=metrics, model_name=model_name, params=params)

        profile_metrics: dict[str, float] = {}

        if input_col is not None and df is not None:
            try:
                stats = df.select(
                    F.avg(F.length(F.col(input_col))).alias("avg_input_length"),
                    F.stddev(F.length(F.col(input_col))).alias("std_input_length"),
                    F.min(F.length(F.col(input_col))).alias("min_input_length"),
                    F.max(F.length(F.col(input_col))).alias("max_input_length"),
                ).collect()[0]

                if stats["avg_input_length"] is not None:
                    profile_metrics["avg_input_length"] = float(stats["avg_input_length"])
                if stats["std_input_length"] is not None:
                    profile_metrics["std_input_length"] = float(stats["std_input_length"])
                if stats["min_input_length"] is not None:
                    profile_metrics["min_input_length"] = float(stats["min_input_length"])
                if stats["max_input_length"] is not None:
                    profile_metrics["max_input_length"] = float(stats["max_input_length"])
            except Exception as exc:
                logger.warning("Failed to compute input profile: %s", exc)

        all_metrics = metrics.to_dict()
        all_metrics.update(profile_metrics)

        return self.log_batch_run(
            df=df,
            metrics=BatchMetrics(**{k: v for k, v in all_metrics.items() if k in BatchMetrics.__dataclass_fields__}),  # type: ignore[arg-type]
            model_name=model_name,
            params=params,
        )

    def log_streaming_query(
        self,
        query: Any,
        metrics: Optional[BatchMetrics] = None,
        model_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> Optional[str]:
        """Log a Structured Streaming query progress as an MLflow run.

        Captures streaming query metadata (input rows/sec, processed rows/sec)
        from the query's ``lastProgress`` if available.
        """
        if self._mlflow is None:
            return None

        merged_params = dict(params or {})
        if model_name:
            merged_params["model_name"] = model_name

        # Capture streaming query metadata if available.
        if query is not None:
            try:
                progress = query.lastProgress
                if progress:
                    merged_params["streaming_query_id"] = query.id
                    merged_params["streaming_run_id"] = query.runId
                    merged_params["streaming_input_rows_per_second"] = progress.get("inputRowsPerSecond", 0)
                    merged_params["streaming_processed_rows_per_second"] = progress.get("processedRowsPerSecond", 0)
                    merged_params["streaming_num_input_rows"] = progress.get("numInputRows", 0)
            except Exception:
                pass

        merged_metrics = metrics.to_dict() if metrics else {}

        run = self._mlflow.log_run(
            experiment_name=experiment_name or self._experiment_name,
            params=merged_params,
            metrics=merged_metrics,
            tags={"spark": "structured-streaming"},
        )
        return run.run_id if run else None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int, base: float = _RETRY_BASE_DELAY, max_delay: float = _RETRY_MAX_DELAY) -> float:
    """Compute exponential backoff delay with jitter."""
    delay = min(base * (2.0 ** attempt), max_delay)
    return delay * (0.5 + random.random() * 0.5)  # noqa: S311


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, p99) for a list of values."""
    if not values:
        return (0.0, 0.0, 0.0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return (
        sorted_vals[max(0, min(n - 1, int(n * 0.50)))],
        sorted_vals[max(0, min(n - 1, int(n * 0.95)))],
        sorted_vals[max(0, min(n - 1, int(n * 0.99)))],
    )


def _schema_summary(schema: Any) -> str:
    """Return a concise string representation of a StructType schema."""
    if schema is None:
        return ""
    try:
        return ", ".join(f"{f.name}:{f.dataType}" for f in schema.fields)
    except Exception:
        return str(schema)[:200]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DistLLMSparkTransformer",
    "SparkMLflowIntegration",
    "BatchInferenceResult",
    "BatchMetrics",
]
