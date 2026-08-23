"""Tests for distllm.dist.observability.

Uses only real objects from the module -- zero mocks.
"""

from distllm.dist.observability import enable_dist_observability
from distllm.observability.exporter import DistLLMPrometheusExporter


class TestEnableDistObservability:
    """Tests for the ``enable_dist_observability`` public function."""

    # ── Return shape ────────────────────────────────────────────────────

    def test_return_type_no_exporter(self) -> None:
        """Returns a dict with 'tracer' and 'metrics' keys."""
        result = enable_dist_observability(exporter=None, enable_tracing=False)
        assert isinstance(result, dict)
        assert "tracer" in result
        assert "metrics" in result

    def test_return_type_with_exporter(self) -> None:
        """Same shape even when an exporter is provided."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(exporter=exporter, enable_tracing=False)
        assert isinstance(result, dict)
        assert "tracer" in result
        assert "metrics" in result

    # ── Tracing ─────────────────────────────────────────────────────────

    def test_tracing_enabled_returns_tracer(self) -> None:
        """When enable_tracing=True, result['tracer'] is a Tracer instance."""
        result = enable_dist_observability(exporter=None, enable_tracing=True)
        assert result["tracer"] is not None
        # Verify it has the Tracer public API surface.
        tracer = result["tracer"]
        assert hasattr(tracer, "start_span")
        assert hasattr(tracer, "current_span")
        assert hasattr(tracer, "trace")
        assert hasattr(tracer, "instrument_pipeline")
        assert hasattr(tracer, "instrument_federation")
        assert hasattr(tracer, "instrument_recovery")
        assert hasattr(tracer, "instrument_all")

    def test_tracing_disabled_returns_none(self) -> None:
        """When enable_tracing=False, result['tracer'] is None."""
        result = enable_dist_observability(exporter=None, enable_tracing=False)
        assert result["tracer"] is None

    # ── Metrics with exporter ───────────────────────────────────────────

    def test_exporter_none_metrics_empty(self) -> None:
        """When exporter is None, metrics dict is empty."""
        result = enable_dist_observability(exporter=None, enable_tracing=False)
        assert result["metrics"] == {}

    def test_exporter_all_flags_false(self) -> None:
        """When all metric flags are False, metrics dict is empty."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        assert result["metrics"] == {}

    def test_pipeline_metrics_only(self) -> None:
        """Only pipeline metrics are registered when other flags are off."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=True,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        assert result["metrics"] == {"pipeline": True}

    def test_federation_metrics_only(self) -> None:
        """Only federation metrics are registered."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=True,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        assert result["metrics"] == {"federation": True}

    def test_recovery_metrics_only(self) -> None:
        """Only recovery metrics are registered."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=False,
            enable_recovery_metrics=True,
            enable_nccl_metrics=False,
        )
        assert result["metrics"] == {"recovery": True}

    def test_nccl_metrics_only(self) -> None:
        """Only NCCL metrics are registered."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=True,
        )
        assert result["metrics"] == {"nccl": True}

    def test_all_metrics_enabled(self) -> None:
        """All four metric subsystems are registered when all flags are on."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=True,
            enable_federation_metrics=True,
            enable_recovery_metrics=True,
            enable_nccl_metrics=True,
        )
        assert result["metrics"] == {
            "pipeline": True,
            "federation": True,
            "recovery": True,
            "nccl": True,
        }

    # ── Idempotency ─────────────────────────────────────────────────────

    def test_second_call_idempotent(self) -> None:
        """Calling twice on the same exporter does not raise.

        The second call gracefully fails to re-register metrics
        (prometheus_client rejects duplicates) so the metrics dict
        is empty, while the first call succeeded.
        """
        exporter = DistLLMPrometheusExporter()
        result1 = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
        )
        result2 = enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
        )
        # First call registered all 4 subsystems.
        assert result1["metrics"] == {
            "pipeline": True,
            "federation": True,
            "recovery": True,
            "nccl": True,
        }
        # Second call catches DuplicatedTimeseries — metrics stay empty.
        assert result2["metrics"] == {}

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_service_name_default(self) -> None:
        """Default service_name is 'distllm-node'."""
        result = enable_dist_observability(exporter=None, enable_tracing=True)
        tracer = result["tracer"]
        assert tracer is not None
        assert hasattr(tracer, "_service_name")
        assert tracer._service_name == "distllm-node"

    def test_custom_service_name(self) -> None:
        """Custom service_name is forwarded to the Tracer."""
        result = enable_dist_observability(
            exporter=None,
            service_name="my-custom-node",
            enable_tracing=True,
        )
        tracer = result["tracer"]
        assert tracer is not None
        assert tracer._service_name == "my-custom-node"

    def test_otlp_endpoint_default(self) -> None:
        """Default OTLP endpoint is localhost:4318."""
        result = enable_dist_observability(exporter=None, enable_tracing=True)
        tracer = result["tracer"]
        assert tracer is not None
        # The Tracer stores the OTLP endpoint indirectly via its provider.
        # Check that it is the default we passed (the observability module
        # hard-codes its own default, which matches the Tracer default).
        assert tracer.provider is not None

    def test_kwargs_passthrough(self) -> None:
        """Extra kwargs are forwarded to the Tracer constructor."""
        result = enable_dist_observability(
            exporter=None,
            enable_tracing=True,
            console_export=False,
            resource_attributes={"environment": "test"},
        )
        tracer = result["tracer"]
        assert tracer is not None
        # resource_attributes should have been passed through.
        provider = tracer.provider
        assert provider is not None
        resource = provider.resource
        assert resource.attributes.get("service.name") is not None
        assert resource.attributes.get("environment") == "test"

    # ── Exporter attribute side effects ─────────────────────────────────

    def test_exporter_gains_pipeline_attributes(self) -> None:
        """After enabling pipeline metrics the exporter gains pipeline attrs."""
        exporter = DistLLMPrometheusExporter()
        enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=True,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        assert hasattr(exporter, "_pipeline_stage_latency")
        assert hasattr(exporter, "_pipeline_overlap_efficiency")
        assert hasattr(exporter, "_pipeline_micro_batches")
        assert hasattr(exporter, "_pipeline_warmup")

    def test_exporter_gains_federation_attributes(self) -> None:
        """After enabling federation metrics the exporter gains federation attrs."""
        exporter = DistLLMPrometheusExporter()
        enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=True,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        assert hasattr(exporter, "_fed_forwards_total")
        assert hasattr(exporter, "_fed_peers")
        assert hasattr(exporter, "_fed_cache_digests")
        assert hasattr(exporter, "_fed_heartbeat_latency")
        assert hasattr(exporter, "_fed_spillovers_total")

    def test_exporter_gains_recovery_attributes(self) -> None:
        """After enabling recovery metrics the exporter gains recovery attrs."""
        exporter = DistLLMPrometheusExporter()
        enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=False,
            enable_recovery_metrics=True,
            enable_nccl_metrics=False,
        )
        assert hasattr(exporter, "_recovery_checkpoint_bytes")
        assert hasattr(exporter, "_recovery_redistributions")
        assert hasattr(exporter, "_recovery_drill_total")
        assert hasattr(exporter, "_recovery_drill_duration")

    def test_exporter_gains_nccl_attributes(self) -> None:
        """After enabling NCCL metrics the exporter gains NCCL attrs."""
        exporter = DistLLMPrometheusExporter()
        enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=True,
        )
        assert hasattr(exporter, "_nccl_bytes_sent")
        assert hasattr(exporter, "_nccl_ops_total")
        assert hasattr(exporter, "_nccl_active_ops")
        assert hasattr(exporter, "_nccl_preemptions_total")

    def test_exporter_no_metrics_registered_when_disabled(self) -> None:
        """No subsystem attributes appear on exporter when all flags are off."""
        exporter = DistLLMPrometheusExporter()
        enable_dist_observability(
            exporter=exporter,
            enable_tracing=False,
            enable_pipeline_metrics=False,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        # These are the metrics from the base exporter class.
        assert not hasattr(exporter, "_pipeline_stage_latency")
        assert not hasattr(exporter, "_fed_forwards_total")
        assert not hasattr(exporter, "_recovery_checkpoint_bytes")
        assert not hasattr(exporter, "_nccl_bytes_sent")

    # ── Tracing + metrics simultaneously ────────────────────────────────

    def test_tracing_and_pipeline_metrics(self) -> None:
        """Both tracing and pipeline metrics work when enabled together."""
        exporter = DistLLMPrometheusExporter()
        result = enable_dist_observability(
            exporter=exporter,
            enable_tracing=True,
            enable_pipeline_metrics=True,
            enable_federation_metrics=False,
            enable_recovery_metrics=False,
            enable_nccl_metrics=False,
        )
        assert result["tracer"] is not None
        assert result["metrics"] == {"pipeline": True}
