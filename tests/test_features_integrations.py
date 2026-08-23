"""Regression tests -- production library integrations (roadmap table).

Verifies the wiring + fail-closed behaviour for each integration. Libraries
that are not installed here (opentelemetry, sigstore, opacus, redis,
prometheus_client) are exercised through their fallback/no-op paths, which is
the honest, verifiable contract (we never fabricate a live connection).

1. OpenTelemetry  -> telemetry exporter wiring (events actually transmit)
2. sigstore/cosign -> artifact signature verification (fail-closed)
3. opacus/Google DP -> pluggable accounting backend (auto -> inhouse fallback)
4. Redis           -> prompt-cache tier is a real adapter, safe no-op w/o dep
5. Prometheus      -> real histogram percentiles + prometheus export shape
6. pydantic config -> CoordinatorConfig extra='forbid' catches silent drops
"""

from distllm.core.coordinator_metrics import Histogram, MetricsManager


# ── 5. Prometheus-style histogram (real percentiles) ──

def test_histogram_percentiles():
    h = Histogram("latency_ms", max_samples=1000)
    for v in range(1, 101):  # 1..100
        h.record(float(v))
    assert h.count == 100
    assert abs(h.p50 - 50) <= 2
    assert abs(h.p95 - 95) <= 2
    assert abs(h.p99 - 99) <= 2
    assert h.mean == 50.5


def test_histogram_prometheus_shape():
    h = Histogram("latency_ms")
    for v in (1.0, 5.0, 50.0, 1000.0):
        h.record(v)
    prom = h.to_prometheus()
    assert prom["type"] == "histogram"
    assert prom["sample_count"] == 4
    assert "le_10.0" in prom["buckets"]
    assert prom["buckets"]["le_+Inf"] == 4


def test_metrics_manager_observe_and_export():
    mm = MetricsManager()
    for v in range(1, 21):
        mm.observe("request_latency_ms", float(v))
    flat = mm.get()
    assert flat["request_latency_ms_p95"] >= 19.0
    prom = mm.get_prometheus()["request_latency_ms"]
    assert prom["type"] == "histogram"
    assert prom["p99"] >= 20.0


# ── 6. pydantic config strictness ──

def _pydantic_or_skip():
    try:
        import pydantic  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("pydantic not importable in this interpreter (broken venv ext)")


def test_coordinator_config_forbids_unknown_field():
    _pydantic_or_skip()
    from pydantic import ValidationError
    from distllm.core.coordinator_config import CoordinatorConfig

    try:
        CoordinatorConfig(model_name="x", not_a_real_field=1)
        raise AssertionError("expected ValidationError (extra='forbid')")
    except ValidationError:
        pass


def test_coordinator_config_from_settings_importable():
    _pydantic_or_skip()
    from distllm.core.coordinator_config import CoordinatorConfig

    assert callable(CoordinatorConfig.from_settings)


# ── 1. OpenTelemetry / exporter wiring ──

def test_telemetry_transmits_via_exporter():
    from distllm.core.telemetry import TelemetryCollector

    sent = []
    collector = TelemetryCollector(enabled=True, exporter=lambda events: sent.extend(events))
    collector.record_request(tokens=10, latency_ms=5.0)
    collector.flush()
    assert len(sent) == 1
    assert sent[0].event_type == "request"


def test_telemetry_no_exporter_still_flushes_to_file(tmp_path):
    from distllm.core.telemetry import TelemetryCollector

    collector = TelemetryCollector(enabled=True, data_dir=str(tmp_path))
    collector.record_request(tokens=1, latency_ms=1.0)
    # No exporter -> must not raise; persists locally (best-effort).
    collector.flush()
    assert any(tmp_path.glob("events_*.jsonl"))


# ── 2. sigstore / cosign (fail-closed) ──

def test_verify_artifact_signature_fail_closed_without_verifier(tmp_path):
    from distllm.core.plugin_sandbox import verify_artifact_signature

    artifact = tmp_path / "model.safetensors"
    artifact.write_text("weights")
    # No sigstore SDK, no cosign CLI -> must report unverified (never "safe").
    assert verify_artifact_signature(str(artifact)) is False


# ── 3. opacus / Google DP backend selection ──

def test_dp_backend_auto_falls_back_to_inhouse():
    from distllm.core.differential_privacy import DifferentialPrivacy

    dp = DifferentialPrivacy(accounting_backend="auto")
    # opacus/dp_accounting not installed here -> inhouse.
    assert dp.accounting_backend == "inhouse"


def test_dp_backend_explicit_missing_raises():
    from distllm.core.differential_privacy import DifferentialPrivacy

    try:
        DifferentialPrivacy(accounting_backend="opacus")
        raise AssertionError("expected RuntimeError (opacus not installed)")
    except RuntimeError:
        pass


def test_dp_inhouse_noise_still_works():
    import torch
    from distllm.core.differential_privacy import DifferentialPrivacy, DifferentialPrivacyConfig

    dp = DifferentialPrivacy(DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5))
    t = torch.zeros(4, 4)
    noisy = dp.add_noise_to_tensor(t)
    assert noisy.shape == t.shape
    # Noise is non-zero (privacy actually applied).
    assert noisy.abs().sum() > 0


# ── 4. Redis prompt cache (real adapter, safe no-op w/o dep) ──

def test_redis_prompt_cache_imports_and_is_safe_noop():
    from distllm.core.redis_prompt_cache import RedisPromptCache

    cache = RedisPromptCache(url="redis://localhost:6379/0")
    # Without the redis driver installed, connect() fails closed and store()
    # is a safe no-op (the "dead tier" is the missing dependency, not missing code).
    assert cache.connect() is False
    # store must not raise even though we're not connected.
    key = cache.store([1, 2, 3], "kv-ref")
    assert key == ""  # no-op when unconnected
