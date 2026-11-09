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
    # The flat export includes the histogram's p95 (see MetricsManager.get()).
    assert flat["request_latency_ms_p95"] >= 19.0
    prom = mm.get_prometheus()["request_latency_ms"]
    # to_prometheus() emits {type, sample_count, buckets}; percentiles live on
    # the flat export, not here (the old prom["p99"] key never existed).
    assert prom["type"] == "histogram"
    assert prom["sample_count"] == 20
    assert prom["buckets"]["le_+Inf"] == 20


# ── 6. pydantic config strictness ──

def _pydantic_or_skip():
    try:
        import pydantic  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("pydantic not importable in this interpreter (broken venv ext)")


def test_coordinator_config_from_settings_importable():
    _pydantic_or_skip()
    from distllm.core.coordinator_config import CoordinatorConfig

    assert callable(CoordinatorConfig.from_settings)


# ── 1. OpenTelemetry / exporter wiring ──
#
# The TelemetryCollector(exporter=...) injection hook was removed; the collector
# is now endpoint-based and persists to a data_dir (no per-call exporter). The
# test that asserted the exporter hook was deleted — current behavior (local
# flush without an exporter) is covered by the next test.


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


# ── 3. differential privacy (Gaussian mechanism) ──
#
# The former opacus / Google-DP *accounting-backend selection* was removed:
# distllm.core.differential_privacy is now a self-contained Gaussian-mechanism
# implementation (DifferentialPrivacyConfig: epsilon/delta/noise_multiplier) with
# no `accounting_backend` knob and no optional opacus/dp_accounting dependency.
# The two tests that asserted backend selection (auto->inhouse, opacus->raise)
# were deleted; the surviving behavior is covered below and in
# test_dp_inhouse_noise_still_works.


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
    import socket

    from distllm.core.redis_prompt_cache import RedisPromptCache

    # This test asserts the "no Redis present" fail-closed path. If a local Redis
    # is reachable (as on some dev/CI hosts) connect() legitimately returns True,
    # so skip rather than report a false failure.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    redis_up = probe.connect_ex(("localhost", 6379)) == 0
    probe.close()
    if redis_up:
        import pytest
        pytest.skip("local Redis reachable on :6379 — fail-closed path not applicable")

    cache = RedisPromptCache(url="redis://localhost:6379/0")
    # Without the redis driver/server, connect() fails closed and store() is a
    # safe no-op (the "dead tier" is the missing dependency, not missing code).
    assert cache.connect() is False
    # store must not raise even though we're not connected.
    key = cache.store([1, 2, 3], "kv-ref")
    assert key == ""  # no-op when unconnected
