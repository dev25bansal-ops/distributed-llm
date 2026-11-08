"""`distllm-api` must start without the optional observability extra.

`pip install "distributed-llm[cli,self-hosted]"` installed the `distllm-api`
console script but running it died at import:

    File ".../distllm/api/server.py", line 117, in <module>
    File ".../distllm/observability/__init__.py", line 40
    File ".../distllm/observability/metrics.py", line 4
    ModuleNotFoundError: No module named 'opentelemetry.sdk'

The OTel SDK lives in the [observability] extra, and both metrics.py and
tracing.py imported it at module scope. Anything that imported the server --
including the API entry point, the coordinator and every test client -- failed
for users who did not know to install an extra they never asked for.

Found by `distllm selftest` on a clean install of the published wheel.
"""

from __future__ import annotations

import builtins

import pytest


class TestOptionalObservabilitySDK:
    def test_metrics_imports_without_the_sdk(self, monkeypatch):
        """The module must import; only calling into it requires the SDK."""
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name.startswith("opentelemetry.sdk"):
                raise ImportError("simulated: no opentelemetry.sdk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)

        import importlib

        import distllm.observability.metrics as metrics

        reloaded = importlib.reload(metrics)
        try:
            assert reloaded._HAS_SDK is False
            with pytest.raises(RuntimeError) as info:
                reloaded.setup_metrics()
            assert "observability" in str(info.value)
        finally:
            monkeypatch.undo()
            importlib.reload(metrics)

    def test_tracing_imports_without_the_sdk(self, monkeypatch):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name.startswith("opentelemetry.sdk"):
                raise ImportError("simulated: no opentelemetry.sdk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)

        import importlib

        import distllm.observability.tracing as tracing

        reloaded = importlib.reload(tracing)
        try:
            assert reloaded._HAS_TRACING_SDK is False
            with pytest.raises(RuntimeError) as info:
                reloaded.setup_tracing()
            assert "observability" in str(info.value)
        finally:
            monkeypatch.undo()
            importlib.reload(tracing)

    def test_error_message_names_the_extra(self):
        from distllm.observability.metrics import _MISSING_SDK_MESSAGE

        assert "observability" in _MISSING_SDK_MESSAGE
        assert "pip install" in _MISSING_SDK_MESSAGE


class TestServerStartsWithNoSDK:
    """The API lifespan called setup_tracing() unguarded.

    A deployment without the [observability] extra therefore failed to start at
    all -- the server bound no port instead of serving without tracing.
    """

    def test_lifespan_survives_missing_tracing_sdk(self, monkeypatch):
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name.startswith("opentelemetry.sdk"):
                raise ImportError("simulated: no opentelemetry.sdk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        try:
            from fastapi.testclient import TestClient

            from distllm.api.server import app

            with TestClient(app) as client:
                response = client.get("/health")
            # 200 with a model loaded, 503 without: either proves the app
            # answered, which is what "the server starts" means here.
            assert response.status_code in (200, 503)
        finally:
            monkeypatch.undo()