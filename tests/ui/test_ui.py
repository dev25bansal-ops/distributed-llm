"""Tests for distllm.ui.app -- FastAPI web UI.

Covers:
    - Module-level constants (UI_DIR, API_URL)
    - FastAPI app construction (title, description, version, routes)
    - get_http_client() singleton behaviour
    - main() CLI argument parsing
    - Health proxy fallback when API is unreachable

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real pathlib, real httpx, minimal stubs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from types import ModuleType


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _list_routes(app) -> set[str]:
    """Return the set of URL path strings registered on a FastAPI app."""
    return {r.path for r in app.routes}


# =================================================================== #
# Construction & defaults                                               #
# =================================================================== #


class TestAppConstruction:
    """Verify the FastAPI application object and its module-level constants."""

    def test_module_constants(self, ui_app_module: ModuleType) -> None:
        """UI_DIR points at the ui package directory and API_URL has the expected default."""
        assert isinstance(ui_app_module.UI_DIR, Path)
        assert ui_app_module.UI_DIR.name == "ui"
        assert (ui_app_module.UI_DIR / "app.py").exists()
        assert ui_app_module.API_URL == "http://localhost:8000"

    def test_app_title_and_version(self, ui_app_module: ModuleType) -> None:
        """FastAPI app is pre-built with the expected metadata."""
        app = ui_app_module.ui_app
        assert app.title == "DistLLM UI"
        assert app.description == "Web interface for Distributed LLM"
        assert app.version == "0.3.0"

    def test_routes_registered(self, ui_app_module: ModuleType) -> None:
        """All expected route paths are present on the app."""
        routes = _list_routes(ui_app_module.ui_app)
        for expected in ("/", "/dashboard", "/models", "/compare", "/api/health"):
            assert expected in routes, f"Missing route: {expected}"


# =================================================================== #
# get_http_client                                                       #
# =================================================================== #


class TestHttpClient:
    """Singleton HTTP client for proxied API calls."""

    def test_get_http_client_returns_client(self, ui_app_module: ModuleType) -> None:
        """First call creates and returns an AsyncClient."""
        client = ui_app_module.get_http_client()
        assert client is not None
        assert hasattr(client, "get")

    def test_get_http_client_is_singleton(self, ui_app_module: ModuleType) -> None:
        """Subsequent calls return the same instance."""
        ui_app_module._client = None
        first = ui_app_module.get_http_client()
        second = ui_app_module.get_http_client()
        assert first is second

    def test_get_http_client_timeout(self, ui_app_module: ModuleType) -> None:
        """Client is created with the expected default timeout."""
        ui_app_module._client = None
        client = ui_app_module.get_http_client()
        assert client._timeout.connect == 10.0


# =================================================================== #
# CLI argument parsing (main)                                           #
# =================================================================== #


class TestMainCLI:
    """Argparse interface of the ``main()`` entrypoint."""

    def test_main_creates_parser(self, ui_app_module: ModuleType) -> None:
        """main() builds a parser with the expected arguments (before parsing)."""
        captured_parser: argparse.ArgumentParser | None = None

        original_init = argparse.ArgumentParser.__init__

        def tracking_init(self, *args, **kwargs):
            nonlocal captured_parser
            original_init(self, *args, **kwargs)
            captured_parser = self

        # Replace __init__ inline rather than via mock.patch
        argparse.ArgumentParser.__init__ = tracking_init
        try:
            original_argv = sys.argv
            sys.argv = ["app.py", "--host", "0.0.0.0", "--port", "9999"]
            try:
                ui_app_module.main()
            except (SystemExit, TypeError):
                pass
            finally:
                sys.argv = original_argv
        finally:
            argparse.ArgumentParser.__init__ = original_init

        assert captured_parser is not None
        namespace, _ = captured_parser.parse_known_args(
            ["--host", "0.0.0.0", "--port", "9999", "--api-url", "http://example:8000"]
        )
        assert namespace.host == "0.0.0.0"
        assert namespace.port == 9999
        assert namespace.api_url == "http://example:8000"

    def test_main_defaults(self, ui_app_module: ModuleType) -> None:
        """Argument defaults match the specification."""
        # Capture the uvicorn.run call by replacing it with a spy
        calls = []

        def _fake_uvicorn_run(app, **kwargs):
            calls.append((app, kwargs))

        original_run = getattr(ui_app_module.uvicorn, "run", None)
        ui_app_module.uvicorn.run = _fake_uvicorn_run
        original_argv = sys.argv
        sys.argv = ["app.py"]
        try:
            ui_app_module.main()
        except (SystemExit, TypeError):
            pass
        finally:
            sys.argv = original_argv
            if original_run is not None:
                ui_app_module.uvicorn.run = original_run
            else:
                del ui_app_module.uvicorn.run

        assert len(calls) == 1
        app, kwargs = calls[0]
        assert app is ui_app_module.ui_app
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8500
        assert kwargs["log_level"] == "info"

    def test_main_sets_global_api_url(self, ui_app_module: ModuleType) -> None:
        """The --api-url argument is propagated to the module-level API_URL."""
        original_argv = sys.argv
        sys.argv = ["app.py", "--api-url", "http://other:8080"]
        try:
            ui_app_module.main()
        except (SystemExit, TypeError):
            pass
        finally:
            sys.argv = original_argv
        assert ui_app_module.API_URL == "http://other:8080"
        ui_app_module.API_URL = "http://localhost:8000"


# =================================================================== #
# Health proxy                                                          #
# =================================================================== #


class TestHealthProxy:
    """/api/health endpoint that proxies the backend API."""

    @pytest.mark.asyncio
    async def test_health_proxy_unavailable(self, ui_app_module: ModuleType) -> None:
        """Returns 'unavailable' when the API server is unreachable."""

        # Replace get_http_client with a factory that returns a failing client
        original_get_http_client = ui_app_module.get_http_client

        def make_failing_client():
            def handler(request):
                raise httpx.RequestError("connection refused")

            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            return client

        ui_app_module.get_http_client = make_failing_client
        try:
            result = await ui_app_module.health_proxy()
        finally:
            ui_app_module.get_http_client = original_get_http_client

        assert result == {"status": "unavailable", "reason": "API server not reachable"}


# =================================================================== #
# __init__.py re-exports                                                #
# =================================================================== #


class TestInitReExports:
    """The ``distllm.ui`` package's __init__.py re-exports the right symbols."""

    def test_main_re_exported(self) -> None:
        """``from distllm.ui import main`` works via the __init__ shim."""
        from distllm.ui import main as init_main

        assert callable(init_main)
