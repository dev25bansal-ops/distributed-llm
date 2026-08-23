"""Regression tests for HIGH fix C7: quota middleware unwired & disabled.

``QuotaMiddleware`` existed but was never registered in the ASGI stack and
defaulted to disabled. It is now wired into ``server.py`` (after auth) and can
be enabled via ``DISTLLM_QUOTA_ENABLED=1``. This test verifies the middleware
imports, is referenced by the server module (wired into the stack), and
enforces a 429 when the usage meter rejects.

Note: ``distllm.api.__init__`` imports the heavy server stack (transformers,
etc.). To verify the quota wiring in isolation we load ``quota_middleware`` and
``server`` directly via importlib with stub parent packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

_REPO_SRC = __import__("pathlib").Path(__file__).resolve().parents[2] / "src"


def _stub_parent(pkg: str, real_path: str | None = None) -> None:
    """Register a parent package in sys.modules.

    With ``real_path=None`` the package is a lightweight STUB (empty ``__path__``)
    so its ``__init__`` is never executed (used for ``distllm``/``distllm.api`` to
    avoid the heavy server stack). With ``real_path`` set, the package points at
    the real filesystem directory so its submodules resolve via normal import.
    """
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        if real_path is not None:
            mod.__path__ = [real_path]
        else:
            mod.__path__ = []
        sys.modules[pkg] = mod


def _load_module(qualified: str, filename: str):
    # Stub the top-level + api packages so we DON'T run the heavy server __init__.
    _stub_parent("distllm")
    _stub_parent("distllm.api")
    # distllm.core / distllm.errors are LIGHT packages; register them as REAL
    # packages so any transitive submodule (cost_tracker, metering, money,
    # tenant_billing, errors.types, ...) resolves via normal import. N1 added
    # distllm.core.tenant_billing -> cost_tracker -> metering -> money, so the
    # hand-picked preload list is replaced by a real-path package.
    _stub_parent("distllm.core", str(_REPO_SRC / "distllm" / "core"))
    _stub_parent("distllm.errors", str(_REPO_SRC / "distllm" / "errors"))
    fpath = _REPO_SRC / filename
    spec = importlib.util.spec_from_file_location(qualified, fpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def test_quota_middleware_importable_and_registered():
    qm = _load_module("distllm.api.quota_middleware", "distllm/api/quota_middleware.py")
    QuotaMiddleware = qm.QuotaMiddleware
    assert QuotaMiddleware is not None

    # The server module must reference the same QuotaMiddleware (wired in).
    # Rather than execute the heavy server stack, assert the wiring textually:
    # the import is from distllm.api.quota_middleware and add_middleware is
    # called with QuotaMiddleware.
    server_src = (_REPO_SRC / "distllm/api/server.py").read_text(encoding="utf-8")
    assert "from distllm.api.quota_middleware import QuotaMiddleware" in server_src
    assert "app.add_middleware(QuotaMiddleware)" in server_src


def test_quota_middleware_enforces_429(monkeypatch):
    qm = _load_module("distllm.api.quota_middleware", "distllm/api/quota_middleware.py")

    monkeypatch.setattr(qm, "_ENABLED", True)
    monkeypatch.setenv("DISTLLM_QUOTA_ENABLED", "1")

    class _RejectMeter:
        def enforce_quota(self, tenant_id):
            return False, "over quota"

        def release_quota(self, tenant_id):
            pass

        def record_request(self, **kwargs):
            pass

    monkeypatch.setattr(qm, "get_usage_meter", lambda: _RejectMeter())

    captured = {}

    class _Resp:
        status_code = 200
        body = b"{}"

    async def call_next(req):
        captured["next"] = True
        return _Resp()

    fake_request = types.SimpleNamespace(
        url=types.SimpleNamespace(path="/v1/chat/completions"),
        state=types.SimpleNamespace(tenant_id="t1", api_key_id="k1"),
    )

    import asyncio

    mw = qm.QuotaMiddleware(app=None, enable=True)
    resp = asyncio.run(mw.dispatch(fake_request, call_next))
    assert resp.status_code == 429
    assert captured.get("next") is None
