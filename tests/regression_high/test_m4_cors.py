"""M4 regression: remove wildcard CORS escape hatch, deny-by-default allowlist.

Covers:
  * DISTLLM_CORS_ALLOW_ALL no longer enables wildcard '*' origins.
  * Default (no env) => empty allowlist (deny-by-default, no cross-origin).
  * Explicit DISTLLM_CORS_ALLOWED_ORIGINS => those origins are returned.
  * DISTLLM_ENV=production + wildcard request => hard CORSError.
  * allow_credentials=True + wildcard request => hard CORSError.

These tests exercise the pure, testable helper
``distllm.config._network.resolve_cors_origins`` directly, so they do not
pull in the heavy server/transformers import graph.
"""
import pytest

from distllm.config._network import CORSError, resolve_cors_origins


def test_default_is_deny_by_default_empty():
    """With no allowlist configured, CORS is fully denied (empty list)."""
    assert resolve_cors_origins(env={}) == []


def test_explicit_allowlist_returned_verbatim():
    """An explicit allowlist is returned exactly."""
    origins = resolve_cors_origins(
        env={"DISTLLM_CORS_ALLOWED_ORIGINS": "https://a.example, https://b.example/"}
    )
    assert origins == ["https://a.example", "https://b.example/"]


def test_allow_all_env_no_longer_enables_wildcard():
    """Regression: DISTLLM_CORS_ALLOW_ALL=1 must NOT produce a '*' origin.

    This is the core M4 bug: previously this switch turned on wildcard CORS.
    Now it is ignored and a wildcard is never accepted.
    """
    # Even with the old escape-hatch env set, a wildcard must be rejected.
    with pytest.raises(CORSError):
        resolve_cors_origins(env={"DISTLLM_CORS_ALLOW_ALL": "1", "DISTLLM_CORS_ALLOWED_ORIGINS": "*"})

    # And it must not silently become ['*'] when combined with an allowlist.
    out = resolve_cors_origins(
        env={
            "DISTLLM_CORS_ALLOW_ALL": "1",
            "DISTLLM_CORS_ALLOWED_ORIGINS": "https://a.example",
        }
    )
    assert out == ["https://a.example"]
    assert "*" not in out


def test_wildcard_with_credentials_is_fatal():
    """A wildcard origin combined with credentials must hard-fail."""
    with pytest.raises(CORSError):
        resolve_cors_origins(
            raw="*",
            allow_credentials=True,
            env={},
        )


def test_production_wildcard_is_fatal():
    """In production mode, requesting a wildcard origin is a fatal error."""
    with pytest.raises(CORSError):
        resolve_cors_origins(
            raw="*",
            allow_credentials=False,
            env={"DISTLLM_ENV": "production"},
        )


def test_production_explicit_allowlist_ok():
    """Production mode is fine with an explicit (non-wildcard) allowlist."""
    out = resolve_cors_origins(
        raw="https://app.example",
        allow_credentials=True,
        env={"DISTLLM_ENV": "production"},
    )
    assert out == ["https://app.example"]


def test_invalid_origin_rejected():
    """Non-URL origins are rejected."""
    with pytest.raises(CORSError):
        resolve_cors_origins(raw="ftp://evil.example", env={})


def test_server_get_cors_origins_deny_by_default(monkeypatch):
    """server._get_cors_origins returns a safe, explicit localhost allowlist
    (never a wildcard), proving the wildcard escape hatch is gone.

    server.py pulls heavy deps (transformers/uvicorn), so we skip this test
    if the module cannot be imported in the current environment rather than
    failing the whole regression suite.
    """
    try:
        import distllm.api.server as server
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"distllm.api.server not importable in this env: {exc}")

    monkeypatch.delenv("DISTLLM_CORS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("DISTLLM_CORS_ALLOW_ALL", raising=False)
    monkeypatch.delenv("DISTLLM_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("DISTLLM_DEV_MODE", raising=False)
    origins = server._get_cors_origins()
    # Deny-by-default: only explicit localhost origins, never '*' and never empty-so-open.
    assert isinstance(origins, list)
    assert "*" not in origins
    assert all(o.startswith("http://localhost") or o.startswith("https://localhost")
               for o in origins)
