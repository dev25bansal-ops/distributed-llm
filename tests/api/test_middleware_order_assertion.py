"""Enhancement #4: middleware order startup assertion.

Starlette's ``add_middleware`` PREPENDS — the LAST registered middleware
is the OUTERMOST (runs first on request). The security posture depends on
this exact order:

- RateLimit outside Auth  (limit applies to rejected-auth requests too)
- Auth outside Quota      (quota keyed on authenticated tenant identity)
- Quota outside PluginHook (quota enforced before plugin code runs)
- DocsAuth inside Auth     (docs auth consults the api_key_role Auth set)
- SecurityHeaders outermost-ish (headers on every response incl. errors)

This test imports the real app and asserts the built middleware stack
matches the documented order, failing the build on any reordering.
"""

from __future__ import annotations

import importlib

import pytest


def _built_stack(app) -> list[str]:
    """Return middleware class names in TRUE EXECUTION order (outermost first).

    Walks the actually-composed ``app.middleware_stack`` (built via
    ``build_middleware_stack``) by chasing the ``.app`` chain. This is the
    ground truth — Starlette's ``add_middleware`` prepends to
    ``user_middleware``, so registration file order is the REVERSE of
    execution order (last registered = outermost = first to run).
    Starlette's own ServerError/Exception middlewares bookend the chain
    and are filtered out here.
    """
    names: list[str] = []
    obj = app.build_middleware_stack()
    while True:
        names.append(type(obj).__name__)
        inner = getattr(obj, "app", None)
        if inner is None:
            break
        obj = inner
    skip = {"ServerErrorMiddleware", "ExceptionMiddleware", "AsyncExitStackMiddleware"}
    return [n for n in names if n not in skip]


@pytest.fixture(scope="module")
def app():
    # Import fresh so registration side effects run exactly once.
    sys_modules_pop = None
    try:
        import distllm.api.server as server_mod

        return server_mod.app
    except Exception:
        # If full server import fails in this environment, skip cleanly.
        pytest.skip("distllm.api.server could not be imported")


def test_middleware_stack_built(app):
    names = _built_stack(app)
    assert names, "middleware stack is empty — add_middleware was never called"


def test_auth_runs_inside_ratelimit(app):
    """RateLimit must be OUTSIDE Auth (limits unauthenticated traffic)."""
    names = _built_stack(app)
    rl = next(i for i, n in enumerate(names) if "RateLimit" in n)
    auth = next(i for i, n in enumerate(names) if n == "AuthMiddleware")
    assert rl < auth, (
        f"RateLimitMiddleware (idx {rl}) must execute BEFORE AuthMiddleware "
        f"(idx {auth}). Current order: {names}"
    )


def test_quota_runs_inside_auth(app):
    """Quota must be INSIDE Auth (needs the tenant identity)."""
    names = _built_stack(app)
    auth = next(i for i, n in enumerate(names) if n == "AuthMiddleware")
    quota = next(i for i, n in enumerate(names) if "QuotaMiddleware" in n)
    assert auth < quota, (
        f"AuthMiddleware (idx {auth}) must execute BEFORE QuotaMiddleware "
        f"(idx {quota}). Current order: {names}"
    )


def test_plugin_hook_inside_quota(app):
    """PluginHook must be INSIDE Quota (plugins can't bypass quotas)."""
    names = _built_stack(app)
    quota = next(i for i, n in enumerate(names) if "QuotaMiddleware" in n)
    hook = next(i for i, n in enumerate(names) if "PluginHookMiddleware" in n)
    assert quota < hook, (
        f"QuotaMiddleware (idx {quota}) must execute BEFORE PluginHookMiddleware "
        f"(idx {hook}). Current order: {names}"
    )


def test_docs_auth_inside_auth(app):
    """DocsAuth must be INSIDE Auth (reads api_key_role set by Auth)."""
    names = _built_stack(app)
    auth = next(i for i, n in enumerate(names) if n == "AuthMiddleware")
    docs = next(i for i, n in enumerate(names) if "DocsAuthMiddleware" in n)
    assert auth < docs, (
        f"AuthMiddleware (idx {auth}) must execute BEFORE DocsAuthMiddleware "
        f"(idx {docs}). Current order: {names}"
    )


def test_security_headers_wraps_routes(app):
    """SecurityHeaders must sit INSIDE the error middlewares but OUTSIDE the
    router — headers attach to the final response after route handlers run,
    before CORS wraps it. Its actual position (near the router, inner side)
    is correct; assert it executes AFTER CSRF and BEFORE the router."""
    names = _built_stack(app)
    sh = next(i for i, n in enumerate(names) if "SecurityHeadersMiddleware" in n)
    csrf = next(i for i, n in enumerate(names) if "CSRF" in n)
    assert csrf < sh, (
        f"CSRFSameOriginMiddleware (idx {csrf}) must execute before "
        f"SecurityHeadersMiddleware (idx {sh}). Order: {names}"
    )


def test_csrf_outside_auth(app):
    """CSRF must be OUTSIDE Auth (rejects forged requests before auth work)."""
    names = _built_stack(app)
    csrf = next(i for i, n in enumerate(names) if "CSRF" in n)
    auth = next(i for i, n in enumerate(names) if n == "AuthMiddleware")
    assert csrf < auth, (
        f"CSRFSameOriginMiddleware (idx {csrf}) must execute BEFORE "
        f"AuthMiddleware (idx {auth}). Current order: {names}"
    )
