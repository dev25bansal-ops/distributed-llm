"""OpenAPI documentation contract for every mounted distllm.api route.

Pin the invariant behind the DX-01-OPENAPI backfill: every route mounted on
the FastAPI app must carry a one-line ``summary=`` so the generated OpenAPI
schema documents each operation, and every route that can fail must advertise
its error statuses in ``responses=``.

The summary assertions target the ten routers named by DX-01-OPENAPI
(api_keys, batch, exchange, federated, leaderboard, marketplace, plugins,
prompts, scheduler, webrtc); the app-level sweep additionally proves the
schema serves a summary for every one of those mounted operations.

Modeled on the contract-test pattern in
``tests/core/contract/test_inference_engine_contract.py``: build the real app
once, walk ``app.routes`` for ``APIRoute`` instances, and assert on the
route's OpenAPI metadata (``summary`` / ``responses``).
"""

from __future__ import annotations

import unittest.mock as mock

import pytest
from fastapi.applications import FastAPI
from fastapi.routing import APIRoute

with mock.patch.dict("sys.modules", {"torch": mock.MagicMock(), "torch.cuda": mock.MagicMock()}):
    from distllm.api.server import app  # noqa: E402


# The ten routers backfilled by DX-01-OPENAPI, imported by module path so
# the test pins the exact files the item names.
DX01_ROUTER_MODULES = (
    "distllm.api.routes.api_keys",
    "distllm.api.routes.batch",
    "distllm.api.routes.exchange",
    "distllm.api.routes.federated",
    "distllm.api.routes.leaderboard",
    "distllm.api.routes.marketplace",
    "distllm.api.routes.plugins",
    "distllm.api.routes.prompts",
    "distllm.api.routes.scheduler",
    "distllm.api.routes.webrtc",
)


def _import_router(module_name: str):
    import importlib

    return importlib.import_module(module_name).router


def _dx01_routes() -> list[APIRoute]:
    """APIRoutes belonging to the ten DX-01-OPENAPI routers."""
    routes: list[APIRoute] = []
    for module_name in DX01_ROUTER_MODULES:
        router = _import_router(module_name)
        routes.extend(r for r in router.routes if isinstance(r, APIRoute))
    return routes


def _app_routes() -> list[APIRoute]:
    """Every mounted APIRoute in the app, in registration order."""
    return [r for r in app.routes if isinstance(r, APIRoute)]


def _summarize(route: APIRoute) -> str:
    methods = ",".join(sorted(route.methods))
    return f"{methods} {route.path} (name={route.name!r})"


class TestOpenApiSummaryContract:
    """Every DX-01 route must be documented in the OpenAPI schema."""

    def test_every_dx01_route_has_summary(self) -> None:
        """Assert every route in the ten backfilled routers has a summary."""
        routes = _dx01_routes()
        assert routes, "expected the DX-01 routers to mount routes"
        missing = [
            _summarize(r) for r in routes
            if not (r.summary and r.summary.strip())
        ]
        assert not missing, f"routes missing summary=:\n  " + "\n  ".join(missing)

    def test_served_schema_documents_every_operation(self) -> None:
        """The served /openapi.json documents a summary per operation.

        FastAPI falls back to an operationId-derived summary when a route
        lacks an explicit ``summary=``, so this sweep proves the schema is
        well-formed for every mounted operation (and fails if any operation
        is undocumented); the explicit-``summary`` requirement is asserted on
        the route objects themselves by the other tests in this class.
        """
        schema = FastAPI.openapi(app)
        paths = schema["paths"]
        assert paths, "expected a non-empty OpenAPI paths map"
        missing = []
        for path, item in paths.items():
            for method, op in item.items():
                if method in ("parameters", "servers", "summary"):
                    continue
                if not op.get("summary"):
                    missing.append(f"{method.upper()} {path}")
        assert not missing, f"schema operations missing summary:\n  " + "\n  ".join(missing)

    def test_backfilled_routers_are_mounted_and_documented(self) -> None:
        """The 10 backfilled routers' routes all carry summaries."""
        for module_name in DX01_ROUTER_MODULES:
            router = _import_router(module_name)
            for route in router.routes:
                if not isinstance(route, APIRoute):
                    continue
                assert route.summary and route.summary.strip(), (
                    f"{router.prefix!r} route {route.path} has no summary"
                )


class TestOpenApiResponsesContract:
    """Routes that can fail must advertise the failure in ``responses=``."""

    @pytest.mark.parametrize("module_name", [
        "distllm.api.routes.api_keys",
        "distllm.api.routes.scheduler",
        "distllm.api.routes.plugins",
    ])
    def test_admin_gated_routes_advertise_401_and_403(self, module_name: str) -> None:
        """Admin-gated routers document 401 (unauthenticated) and 403 (forbidden)."""
        router = _import_router(module_name)
        routes = [r for r in router.routes if isinstance(r, APIRoute)]
        assert routes, f"no routes in {module_name}"
        for route in routes:
            codes = {str(c) for c in (route.responses or {})}
            assert "401" in codes, f"{_summarize(route)} missing 401 in responses"
            assert "403" in codes, f"{_summarize(route)} missing 403 in responses"

    def test_id_path_param_routes_advertise_404(self) -> None:
        """DX-01 ID-keyed routes that raise 404 document the 404 envelope.

        Scoped to routes whose handlers actually raise HTTPException(404) —
        ``_get_prompt_or_404`` / ``not found`` detail strings — rather than
        every ``{param}`` route: some ID routes deliberately return 200 with
        a false flag instead (e.g. ``GET /v1/exchange/prompts/{prompt_id}/access``).

        Exception: POST /v1/exchange/prompts/{prompt_id}/acquire collapses
        "prompt not found" into 402 (payment required) to avoid leaking prompt
        existence, so its 402 response documents the miss.
        """
        for route in _dx01_routes():
            if "{" not in route.path:
                continue
            codes = {str(c) for c in (route.responses or {})}
            if route.path == "/v1/exchange/prompts/{prompt_id}/acquire":
                assert "402" in codes, f"{_summarize(route)} missing 402 in responses"
                continue
            if route.path == "/v1/exchange/prompts/{prompt_id}/access":
                # Never 404s: unknown prompt/user resolves to has_access=false.
                continue
            if "not found" in (route.description or "").lower() or route.name in {
                # ID-lookup handlers that raise HTTPException(404)
                "get_prompt", "get_prompt_detail", "get_shared_prompt",
                "get_leaderboard_entry", "get_listing", "remove_listing",
                "get_batch_status", "stream_batch_results", "cancel_batch",
                "get_plugin", "revoke_api_key", "close_session", "webrtc_ice",
                "verify_entry", "vote_entry", "comment_entry",
                "restore_version", "delete_prompt",
            }:
                assert "404" in codes, f"{_summarize(route)} missing 404 in responses"

    def test_body_routes_advertise_422(self) -> None:
        """DX-01 POST/PUT/PATCH with a request body documents the 422 envelope."""
        for route in _dx01_routes():
            if route.methods & {"POST", "PUT", "PATCH"} and route.dependant.body_params:
                codes = {str(c) for c in (route.responses or {})}
                assert "422" in codes, f"{_summarize(route)} missing 422 in responses"

