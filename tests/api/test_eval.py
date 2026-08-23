"""Tests for evaluation API endpoints.

Covers all four routes defined in ``distllm.api.routes.eval``:

- POST /v1/eval/run
- GET /v1/eval/results
- GET /v1/eval/results/{report_id}
- DELETE /v1/eval/results/{report_id}

Uses ONLY real objects — no ``MagicMock``, ``Mock``, or ``AsyncMock``.
The ``conftest.py`` autouse ``reset_app_state`` fixture cleans the shared
``AppState`` before each test.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest

from distllm.api.api_state import g
from distllm.api.routes.eval import router as eval_router
from distllm.api.services.eval_service import EvalService


# ---------------------------------------------------------------------------
# Test doubles  (no MagicMock / Mock / AsyncMock)
# ---------------------------------------------------------------------------


class CoordinatorMock:
    """Minimal coordinator stand-in with no MagicMock dependency.

    Provides the attributes and methods that ``EvalService`` / ``EvalRunner``
    reference during construction and lazy initialization.
    """

    model_name = "test-model"
    nodes: dict = {}
    _shutting_down = False

    def generate(self, prompt: str, **kwargs: object) -> str:
        return "mock generation output"


class _TestEvalService(EvalService):
    """``EvalService`` subclass that returns canned responses.

    Overrides every public method so that no real coordinator,
    eval runner, or SQLite database is needed during tests.
    """

    def run_benchmarks(
        self,
        model_id: str,
        benchmarks: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for b in benchmarks or ["mmlu"]:
            results[b] = {
                "report_id": f"report-{b}",
                "model_id": model_id,
                "dataset": b,
                "status": "completed",
                "metrics": {"accuracy": 0.85},
                "config": {},
                "created_at": 1000.0,
                "duration_s": 5.0,
                "results": [],
            }
        return results

    def list_reports(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        return [
            {
                "report_id": "report-1",
                "model_id": model_id or "model-x",
                "dataset": dataset or "mmlu",
                "status": "completed",
                "metrics": '{"accuracy": 0.85}',
                "config": "{}",
                "created_at": 1000.0,
                "duration_s": 5.0,
            },
        ]

    def get_report(self, report_id: str) -> dict | None:
        if report_id == "report-1":
            return {
                "report_id": "report-1",
                "model_id": "model-x",
                "dataset": "mmlu",
                "status": "completed",
                "metrics": '{"accuracy": 0.85}',
                "config": "{}",
                "created_at": 1000.0,
                "duration_s": 5.0,
            }
        return None

    def get_report_results(self, report_id: str) -> list[dict]:
        return [
            {
                "question": "What is 2+2?",
                "answer": "4",
                "prediction": "4",
                "score": 1.0,
                "category": "math",
                "latency_ms": 150.0,
                "prompt_tokens": 10,
                "generated_tokens": 1,
                "error": None,
                "metadata": {},
            },
        ]

    def delete_report(self, report_id: str) -> bool:
        return report_id == "report-1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinator() -> CoordinatorMock:
    """Set ``g.coordinator`` to a plain ``CoordinatorMock``."""
    coord = CoordinatorMock()
    g.coordinator = coord
    return coord


@pytest.fixture
def admin_client(monkeypatch: pytest.MonkeyPatch, coordinator: CoordinatorMock) -> TestClient:
    """TestClient with ``require_role("admin")`` satisfied.

    Patches the ``EvalService`` reference in the route module so that
    every handler receives ``_TestEvalService`` instead of the real
    service class.  Also injects a middleware that stamps the request
    with ``api_key_role = "admin"`` so the DELETE endpoint's auth
    dependency passes.
    """
    monkeypatch.setattr("distllm.api.routes.eval.EvalService", _TestEvalService)

    app = FastAPI()

    @app.middleware("http")
    async def _inject_admin_role(request: Request, call_next):
        request.state.api_key_role = "admin"
        request.state.api_key_id = "test-admin-key"
        return await call_next(request)

    app.include_router(eval_router)
    return TestClient(app)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, coordinator: CoordinatorMock) -> TestClient:
    """TestClient *without* admin role injection.

    GET/POST endpoints do not require auth, so this is sufficient for
    success-path tests.  The DELETE endpoint returns 401 when used
    with this client.
    """
    monkeypatch.setattr("distllm.api.routes.eval.EvalService", _TestEvalService)

    app = FastAPI()
    app.include_router(eval_router)
    return TestClient(app)


# ===================================================================
# POST /v1/eval/run
# ===================================================================


class TestEvalRun:
    """POST /v1/eval/run -- run evaluation benchmarks."""

    def test_success_single_benchmark(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/eval/run",
            json={"model_id": "my-model", "benchmarks": ["mmlu"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "mmlu" in data["reports"]
        assert data["reports"]["mmlu"]["metrics"]["accuracy"] == 0.85

    def test_success_multiple_benchmarks(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/eval/run",
            json={"model_id": "my-model", "benchmarks": ["mmlu", "gsm8k"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "mmlu" in data["reports"]
        assert "gsm8k" in data["reports"]

    def test_invalid_benchmark_name(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/eval/run",
            json={
                "model_id": "my-model",
                "benchmarks": ["not-a-real-benchmark"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "Unknown benchmark" in data["error"]

    def test_ssrf_localhost_coordinator_url(self, client: TestClient) -> None:
        """``coordinator_url`` pointing at localhost is rejected."""
        resp = client.post(
            "/v1/eval/run",
            json={
                "model_id": "my-model",
                "benchmarks": ["mmlu"],
                "coordinator_url": "http://localhost:8000/v1/chat/completions",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "localhost" in data["error"].lower()

    def test_ssrf_private_ip_coordinator_url(self, client: TestClient) -> None:
        """A private IP in ``coordinator_url`` is rejected."""
        resp = client.post(
            "/v1/eval/run",
            json={
                "model_id": "my-model",
                "benchmarks": ["mmlu"],
                "coordinator_url": "http://10.0.0.5:8000",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "private" in data["error"].lower()

    def test_ssrf_localhost_coordinator_url_b(self, client: TestClient) -> None:
        """``coordinator_url_b`` pointing at localhost is rejected."""
        resp = client.post(
            "/v1/eval/run",
            json={
                "model_id": "my-model",
                "benchmarks": ["arena"],
                "coordinator_url": "http://8.8.8.8:8000",
                "model_b": "other-model",
                "coordinator_url_b": "http://localhost:8000",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "localhost" in data["error"].lower()

    def test_public_ip_coordinator_url_succeeds(self, client: TestClient) -> None:
        """A public IP ``coordinator_url`` passes SSRF check and reaches the
        service layer."""
        resp = client.post(
            "/v1/eval/run",
            json={
                "model_id": "my-model",
                "benchmarks": ["mmlu"],
                "coordinator_url": "http://8.8.8.8:8000",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


# ===================================================================
# GET /v1/eval/results
# ===================================================================


class TestEvalResults:
    """GET /v1/eval/results -- list evaluation reports."""

    def test_list_results(self, client: TestClient) -> None:
        resp = client.get("/v1/eval/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["reports"], list)
        assert len(data["reports"]) >= 1

    def test_list_results_with_filters(self, client: TestClient) -> None:
        resp = client.get(
            "/v1/eval/results",
            params={"model_id": "my-model", "dataset": "mmlu", "limit": 10, "offset": 0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["reports"], list)


# ===================================================================
# GET /v1/eval/results/{report_id}
# ===================================================================


class TestEvalResultDetail:
    """GET /v1/eval/results/{report_id} -- get a single report."""

    def test_get_existing_report(self, client: TestClient) -> None:
        resp = client.get("/v1/eval/results/report-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["report"]["report_id"] == "report-1"
        assert data["report"]["model_id"] == "model-x"

    def test_get_nonexistent_report(self, client: TestClient) -> None:
        resp = client.get("/v1/eval/results/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"].lower()


# ===================================================================
# DELETE /v1/eval/results/{report_id}
# ===================================================================


class TestEvalResultDelete:
    """DELETE /v1/eval/results/{report_id} -- requires admin role."""

    def test_delete_requires_auth(self, client: TestClient) -> None:
        """Without the admin middleware the endpoint returns 401."""
        resp = client.delete("/v1/eval/results/report-1")
        assert resp.status_code == 401

    def test_delete_existing_report(self, admin_client: TestClient) -> None:
        resp = admin_client.delete("/v1/eval/results/report-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["deleted"] == "report-1"

    def test_delete_nonexistent_report(self, admin_client: TestClient) -> None:
        resp = admin_client.delete("/v1/eval/results/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"].lower()
