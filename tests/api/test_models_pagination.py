"""Regression: GET /v1/models supports pagination and filtering.

Query parameters:
  * ``limit`` -- 1..500 (default 100). Caps the number of items in ``data``.
  * ``offset`` -- >= 0 (default 0). Skips the first N matching items.
  * ``family`` -- exact match on the model ID's first ``/``-separated segment.
  * ``q`` -- case-insensitive substring filter on the model ID.

Behaviour pinned by these tests:
  * Default response shape is unchanged when no query string is supplied.
  * ``limit`` and ``offset`` apply to the *filtered* result set, not the
    unfiltered catalogue.
  * ``family`` and ``q`` are *and*-combined (intersection, not union).
  * Invalid ``limit`` (< 1 or > 500) and invalid ``offset`` (< 0) return 400.
  * Empty catalogue still returns 200 with empty data.
  * Empty-string ``family`` / ``q`` are treated as not-set.
"""

from __future__ import annotations

import secrets
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store


SAMPLE_CATALOGUE: list[str] = [
    "meta-llama/Llama-3-8B",
    "meta-llama/Llama-3-70B",
    "meta-llama/Llama-2-7B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B",
    "google/gemma-7b",
    "google/gemma-2b",
    "OpenAI/gpt-oss-20B",
]


@pytest.fixture(autouse=True)
def _setup_auth(monkeypatch):
    """Set up a valid API key for tests that need auth."""
    test_api_key = secrets.token_hex(32)
    monkeypatch.setenv("API_KEY", test_api_key)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    reset_api_key_store()
    g._startup_time = time.time()
    yield test_api_key
    monkeypatch.delenv("API_KEY", raising=False)


def _authed_get(api_key: str, path: str):
    """Helper: send an authenticated GET to /v1/models*."""
    return TestClient(app).get(
        path,
        headers={"Authorization": f"Bearer {api_key}"},
    )


@pytest.fixture
def coordinator():
    coord = MagicMock()
    coord.model_name = "meta-llama/Llama-3-8B"
    coord.nodes = {}
    coord._shutting_down = False
    coord.list_models = MagicMock(return_value=list(SAMPLE_CATALOGUE))
    coord.health_check = MagicMock(return_value={})
    coord.get_metrics = MagicMock(return_value={"requests_total": 42})
    coord.metrics_exporter = None
    coord.scheduler = None
    coord.prefix_cache = None
    return coord


def _ids(resp) -> list[str]:
    return [m["id"] for m in resp.json()["data"]]


class TestV1ModelsDefaults:
    """Default behaviour (no query string) is unchanged from the legacy contract."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator, _setup_auth):
        self._prev = g.coordinator
        g.coordinator = coordinator
        self._api_key = _setup_auth
        yield
        g.coordinator = self._prev

    def test_default_returns_all_models(self):
        resp = _authed_get(self._api_key, "/v1/models")
        assert resp.status_code == 200
        assert _ids(resp) == SAMPLE_CATALOGUE

    def test_response_shape_preserved(self):
        """Even with no query string the response has 'object=list' and 'data'."""
        resp = _authed_get(self._api_key, "/v1/models")
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        for item in body["data"]:
            assert item["object"] == "model"
            assert "id" in item
            assert "created" in item


class TestV1ModelsPagination:
    """limit / offset behave correctly on the filtered result set."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator, _setup_auth):
        self._prev = g.coordinator
        g.coordinator = coordinator
        self._api_key = _setup_auth
        yield
        g.coordinator = self._prev

    def test_limit_caps_result_size(self):
        resp = _authed_get(self._api_key, "/v1/models?limit=3")
        assert resp.status_code == 200
        assert len(_ids(resp)) == 3
        assert _ids(resp) == SAMPLE_CATALOGUE[:3]

    def test_offset_skips_first_n(self):
        resp = _authed_get(self._api_key, "/v1/models?offset=2&limit=2")
        assert resp.status_code == 200
        assert _ids(resp) == SAMPLE_CATALOGUE[2:4]

    def test_limit_at_default_100(self):
        """Default limit of 100 — catalogue has 9 entries, all returned."""
        resp = _authed_get(self._api_key, "/v1/models")
        assert len(_ids(resp)) == 9

    def test_offset_past_end_returns_empty_data(self):
        resp = _authed_get(self._api_key, "/v1/models?offset=999")
        assert resp.status_code == 200
        assert _ids(resp) == []

    def test_offset_applied_to_filtered_set(self):
        """Filter to family=meta-llama (4 entries), then offset 1, limit 2."""
        resp = _authed_get(
            self._api_key, "/v1/models?family=meta-llama&offset=1&limit=2"
        )
        assert resp.status_code == 200
        assert _ids(resp) == [
            "meta-llama/Llama-3-70B",
            "meta-llama/Llama-2-7B",
        ]

    def test_limit_zero_rejected(self):
        resp = _authed_get(self._api_key, "/v1/models?limit=0")
        # FastAPI's Query(ge=1, le=500) returns 422 on out-of-range values.
        assert resp.status_code in (400, 422)

    def test_limit_too_large_rejected(self):
        resp = _authed_get(self._api_key, "/v1/models?limit=501")
        assert resp.status_code in (400, 422)

    def test_negative_offset_rejected(self):
        resp = _authed_get(self._api_key, "/v1/models?offset=-1")
        assert resp.status_code in (400, 422)


class TestV1ModelsFamilyFilter:
    """family matches the first '/' segment of the model ID."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator, _setup_auth):
        self._prev = g.coordinator
        g.coordinator = coordinator
        self._api_key = _setup_auth
        yield
        g.coordinator = self._prev

    def test_family_meta_llama(self):
        resp = _authed_get(self._api_key, "/v1/models?family=meta-llama")
        assert resp.status_code == 200
        assert _ids(resp) == [
            "meta-llama/Llama-3-8B",
            "meta-llama/Llama-3-70B",
            "meta-llama/Llama-2-7B",
        ]

    def test_family_mistralai(self):
        resp = _authed_get(self._api_key, "/v1/models?family=mistralai")
        assert resp.status_code == 200
        assert _ids(resp) == [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B",
        ]

    def test_family_case_sensitive_match(self):
        """Family match is exact, case-sensitive on the first segment."""
        resp = _authed_get(self._api_key, "/v1/models?family=Meta-Llama")
        assert resp.status_code == 200
        # None of the sample IDs use the capitalised form, so this is empty.
        assert _ids(resp) == []

    def test_family_no_match(self):
        resp = _authed_get(self._api_key, "/v1/models?family=nonexistent-org")
        assert resp.status_code == 200
        assert _ids(resp) == []

    def test_family_empty_string_treated_as_not_set(self):
        resp = _authed_get(self._api_key, "/v1/models?family=")
        assert resp.status_code == 200
        # Empty family filter must NOT filter the catalogue to empty.
        assert _ids(resp) == SAMPLE_CATALOGUE


class TestV1ModelsQueryFilter:
    """q is a case-insensitive substring filter on the model ID."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator, _setup_auth):
        self._prev = g.coordinator
        g.coordinator = coordinator
        self._api_key = _setup_auth
        yield
        g.coordinator = self._prev

    def test_q_substring_match(self):
        resp = _authed_get(self._api_key, "/v1/models?q=Mistral")
        assert resp.status_code == 200
        # "Mistral" matches all three Mistral-family IDs (including Mixtral,
        # case-insensitive substring match).  Sort the IDs so the test is
        # stable against future catalogue re-orderings.
        ids = sorted(_ids(resp))
        assert ids == sorted([
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B",
        ])

    def test_q_case_insensitive(self):
        """Substring match is case-insensitive."""
        resp_lower = _authed_get(self._api_key, "/v1/models?q=mistral")
        resp_upper = _authed_get(self._api_key, "/v1/models?q=MISTRAL")
        assert _ids(resp_lower) == _ids(resp_upper)
        assert len(_ids(resp_lower)) >= 1

    def test_q_no_match(self):
        resp = _authed_get(self._api_key, "/v1/models?q=definitely-not-a-model")
        assert resp.status_code == 200
        assert _ids(resp) == []

    def test_q_empty_string_treated_as_not_set(self):
        resp = _authed_get(self._api_key, "/v1/models?q=")
        assert resp.status_code == 200
        assert _ids(resp) == SAMPLE_CATALOGUE

    def test_q_whitespace_only_treated_as_not_set(self):
        resp = _authed_get(self._api_key, "/v1/models?q=%20%20")
        assert resp.status_code == 200
        assert _ids(resp) == SAMPLE_CATALOGUE


class TestV1ModelsCombinedFilters:
    """family and q combine with AND; pagination applies to the intersection."""

    @pytest.fixture(autouse=True)
    def setup(self, coordinator, _setup_auth):
        self._prev = g.coordinator
        g.coordinator = coordinator
        self._api_key = _setup_auth
        yield
        g.coordinator = self._prev

    def test_family_and_q_intersect(self):
        resp = _authed_get(self._api_key, "/v1/models?family=meta-llama&q=Llama-3")
        assert resp.status_code == 200
        assert _ids(resp) == [
            "meta-llama/Llama-3-8B",
            "meta-llama/Llama-3-70B",
        ]

    def test_family_q_pagination(self):
        resp = _authed_get(
            self._api_key, "/v1/models?family=meta-llama&q=Llama&limit=2&offset=1"
        )
        assert resp.status_code == 200
        assert _ids(resp) == [
            "meta-llama/Llama-3-70B",
            "meta-llama/Llama-2-7B",
        ]


class TestV1ModelsEmptyCatalogue:
    """No coordinator or empty list still returns 200 with empty data."""

    @pytest.fixture(autouse=True)
    def setup(self, _setup_auth):
        self._prev = g.coordinator
        self._api_key = _setup_auth
        yield
        g.coordinator = self._prev

    def test_no_coordinator_returns_empty(self):
        g.coordinator = None
        resp = _authed_get(self._api_key, "/v1/models?limit=10")
        assert resp.status_code == 200
        assert _ids(resp) == []

    def test_empty_list_models_returns_empty(self):
        coord = MagicMock()
        coord.model_name = ""
        coord.list_models = MagicMock(return_value=[])
        coord._shutting_down = False
        coord.health_check = MagicMock(return_value={})
        g.coordinator = coord
        resp = _authed_get(self._api_key, "/v1/models?family=meta-llama")
        assert resp.status_code == 200
        assert _ids(resp) == []