"""Content moderation tests: POST /v1/moderations."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


class TestModerate:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        self.coord = MagicMock()
        self.coord.model_name = "test-model"
        self.coord.nodes = {}
        self.coord._shutting_down = False
        self.coord._moderation_model = None
        g.coordinator = self.coord
        yield
        g.coordinator = original

    def test_moderate_clean_text(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "The weather is nice today."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["flagged"] is False
        for cat, val in result["categories"].items():
            assert val is False, f"{cat} should be False for clean text"

    def test_moderate_hateful_text(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "This is a hate speech slur against an inferior race."},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"]["hate"] is True
        assert result["category_scores"]["hate"] > 0.5

    def test_moderate_violent_text(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "I will kill and murder them with a weapon."},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"]["violence"] is True
        assert result["category_scores"]["violence"] > 0.5

    def test_heuristic_hate_keywords(self):
        for text in ["hate speech and ethnic cleansing", "that slur is hate speech", "inferior race slur"]:
            resp = TestClient(app).post(
                "/v1/moderations",
                json={"input": text},
            )
            result = resp.json()["results"][0]
            assert result["flagged"] is True, f"should flag: {text}"
            assert result["categories"]["hate"] is True, f"hate category: {text}"

    def test_heuristic_self_harm_keywords(self):
        for text in ["suicide and self harm", "cut myself and suicide", "kill myself self harm"]:
            resp = TestClient(app).post(
                "/v1/moderations",
                json={"input": text},
            )
            result = resp.json()["results"][0]
            assert result["flagged"] is True, f"should flag: {text}"
            assert result["categories"]["self_harm"] is True, f"self_harm category: {text}"

    def test_heuristic_sexual_keywords(self):
        for text in ["explicit nude content", "naked sex", "porn and explicit material"]:
            resp = TestClient(app).post(
                "/v1/moderations",
                json={"input": text},
            )
            result = resp.json()["results"][0]
            assert result["flagged"] is True, f"should flag: {text}"
            assert result["categories"]["sexual"] is True, f"sexual category: {text}"

    def test_heuristic_violence_keywords(self):
        for text in ["kill and murder", "torture and violence", "murder with a weapon"]:
            resp = TestClient(app).post(
                "/v1/moderations",
                json={"input": text},
            )
            result = resp.json()["results"][0]
            assert result["flagged"] is True, f"should flag: {text}"
            assert result["categories"]["violence"] is True, f"violence category: {text}"

    def test_moderate_multiple_inputs(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": ["The weather is nice.", "I will kill and murder you with violence.", "Hello world"]},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 3
        assert results[0]["flagged"] is False
        assert results[1]["flagged"] is True
        assert results[2]["flagged"] is False

    def test_moderate_with_model(self):
        from distllm.api.routes.moderations import ModerationResult, ModerationCategories, ModerationCategoryScores
        expected = ModerationResult(
            flagged=True,
            categories=ModerationCategories(hate=True, violence=True),
            category_scores=ModerationCategoryScores(hate=0.95, violence=0.85),
        )
        self.coord._moderation_model = MagicMock()
        with patch("distllm.api.routes.moderations._moderate_with_model", return_value=[expected]):
            resp = TestClient(app).post(
                "/v1/moderations",
                json={"input": "some text"},
            )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"]["hate"] is True
        assert result["categories"]["violence"] is True
        assert result["category_scores"]["hate"] == 0.95
        assert result["category_scores"]["violence"] == 0.85

    def test_heuristic_fallback(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "I want to commit suicide and self harm."},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"]["self_harm"] is True
        assert result["category_scores"]["self_harm"] > 0.5

    def test_threshold_below(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "explicitly"},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is False
        assert result["categories"]["sexual"] is False
        assert 0 < result["category_scores"]["sexual"] < 0.5

    def test_threshold_above(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "explicitly explicit sex"},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"]["sexual"] is True
        assert result["category_scores"]["sexual"] > 0.5

    def test_all_categories_populated(self):
        resp = TestClient(app).post(
            "/v1/moderations",
            json={"input": "explicit sex and hate speech slur kill murder"},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        cats = result["categories"]
        scores = result["category_scores"]
        expected = [
            "sexual", "hate", "harassment", "self_harm",
            "sexual_minors", "hate_threatening", "violence_graphic",
            "self_harm_intent", "self_harm_instructions",
            "harassment_threatening", "violence",
        ]
        for cat in expected:
            assert cat in cats, f"missing category: {cat}"
            assert cat in scores, f"missing category_scores: {cat}"

    def test_moderate_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/moderations",
                json={"input": "test"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original
