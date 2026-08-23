"""Tests for distllm.api.routes.exchange — prompt exchange endpoints.

Covers all 15 endpoints with:
- Success paths (exchange coordinator set)
- 503 error paths (exchange not available)
- Endpoint-specific error paths (404, 402, 400, 401, 403, 422)
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.api_state import _state, reset_app_state_for_testing
from distllm.api.auth_deps import require_role
from distllm.api.routes.exchange import router as exchange_router
from distllm.prompts.exchange import (
    PromptLicense,
    PromptMetrics,
    PromptReview,
    PromptStatus,
    PublishedPrompt,
    UserWallet,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock exchange coordinator (no MagicMock / Mock / AsyncMock)
# ═══════════════════════════════════════════════════════════════════════════════

class ExchangeMock:
    """Minimal in-memory PromptExchange mock using real dataclasses."""

    def __init__(self) -> None:
        self._id = 0
        self.prompts: dict[str, PublishedPrompt] = {}
        self.reviews: dict[str, list[PromptReview]] = {}
        self.wallets: dict[str, UserWallet] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _next_id(self) -> str:
        self._id += 1
        return f"p-{self._id}"

    # ── publishing ────────────────────────────────────────────────────────

    def publish_prompt(
        self,
        author_id: str,
        name: str,
        description: str,
        category: str,
        system_prompt: str,
        tags: list[str] | None = None,
        license: PromptLicense = PromptLicense.FREE,
        price_tokens: int = 0,
        examples: list[dict[str, str]] | None = None,
        parent_id: str = "",
    ) -> PublishedPrompt:
        pid = self._next_id()
        prompt = PublishedPrompt(
            prompt_id=pid,
            author_id=author_id,
            name=name,
            description=description,
            category=category,
            system_prompt=system_prompt,
            tags=tags or [],
            license=license,
            price_tokens=price_tokens,
            examples=examples or [],
            parent_id=parent_id,
        )
        self.prompts[pid] = prompt
        return prompt

    # ── discovery ─────────────────────────────────────────────────────────

    def browse(
        self,
        category: str = "",
        tags: list[str] | None = None,
        license: PromptLicense | None = None,
        author_id: str = "",
        sort_by: str = "popular",
        limit: int = 50,
        offset: int = 0,
    ) -> list[PublishedPrompt]:
        results = [
            p
            for p in self.prompts.values()
            if p.status in (PromptStatus.PUBLISHED, PromptStatus.FEATURED)
        ]
        if category:
            results = [p for p in results if p.category == category]
        if tags:
            tag_set = set(tags)
            results = [p for p in results if tag_set.intersection(p.tags)]
        if license:
            results = [p for p in results if p.license == license]
        if author_id:
            results = [p for p in results if p.author_id == author_id]
        return results[offset : offset + limit]

    def search(self, q: str, limit: int = 20) -> list[PublishedPrompt]:
        q = q.lower()
        scored: list[tuple[int, PublishedPrompt]] = []
        for p in self.prompts.values():
            score = 0
            if q in p.name.lower():
                score += 3
            if q in p.description.lower():
                score += 2
            if any(q in t.lower() for t in p.tags):
                score += 1
            if score:
                scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:limit]]

    def get_prompt(self, prompt_id: str) -> PublishedPrompt | None:
        return self.prompts.get(prompt_id)

    # ── acquisition ───────────────────────────────────────────────────────

    def acquire_prompt(
        self, user_id: str, prompt_id: str
    ) -> PublishedPrompt | None:
        p = self.prompts.get(prompt_id)
        if p is None:
            return None
        if p.price_tokens > 0:
            w = self.wallets.get(user_id, UserWallet(user_id=user_id))
            if w.balance_tokens < p.price_tokens:
                return None
            w.balance_tokens -= p.price_tokens
            w.total_spent += p.price_tokens
        return p

    def has_access(self, user_id: str, prompt_id: str) -> bool:
        return prompt_id in self.prompts

    # ── reviews ───────────────────────────────────────────────────────────

    def add_review(
        self,
        user_id: str,
        prompt_id: str,
        rating: int,
        comment: str = "",
    ) -> PromptReview | None:
        p = self.prompts.get(prompt_id)
        if p is None or not (1 <= rating <= 5):
            return None
        r = PromptReview(
            review_id=uuid.uuid4().hex[:12],
            prompt_id=prompt_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )
        self.reviews.setdefault(prompt_id, []).append(r)
        p.metrics.rating_count = len(self.reviews[prompt_id])
        p.metrics.avg_rating = (
            sum(rv.rating for rv in self.reviews[prompt_id])
            / p.metrics.rating_count
        )
        return r

    def get_reviews(
        self, prompt_id: str, limit: int = 20
    ) -> list[PromptReview]:
        return sorted(
            self.reviews.get(prompt_id, []),
            key=lambda r: r.helpful_count,
            reverse=True,
        )[:limit]

    # ── usage ─────────────────────────────────────────────────────────────

    def record_usage(
        self,
        prompt_id: str,
        throughput_tok_s: float = 0,
        latency_ms: float = 0,
        tokens_generated: int = 0,
        cost_usd: float = 0,
        quality_score: float = 0,
    ) -> None:
        p = self.prompts.get(prompt_id)
        if p is None:
            return
        m = p.metrics
        n = m.total_uses
        if n == 0:
            m.avg_throughput_tok_s = throughput_tok_s
            m.avg_latency_ms = latency_ms
        else:
            m.avg_throughput_tok_s = (
                m.avg_throughput_tok_s * n + throughput_tok_s
            ) / (n + 1)
            m.avg_latency_ms = (m.avg_latency_ms * n + latency_ms) / (n + 1)
        m.total_tokens_generated += tokens_generated

    # ── forking ───────────────────────────────────────────────────────────

    def fork_prompt(
        self,
        user_id: str,
        prompt_id: str,
        new_name: str = "",
    ) -> PublishedPrompt | None:
        original = self.prompts.get(prompt_id)
        if original is None:
            return None
        return self.publish_prompt(
            author_id=user_id,
            name=new_name or f"Fork of {original.name}",
            description=(
                f"Forked from {original.name} by {original.author_id}"
            ),
            category=original.category,
            system_prompt=original.system_prompt,
            tags=list(original.tags),
            license=original.license,
            price_tokens=original.price_tokens,
            parent_id=prompt_id,
        )

    # ── analytics ─────────────────────────────────────────────────────────

    def get_leaderboard(
        self,
        category: str = "",
        metric: str = "uses",
        limit: int = 20,
    ) -> list[dict]:
        results = [
            p
            for p in self.prompts.values()
            if p.status in (PromptStatus.PUBLISHED, PromptStatus.FEATURED)
        ]
        if category:
            results = [p for p in results if p.category == category]
        if metric == "rating":
            results.sort(key=lambda p: p.metrics.avg_rating, reverse=True)
        else:
            results.sort(key=lambda p: p.metrics.total_uses, reverse=True)
        return [
            {
                "prompt_id": p.prompt_id,
                "name": p.name,
                "author_id": p.author_id,
                "category": p.category,
                "license": p.license.value,
                "total_uses": p.metrics.total_uses,
                "avg_rating": p.metrics.avg_rating,
                "avg_throughput_tok_s": p.metrics.avg_throughput_tok_s,
                "avg_latency_ms": p.metrics.avg_latency_ms,
            }
            for p in results[:limit]
        ]

    def get_exchange_stats(self) -> dict:
        published = [
            p
            for p in self.prompts.values()
            if p.status in (PromptStatus.PUBLISHED, PromptStatus.FEATURED)
        ]
        cats: dict[str, int] = {}
        for p in published:
            cats[p.category] = cats.get(p.category, 0) + 1
        return {
            "total_prompts": len(published),
            "free_prompts": sum(1 for p in published if p.is_free),
            "premium_prompts": sum(1 for p in published if not p.is_free),
            "total_uses": sum(p.metrics.total_uses for p in published),
            "total_tokens_generated": sum(
                p.metrics.total_tokens_generated for p in published
            ),
            "total_wallets": len(self.wallets),
            "total_reviews": sum(len(r) for r in self.reviews.values()),
            "categories": cats,
        }

    # ── wallet ────────────────────────────────────────────────────────────

    def get_wallet(self, user_id: str) -> UserWallet:
        if user_id not in self.wallets:
            self.wallets[user_id] = UserWallet(user_id=user_id)
        return self.wallets[user_id]

    def top_up(self, user_id: str, amount: int) -> UserWallet:
        w = self.get_wallet(user_id)
        w.balance_tokens += amount
        return w

    def get_user_library(self, user_id: str) -> list[PublishedPrompt]:
        return list(self.prompts.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_app_state():
    """Reset shared AppState before every test."""
    reset_app_state_for_testing()


@pytest.fixture
def exchange():
    """Create an ExchangeMock and set it as the prompt_exchange coordinator."""
    ex = ExchangeMock()
    _state.prompt_exchange = ex
    return ex


def _build_app(with_admin: bool = False) -> FastAPI:
    """Minimal FastAPI app with exchange router.

    When *with_admin* is True, adds middleware that sets ``api_key_role``
    to ``"admin"`` so the ``require_role("admin")`` dep passes.
    """
    app = FastAPI()

    if with_admin:

        class _AdminMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.api_key_role = "admin"
                return await call_next(request)

        app.add_middleware(_AdminMiddleware)

    app.include_router(exchange_router)
    return app


@pytest.fixture
def client():
    """TestClient where admin-protected endpoints will pass (admin middleware)."""
    return TestClient(_build_app(with_admin=True))


@pytest.fixture
def auth_client():
    """TestClient with admin middleware AND acquire auth state."""
    app = _build_app(with_admin=True)

    class _AcquireAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.api_key_role = "admin"
            request.state.api_key_id = "key-1"
            request.state.api_key_owner = "owner-1"
            return await call_next(request)

    app.add_middleware(_AcquireAuth)
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

PUBLISH_BODY = {
    "author_id": "alice",
    "name": "Code Reviewer",
    "description": "Reviews pull requests",
    "category": "code",
    "system_prompt": "You are a code reviewer.",
    "tags": ["code-review", "python"],
}

REVIEW_BODY = {"user_id": "bob", "rating": 5, "comment": "Great prompt!"}

USAGE_BODY = {
    "throughput_tok_s": 150.0,
    "latency_ms": 45.0,
    "tokens_generated": 500,
}


# ── 503: exchange not available ──────────────────────────────────────────────

class Test503NoExchange:
    """All 15 endpoints return 503 when prompt_exchange is not set."""

    def test_publish(self, client):
        resp = client.post("/v1/exchange/prompts", json=PUBLISH_BODY)
        assert resp.status_code == 503
        assert "not available" in resp.json()["detail"]

    def test_browse(self, client):
        resp = client.get("/v1/exchange/prompts")
        assert resp.status_code == 503

    def test_search(self, client):
        resp = client.get("/v1/exchange/prompts/search?q=test")
        assert resp.status_code == 503

    def test_get_prompt(self, client):
        resp = client.get("/v1/exchange/prompts/p-1")
        assert resp.status_code == 503

    def test_acquire(self, client):
        resp = client.post("/v1/exchange/prompts/p-1/acquire?user_id=u")
        assert resp.status_code == 503

    def test_check_access(self, client):
        resp = client.get("/v1/exchange/prompts/p-1/access?user_id=u")
        assert resp.status_code == 503

    def test_add_review(self, client):
        resp = client.post(
            "/v1/exchange/prompts/p-1/review", json=REVIEW_BODY
        )
        assert resp.status_code == 503

    def test_get_reviews(self, client):
        resp = client.get("/v1/exchange/prompts/p-1/reviews")
        assert resp.status_code == 503

    def test_record_usage(self, client):
        resp = client.post("/v1/exchange/prompts/p-1/usage", json=USAGE_BODY)
        assert resp.status_code == 503

    def test_fork(self, client):
        resp = client.post("/v1/exchange/prompts/p-1/fork?user_id=u")
        assert resp.status_code == 503

    def test_leaderboard(self, client):
        resp = client.get("/v1/exchange/leaderboard")
        assert resp.status_code == 503

    def test_stats(self, client):
        resp = client.get("/v1/exchange/stats")
        assert resp.status_code == 503

    def test_wallet(self, client):
        resp = client.get("/v1/exchange/wallet/u")
        assert resp.status_code == 503

    def test_topup(self, client):
        resp = client.post("/v1/exchange/wallet/u/topup?amount=100")
        assert resp.status_code == 503

    def test_library(self, client):
        resp = client.get("/v1/exchange/library/u")
        assert resp.status_code == 503


# ── POST /v1/exchange/prompts ────────────────────────────────────────────────
#   Admin-only: publishes a prompt to the exchange.

class TestPublishPrompt:
    def test_success(self, client, exchange):
        resp = client.post("/v1/exchange/prompts", json=PUBLISH_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Code Reviewer"
        assert data["author_id"] == "alice"
        assert data["category"] == "code"
        assert data["license"] == "free"
        assert data["status"] == "published"
        assert data["version"] == 1
        assert data["price_tokens"] == 0
        assert data["prompt_id"] in exchange.prompts

    def test_invalid_license_falls_back_to_free(self, client, exchange):
        body = {**PUBLISH_BODY, "license": "bogus"}
        resp = client.post("/v1/exchange/prompts", json=body)
        assert resp.status_code == 200
        assert resp.json()["license"] == "free"

    def test_premium_prompt(self, client, exchange):
        body = {**PUBLISH_BODY, "license": "premium", "price_tokens": 500}
        resp = client.post("/v1/exchange/prompts", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["license"] == "premium"
        assert data["price_tokens"] == 500

    def test_missing_required_fields_returns_422(self, client, exchange):
        resp = client.post("/v1/exchange/prompts", json={})
        assert resp.status_code == 422

    def test_negative_price_returns_422(self, client, exchange):
        body = {**PUBLISH_BODY, "price_tokens": -1}
        resp = client.post("/v1/exchange/prompts", json=body)
        assert resp.status_code == 422


# ── GET /v1/exchange/prompts ─────────────────────────────────────────────────
#   Public: browse published prompts.

class TestBrowsePrompts:
    def test_empty(self, client, exchange):
        resp = client.get("/v1/exchange/prompts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_data(self, client, exchange):
        exchange.publish_prompt(
            author_id="alice",
            name="Reviewer",
            description="Reviews code",
            category="code",
            system_prompt="Review code.",
        )
        resp = client.get("/v1/exchange/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Reviewer"

    def test_category_filter(self, client, exchange):
        exchange.publish_prompt(
            author_id="a", name="Cat1", description="d", category="code",
            system_prompt="sp",
        )
        exchange.publish_prompt(
            author_id="b", name="Cat2", description="d", category="writing",
            system_prompt="sp",
        )
        resp = client.get("/v1/exchange/prompts?category=writing")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["category"] == "writing"

    def test_author_filter(self, client, exchange):
        exchange.publish_prompt(
            author_id="alice", name="P1", description="d", category="code",
            system_prompt="sp",
        )
        exchange.publish_prompt(
            author_id="bob", name="P2", description="d", category="code",
            system_prompt="sp",
        )
        resp = client.get("/v1/exchange/prompts?author_id=alice")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_limit_and_offset(self, client, exchange):
        for i in range(5):
            exchange.publish_prompt(
                author_id="a", name=f"N{i}", description="d",
                category="code", system_prompt="sp",
            )
        resp = client.get("/v1/exchange/prompts?limit=2&offset=1")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_invalid_license_ignored(self, client, exchange):
        exchange.publish_prompt(
            author_id="a", name="N", description="d", category="code",
            system_prompt="sp",
        )
        resp = client.get("/v1/exchange/prompts?license=invalid")
        assert resp.status_code == 200


# ── GET /v1/exchange/prompts/search ──────────────────────────────────────────
#   Public: full-text search.

class TestSearchPrompts:
    def test_search(self, client, exchange):
        exchange.publish_prompt(
            author_id="a", name="Python Linter", description="Lints Python code",
            category="code", system_prompt="Lint.", tags=["python", "lint"],
        )
        exchange.publish_prompt(
            author_id="b", name="Go Builder", description="Builds Go projects",
            category="code", system_prompt="Build.", tags=["go"],
        )
        resp = client.get("/v1/exchange/prompts/search?q=python")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Python Linter"

    def test_no_results(self, client, exchange):
        resp = client.get("/v1/exchange/prompts/search?q=nonexistent")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_missing_query_returns_422(self, client, exchange):
        resp = client.get("/v1/exchange/prompts/search")
        assert resp.status_code == 422


# ── GET /v1/exchange/prompts/{prompt_id} ─────────────────────────────────────
#   Public: get a single prompt.

class TestGetPrompt:
    def test_success(self, client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="My Prompt", description="d",
            category="code", system_prompt="You are helpful.",
        )
        resp = client.get(f"/v1/exchange/prompts/{prompt.prompt_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Prompt"

    def test_not_found(self, client, exchange):
        resp = client.get("/v1/exchange/prompts/nonexistent")
        assert resp.status_code == 404


# ── POST /v1/exchange/prompts/{prompt_id}/acquire ────────────────────────────
#   Manual auth: acquires a prompt (free or token-gated).

class TestAcquirePrompt:
    def test_success_free(self, auth_client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="Free Prompt", description="d",
            category="code", system_prompt="Free.",
        )
        resp = auth_client.post(
            f"/v1/exchange/prompts/{prompt.prompt_id}/acquire",
            params={"user_id": "owner-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prompt_id"] == prompt.prompt_id
        assert data["acquired"] is True

    def test_not_found(self, auth_client, exchange):
        resp = auth_client.post(
            "/v1/exchange/prompts/nonexistent/acquire",
            params={"user_id": "owner-1"},
        )
        assert resp.status_code == 402
        assert "Insufficient" in resp.json()["detail"]

    def test_owner_mismatch_returns_403(self, auth_client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="P", description="d",
            category="code", system_prompt="SP.",
        )
        # api_key_owner is "owner-1", but we pass a different user_id
        resp = auth_client.post(
            f"/v1/exchange/prompts/{prompt.prompt_id}/acquire",
            params={"user_id": "different-user"},
        )
        assert resp.status_code == 403


# ── GET /v1/exchange/prompts/{prompt_id}/access ──────────────────────────────
#   Public: check if a user has access.

class TestCheckAccess:
    def test_has_access(self, client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="P", description="d",
            category="code", system_prompt="SP.",
        )
        resp = client.get(
            f"/v1/exchange/prompts/{prompt.prompt_id}/access",
            params={"user_id": "alice"},
        )
        assert resp.status_code == 200
        assert resp.json()["has_access"] is True

    def test_no_access(self, client, exchange):
        resp = client.get(
            "/v1/exchange/prompts/nonexistent/access",
            params={"user_id": "alice"},
        )
        assert resp.status_code == 200
        assert resp.json()["has_access"] is False


# ── POST /v1/exchange/prompts/{prompt_id}/review ─────────────────────────────
#   Public: add a review.

class TestAddReview:
    def test_success(self, client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="P", description="d",
            category="code", system_prompt="SP.",
        )
        resp = client.post(
            f"/v1/exchange/prompts/{prompt.prompt_id}/review",
            json={"user_id": "bob", "rating": 4, "comment": "Nice!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["review_id"] is not None
        assert data["rating"] == 4

    def test_missing_prompt_returns_400(self, client, exchange):
        resp = client.post(
            "/v1/exchange/prompts/nonexistent/review",
            json={"user_id": "bob", "rating": 5},
        )
        assert resp.status_code == 400

    def test_invalid_rating_returns_422(self, client, exchange):
        resp = client.post(
            "/v1/exchange/prompts/p-1/review",
            json={"user_id": "bob", "rating": 99},
        )
        assert resp.status_code == 422


# ── GET /v1/exchange/prompts/{prompt_id}/reviews ─────────────────────────────
#   Public: get reviews for a prompt.

class TestGetReviews:
    def test_empty(self, client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="P", description="d",
            category="code", system_prompt="SP.",
        )
        resp = client.get(
            f"/v1/exchange/prompts/{prompt.prompt_id}/reviews"
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_reviews(self, client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="P", description="d",
            category="code", system_prompt="SP.",
        )
        exchange.add_review("bob", prompt.prompt_id, 5, "Great!")
        resp = client.get(
            f"/v1/exchange/prompts/{prompt.prompt_id}/reviews"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["rating"] == 5
        assert data[0]["comment"] == "Great!"


# ── POST /v1/exchange/prompts/{prompt_id}/usage ──────────────────────────────
#   Public: record usage metrics.

class TestRecordUsage:
    def test_success(self, client, exchange):
        prompt = exchange.publish_prompt(
            author_id="a", name="P", description="d",
            category="code", system_prompt="SP.",
        )
        resp = client.post(
            f"/v1/exchange/prompts/{prompt.prompt_id}/usage",
            json=USAGE_BODY,
        )
        assert resp.status_code == 200
        assert resp.json() == {"recorded": True, "prompt_id": prompt.prompt_id}


# ── POST /v1/exchange/prompts/{prompt_id}/fork ───────────────────────────────
#   Admin-only: fork a prompt.

class TestForkPrompt:
    def test_success(self, client, exchange):
        original = exchange.publish_prompt(
            author_id="alice", name="Original", description="d",
            category="code", system_prompt="Original prompt.",
        )
        resp = client.post(
            f"/v1/exchange/prompts/{original.prompt_id}/fork",
            params={"user_id": "bob", "new_name": "Forked"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Forked"
        assert data["author_id"] == "bob"

    def test_not_found(self, client, exchange):
        resp = client.post(
            "/v1/exchange/prompts/nonexistent/fork",
            params={"user_id": "bob"},
        )
        assert resp.status_code == 404


# ── GET /v1/exchange/leaderboard ─────────────────────────────────────────────
#   Public: get the leaderboard.

class TestLeaderboard:
    def test_empty(self, client, exchange):
        resp = client.get("/v1/exchange/leaderboard")
        assert resp.status_code == 200
        assert resp.json() == {"leaderboard": []}

    def test_with_data(self, client, exchange):
        exchange.publish_prompt(
            author_id="a", name="Top Prompt", description="d",
            category="code", system_prompt="SP.",
        )
        resp = client.get("/v1/exchange/leaderboard")
        assert resp.status_code == 200
        data = resp.json()["leaderboard"]
        assert len(data) == 1
        assert data[0]["name"] == "Top Prompt"

    def test_category_filter(self, client, exchange):
        exchange.publish_prompt(
            author_id="a", name="P1", description="d", category="code",
            system_prompt="SP.",
        )
        exchange.publish_prompt(
            author_id="b", name="P2", description="d", category="writing",
            system_prompt="SP.",
        )
        resp = client.get("/v1/exchange/leaderboard?category=writing")
        assert resp.status_code == 200
        assert len(resp.json()["leaderboard"]) == 1


# ── GET /v1/exchange/stats ───────────────────────────────────────────────────
#   Public: get exchange statistics.

class TestExchangeStats:
    def test_empty(self, client, exchange):
        resp = client.get("/v1/exchange/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_prompts"] == 0
        assert data["free_prompts"] == 0

    def test_with_data(self, client, exchange):
        exchange.publish_prompt(
            author_id="a", name="P1", description="d", category="code",
            system_prompt="SP.", license=PromptLicense.FREE,
        )
        exchange.publish_prompt(
            author_id="b", name="P2", description="d", category="writing",
            system_prompt="SP.", license=PromptLicense.PREMIUM,
            price_tokens=100,
        )
        resp = client.get("/v1/exchange/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_prompts"] == 2
        assert data["free_prompts"] == 1
        assert data["premium_prompts"] == 1
        assert data["categories"]["code"] == 1
        assert data["categories"]["writing"] == 1


# ── GET /v1/exchange/wallet/{user_id} ────────────────────────────────────────
#   Public: get a user's wallet.

class TestWallet:
    def test_success(self, client, exchange):
        resp = client.get("/v1/exchange/wallet/bob")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "bob"
        assert data["balance_tokens"] == 0
        assert data["total_earned"] == 0
        assert data["total_spent"] == 0


# ── POST /v1/exchange/wallet/{user_id}/topup ─────────────────────────────────
#   Admin-only: top up a user's wallet.

class TestTopUpWallet:
    def test_success(self, client, exchange):
        resp = client.post(
            "/v1/exchange/wallet/bob/topup", params={"amount": 1000}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "bob"
        assert data["balance_tokens"] == 1000
        assert data["added"] == 1000

    def test_multiple_topups(self, client, exchange):
        client.post("/v1/exchange/wallet/bob/topup", params={"amount": 500})
        resp = client.post(
            "/v1/exchange/wallet/bob/topup", params={"amount": 300}
        )
        assert resp.status_code == 200
        assert resp.json()["balance_tokens"] == 800

    def test_non_positive_amount_returns_422(self, client, exchange):
        resp = client.post(
            "/v1/exchange/wallet/bob/topup", params={"amount": 0}
        )
        assert resp.status_code == 422

        resp = client.post(
            "/v1/exchange/wallet/bob/topup", params={"amount": -50}
        )
        assert resp.status_code == 422


# ── GET /v1/exchange/library/{user_id} ───────────────────────────────────────
#   Public: get a user's library of accessible prompts.

class TestUserLibrary:
    def test_empty(self, client, exchange):
        resp = client.get("/v1/exchange/library/alice")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_prompts(self, client, exchange):
        exchange.publish_prompt(
            author_id="alice", name="My Prompt", description="d",
            category="code", system_prompt="SP.",
        )
        resp = client.get("/v1/exchange/library/alice")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "My Prompt"
