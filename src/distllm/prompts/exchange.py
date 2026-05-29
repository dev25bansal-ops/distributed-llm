"""Prompt Exchange — community prompt marketplace with token-gating.

Combines the prompt library, marketplace, and leaderboard into a
unified prompt exchange where users can:
- Share and discover high-quality prompts
- Token-gate premium prompts (pay per use or subscription)
- Rate and review prompts
- Track prompt performance (throughput, quality, cost)
- Fork and remix prompts

This is novel — no LLM server has a built-in prompt marketplace.

Usage::

    exchange = PromptExchange()
    exchange.publish_prompt(author="alice", prompt_def=my_prompt, price_tokens=100)
    prompt = exchange.acquire_prompt(user="bob", prompt_id="code-review-v2")
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class PromptLicense(str, Enum):
    """Prompt licensing model."""
    FREE = "free"               # Open source, no restrictions
    ATTRIBUTION = "attribution" # Free with attribution required
    PREMIUM = "premium"         # Paid per-use
    SUBSCRIPTION = "subscription"  # Paid monthly subscription
    ENTERPRISE = "enterprise"   # Custom pricing


class PromptStatus(str, Enum):
    """Prompt publication status."""
    DRAFT = "draft"
    PUBLISHED = "published"
    FEATURED = "featured"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass
class PromptMetrics:
    """Performance metrics for a prompt."""
    total_uses: int = 0
    avg_throughput_tok_s: float = 0.0
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    avg_cost_usd: float = 0.0
    total_tokens_generated: int = 0
    unique_users: int = 0
    avg_rating: float = 0.0
    rating_count: int = 0


@dataclass
class PromptReview:
    """A user review of a prompt."""
    review_id: str
    prompt_id: str
    user_id: str
    rating: int  # 1-5 stars
    comment: str = ""
    created_at: float = field(default_factory=time.time)
    helpful_count: int = 0


@dataclass
class PublishedPrompt:
    """A prompt published to the exchange."""
    prompt_id: str
    author_id: str
    name: str
    description: str
    category: str
    system_prompt: str
    tags: list[str] = field(default_factory=list)
    license: PromptLicense = PromptLicense.FREE
    price_tokens: int = 0  # Cost in tokens (0 = free)
    status: PromptStatus = PromptStatus.PUBLISHED
    version: int = 1
    parent_id: str = ""  # For forks/remixes
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metrics: PromptMetrics = field(default_factory=PromptMetrics)
    examples: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.system_prompt.encode()).hexdigest()[:16]

    @property
    def is_free(self) -> bool:
        return self.license == PromptLicense.FREE or self.price_tokens == 0


@dataclass
class UserWallet:
    """Token wallet for prompt exchange transactions."""
    user_id: str
    balance_tokens: int = 0
    total_earned: int = 0
    total_spent: int = 0
    total_purchased: int = 0
    created_at: float = field(default_factory=time.time)


class PromptExchange:
    """Community prompt marketplace with token-gating.

    Enables users to publish, discover, acquire, and review prompts.
    Supports free, attribution, premium, and subscription licensing.
    """

    def __init__(self):
        self._prompts: dict[str, PublishedPrompt] = {}
        self._wallets: dict[str, UserWallet] = {}
        self._reviews: dict[str, list[PromptReview]] = {}  # prompt_id -> reviews
        self._user_purchases: dict[str, set[str]] = {}  # user_id -> set of prompt_ids
        self._user_library: dict[str, list[str]] = {}  # user_id -> list of prompt_ids
        self._featured: list[str] = []

    # ── Publishing ───────────────────────────────────────────────────

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
        """Publish a prompt to the exchange.

        Args:
            author_id: Publisher's user ID.
            name: Prompt display name.
            description: What the prompt does.
            category: Category (code, writing, analysis, etc.).
            system_prompt: The actual system prompt text.
            tags: Searchable tags.
            license: Licensing model.
            price_tokens: Cost in tokens (0 for free).
            examples: Example input/output pairs.
            parent_id: ID of parent prompt (for forks).

        Returns:
            The published PublishedPrompt.
        """
        prompt_id = f"{name.lower().replace(' ', '-')}-v{1}-{uuid.uuid4().hex[:8]}"

        prompt = PublishedPrompt(
            prompt_id=prompt_id,
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

        self._prompts[prompt_id] = prompt
        logger.info(f"Prompt published: {prompt_id} by {author_id} ({license.value})")
        return prompt

    def update_prompt(
        self,
        prompt_id: str,
        author_id: str,
        system_prompt: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        price_tokens: int | None = None,
    ) -> PublishedPrompt | None:
        """Update an existing prompt (author only)."""
        prompt = self._prompts.get(prompt_id)
        if prompt is None or prompt.author_id != author_id:
            return None

        if system_prompt is not None:
            prompt.system_prompt = system_prompt
            prompt.version += 1
        if description is not None:
            prompt.description = description
        if tags is not None:
            prompt.tags = tags
        if price_tokens is not None:
            prompt.price_tokens = price_tokens
        prompt.updated_at = time.time()
        return prompt

    # ── Discovery ────────────────────────────────────────────────────

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
        """Browse published prompts with filters."""
        results = [
            p for p in self._prompts.values()
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

        if sort_by == "popular":
            results.sort(key=lambda p: p.metrics.total_uses, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda p: p.metrics.avg_rating, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda p: p.created_at, reverse=True)
        elif sort_by == "trending":
            results.sort(key=lambda p: p.metrics.unique_users, reverse=True)

        return results[offset:offset + limit]

    def search(self, query: str, limit: int = 20) -> list[PublishedPrompt]:
        """Full-text search across prompt names, descriptions, and tags."""
        q = query.lower()
        results = []
        for p in self._prompts.values():
            if p.status not in (PromptStatus.PUBLISHED, PromptStatus.FEATURED):
                continue
            score = 0
            if q in p.name.lower():
                score += 3
            if q in p.description.lower():
                score += 2
            if any(q in t.lower() for t in p.tags):
                score += 1
            if q in p.system_prompt.lower():
                score += 1
            if score > 0:
                results.append((score, p))
        results.sort(key=lambda x: -x[0])
        return [p for _, p in results[:limit]]

    def get_prompt(self, prompt_id: str) -> PublishedPrompt | None:
        return self._prompts.get(prompt_id)

    def get_featured(self) -> list[PublishedPrompt]:
        return [self._prompts[pid] for pid in self._featured if pid in self._prompts]

    # ── Acquisition ──────────────────────────────────────────────────

    def acquire_prompt(
        self,
        user_id: str,
        prompt_id: str,
    ) -> PublishedPrompt | None:
        """Acquire a prompt (free purchase or token deduction).

        Returns the prompt if successful, None if insufficient funds
        or prompt not found.
        """
        prompt = self._prompts.get(prompt_id)
        if prompt is None:
            return None

        if prompt.is_free:
            self._record_purchase(user_id, prompt_id)
            prompt.metrics.total_uses += 1
            return prompt

        wallet = self._get_or_create_wallet(user_id)
        if wallet.balance_tokens < prompt.price_tokens:
            logger.warning(f"Insufficient tokens: {user_id} has {wallet.balance_tokens}, needs {prompt.price_tokens}")
            return None

        wallet.balance_tokens -= prompt.price_tokens
        wallet.total_spent += prompt.price_tokens

        author_wallet = self._get_or_create_wallet(prompt.author_id)
        author_wallet.balance_tokens += prompt.price_tokens
        author_wallet.total_earned += prompt.price_tokens

        self._record_purchase(user_id, prompt_id)
        prompt.metrics.total_uses += 1
        logger.info(f"Prompt acquired: {user_id} purchased {prompt_id} for {prompt.price_tokens} tokens")
        return prompt

    def _record_purchase(self, user_id: str, prompt_id: str) -> None:
        if user_id not in self._user_purchases:
            self._user_purchases[user_id] = set()
        self._user_purchases[user_id].add(prompt_id)
        if user_id not in self._user_library:
            self._user_library[user_id] = []
        if prompt_id not in self._user_library[user_id]:
            self._user_library[user_id].append(prompt_id)

    def has_access(self, user_id: str, prompt_id: str) -> bool:
        """Check if a user has access to a prompt."""
        prompt = self._prompts.get(prompt_id)
        if prompt is None:
            return False
        if prompt.is_free:
            return True
        if prompt.author_id == user_id:
            return True
        purchases = self._user_purchases.get(user_id, set())
        return prompt_id in purchases

    def get_user_library(self, user_id: str) -> list[PublishedPrompt]:
        """Get all prompts a user has access to."""
        prompt_ids = self._user_library.get(user_id, [])
        return [self._prompts[pid] for pid in prompt_ids if pid in self._prompts]

    # ── Reviews & Ratings ────────────────────────────────────────────

    def add_review(
        self,
        user_id: str,
        prompt_id: str,
        rating: int,
        comment: str = "",
    ) -> PromptReview | None:
        """Add a review for a prompt."""
        prompt = self._prompts.get(prompt_id)
        if prompt is None:
            return None
        if not (1 <= rating <= 5):
            return None

        review = PromptReview(
            review_id=uuid.uuid4().hex[:12],
            prompt_id=prompt_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )

        if prompt_id not in self._reviews:
            self._reviews[prompt_id] = []
        self._reviews[prompt_id].append(review)

        reviews = self._reviews[prompt_id]
        prompt.metrics.rating_count = len(reviews)
        prompt.metrics.avg_rating = sum(r.rating for r in reviews) / len(reviews)

        return review

    def get_reviews(self, prompt_id: str, limit: int = 20) -> list[PromptReview]:
        reviews = self._reviews.get(prompt_id, [])
        reviews.sort(key=lambda r: r.helpful_count, reverse=True)
        return reviews[:limit]

    # ── Wallet ───────────────────────────────────────────────────────

    def _get_or_create_wallet(self, user_id: str) -> UserWallet:
        if user_id not in self._wallets:
            self._wallets[user_id] = UserWallet(user_id=user_id)
        return self._wallets[user_id]

    def get_wallet(self, user_id: str) -> UserWallet:
        return self._get_or_create_wallet(user_id)

    def top_up(self, user_id: str, amount_tokens: int) -> UserWallet:
        wallet = self._get_or_create_wallet(user_id)
        wallet.balance_tokens += amount_tokens
        return wallet

    # ── Analytics ────────────────────────────────────────────────────

    def record_usage(
        self,
        prompt_id: str,
        throughput_tok_s: float,
        latency_ms: float,
        tokens_generated: int,
        cost_usd: float = 0.0,
        quality_score: float = 0.0,
    ) -> None:
        """Record usage metrics for a prompt."""
        prompt = self._prompts.get(prompt_id)
        if prompt is None:
            return

        m = prompt.metrics
        n = m.total_uses
        if n == 0:
            m.avg_throughput_tok_s = throughput_tok_s
            m.avg_latency_ms = latency_ms
            m.avg_cost_usd = cost_usd
            m.avg_quality_score = quality_score
        else:
            m.avg_throughput_tok_s = (m.avg_throughput_tok_s * n + throughput_tok_s) / (n + 1)
            m.avg_latency_ms = (m.avg_latency_ms * n + latency_ms) / (n + 1)
            m.avg_cost_usd = (m.avg_cost_usd * n + cost_usd) / (n + 1)
            if quality_score > 0:
                m.avg_quality_score = (m.avg_quality_score * n + quality_score) / (n + 1)
        m.total_tokens_generated += tokens_generated

    def get_leaderboard(
        self,
        category: str = "",
        metric: str = "uses",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get prompt leaderboard by various metrics."""
        prompts = [
            p for p in self._prompts.values()
            if p.status in (PromptStatus.PUBLISHED, PromptStatus.FEATURED)
        ]
        if category:
            prompts = [p for p in prompts if p.category == category]

        if metric == "uses":
            prompts.sort(key=lambda p: p.metrics.total_uses, reverse=True)
        elif metric == "rating":
            prompts.sort(key=lambda p: p.metrics.avg_rating, reverse=True)
        elif metric == "throughput":
            prompts.sort(key=lambda p: p.metrics.avg_throughput_tok_s, reverse=True)
        elif metric == "quality":
            prompts.sort(key=lambda p: p.metrics.avg_quality_score, reverse=True)

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
            for p in prompts[:limit]
        ]

    def get_exchange_stats(self) -> dict[str, Any]:
        """Get overall exchange statistics."""
        published = [
            p for p in self._prompts.values()
            if p.status in (PromptStatus.PUBLISHED, PromptStatus.FEATURED)
        ]
        total_uses = sum(p.metrics.total_uses for p in published)
        total_tokens = sum(p.metrics.total_tokens_generated for p in published)
        free_count = sum(1 for p in published if p.is_free)
        premium_count = len(published) - free_count

        categories: dict[str, int] = {}
        for p in published:
            categories[p.category] = categories.get(p.category, 0) + 1

        return {
            "total_prompts": len(published),
            "free_prompts": free_count,
            "premium_prompts": premium_count,
            "total_uses": total_uses,
            "total_tokens_generated": total_tokens,
            "total_wallets": len(self._wallets),
            "total_reviews": sum(len(r) for r in self._reviews.values()),
            "categories": categories,
        }

    # ── Forking & Remixing ───────────────────────────────────────────

    def fork_prompt(
        self,
        user_id: str,
        prompt_id: str,
        new_name: str = "",
        new_system_prompt: str = "",
    ) -> PublishedPrompt | None:
        """Fork an existing prompt to create a variant.

        The fork inherits the original's category and tags but can
        modify the system prompt and pricing.
        """
        original = self._prompts.get(prompt_id)
        if original is None:
            return None

        return self.publish_prompt(
            author_id=user_id,
            name=new_name or f"Fork of {original.name}",
            description=f"Forked from {original.name} by {original.author_id}",
            category=original.category,
            system_prompt=new_system_prompt or original.system_prompt,
            tags=list(original.tags),
            license=original.license,
            price_tokens=original.price_tokens,
            parent_id=prompt_id,
        )
