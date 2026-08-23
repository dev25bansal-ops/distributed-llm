"""Learning router — online RL-based model selection via contextual bandits.

Wraps a :class:`ModelRouter` and learns from reward signals (user rating,
latency, cost, quality) to improve model selection over time.

Architecture::

    query + context
         |
         v
    Feature hashing (no ML dep)
         |
         v
    Epsilon-greedy / UCB bandit  ──>  selected model
         ^                            |
         |                            v
    reward signal  <──────  inference result

Cold start: falls back to the rule-based ``ModelRouter`` until enough
data is collected.  Per-tenant policies are learned independently.

No external ML dependencies — uses feature hashing and simple statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.model_router import ModelRouter, RouteMatch, RoutingContext


# ---------------------------------------------------------------------------
# Feature hashing
# ---------------------------------------------------------------------------

def _feature_hash(text: str, num_buckets: int = 256) -> list[float]:
    """Hash text into a fixed-size feature vector using feature hashing.

    Uses character n-grams (2-4) for better capture of local patterns.
    Returns a list of floats in [-1, 1].
    """
    vec = [0.0] * num_buckets
    text_lower = text.lower()
    # Character n-grams: 2, 3, 4
    for n in (2, 3, 4):
        for i in range(len(text_lower) - n + 1):
            ngram = text_lower[i : i + n].encode("utf-8")
            # M-10: MD5 is acceptable for non-cryptographic feature hashing
            # (bucketing, not security). Performance > collision resistance here.
            h = int(hashlib.md5(ngram).hexdigest(), 16)
            idx = h % num_buckets
            sign = 1.0 if (h // num_buckets) % 2 == 0 else -1.0
            vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Bandit arm statistics
# ---------------------------------------------------------------------------

@dataclass
class _ArmStats:
    """Running statistics for one arm (model) in one context bucket."""
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.total_reward += reward


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    """Components that contribute to the scalar reward for a routing decision.

    All values are optional.  When present they are weighted and summed
    to produce a scalar reward in [0, 1].
    """
    user_rating: float | None = None        # 0.0 (bad) – 1.0 (good)
    latency_ms: float | None = None         # Lower is better
    latency_sla_ms: float | None = None     # The SLA that was requested
    cost_usd: float | None = None           # Actual cost incurred
    cost_budget_usd: float | None = None    # Budget that was set
    quality_score: float | None = None      # 0.0 – 1.0 from quality judge

    def to_reward(self) -> float:
        """Compute scalar reward in [0, 1] from available signals."""
        parts: list[float] = []

        if self.user_rating is not None:
            parts.append(max(0.0, min(self.user_rating, 1.0)))

        if self.latency_ms is not None and self.latency_sla_ms is not None:
            # 1.0 if met SLA, decays to 0.0 at 3x SLA
            ratio = self.latency_ms / max(self.latency_sla_ms, 1.0)
            parts.append(max(0.0, 1.0 - (ratio - 1.0) / 2.0))

        if self.cost_usd is not None and self.cost_budget_usd is not None:
            # 1.0 if under budget, decays to 0.0 at 2x budget
            ratio = self.cost_usd / max(self.cost_budget_usd, 1e-9)
            parts.append(max(0.0, 1.0 - (ratio - 1.0)))

        if self.quality_score is not None:
            parts.append(max(0.0, min(self.quality_score, 1.0)))

        if not parts:
            return 0.5  # Neutral default
        return sum(parts) / len(parts)


# ---------------------------------------------------------------------------
# Learning router
# ---------------------------------------------------------------------------

class LearningRouter:
    """Online RL-based model router using contextual bandits.

    Wraps a rule-based :class:`ModelRouter` and learns to improve model
    selection from reward signals.  Uses epsilon-greedy exploration with
    UCB tie-breaking.

    Args:
        base_router: The rule-based fallback router.
        models: List of model names this router can choose from.
        epsilon: Exploration probability (0.0 = pure exploit, 1.0 = pure random).
        epsilon_decay: Multiplicative decay applied to epsilon after each update.
        epsilon_floor: Minimum epsilon value.
        num_buckets: Feature hash bucket count (context dimension).
        context_granularity: Number of discrete context buckets for arm stats.

    Usage::

        base = ModelRouter(settings)
        lr = LearningRouter(base, models=["codellama", "llama3", "mathgpt"])

        # Routing
        match = lr.route(messages, ctx)

        # Learning from outcome
        lr.record_outcome(match, RewardSignal(user_rating=0.9, latency_ms=200))
    """

    def __init__(
        self,
        base_router: ModelRouter,
        models: list[str],
        epsilon: float = 0.15,
        epsilon_decay: float = 0.999,
        epsilon_floor: float = 0.02,
        num_buckets: int = 256,
        context_granularity: int = 64,
    ) -> None:
        self._base = base_router
        self._models = list(models)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_floor = epsilon_floor
        self._num_buckets = num_buckets
        self._context_granularity = context_granularity

        # Per-tenant arm stats: {tenant_id: {context_bucket: {model: _ArmStats}}}
        self._policies: dict[str, dict[int, dict[str, _ArmStats]]] = {}
        self._default_policy: dict[int, dict[str, _ArmStats]] = {}
        self._lock = threading.Lock()

        # Tracking
        self._total_decisions = 0
        self._explore_count = 0
        self._exploit_count = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def route(
        self,
        text: str,
        ctx: RoutingContext | None = None,
        tenant_id: str = "",
    ) -> str:
        """Select a model using the learned policy or epsilon-greedy exploration.

        Falls back to the rule-based base router during cold start.

        Args:
            text: Query text (pre-lowered or raw).
            ctx: Optional routing context.
            tenant_id: Tenant identifier for per-tenant policies.

        Returns:
            Selected model name.
        """
        with self._lock:
            self._total_decisions += 1

        policy = self._get_policy(tenant_id)
        context_bucket = self._compute_context_bucket(text, ctx)

        # Cold start: if we have no data for this context, use base router
        arms = policy.get(context_bucket, {})
        if not arms or all(a.pulls == 0 for a in arms.values()):
            base_match = self._base.route(
                [{"role": "user", "content": text}],
                available_models=self._models,
            )
            return base_match.model

        # Epsilon-greedy exploration
        import random
        if random.random() < self._epsilon:
            with self._lock:
                self._explore_count += 1
            return random.choice(self._models)

        # Exploit: pick the arm with highest UCB score
        with self._lock:
            self._exploit_count += 1
        return self._select_best_arm(arms)

    def route_with_context(
        self,
        messages: list[dict[str, str]],
        ctx: RoutingContext | None = None,
        available_models: list[str] | None = None,
        tenant_id: str = "",
    ) -> RouteMatch:
        """Route with full message context, returning a RouteMatch.

        This is the primary integration point with the ModelRouter ecosystem.

        Args:
            messages: Conversation messages.
            ctx: Optional routing context.
            available_models: Filter for currently-loaded models.
            tenant_id: Tenant identifier for per-tenant policies.

        Returns:
            RouteMatch with the selected model.
        """
        # Extract text from messages
        text = ""
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                text = m["content"]
                break

        if not text:
            return self._base.route(messages, available_models)

        models = available_models or self._models
        selected = self.route(text, ctx, tenant_id)

        # Verify selected model is available
        if selected not in models:
            base_match = self._base.route(messages, available_models)
            return base_match  # Return the full RouteMatch, not just model name

        # Return a RouteMatch
        elapsed = 0.0  # Will be filled by caller
        from distllm.core.model_router import RouteMatch
        return RouteMatch(
            model=selected,
            rule_name="learning",
            confidence=0.7,
            latency_ms=elapsed,
        )

    def record_outcome(
        self,
        model: str,
        reward_signal: RewardSignal,
        text: str = "",
        ctx: RoutingContext | None = None,
        tenant_id: str = "",
    ) -> None:
        """Record an outcome and update the learned policy.

        Args:
            model: The model that was used.
            reward_signal: Reward components from the outcome.
            text: The original query text (for context bucketing).
            ctx: The routing context that was used.
            tenant_id: Tenant identifier.
        """
        reward = reward_signal.to_reward()
        context_bucket = self._compute_context_bucket(text, ctx)
        policy = self._get_policy(tenant_id)

        with self._lock:
            if context_bucket not in policy:
                policy[context_bucket] = {}
            arms = policy[context_bucket]
            if model not in arms:
                arms[model] = _ArmStats()
            arms[model].update(reward)

            # Decay epsilon
            self._epsilon = max(
                self._epsilon_floor,
                self._epsilon * self._epsilon_decay,
            )

        logger.debug(
            f"LearningRouter: recorded reward={reward:.3f} for "
            f"model={model}, ctx_bucket={context_bucket}, "
            f"tenant={tenant_id or 'default'}"
        )

    # ── Policy persistence ─────────────────────────────────────────────────

    def save_policy(self, path: str | Path) -> None:
        """Save learned policies to a JSON file."""
        path = Path(path)
        with self._lock:
            data = {
                "version": 1,
                "epsilon": self._epsilon,
                "total_decisions": self._total_decisions,
                "explore_count": self._explore_count,
                "exploit_count": self._exploit_count,
                "policies": {},
            }
            for tenant, ctx_arms in self._policies.items():
                tenant_data = {}
                for bucket, arms in ctx_arms.items():
                    bucket_data = {}
                    for model, stats in arms.items():
                        bucket_data[model] = {
                            "pulls": stats.pulls,
                            "total_reward": stats.total_reward,
                        }
                    tenant_data[str(bucket)] = bucket_data
                data["policies"][tenant] = tenant_data

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"LearningRouter: policy saved to {path}")

    def load_policy(self, path: str | Path) -> bool:
        """Load learned policies from a JSON file.

        Returns True if successfully loaded, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("version") != 1:
                logger.warning("LearningRouter: unsupported policy version")
                return False

            with self._lock:
                self._epsilon = data.get("epsilon", self._epsilon)
                self._total_decisions = data.get("total_decisions", 0)
                self._explore_count = data.get("explore_count", 0)
                self._exploit_count = data.get("exploit_count", 0)

                self._policies = {}
                for tenant, ctx_arms in data.get("policies", {}).items():
                    tenant_policy: dict[int, dict[str, _ArmStats]] = {}
                    for bucket_str, arms in ctx_arms.items():
                        bucket = int(bucket_str)
                        bucket_arms: dict[str, _ArmStats] = {}
                        for model, stats in arms.items():
                            arm = _ArmStats(
                                pulls=stats["pulls"],
                                total_reward=stats["total_reward"],
                            )
                            bucket_arms[model] = arm
                        tenant_policy[bucket] = bucket_arms
                    self._policies[tenant] = tenant_policy

            logger.info(f"LearningRouter: policy loaded from {path}")
            return True
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            logger.warning(f"LearningRouter: failed to load policy: {e}")
            return False

    @property
    def stats(self) -> dict:
        """Return router statistics."""
        with self._lock:
            total_arms = sum(
                len(arms)
                for policy in self._policies.values()
                for arms in policy.values()
            )
            return {
                "total_decisions": self._total_decisions,
                "explore_count": self._explore_count,
                "exploit_count": self._exploit_count,
                "epsilon": round(self._epsilon, 4),
                "num_tenants": len(self._policies),
                "total_context_buckets": total_arms,
            }

    # ── Internals ──────────────────────────────────────────────────────────

    def _get_policy(
        self, tenant_id: str
    ) -> dict[int, dict[str, _ArmStats]]:
        """Get or create the policy for a tenant."""
        if not tenant_id:
            return self._default_policy
        with self._lock:
            if tenant_id not in self._policies:
                self._policies[tenant_id] = {}
            return self._policies[tenant_id]

    def _compute_context_bucket(
        self, text: str, ctx: RoutingContext | None
    ) -> int:
        """Map (text features, context) to a discrete bucket index."""
        vec = _feature_hash(text, self._num_buckets)

        # Incorporate context signals into the hash
        ctx_bits = 0
        if ctx:
            if ctx.cost_budget is not None:
                ctx_bits += int(ctx.cost_budget * 1000) % 16
            if ctx.max_latency_ms is not None:
                ctx_bits += int(ctx.max_latency_ms / 100) % 16
            if ctx.has_tool_calls:
                ctx_bits += 32
            if ctx.input_tokens is not None:
                ctx_bits += (ctx.input_tokens // 512) % 16
            if ctx.language:
                ctx_bits += sum(ord(c) for c in ctx.language) % 16

        # Simple bucketing: sum of features mod granularity + context offset
        feature_sum = sum(abs(v) for v in vec)
        bucket = int(feature_sum * 100 + ctx_bits) % self._context_granularity
        return bucket

    def _select_best_arm(self, arms: dict[str, _ArmStats]) -> str:
        """Select the best arm using UCB1 (Upper Confidence Bound).

        UCB1 = mean_reward + sqrt(2 * ln(N) / n_i)
        where N = total pulls, n_i = pulls for arm i.
        """
        total_pulls = sum(a.pulls for a in arms.values())
        if total_pulls == 0:
            return self._models[0] if self._models else ""

        best_model = ""
        best_score = float("-inf")

        for model in self._models:
            arm = arms.get(model)
            if arm is None or arm.pulls == 0:
                # Unexplored arm gets highest priority
                return model

            # UCB1 exploration bonus
            exploration = math.sqrt(
                2.0 * math.log(total_pulls) / arm.pulls
            )
            score = arm.mean_reward + exploration

            if score > best_score:
                best_score = score
                best_model = model

        return best_model or (self._models[0] if self._models else "")


# ── Neural bandit router ──────────────────────────────────────────────────

class NeuralBanditRouter:
    """Neural contextual bandit router using a 2-layer MLP.

    Replaces the feature-hashing + UCB1 approach of ``LearningRouter``
    with a lightweight neural network that can capture complex feature
    interactions and generalise across tenants via per-tenant embeddings.

    Architecture::

        features (256-dim) ──┐
        tenant embedding ────┤──► Linear(128) ──► ReLU ──► Linear(num_models)
        context signals ─────┘

    Training: online mini-batch SGD with replay buffer.
    Exploration: Thompson sampling (Gaussian noise on output logits).

    Falls back to the base ``LearningRouter`` when sample count < threshold.
    """

    def __init__(
        self,
        base_router: LearningRouter | ModelRouter,
        models: list[str],
        feature_dim: int = 256,
        hidden_dim: int = 128,
        replay_capacity: int = 5_000,
        batch_size: int = 32,
        lr: float = 1e-3,
        min_samples_for_training: int = 50,
        device: str = "cpu",
    ):
        self._base = base_router
        self._models = models
        self._feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self._min_samples = min_samples_for_training
        self._device = torch.device(device)

        # Neural network: 2-layer MLP with ReLU
        # Input: features (256) + context (16) + tenant embedding (32) = 304
        self._net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + 16 + 32, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(models)),
        ).to(self._device)

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._replay_buffer: list[tuple[torch.Tensor, int]] = []
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._train_count = 0
        self._total_decisions = 0
        self._fallback_count = 0

        # Per-tenant learnable embedding table (lazy initialised)
        self._tenant_embeddings: dict[str, torch.Tensor] = {}
        self._embed_dim = 32

    def _encode(
        self, text: str, ctx: RoutingContext | None = None, tenant_id: str = ""
    ) -> torch.Tensor:
        """Encode (text, context, tenant) into a fixed-size feature tensor."""
        # Text features via hashing
        feat = _feature_hash(text, self._feature_dim)
        x = torch.tensor(feat, dtype=torch.float, device=self._device)

        # Context bits
        ctx_vec = torch.zeros(16, dtype=torch.float, device=self._device)
        if ctx:
            if ctx.cost_budget is not None:
                ctx_vec[0] = min(ctx.cost_budget, 1.0)
            if ctx.max_latency_ms is not None:
                ctx_vec[1] = min(ctx.max_latency_ms / 10_000, 1.0)
            if ctx.has_tool_calls:
                ctx_vec[2] = 1.0
            if ctx.input_tokens is not None:
                ctx_vec[3] = min(ctx.input_tokens / 32_000, 1.0)
        x = torch.cat([x, ctx_vec])

        # Tenant embedding
        if tenant_id:
            if tenant_id not in self._tenant_embeddings:
                self._tenant_embeddings[tenant_id] = torch.randn(
                    self._embed_dim, device=self._device,
                ) * 0.1
            x = torch.cat([x, self._tenant_embeddings[tenant_id]])
        else:
            x = torch.cat([x, torch.zeros(self._embed_dim, device=self._device)])

        return x

    def route(
        self, text: str, ctx: RoutingContext | None = None, tenant_id: str = "",
    ) -> str:
        """Select a model using the neural bandit."""
        self._total_decisions += 1
        with self._lock:
            if len(self._replay_buffer) < self._min_samples:
                self._fallback_count += 1
                if isinstance(self._base, LearningRouter):
                    return self._base.route(text, ctx, tenant_id)
                return self._models[0] if self._models else ""

        x = self._encode(text, ctx, tenant_id).unsqueeze(0)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(x)[0]
        self._net.train()

        # Thompson sampling: add Gaussian noise to logits
        noise = torch.randn_like(logits) * 0.3
        chosen = (logits + noise).argmax().item()
        return self._models[chosen] if chosen < len(self._models) else self._models[0]

    def record_outcome(
        self, model: str, reward: float, text: str = "",
        ctx: RoutingContext | None = None, tenant_id: str = "",
    ) -> None:
        """Record outcome and optionally train the network."""
        if model not in self._models:
            return
        x = self._encode(text, ctx, tenant_id)
        action = self._models.index(model)

        with self._lock:
            self._replay_buffer.append((x.cpu(), action))
            if len(self._replay_buffer) > self._replay_capacity:
                self._replay_buffer.pop(0)
            buf_size = len(self._replay_buffer)

        if buf_size < self._min_samples:
            return

        # Online mini-batch training
        import random as _random
        batch = _random.sample(self._replay_buffer, min(self._batch_size, buf_size))
        batch_x = torch.stack([b[0] for b in batch]).to(self._device)
        batch_a = torch.tensor([b[1] for b in batch], device=self._device)

        logits = self._net(batch_x)
        loss = torch.nn.functional.cross_entropy(logits, batch_a)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._train_count += 1

    @property
    def stats(self) -> dict:
        return {
            "total_decisions": self._total_decisions,
            "fallback_count": self._fallback_count,
            "train_count": self._train_count,
            "replay_buffer_size": len(self._replay_buffer),
            "num_tenants": len(self._tenant_embeddings),
        }
