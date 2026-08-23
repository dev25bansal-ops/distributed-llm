"""Global inference mesh — route each request across a federated graph of
clusters, optimising latency, cost, reliability, and carbon intensity.

Architecture::

    request
       |
       v
    AtlasMesh.route(request)
       |
       ├── ClusterGraph        — federated graph of available clusters
       ├── LatencyCostReliabilityScorer  — multi-objective score per cluster
       ├── ContextualBanditRewardModel   — learned reward estimates
       ├── LPSolverRouter      — LP / greedy assignment
       |
       v
    (selected_cluster, expected_cost, expected_latency)

Usage::

    mesh = AtlasMesh()
    mesh.cluster_graph.add_cluster("c1", region="us-east-1", provider="aws",
                                    cost_per_token=0.002, latency_baseline=50.0)
    cluster, cost, lat = mesh.route(model="llama-70b", prompt_length=512)
    print(mesh.stats())
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Optional LP solver libraries
# ---------------------------------------------------------------------------

try:
    import scipy.optimize as _sp_opt

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cvxpy as _cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cluster:
    """A single cluster (on-prem, spot, cloud, edge) in the federated graph.

    Attributes:
        id: Unique cluster identifier.
        region: Geographic region (e.g. ``"us-east-1"``, ``"eu-west-2"``).
        provider: Cloud / on-prem provider name (e.g. ``"aws"``, ``"gcp"``,
            ``"on-prem"``, ``"edge"``).
        cost_per_token: USD per token for inference on this cluster.
        latency_baseline: Baseline latency in milliseconds for a reference
            request on this cluster.
        reliability_history: Moving-average reliability in [0.0, 1.0],
            where 1.0 means 100 % uptime / no errors.
        carbon_intensity: Grams of CO₂ equivalent per kWh in the region
            (0.0 = cleanest).
        current_load: Current cluster utilisation in [0.0, 1.0].
        gpu_type: GPU hardware identifier (e.g. ``"A100"``, ``"H100"``).
        gpu_count: Number of GPUs in the cluster.
    """

    id: str
    region: str
    provider: str
    cost_per_token: float = 0.0
    latency_baseline: float = 100.0
    reliability_history: float = 1.0
    carbon_intensity: float = 0.0
    current_load: float = 0.0
    gpu_type: str = "unknown"
    gpu_count: int = 1


@dataclass
class RoutingRequest:
    """Details of an inference request to be routed.

    Attributes:
        model: Model name (e.g. ``"llama-70b"``).
        prompt_length: Number of input tokens.
        max_tokens: Maximum output tokens to generate.
        complexity: Estimated request complexity in [0.0, 1.0]
            (auto-computed if not set).
        priority: Request priority (higher = more important).
        max_latency_ms: Optional latency SLA in milliseconds.
        max_budget_usd: Optional cost budget in USD.
        max_carbon_g: Optional carbon budget in grams of CO₂e.
    """

    model: str
    prompt_length: int = 0
    max_tokens: int = 0
    complexity: float | None = None
    priority: float = 1.0
    max_latency_ms: float | None = None
    max_budget_usd: float | None = None
    max_carbon_g: float | None = None


@dataclass
class RoutingAssignment:
    """Assignment of a single request to a cluster.

    Attributes:
        request: The routed request.
        cluster: The chosen cluster.
        score: Composite score from the multi-objective scorer.
        reward_estimate: Expected reward from the bandit model.
        expected_cost_usd: Predicted cost in USD.
        expected_latency_ms: Predicted latency in milliseconds.
        expected_carbon_g: Predicted carbon emission in grams of CO₂e.
    """

    request: RoutingRequest
    cluster: Cluster
    score: float = 0.0
    reward_estimate: float = 0.0
    expected_cost_usd: float = 0.0
    expected_latency_ms: float = 0.0
    expected_carbon_g: float = 0.0


@dataclass
class Observation:
    """A single training observation for the bandit reward model.

    Attributes:
        cluster_id: ID of the cluster that was selected.
        features: Feature vector used for prediction.
        reward: Observed scalar reward in [0.0, 1.0].
        timestamp: Unix timestamp of the observation.
    """

    cluster_id: str
    features: list[float]
    reward: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MeshStats:
    """Aggregated routing statistics for ``AtlasMesh.stats()``."""

    total_routed: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_carbon_g: float = 0.0
    avg_score: float = 0.0
    avg_reward_estimate: float = 0.0
    cost_savings_vs_baseline_pct: float = 0.0
    latency_savings_vs_baseline_pct: float = 0.0
    carbon_savings_vs_baseline_pct: float = 0.0
    cluster_usage: dict[str, int] = field(default_factory=dict)
    solver_mode: str = "unknown"


# ---------------------------------------------------------------------------
# Canonical request key
# ---------------------------------------------------------------------------


def _canonical_request_key(request: RoutingRequest) -> str:
    """Canonical key identifying a routing request.

    Both :class:`AtlasMesh` and :class:`LPSolverRouter` build their
    ``(request_key, cluster_id)`` score-dict keys with this exact format, so
    the multi-objective scores pre-computed by :class:`AtlasMesh` are always
    found by the solver's lookups.  The batch index is intentionally NOT part
    of the key — the scores dict is keyed per request by model/prompt/max.
    """
    return f"{request.model}:{request.prompt_length}:{request.max_tokens}"


# ---------------------------------------------------------------------------
# ClusterGraph
# ---------------------------------------------------------------------------


class ClusterGraph:
    """Federated graph of clusters (on-prem, spot, cloud, edge).

    Maintains a thread-safe registry of available clusters and supports
    filtering by region, provider, or other attributes.

    Usage::

        cg = ClusterGraph()
        cg.add_cluster("c1", region="us-east-1", provider="aws",
                        cost_per_token=0.002, latency_baseline=50.0)
        clusters = cg.get_clusters(region="us-east-1")
    """

    def __init__(self) -> None:
        self._clusters: dict[str, Cluster] = {}
        self._lock = threading.RLock()

    # ── Mutators ─────────────────────────────────────────────────────────

    def add_cluster(
        self,
        cluster_id: str,
        region: str,
        provider: str,
        cost_per_token: float = 0.0,
        latency_baseline: float = 100.0,
        reliability_history: float = 1.0,
        carbon_intensity: float = 0.0,
        current_load: float = 0.0,
        gpu_type: str = "unknown",
        gpu_count: int = 1,
    ) -> Cluster:
        """Register a new cluster or replace an existing one.

        Returns:
            The newly created :class:`Cluster` instance.
        """
        cluster = Cluster(
            id=cluster_id,
            region=region,
            provider=provider,
            cost_per_token=cost_per_token,
            latency_baseline=latency_baseline,
            reliability_history=reliability_history,
            carbon_intensity=carbon_intensity,
            current_load=current_load,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )
        with self._lock:
            self._clusters[cluster_id] = cluster
        logger.debug("Cluster added: {!r} in {} ({})", cluster_id, region, provider)
        return cluster

    def remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from the graph.

        Returns:
            ``True`` if the cluster existed and was removed.
        """
        with self._lock:
            if cluster_id in self._clusters:
                del self._clusters[cluster_id]
                logger.debug("Cluster removed: {!r}", cluster_id)
                return True
        return False

    def update_cluster(self, cluster_id: str, **kwargs: Any) -> bool:
        """Update attributes of an existing cluster.

        Accepts any field of :class:`Cluster` as a keyword argument.
        Returns ``True`` if the cluster was found and updated.
        """
        with self._lock:
            cluster = self._clusters.get(cluster_id)
            if cluster is None:
                return False
            # Frozen dataclass → replace with new instance
            current = {
                "id": cluster.id,
                "region": cluster.region,
                "provider": cluster.provider,
                "cost_per_token": cluster.cost_per_token,
                "latency_baseline": cluster.latency_baseline,
                "reliability_history": cluster.reliability_history,
                "carbon_intensity": cluster.carbon_intensity,
                "current_load": cluster.current_load,
                "gpu_type": cluster.gpu_type,
                "gpu_count": cluster.gpu_count,
            }
            current.update(kwargs)
            self._clusters[cluster_id] = Cluster(**current)
        return True

    # ── Accessors ────────────────────────────────────────────────────────

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        """Look up a single cluster by ID."""
        with self._lock:
            return self._clusters.get(cluster_id)

    def get_clusters(
        self,
        region: str | None = None,
        provider: str | None = None,
        min_reliability: float | None = None,
        max_cost_per_token: float | None = None,
    ) -> list[Cluster]:
        """Return clusters matching the given filters.

        When a filter is ``None`` it is not applied.  Returns all clusters
        when no filters are specified.
        """
        with self._lock:
            result = list(self._clusters.values())

        if region is not None:
            result = [c for c in result if c.region == region]
        if provider is not None:
            result = [c for c in result if c.provider == provider]
        if min_reliability is not None:
            result = [c for c in result if c.reliability_history >= min_reliability]
        if max_cost_per_token is not None:
            result = [c for c in result if c.cost_per_token <= max_cost_per_token]

        return result

    def all_clusters(self) -> list[Cluster]:
        """Return every registered cluster (cheaper than calling with no filters)."""
        with self._lock:
            return list(self._clusters.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._clusters)

    def __contains__(self, cluster_id: str) -> bool:
        with self._lock:
            return cluster_id in self._clusters

    def __repr__(self) -> str:
        with self._lock:
            ids = ", ".join(sorted(self._clusters))
            return f"ClusterGraph({len(self._clusters)} clusters: [{ids}])"


# ---------------------------------------------------------------------------
# LatencyCostReliabilityScorer
# ---------------------------------------------------------------------------


@dataclass
class ScoringWeights:
    """Configurable weights for the multi-objective scorer.

    Higher weight = more importance.  Values are normalised to a sum of 1.0
    internally before scoring so they need not sum to any particular value.
    """

    latency: float = 0.30
    cost: float = 0.30
    reliability: float = 0.25
    carbon: float = 0.15


class LatencyCostReliabilityScorer:
    """Multi-objective scorer that produces a composite score for each
    (cluster, request) pair.

    Scores are in [0.0, 1.0] where **higher is better**.  Each dimension is
    independently normalised and then combined via configurable weights.

    Usage::

        scorer = LatencyCostReliabilityScorer()
        score = scorer.score(cluster, request)
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights if weights is not None else ScoringWeights()
        self._lock = threading.RLock()

    def score(self, cluster: Cluster, request: RoutingRequest) -> float:
        """Compute a composite score for routing *request* to *cluster*.

        Returns:
            A float in [0.0, 1.0], higher = better.
        """
        w = self.weights
        total_weight = w.latency + w.cost + w.reliability + w.carbon
        if total_weight <= 0.0:
            return 0.5  # neutral default when all weights are zero

        latency_score = self._score_latency(cluster, request)
        cost_score = self._score_cost(cluster, request)
        reliability_score = self._score_reliability(cluster)
        carbon_score = self._score_carbon(cluster)

        composite = (
            w.latency * latency_score
            + w.cost * cost_score
            + w.reliability * reliability_score
            + w.carbon * carbon_score
        ) / total_weight

        return max(0.0, min(composite, 1.0))

    # ── Per-dimension scorers (all return [0, 1], higher = better) ───────

    @staticmethod
    def _score_latency(cluster: Cluster, request: RoutingRequest) -> float:
        """Predicted latency score.

        Baseline latency adjusted by current load, then compared to
        ``max_latency_ms`` if provided.  Lower is better.
        """
        # Load-adjusted latency
        load_penalty = 1.0 + cluster.current_load * 0.5
        predicted_ms = cluster.latency_baseline * load_penalty

        if request.max_latency_ms is not None and request.max_latency_ms > 0:
            ratio = predicted_ms / request.max_latency_ms
            # 1.0 if well under SLA, decays to 0.0 at 5x SLA
            score = max(0.0, 1.0 - (ratio - 0.2) / 4.8)
        else:
            # No explicit SLA — score relative to absolute scale (0–5000 ms)
            score = max(0.0, 1.0 - predicted_ms / 5000.0)

        return score

    @staticmethod
    def _score_cost(cluster: Cluster, request: RoutingRequest) -> float:
        """Cost score.

        Uses ``cost_per_token`` and estimated token count.  Higher cost = lower
        score.  Compared to request budget if available.
        """
        estimated_total_tokens = request.prompt_length + request.max_tokens
        estimated_cost = cluster.cost_per_token * max(estimated_total_tokens, 1)

        if request.max_budget_usd is not None and request.max_budget_usd > 0:
            ratio = estimated_cost / request.max_budget_usd
            score = max(0.0, 1.0 - (ratio - 0.1) / 1.9)
        else:
            # Score relative to a reference cost of $0.01 per request
            score = max(0.0, 1.0 - estimated_cost / 0.01)

        return score

    @staticmethod
    def _score_reliability(cluster: Cluster) -> float:
        """Reliability score — directly uses ``reliability_history``."""
        return max(0.0, min(cluster.reliability_history, 1.0))

    @staticmethod
    def _score_carbon(cluster: Cluster) -> float:
        """Carbon intensity score.

        Lower carbon intensity = higher score.  Scaled against a reference of
        500 gCO₂e/kWh (roughly the global average grid carbon intensity).
        """
        intensity = max(cluster.carbon_intensity, 0.0)
        score = max(0.0, 1.0 - intensity / 500.0)
        return score


# ---------------------------------------------------------------------------
# ContextualBanditRewardModel
# ---------------------------------------------------------------------------


class ContextualBanditRewardModel:
    """Online learned reward model for routing decisions.

    Uses per-cluster linear weights with online SGD updates (no external ML
    dependency).  The feature vector captures:

    * ``cluster_load`` — current cluster utilisation [0, 1]
    * ``request_complexity`` — estimated complexity [0, 1]
    * ``time_sin`` / ``time_cos`` — sin/cos encoding of hour-of-day
    * ``recent_latency`` — recent observed latency [0, 1]

    **Training** (``train(observation)``): updates the weight vector for the
    cluster that was selected using a gradient step (R²-style squared loss).

    **Prediction** (``predict(request, clusters)``): returns a dict mapping
    cluster ID → estimated reward in [0.0, 1.0].

    Usage::

        bandit = ContextualBanditRewardModel()
        estimates = bandit.predict(request, clusters)
        bandit.train(Observation(cluster_id="c1", features=[...], reward=0.8))
    """

    def __init__(
        self,
        feature_dim: int = 5,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
    ) -> None:
        self._feature_dim = feature_dim
        self._learning_rate = learning_rate
        self._regularization = regularization

        # Per-cluster linear weights: cluster_id → list[float]
        self._weights: dict[str, list[float]] = {}
        # Per-cluster update count for annealing the learning rate
        self._updates: dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

        # Observation buffer for stats / diagnostics
        self._observations: list[Observation] = []

    # ── Public API ───────────────────────────────────────────────────────

    def predict(
        self,
        request: RoutingRequest,
        clusters: list[Cluster],
    ) -> dict[str, float]:
        """Estimate the reward for routing *request* to each cluster.

        Returns:
            Mapping of ``cluster_id → estimated_reward`` in [0.0, 1.0].
        """
        with self._lock:
            results: dict[str, float] = {}
            for cluster in clusters:
                features = self._build_features(cluster, request)
                weights = self._weights.get(cluster.id)
                if weights is None:
                    # Cold start — use a default of 0.5
                    results[cluster.id] = 0.5
                else:
                    raw = self._predict_from_weights(weights, features)
                    results[cluster.id] = max(0.0, min(raw, 1.0))
            return results

    def train(self, observation: Observation) -> None:
        """Update the model with a new observation.

        Performs a single step of online SGD (ridge regression style) on the
        weight vector for the observed cluster.

        Args:
            observation: The observed (features, reward) pair.
        """
        feat = observation.features
        target = max(0.0, min(observation.reward, 1.0))
        cluster_id = observation.cluster_id

        with self._lock:
            # Lazy initialise weights
            if cluster_id not in self._weights:
                self._weights[cluster_id] = [0.0] * self._feature_dim

            weights = self._weights[cluster_id]
            pred = self._predict_from_weights(weights, feat)
            error = pred - target

            # Annealed learning rate
            n = self._updates[cluster_id]
            lr = self._learning_rate / (1.0 + math.sqrt(n))
            self._updates[cluster_id] = n + 1

            # Gradient descent with L2 regularisation
            for i in range(self._feature_dim):
                gradient = error * feat[i] + self._regularization * weights[i]
                weights[i] -= lr * gradient

            self._observations.append(observation)
            # Keep only recent 10 000 observations
            if len(self._observations) > 10000:
                self._observations = self._observations[-5000:]

    # ── Feature engineering ──────────────────────────────────────────────

    def _build_features(
        self,
        cluster: Cluster,
        request: RoutingRequest,
    ) -> list[float]:
        """Construct the 5-dimensional feature vector for a (cluster, request)
        pair.

        Feature layout: [cluster_load, request_complexity, time_sin, time_cos,
        recent_latency].
        """
        # 1. Cluster load (already in [0, 1])
        cluster_load = max(0.0, min(cluster.current_load, 1.0))

        # 2. Request complexity — estimate from prompt_length / max_tokens
        complexity = request.complexity
        if complexity is None:
            if request.max_tokens > 0:
                ratio = request.prompt_length / max(request.max_tokens, 1)
                complexity = min(ratio, 1.0)
            else:
                complexity = 0.5
        complexity = max(0.0, min(complexity, 1.0))

        # 3–4. Time-of-day cyclic encoding
        now = time.localtime()
        hour_angle = 2.0 * math.pi * (now.tm_hour + now.tm_min / 60.0) / 24.0
        time_sin = math.sin(hour_angle)
        time_cos = math.cos(hour_angle)

        # 5. Recent latency as a normalised feature
        latency = cluster.latency_baseline * (1.0 + cluster.current_load * 0.5)
        recent_latency = max(0.0, min(latency / 5000.0, 1.0))

        return [cluster_load, complexity, time_sin, time_cos, recent_latency]

    @staticmethod
    def _predict_from_weights(
        weights: list[float],
        features: list[float],
    ) -> float:
        """Dot-product prediction, squashed to [0, 1] via sigmoid-like
        clamp (avoids the need for an external activation function)."""
        raw = sum(w * f for w, f in zip(weights, features))
        # Sigmoid approximation via clamp to keep output in [0, 1]
        # (weights are regularised so values stay bounded)
        return 1.0 / (1.0 + math.exp(-raw))

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def training_count(self) -> int:
        """Total number of observations trained on."""
        return len(self._observations)

    @property
    def cluster_count(self) -> int:
        """Number of clusters with learned weights."""
        with self._lock:
            return len(self._weights)

    def get_weights(self, cluster_id: str) -> list[float] | None:
        """Return the learned weight vector for a cluster, or ``None``."""
        with self._lock:
            return self._weights.get(cluster_id)


# ---------------------------------------------------------------------------
# LPSolverRouter
# ---------------------------------------------------------------------------


@dataclass
class ConstraintViolation(Exception):
    """Raised when the LP solver cannot find a feasible assignment."""

    message: str


class LPSolverRouter:
    """Online linear programming solver for optimal request-to-cluster routing.

    Uses SciPy's ``linprog`` or CVXPY when available, falling back to a
    greedy heuristic that assigns each request to the highest-score cluster
    satisfying constraints.

    Constraints:
        * ``max_latency`` — per-request latency SLA
        * ``budget`` — total cost budget across all requests
        * ``carbon_budget`` — total carbon budget across all requests

    Usage::

        router = LPSolverRouter()
        assignments = router.solve(requests, clusters, latency_weight=0.5,
                                   cost_weight=0.3, carbon_weight=0.2)
    """

    def __init__(self) -> None:
        self._mode: str = "unknown"
        if HAS_CVXPY:
            self._mode = "cvxpy"
            logger.debug("LPSolverRouter using CVXPY")
        elif HAS_SCIPY:
            self._mode = "scipy"
            logger.debug("LPSolverRouter using SciPy")
        else:
            self._mode = "greedy"
            logger.info(
                "LPSolverRouter using greedy fallback — "
                "install scipy or cvxpy for optimal LP routing"
            )

    @property
    def mode(self) -> str:
        """The active solver mode: ``"cvxpy"``, ``"scipy"``, or ``"greedy"``."""
        return self._mode

    def solve(
        self,
        requests: list[RoutingRequest],
        clusters: list[Cluster],
        scores: dict[tuple[str, str], float],
        reward_estimates: dict[str, float] | None = None,
        latency_weight: float = 0.5,
        cost_weight: float = 0.3,
        carbon_weight: float = 0.2,
    ) -> list[RoutingAssignment]:
        """Solve the optimal assignment of requests to clusters.

        Args:
            requests: List of requests to route.
            clusters: Available clusters.
            scores: Pre-computed composite scores keyed by
                ``(request_id, cluster_id)``.
            reward_estimates: Optional bandit reward estimates keyed by
                ``cluster_id``, blended into the objective.
            latency_weight: Weight for latency in the combined objective.
            cost_weight: Weight for cost in the combined objective.
            carbon_weight: Weight for carbon in the combined objective.

        Returns:
            List of :class:`RoutingAssignment` (one per request).

        Raises:
            ConstraintViolation: If no feasible assignment exists.
        """
        if not requests:
            return []
        if not clusters:
            raise ConstraintViolation("No clusters available to route requests")

        if self._mode in ("cvxpy", "scipy") and len(requests) > 1:
            return self._solve_lp(
                requests, clusters, scores, reward_estimates,
                latency_weight, cost_weight, carbon_weight,
            )
        return self._solve_greedy(
            requests, clusters, scores, reward_estimates,
            latency_weight, cost_weight, carbon_weight,
        )

    # ── LP solver (batch) ───────────────────────────────────────────────

    def _solve_lp(
        self,
        requests: list[RoutingRequest],
        clusters: list[Cluster],
        scores: dict[tuple[str, str], float],
        reward_estimates: dict[str, float] | None = None,
        latency_weight: float = 0.5,
        cost_weight: float = 0.3,
        carbon_weight: float = 0.2,
    ) -> list[RoutingAssignment]:
        if self._mode == "cvxpy" and HAS_CVXPY:
            return self._solve_cvxpy(
                requests, clusters, scores, reward_estimates,
                latency_weight, cost_weight, carbon_weight,
            )
        if self._mode == "scipy" and HAS_SCIPY:
            return self._solve_scipy(
                requests, clusters, scores, reward_estimates,
                latency_weight, cost_weight, carbon_weight,
            )
        # Fallback if mode was set optimistically but import disappeared
        return self._solve_greedy(
            requests, clusters, scores, reward_estimates,
            latency_weight, cost_weight, carbon_weight,
        )

    def _solve_cvxpy(
        self,
        requests: list[RoutingRequest],
        clusters: list[Cluster],
        scores: dict[tuple[str, str], float],
        reward_estimates: dict[str, float] | None = None,
        latency_weight: float = 0.5,
        cost_weight: float = 0.3,
        carbon_weight: float = 0.2,
    ) -> list[RoutingAssignment]:
        n = len(requests)
        m = len(clusters)

        # Decision variable: x[i, j] ∈ [0, 1]
        x = _cp.Variable((n, m), nonneg=True)

        # Objective: maximise composite utility
        obj_terms: list = []
        for i, req in enumerate(requests):
            for j, cluster in enumerate(clusters):
                key = (self._req_key(req, i), cluster.id)
                base_score = scores.get(key, 0.5)
                bandit_bonus = (
                    reward_estimates.get(cluster.id, 0.5) if reward_estimates else 0.5
                )
                utility = 0.7 * base_score + 0.3 * bandit_bonus
                obj_terms.append(utility * x[i, j])

        objective = _cp.Maximize(_cp.sum(obj_terms))

        # Constraints
        constraints: list = []

        # Each request assigned to exactly one cluster
        for i in range(n):
            constraints.append(_cp.sum(x[i, :]) == 1)

        # Max latency per request
        for i, req in enumerate(requests):
            if req.max_latency_ms is not None and req.max_latency_ms > 0:
                for j, cluster in enumerate(clusters):
                    predicted_ms = self._predict_latency(cluster, req)
                    constraints.append(
                        x[i, j] * predicted_ms <= req.max_latency_ms + 1e-6
                    )

        # Total cost budget
        total_budget = sum(
            req.max_budget_usd or float("inf") for req in requests
        )
        if total_budget < float("inf"):
            cost_expr = _cp.sum(
                x[i, j] * self._predict_cost(clusters[j], requests[i])
                for i in range(n)
                for j in range(m)
            )
            constraints.append(cost_expr <= total_budget)

        # Total carbon budget
        total_carbon_budget = sum(
            req.max_carbon_g or float("inf") for req in requests
        )
        if total_carbon_budget < float("inf"):
            carbon_expr = _cp.sum(
                x[i, j] * self._predict_carbon(clusters[j], requests[i])
                for i in range(n)
                for j in range(m)
            )
            constraints.append(carbon_expr <= total_carbon_budget)

        problem = _cp.Problem(objective, constraints)
        try:
            problem.solve(verbose=False)
        except Exception as exc:
            raise ConstraintViolation(
                f"CVXPY solver failed: {exc}"
            ) from exc

        if x.value is None:
            raise ConstraintViolation("CVXPY found no feasible assignment")

        # Build assignments from the solution
        assignments: list[RoutingAssignment] = []
        for i, req in enumerate(requests):
            row = x.value[i, :]
            j = int(row.argmax())
            cluster = clusters[j]
            assignments.append(self._build_assignment(req, cluster, scores, reward_estimates))

        return assignments

    def _solve_scipy(
        self,
        requests: list[RoutingRequest],
        clusters: list[Cluster],
        scores: dict[tuple[str, str], float],
        reward_estimates: dict[str, float] | None = None,
        latency_weight: float = 0.5,
        cost_weight: float = 0.3,
        carbon_weight: float = 0.2,
    ) -> list[RoutingAssignment]:
        n = len(requests)
        m = len(clusters)
        n_vars = n * m

        # Objective coefficients (flattened, we maximise so negate for scipy)
        c = []
        for i, req in enumerate(requests):
            for j, cluster in enumerate(clusters):
                key = (self._req_key(req, i), cluster.id)
                base_score = scores.get(key, 0.5)
                bandit_bonus = (
                    reward_estimates.get(cluster.id, 0.5) if reward_estimates else 0.5
                )
                utility = 0.7 * base_score + 0.3 * bandit_bonus
                c.append(-utility)  # scipy minimises

        # Bounds: 0 <= x[i,j] <= 1
        bounds = [(0.0, 1.0)] * n_vars

        # Equality constraints: each request assigned to exactly one cluster
        A_eq = []
        b_eq = []
        for i in range(n):
            row = [0.0] * n_vars
            for j in range(m):
                row[i * m + j] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)

        # Inequality constraints: latency, budget, carbon
        A_ub: list[list[float]] = []
        b_ub: list[float] = []

        for i, req in enumerate(requests):
            if req.max_latency_ms is not None and req.max_latency_ms > 0:
                row = [0.0] * n_vars
                for j, cluster in enumerate(clusters):
                    predicted_ms = self._predict_latency(cluster, req)
                    row[i * m + j] = predicted_ms
                A_ub.append(row)
                b_ub.append(req.max_latency_ms)

        total_budget = sum(
            req.max_budget_usd or float("inf") for req in requests
        )
        if total_budget < float("inf"):
            row = [0.0] * n_vars
            for i, req in enumerate(requests):
                for j, cluster in enumerate(clusters):
                    row[i * m + j] = self._predict_cost(cluster, req)
            A_ub.append(row)
            b_ub.append(total_budget)

        total_carbon_budget = sum(
            req.max_carbon_g or float("inf") for req in requests
        )
        if total_carbon_budget < float("inf"):
            row = [0.0] * n_vars
            for i, req in enumerate(requests):
                for j, cluster in enumerate(clusters):
                    row[i * m + j] = self._predict_carbon(cluster, req)
            A_ub.append(row)
            b_ub.append(total_carbon_budget)

        try:
            result = _sp_opt.linprog(
                c,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
        except Exception as exc:
            raise ConstraintViolation(
                f"SciPy linprog failed: {exc}"
            ) from exc

        if not result.success:
            raise ConstraintViolation(
                f"SciPy linprog found no feasible assignment: {result.message}"
            )

        assignments: list[RoutingAssignment] = []
        for i, req in enumerate(requests):
            start = i * m
            row = result.x[start : start + m]
            j = int(row.argmax())
            cluster = clusters[j]
            assignments.append(self._build_assignment(req, cluster, scores, reward_estimates))

        return assignments

    # ── Greedy fallback ─────────────────────────────────────────────────

    def _solve_greedy(
        self,
        requests: list[RoutingRequest],
        clusters: list[Cluster],
        scores: dict[tuple[str, str], float],
        reward_estimates: dict[str, float] | None = None,
        latency_weight: float = 0.5,
        cost_weight: float = 0.3,
        carbon_weight: float = 0.2,
    ) -> list[RoutingAssignment]:
        """Assign each request greedily to the best feasible cluster sorted
        by combined score.
        """
        # Pre-compute combined scores for all pairs
        combined: dict[tuple[str, str], float] = {}
        for req in requests:
            for cluster in clusters:
                key = (self._req_key(req), cluster.id)
                base = scores.get(key, 0.5)
                bandit = (
                    reward_estimates.get(cluster.id, 0.5) if reward_estimates else 0.5
                )
                combined[key] = 0.7 * base + 0.3 * bandit

        used_budget = 0.0
        used_carbon = 0.0
        assignments: list[RoutingAssignment] = []

        for req in sorted(
            requests, key=lambda r: r.priority, reverse=True
        ):
            # Sort clusters by combined score descending
            candidate_clusters = sorted(
                clusters,
                key=lambda c: combined.get((self._req_key(req), c.id), 0.5),
                reverse=True,
            )

            best: RoutingAssignment | None = None
            for cluster in candidate_clusters:
                # Check latency constraint
                if req.max_latency_ms is not None and req.max_latency_ms > 0:
                    predicted_ms = self._predict_latency(cluster, req)
                    if predicted_ms > req.max_latency_ms:
                        continue

                # Check budget
                cost = self._predict_cost(cluster, req)
                if req.max_budget_usd is not None:
                    if cost > req.max_budget_usd:
                        continue
                total_budget = sum(
                    r.max_budget_usd or float("inf") for r in requests
                )
                if total_budget < float("inf") and used_budget + cost > total_budget:
                    continue

                # Check carbon budget
                carbon = self._predict_carbon(cluster, req)
                if req.max_carbon_g is not None and carbon > req.max_carbon_g:
                    continue
                total_carbon = sum(
                    r.max_carbon_g or float("inf") for r in requests
                )
                if total_carbon < float("inf") and used_carbon + carbon > total_carbon:
                    continue

                # Feasible — assign
                best = self._build_assignment(
                    req, cluster, scores, reward_estimates
                )
                used_budget += cost
                used_carbon += carbon
                break

            if best is None:
                raise ConstraintViolation(
                    f"No feasible cluster for request {self._req_key(req)} "
                    f"(priority={req.priority})"
                )

            assignments.append(best)

        return assignments

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _predict_latency(cluster: Cluster, request: RoutingRequest) -> float:
        """Estimated latency in ms for a request on a cluster."""
        load_penalty = 1.0 + cluster.current_load * 0.5
        token_ratio = max(request.prompt_length, 1) / 1024.0
        return cluster.latency_baseline * load_penalty * math.sqrt(token_ratio)

    @staticmethod
    def _predict_cost(cluster: Cluster, request: RoutingRequest) -> float:
        """Estimated cost in USD for a request on a cluster."""
        total_tokens = max(request.prompt_length + request.max_tokens, 1)
        return cluster.cost_per_token * total_tokens

    @staticmethod
    def _predict_carbon(cluster: Cluster, request: RoutingRequest) -> float:
        """Estimated carbon emission in gCO₂e for a request on a cluster."""
        total_tokens = max(request.prompt_length + request.max_tokens, 1)
        # Rough model: cost × carbon_intensity × conversion factor
        energy_kwh = total_tokens * 0.000_003  # ~3e-6 kWh per token
        return energy_kwh * cluster.carbon_intensity

    @staticmethod
    def _req_key(request: RoutingRequest, idx: int = 0) -> str:
        """Canonical key identifying a request.

        Matches :meth:`AtlasMesh._req_key` exactly so the scores dict built
        by :meth:`AtlasMesh.route_batch` is found by the solver's lookups.
        ``idx`` is accepted for backward compatibility but is intentionally
        excluded from the key (see :func:`_canonical_request_key`).
        """
        return _canonical_request_key(request)

    def _build_assignment(
        self,
        request: RoutingRequest,
        cluster: Cluster,
        scores: dict[tuple[str, str], float],
        reward_estimates: dict[str, float] | None = None,
    ) -> RoutingAssignment:
        key = (self._req_key(request), cluster.id)
        return RoutingAssignment(
            request=request,
            cluster=cluster,
            score=scores.get(key, 0.5),
            reward_estimate=(
                reward_estimates.get(cluster.id, 0.5) if reward_estimates else 0.5
            ),
            expected_cost_usd=self._predict_cost(cluster, request),
            expected_latency_ms=self._predict_latency(cluster, request),
            expected_carbon_g=self._predict_carbon(cluster, request),
        )


# ---------------------------------------------------------------------------
# AtlasMesh
# ---------------------------------------------------------------------------


class AtlasMesh:
    """Global inference mesh combining cluster management, multi-objective
    scoring, learned bandit rewards, and LP-optimal routing.

    Usage::

        mesh = AtlasMesh()
        mesh.cluster_graph.add_cluster(
            "c1", region="us-east-1", provider="aws",
            cost_per_token=0.002, latency_baseline=50.0,
        )
        cluster, cost, lat = mesh.route(model="llama-70b", prompt_length=512)
        print(mesh.stats())
    """

    def __init__(
        self,
        cluster_graph: ClusterGraph | None = None,
        scorer: LatencyCostReliabilityScorer | None = None,
        bandit: ContextualBanditRewardModel | None = None,
        lp_router: LPSolverRouter | None = None,
    ) -> None:
        self.cluster_graph = cluster_graph or ClusterGraph()
        self.scorer = scorer or LatencyCostReliabilityScorer()
        self.bandit = bandit or ContextualBanditRewardModel()
        self.lp_router = lp_router or LPSolverRouter()

        # Internal tracking
        self._assignments: list[RoutingAssignment] = []
        self._lock = threading.RLock()

        # Baseline comparison tracking
        self._baseline_cost: float = 0.0  # naive cost (cheapest cluster)
        self._baseline_latency: float = 0.0  # naive latency (fastest cluster)
        self._baseline_carbon: float = 0.0

    # ── Routing ──────────────────────────────────────────────────────────

    def route(
        self,
        model: str,
        prompt_length: int = 0,
        max_tokens: int = 0,
        complexity: float | None = None,
        priority: float = 1.0,
        max_latency_ms: float | None = None,
        max_budget_usd: float | None = None,
        max_carbon_g: float | None = None,
    ) -> tuple[Cluster, float, float]:
        """Route a single inference request through the mesh.

        Returns:
            ``(selected_cluster, expected_cost_usd, expected_latency_ms)``.

        This is the primary entry-point for single-request routing.
        Internally defers to :meth:`route_batch` with a single-element list.
        """
        results = self.route_batch([
            RoutingRequest(
                model=model,
                prompt_length=prompt_length,
                max_tokens=max_tokens,
                complexity=complexity,
                priority=priority,
                max_latency_ms=max_latency_ms,
                max_budget_usd=max_budget_usd,
                max_carbon_g=max_carbon_g,
            )
        ])
        if not results:
            raise RuntimeError("AtlasMesh.route() returned empty assignment")
        r = results[0]
        return r.cluster, r.expected_cost_usd, r.expected_latency_ms

    def route_batch(
        self,
        requests: list[RoutingRequest],
    ) -> list[RoutingAssignment]:
        """Route a batch of requests through the mesh using LP-optimal
        assignment (or greedy fallback).

        Steps:
            1. Fetch eligible clusters from ``ClusterGraph``.
            2. Score every (request, cluster) pair via the multi-objective
               scorer.
            3. Obtain learned reward estimates from the bandit model.
            4. Solve the assignment problem via ``LPSolverRouter``.
            5. Record routing observations for future training.

        Args:
            requests: One or more requests to route.

        Returns:
            List of :class:`RoutingAssignment` matching the order of
            *requests*.
        """
        if not requests:
            return []

        clusters = self.cluster_graph.all_clusters()
        if not clusters:
            logger.error("AtlasMesh: no clusters registered")
            raise RuntimeError(
                "AtlasMesh has no clusters — call cluster_graph.add_cluster() first"
            )

        # 1. Score every (request, cluster) pair
        scores: dict[tuple[str, str], float] = {}
        for req in requests:
            for cluster in clusters:
                key = (self._req_key(req), cluster.id)
                scores[key] = self.scorer.score(cluster, req)

        # 2. Bandit reward estimates
        reward_estimates: dict[str, float] = {}
        for cluster in clusters:
            estimates = self.bandit.predict(
                requests[-1], [cluster]
            )  # use last request for features
            reward_estimates[cluster.id] = estimates.get(cluster.id, 0.5)

        # More accurate: compute per-request estimates and average
        per_cluster_rewards: dict[str, list[float]] = {}
        for req in requests:
            estimates = self.bandit.predict(req, clusters)
            for cid, est in estimates.items():
                per_cluster_rewards.setdefault(cid, []).append(est)
        reward_estimates = {
            cid: statistics.mean(rewards) for cid, rewards in per_cluster_rewards.items()
        }

        # 3. Solve assignment
        assignments = self.lp_router.solve(
            requests, clusters, scores, reward_estimates,
        )

        # 4. Record observations and update tracking
        with self._lock:
            for assignment in assignments:
                self._assignments.append(assignment)

                # Train bandit with the observed outcome
                features = self.bandit._build_features(
                    assignment.cluster, assignment.request
                )
                # Reward = weighted combination of normalised metrics
                reward = self._compute_outcome_reward(assignment)
                obs = Observation(
                    cluster_id=assignment.cluster.id,
                    features=features,
                    reward=reward,
                )
                self.bandit.train(obs)

                # Track baseline counters
                self._track_baseline(assignment)

        # Drop stale baseline comparisons
        if len(self._assignments) > 10000:
            with self._lock:
                self._assignments = self._assignments[-5000:]

        return assignments

    # ── Statistics ───────────────────────────────────────────────────────

    def stats(self) -> MeshStats:
        """Return aggregated routing statistics.

        Includes total requests routed, cumulative cost/latency/carbon,
        average scores, and savings percentages relative to a naive
        baseline (cheapest-cost cluster for cost, lowest-latency cluster
        for latency, cleanest cluster for carbon).
        """
        with self._lock:
            total = len(self._assignments)
            if total == 0:
                return MeshStats(solver_mode=self.lp_router.mode)

            total_cost = sum(a.expected_cost_usd for a in self._assignments)
            total_latency = sum(a.expected_latency_ms for a in self._assignments)
            total_carbon = sum(a.expected_carbon_g for a in self._assignments)
            avg_score = statistics.mean(a.score for a in self._assignments)
            avg_reward = statistics.mean(
                a.reward_estimate for a in self._assignments
            )

            # Cluster usage breakdown
            usage: dict[str, int] = defaultdict(int)
            for a in self._assignments:
                usage[a.cluster.id] += 1

            # Savings vs baseline
            cost_savings = 0.0
            lat_savings = 0.0
            carb_savings = 0.0
            if self._baseline_cost > 0:
                cost_savings = (
                    (self._baseline_cost - total_cost) / self._baseline_cost * 100.0
                )
            if self._baseline_latency > 0:
                lat_savings = (
                    (self._baseline_latency - total_latency)
                    / self._baseline_latency
                    * 100.0
                )
            if self._baseline_carbon > 0:
                carb_savings = (
                    (self._baseline_carbon - total_carbon)
                    / self._baseline_carbon
                    * 100.0
                )

            return MeshStats(
                total_routed=total,
                total_cost_usd=total_cost,
                total_latency_ms=total_latency,
                total_carbon_g=total_carbon,
                avg_score=avg_score,
                avg_reward_estimate=avg_reward,
                cost_savings_vs_baseline_pct=cost_savings,
                latency_savings_vs_baseline_pct=lat_savings,
                carbon_savings_vs_baseline_pct=carb_savings,
                cluster_usage=dict(usage),
                solver_mode=self.lp_router.mode,
            )

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _req_key(request: RoutingRequest) -> str:
        """Canonical key identifying a request.

        Must match :meth:`LPSolverRouter._req_key` so the scores dict built
        here is found by the solver (see :func:`_canonical_request_key`).
        """
        return _canonical_request_key(request)

    @staticmethod
    def _compute_outcome_reward(assignment: RoutingAssignment) -> float:
        """Compute a scalar reward in [0, 1] from an assignment for
        bandit training.

        Blends score (quality of routing decision) and normalised inverse
        cost/latency/carbon into a single reward.
        """
        # Score already captures the multi-objective quality
        score_component = assignment.score * 0.5

        # Cost component: cheaper = higher reward
        cost = assignment.expected_cost_usd
        cost_component = max(0.0, 1.0 - cost / 0.01) * 0.2

        # Latency component: faster = higher reward
        lat = assignment.expected_latency_ms
        lat_component = max(0.0, 1.0 - lat / 5000.0) * 0.2

        # Carbon component: cleaner = higher reward
        carb = assignment.expected_carbon_g
        carb_component = max(0.0, 1.0 - carb / 10.0) * 0.1

        return min(score_component + cost_component + lat_component + carb_component, 1.0)

    def _track_baseline(self, assignment: RoutingAssignment) -> None:
        """Accumulate what it *would* have cost if we had always chosen the
        cheapest, fastest, or cleanest cluster.
        """
        clusters = self.cluster_graph.all_clusters()

        # Cheapest cluster for this request
        cheapest = min(clusters, key=lambda c: c.cost_per_token)
        self._baseline_cost += self.lp_router._predict_cost(cheapest, assignment.request)

        # Fastest cluster for this request
        fastest = min(clusters, key=lambda c: c.latency_baseline)
        self._baseline_latency += self.lp_router._predict_latency(
            fastest, assignment.request
        )

        # Cleanest cluster for this request
        cleanest = min(clusters, key=lambda c: c.carbon_intensity)
        self._baseline_carbon += self.lp_router._predict_carbon(
            cleanest, assignment.request
        )

    # ── Convenience ──────────────────────────────────────────────────────

    def reset_stats(self) -> None:
        """Reset all accumulated routing statistics and baseline counters."""
        with self._lock:
            self._assignments.clear()
            self._baseline_cost = 0.0
            self._baseline_latency = 0.0
            self._baseline_carbon = 0.0

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"AtlasMesh(clusters={len(self.cluster_graph)}, "
            f"routed={s.total_routed}, "
            f"solver={s.solver_mode})"
        )


# ---------------------------------------------------------------------------
# Public API symbols
# ---------------------------------------------------------------------------

__all__ = [
    # Data classes
    "Cluster",
    "RoutingRequest",
    "RoutingAssignment",
    "Observation",
    "MeshStats",
    "ScoringWeights",
    "ConstraintViolation",
    # Components
    "ClusterGraph",
    "LatencyCostReliabilityScorer",
    "ContextualBanditRewardModel",
    "LPSolverRouter",
    # Composite
    "AtlasMesh",
]
