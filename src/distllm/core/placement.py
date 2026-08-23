"""WAN/topology-aware placement, carbon-aware live-migration planning, and canary/AB routing.

This module is the single source of truth for *where* a request / model replica
should be placed across a cluster of nodes spread over a WAN.  It is deliberately
**pure** (no network, no GPU, no global state) so that it can be unit-tested in
isolation and reused by:

  * :class:`distllm.core.batch_scheduler.BatchScheduler` (a thin ``set_placement_policy``
    hook delegates here without touching ``Backends.select()`` — that is owned by E9),
  * the API router for canary/AB request routing, and
  * the carbon-aware live-migration scheduler (``plan_migration``).

Concepts
--------
``NodeTopology``
    A map of ``node_id -> LinkInfo`` describing, *from the coordinator's point of
    view*, the RTT (``latency_ms``), link capacity (``bandwidth_gbps``), region,
    and grid carbon intensity (``carbon_intensity`` in gCO₂/kWh) for each candidate
    node.  It can be built directly, or adapted from the existing
    ``distllm.dist.network.Topology`` bandwidth matrix.

``PlacementPolicy``
    Tunable weights and thresholds.  Two independent behaviours:

      1. **Topology-aware ranking** — prefer low-latency / high-bandwidth links.
      2. **Carbon-aware tie-break** — when a candidate's latency is within
         ``latency_threshold_ms`` of the best (lowest-latency) candidate, the
         carbon term is allowed to flip the ordering so greener regions win.

``CanaryRouter`` / ``route_canary``
    Deterministic (seedable) A/B split: a ``canary_weight`` fraction of request
    ids is routed to ``canary_node_ids`` while the remainder goes to stable nodes.

``plan_migration``
    Pure planner that, given the current node set and topology, returns an *ordered*
    list of migration steps that most reduce the combined carbon+latency cost.
    It performs **no** actual migration.

Scoring formula
--------------
For a single node ``i``::

    net_i       = latency_i / ref_latency + ref_bandwidth / bandwidth_i
    carbon_i    = carbon_weight * (carbon_i / ref_carbon)   # only if carbon-aware
                                                     # AND latency_i <= min_latency + threshold
    score_i     = net_i + carbon_i                       # LOWER IS BETTER

``min_latency`` is the minimum ``latency_ms`` across all candidates, so the carbon
term only activates for nodes that are "close enough" in latency — guaranteeing the
topology-aware requirement dominates when latency differs beyond the threshold.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable


# ── Default reference points (normalisation constants) ─────────────────────
# Chosen so a "typical" WAN edge node (10 ms, 100 Gbps, 400 gCO₂/kWh) scores ~1.0.
DEFAULT_REF_LATENCY_MS = 10.0
DEFAULT_REF_BANDWIDTH_GBPS = 100.0
DEFAULT_REF_CARBON_INTENSITY = 400.0


@dataclass(frozen=True)
class LinkInfo:
    """Link characteristics from the coordinator to a candidate node."""

    node_id: str
    region: str = "default"
    latency_ms: float = 0.0
    bandwidth_gbps: float = 10.0
    carbon_intensity: float = 0.0

    def __post_init__(self) -> None:
        if self.bandwidth_gbps <= 0:
            raise ValueError(
                f"LinkInfo.bandwidth_gbps must be > 0 (got {self.bandwidth_gbps})"
            )
        if self.latency_ms < 0:
            raise ValueError(
                f"LinkInfo.latency_ms must be >= 0 (got {self.latency_ms})"
            )


@dataclass(frozen=True)
class NodeTopology:
    """Coordinator-centric view of all candidate nodes and their links."""

    coordinator_region: str = "default"
    links: dict[str, LinkInfo] = field(default_factory=dict)

    def get(self, node_id: str) -> LinkInfo:
        link = self.links.get(node_id)
        if link is None:
            raise KeyError(f"Node '{node_id}' not present in topology")
        return link

    def node_ids(self) -> list[str]:
        return list(self.links.keys())

    # ── Adapters from existing dist modules (lazy imports, optional) ──

    @classmethod
    def from_network(
        cls,
        topology: "object",
        regions: dict[int, str] | None = None,
        carbon: dict[str, float] | None = None,
        coordinator_idx: int = 0,
    ) -> "NodeTopology":
        """Build a ``NodeTopology`` from the existing ``distllm.dist.network.Topology``.

        Args:
            topology: A ``distllm.dist.network.Topology`` (numpy bandwidth matrix +
                adjacency list).  Imported lazily so this module stays import-light.
            regions: Optional map ``node_index -> region`` string.  Defaults to the
                matrix index as the region name.
            carbon: Optional map ``region -> carbon_intensity`` (gCO₂/kWh).  Defaults
                to neutral (0.0) for every region.
            coordinator_idx: Index of the coordinator node (latency to itself = 0).

        Returns:
            A ``NodeTopology`` keyed by ``"node{idx}"`` strings.
        """
        import numpy as np  # noqa: WPS433 (lazy)

        n = int(topology.num_nodes)
        bw: np.ndarray = np.asarray(topology.bandwidth_matrix, dtype=float)
        regions = regions or {i: f"region-{i}" for i in range(n)}
        carbon = carbon or {}
        links: dict[str, LinkInfo] = {}
        for i in range(n):
            # Latency proxy: derived from the inverse of available bandwidth on the
            # coordinator->i link (no direct latency matrix exists in network.Topology).
            link_bw = float(bw[coordinator_idx, i]) if bw.shape[0] > coordinator_idx else 0.0
            link_bw = link_bw if link_bw > 0 else DEFAULT_REF_BANDWIDTH_GBPS * 0.1
            region = regions.get(i, f"region-{i}")
            links[f"node{i}"] = LinkInfo(
                node_id=f"node{i}",
                region=region,
                latency_ms=DEFAULT_REF_LATENCY_MS * (DEFAULT_REF_BANDWIDTH_GBPS / link_bw),
                bandwidth_gbps=link_bw,
                carbon_intensity=carbon.get(region, 0.0),
            )
        return cls(coordinator_region=regions.get(coordinator_idx, "default"), links=links)


@dataclass(frozen=True)
class PlacementPolicy:
    """Tunable placement weights and thresholds."""

    carbon_aware: bool = True
    carbon_weight: float = 0.3
    latency_threshold_ms: float = 50.0
    ref_latency_ms: float = DEFAULT_REF_LATENCY_MS
    ref_bandwidth_gbps: float = DEFAULT_REF_BANDWIDTH_GBPS
    ref_carbon_intensity: float = DEFAULT_REF_CARBON_INTENSITY
    # ── Canary / A-B routing ──
    canary_weight: float = 0.0
    canary_node_ids: tuple[str, ...] = ()
    canary_seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.canary_weight <= 1.0):
            raise ValueError("canary_weight must be in [0, 1]")
        if self.carbon_weight < 0:
            raise ValueError("carbon_weight must be >= 0")


# ── Core scoring / ranking ────────────────────────────────────────────────

def score_node(link: LinkInfo, policy: PlacementPolicy) -> float:
    """Return the placement cost for a single node. Lower is better.

    See module docstring for the exact formula.
    """
    net = (link.latency_ms / policy.ref_latency_ms) + (
        policy.ref_bandwidth_gbps / link.bandwidth_gbps
    )
    return net  # carbon term handled in select_placement (needs min latency)


def _carbon_term(link: LinkInfo, policy: PlacementPolicy) -> float:
    if not policy.carbon_aware:
        return 0.0
    return policy.carbon_weight * (link.carbon_intensity / policy.ref_carbon_intensity)


def select_placement(
    candidates: Iterable[str],
    topology: NodeTopology,
    policy: PlacementPolicy,
) -> list[str]:
    """Rank candidate nodes by combined cost (topology-aware + carbon-aware).

    Ordering:
      1. Compute ``net`` cost for every candidate.
      2. Find ``min_latency`` across candidates.
      3. Add the carbon term **only** for candidates whose latency is within
         ``latency_threshold_ms`` of ``min_latency`` — so topology awareness
         dominates when latency differs beyond the threshold, but greener nodes
         win ties within the threshold.
      4. Return node ids sorted ascending by total score (best first).

    Args:
        candidates: Iterable of node ids present in ``topology``.
        topology: Coordinator-centric link view.
        policy: Placement weights/thresholds.

    Returns:
        Node ids ordered best→worst (lowest cost first).
    """
    ids = [c for c in candidates if c in topology.links]
    if not ids:
        return []

    links = [topology.get(c) for c in ids]
    min_latency = min(l.latency_ms for l in links)

    scored: list[tuple[float, str]] = []
    for link in links:
        net = score_node(link, policy)
        carbon = _carbon_term(link, policy) if (
            policy.carbon_aware
            and link.latency_ms <= min_latency + policy.latency_threshold_ms
        ) else 0.0
        scored.append((net + carbon, link.node_id))

    # Stable sort ascending by cost; ties broken by node id for determinism.
    scored.sort(key=lambda t: (t[0], t[1]))
    return [node_id for _, node_id in scored]


# ── Canary / A-B routing ─────────────────────────────────────────────────

def route_canary(
    request_ids: Iterable[str],
    stable_nodes: Iterable[str],
    canary_nodes: Iterable[str],
    canary_weight: float = 0.0,
    seed: int = 0,
) -> dict[str, str]:
    """Deterministically split requests between stable and canary node sets.

    Exactly ``round(len(requests) * canary_weight)`` requests are routed to the
    canary set (round-robin across ``canary_nodes``); the rest go to the stable
    set (round-robin across ``stable_nodes``).  Assignment is a pure function of
    ``(request_ids, canary_nodes, stable_nodes, canary_weight, seed)`` — given the
    same inputs it always produces the same mapping, so A/B experiments are
    reproducible.

    Args:
        request_ids: Request identifiers to route.
        stable_nodes: Default/stable node ids.
        canary_nodes: Canary/experiment node ids.
        canary_weight: Fraction (0–1) of traffic to send to canary nodes.
        seed: RNG seed for the deterministic shuffle that picks *which* requests
            become canary traffic (avoids contiguous blocks of canary requests).

    Returns:
        Mapping ``request_id -> assigned_node_id``.

    Raises:
        ValueError: If ``canary_weight > 0`` but ``canary_nodes`` is empty, or if
            ``canary_weight < 1`` but ``stable_nodes`` is empty.
    """
    reqs = list(request_ids)
    canary = list(canary_nodes)
    stable = list(stable_nodes)

    if canary_weight > 0 and not canary:
        raise ValueError("canary_weight > 0 but canary_nodes is empty")
    if canary_weight < 1 and not stable:
        raise ValueError("canary_weight < 1 but stable_nodes is empty")

    if not reqs:
        return {}

    n_canary = int(round(len(reqs) * canary_weight))

    rng = random.Random(seed)
    order = reqs[:]
    rng.shuffle(order)

    assignment: dict[str, str] = {}
    for idx, rid in enumerate(order):
        if idx < n_canary:
            assignment[rid] = canary[idx % len(canary)]
        else:
            assignment[rid] = stable[(idx - n_canary) % len(stable)]
    return assignment


class CanaryRouter:
    """Stateful wrapper around :func:`route_canary` for the API router.

    Holds the configured canary node set and weight, and exposes a
    ``route(request_ids)`` method.  Pure aside from the seedable RNG.
    """

    def __init__(
        self,
        stable_nodes: Iterable[str],
        canary_nodes: Iterable[str],
        canary_weight: float = 0.0,
        seed: int = 0,
    ):
        self.stable_nodes = list(stable_nodes)
        self.canary_nodes = list(canary_nodes)
        self.canary_weight = canary_weight
        self.seed = seed

    def route(self, request_ids: Iterable[str]) -> dict[str, str]:
        return route_canary(
            request_ids,
            self.stable_nodes,
            self.canary_nodes,
            canary_weight=self.canary_weight,
            seed=self.seed,
        )

    def canary_fraction(self, n_requests: int) -> int:
        """How many of ``n_requests`` would be routed to canary nodes."""
        return int(round(n_requests * self.canary_weight))


# ── Carbon-aware live migration planner ──────────────────────────────────

@dataclass(frozen=True)
class MigrationStep:
    """One ordered step in a live-migration plan (no actual migration performed)."""

    order: int
    from_node: str
    to_node: str
    cost_before: float
    cost_after: float
    cost_reduction: float
    reason: str


def _node_cost(link: LinkInfo, policy: PlacementPolicy) -> float:
    """Combined placement cost of a single node (with carbon fully enabled)."""
    net = score_node(link, policy)
    carbon = _carbon_term(link, policy)
    return net + carbon


def plan_migration(
    nodes: Iterable[str],
    topology: NodeTopology,
    policy: PlacementPolicy | None = None,
) -> list[MigrationStep]:
    """Return an ordered plan that migrates load toward lower-cost nodes.

    Greedy planner: at each step it pairs the highest-cost node (a migration
    *source*) with the lowest-cost node (a migration *target*) and records the
    expected cost reduction.  Steps are ordered by descending ``cost_reduction``
    so the most beneficial moves happen first.  This is a *pure* planning
    function — it never touches a real cluster.

    A migration is only emitted when it actually reduces cost
    (``cost_after < cost_before``); if nothing can be improved the result is empty.

    Args:
        nodes: Current set of node ids (sources and potential targets).
        topology: Coordinator-centric link view.
        policy: Optional policy (defaults to ``PlacementPolicy()``).  Note the
            carbon term is always considered here (migration is a slow, deliberate
            decision, so carbon awareness always applies).

    Returns:
        List of :class:`MigrationStep` ordered best→worst by cost reduction.
    """
    policy = policy or PlacementPolicy()
    ids = [n for n in nodes if n in topology.links]
    if len(ids) < 2:
        return []

    costs = {nid: _node_cost(topology.get(nid), policy) for nid in ids}

    # Pair every (source, target) where moving source→target reduces cost, and
    # keep only the *best* (max-reduction) target for each source.
    best_per_source: dict[str, MigrationStep] = {}
    for src in ids:
        for tgt in ids:
            if src == tgt:
                continue
            reduction = costs[src] - costs[tgt]
            if reduction <= 0:
                continue
            candidate = MigrationStep(
                order=0,
                from_node=src,
                to_node=tgt,
                cost_before=costs[src],
                cost_after=costs[tgt],
                cost_reduction=reduction,
                reason=(
                    f"move load off high-cost node {src} "
                    f"(cost={costs[src]:.3f}) onto greener/lower-latency "
                    f"node {tgt} (cost={costs[tgt]:.3f})"
                ),
            )
            cur = best_per_source.get(src)
            if cur is None or reduction > cur.cost_reduction:
                best_per_source[src] = candidate

    # Order by descending reduction so the most impactful move is first.
    ordered = sorted(
        best_per_source.values(),
        key=lambda s: (-s.cost_reduction, s.from_node, s.to_node),
    )
    return [
        MigrationStep(
            order=i,
            from_node=s.from_node,
            to_node=s.to_node,
            cost_before=s.cost_before,
            cost_after=s.cost_after,
            cost_reduction=s.cost_reduction,
            reason=s.reason,
        )
        for i, s in enumerate(ordered, start=1)
    ]


__all__ = [
    "LinkInfo",
    "NodeTopology",
    "PlacementPolicy",
    "MigrationStep",
    "score_node",
    "select_placement",
    "route_canary",
    "CanaryRouter",
    "plan_migration",
]
