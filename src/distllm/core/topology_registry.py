"""TopologyRegistry — the single source of truth for layer assignments (M-06).

Previously, layer-assignment logic was split between
``distllm.dist.pipeline.orchestrator.PipelineOrchestrator.validate_layer_assignment``
(overlap detection) and ``core.cluster_manager.ClusterManager`` (auto-split
computation via the node registrar), with duplicated invariants and no
shared tests. This module consolidates the pure topology math:

- ``validate_assignment`` — no-overlap invariant for a new node
- ``compute_even_split``  — N nodes across L layers with remainder spread
- ``compute_vram_weighted_split`` — proportional to per-node capacity hints
- ``coverage_gaps`` — detect holes in a full layer coverage

Both the orchestrator and the cluster manager delegate here. The functions
are pure (no I/O, no locks) so they are trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeRange:
    node_id: str
    start_layer: int
    end_layer: int  # inclusive

    @property
    def size(self) -> int:
        return self.end_layer - self.start_layer + 1

    def overlaps(self, other: NodeRange) -> bool:
        return self.start_layer <= other.end_layer and self.end_layer >= other.start_layer


def validate_assignment(
    new_id: str,
    start_layer: int,
    end_layer: int,
    existing: list[NodeRange],
) -> None:
    """Raise ``ValueError`` if ``[start_layer, end_layer]`` overlaps any node.

    Same message format as the orchestrator's historical check so existing
    callers that match on error text keep working.
    """
    candidate = NodeRange(new_id, start_layer, end_layer)
    for node in existing:
        if node.node_id == new_id:
            continue
        if candidate.overlaps(node):
            raise ValueError(
                f"Layer overlap: {new_id}[{start_layer}-{end_layer}] "
                f"overlaps with {node.node_id}[{node.start_layer}-{node.end_layer}]"
            )


def compute_even_split(
    num_layers: int,
    node_ids: list[str],
) -> list[NodeRange]:
    """Split ``num_layers`` across ``node_ids`` in order, spreading remainder.

    E.g. 10 layers over 3 nodes -> [0-3], [4-6], [7-9] (the first node(s)
    absorb the remainder, matching integer-div floor semantics used by
    the original cluster-manager auto-split).
    """
    n = len(node_ids)
    if n <= 0:
        return []
    base, rem = divmod(num_layers, n)
    ranges: list[NodeRange] = []
    cursor = 0
    for i, nid in enumerate(node_ids):
        size = base + (1 if i < rem else 0)
        ranges.append(NodeRange(nid, cursor, cursor + size - 1))
        cursor += size
    return ranges


def compute_vram_weighted_split(
    num_layers: int,
    node_capacity_gb: dict[str, float],
) -> list[NodeRange]:
    """Split layers proportionally to each node's capacity hint.

    Nodes with larger ``capacity_gb`` receive proportionally more layers.
    Falls back to an even split when the total capacity is non-positive.
    """
    total = sum(node_capacity_gb.values())
    if total <= 0 or not node_capacity_gb:
        return compute_even_split(num_layers, list(node_capacity_gb))
    # Largest-remainder allocation
    raw = {nid: cap / total * num_layers for nid, cap in node_capacity_gb.items()}
    counts = {nid: int(v) for nid, v in raw.items()}
    assigned = sum(counts.values())
    # Distribute the remainder to the largest fractional parts
    order = sorted(node_capacity_gb, key=lambda nid: raw[nid] - counts[nid], reverse=True)
    i = 0
    while assigned < num_layers:
        counts[order[i % len(order)]] += 1
        assigned += 1
        i += 1
    ranges: list[NodeRange] = []
    cursor = 0
    for nid in order:
        size = counts[nid]
        if size <= 0:
            continue
        ranges.append(NodeRange(nid, cursor, cursor + size - 1))
        cursor += size
    return ranges


def coverage_gaps(ranges: list[NodeRange], num_layers: int) -> list[tuple[int, int]]:
    """Return the inclusive layer intervals NOT covered by ``ranges``."""
    if not ranges:
        return [(0, num_layers - 1)] if num_layers > 0 else []
    covered = sorted((r.start_layer, r.end_layer) for r in ranges)
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in covered:
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor < num_layers:
        gaps.append((cursor, num_layers - 1))
    return [g for g in gaps if g[0] <= g[1]]
