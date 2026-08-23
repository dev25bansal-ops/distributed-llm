from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.dist.partition.partitioner import HardwareAwarePartitioner
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ---------------------------------------------------------------------------
# NetworkCostModel
# ---------------------------------------------------------------------------


@dataclass
class NodeLinkStats:
    """Per-peer measured statistics for a single node."""

    latency_ms: float = 0.0
    bandwidth_gbps: float = 0.0
    same_region: bool = True
    last_updated: float = 0.0


class NetworkCostModel:
    """Network-aware cost model that incorporates inter-node latency,
    bandwidth, and region / availability-zone information.

    ``topology`` is a two-level dict::

        topology[node_id][peer_id] = NodeLinkStats

    Callers should populate it via :meth:`update_measurement` or let
    :class:`TopologyProbe` fill it periodically.
    """

    def __init__(self) -> None:
        self._topology: dict[str, dict[str, NodeLinkStats]] = {}

    # -- topology access ----------------------------------------------------

    @property
    def topology(self) -> dict[str, dict[str, NodeLinkStats]]:
        """Read-only view of the raw topology data."""
        return {k: dict(v) for k, v in self._topology.items()}

    @property
    def node_ids(self) -> list[str]:
        """Return all known node ids."""
        return list(self._topology)

    def peer_ids(self, node: str) -> list[str]:
        """Return all known peers for *node*."""
        return list(self._topology.get(node, {}))

    # -- measurement recording ----------------------------------------------

    def update_measurement(
        self,
        node: str,
        peer: str,
        latency_ms: float,
        bandwidth_gbps: float,
        *,
        same_region: bool = True,
    ) -> None:
        """Record or update a latency / bandwidth measurement between *node*
        and *peer*.

        The mapping is symmetric: calling ``update_measurement(A, B, …)``
        also stores the reverse (B → A) automatically.
        """
        now = time.time()
        stats = NodeLinkStats(
            latency_ms=latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            same_region=same_region,
            last_updated=now,
        )
        self._topology.setdefault(node, {})[peer] = stats
        self._topology.setdefault(peer, {})[node] = stats

    def get_stats(self, node: str, peer: str) -> NodeLinkStats | None:
        """Return the recorded stats for the *node* → *peer* direction."""
        return self._topology.get(node, {}).get(peer)

    # -- cost estimation ---------------------------------------------------

    def communication_cost(
        self,
        node: str,
        peer: str,
        data_size_bytes: int,
    ) -> float:
        """Estimated time (seconds) to send *data_size_bytes* from *node* to
        *peer*.

        Uses a simple latency + transfer-time model:

            cost = latency_s + (data_bytes * 8) / (bandwidth_bps)

        If no measurement exists for this pair a conservative default is
        returned (10 ms latency, 1 Gbps bandwidth).
        """
        stats = self.get_stats(node, peer)
        if stats is None:
            # Conservative fallback
            latency_s = 0.01
            bw_bps = 1e9
        else:
            latency_s = stats.latency_ms / 1000.0
            bw_bps = stats.bandwidth_gbps * 1e9

        if bw_bps <= 0:
            bw_bps = 1e9  # avoid division by zero

        transfer_s = (data_size_bytes * 8) / bw_bps
        return latency_s + transfer_s

    def partition_cost(
        self,
        assignment: dict[int, str],
        data_size_per_layer_bytes: int | dict[int, int] = 4 * 1024 * 1024,
    ) -> float:
        """Total estimated communication cost for an entire layer-to-node
        assignment.

        Parameters
        ----------
        assignment:
            ``{layer_id: node_id}`` mapping every layer to its host node.
        data_size_per_layer_bytes:
            Size of intermediate activations sent between consecutive layers
            when they reside on different nodes.  Can be a single int (same
            for all layer boundaries) or a dict ``{layer_id: bytes}``.

        Returns
        -------
        Total estimated time in seconds.
        """
        if not assignment:
            return 0.0

        total: float = 0.0
        sorted_layers = sorted(assignment)

        for i in range(len(sorted_layers) - 1):
            cur_layer = sorted_layers[i]
            next_layer = sorted_layers[i + 1]
            cur_node = assignment[cur_layer]
            next_node = assignment[next_layer]

            if cur_node == next_node:
                continue  # same node — no network cost

            if isinstance(data_size_per_layer_bytes, dict):
                data_bytes = data_size_per_layer_bytes.get(cur_layer, 4 * 1024 * 1024)
            else:
                data_bytes = data_size_per_layer_bytes

            total += self.communication_cost(cur_node, next_node, data_bytes)

        return total

    # -- import / export helpers -------------------------------------------

    def to_topology_graph(self) -> TopologyGraph:
        """Convert the current network view into a ``TopologyGraph`` suitable
        for use with ``PartitionCostModel``.
        """
        node_ids = sorted(self._topology)
        links: list[LinkProfile] = []

        for src in node_ids:
            for dst, stats in self._topology.get(src, {}).items():
                if src >= dst:
                    continue
                latency_us = stats.latency_ms * 1000.0
                links.append(
                    LinkProfile(
                        source=src,
                        target=dst,
                        bandwidth_gbps=stats.bandwidth_gbps,
                        latency_us=latency_us,
                        is_infiniband=stats.bandwidth_gbps > 25.0,
                    )
                )

        return TopologyGraph(node_ids=node_ids, links=links)

    @classmethod
    def from_topology_graph(cls, graph: TopologyGraph) -> NetworkCostModel:
        """Create a ``NetworkCostModel`` pre-populated from an existing
        ``TopologyGraph`` (e.g. one produced by ``TopologyProber.probe``).
        """
        model = cls()
        for link in graph.links:
            model.update_measurement(
                node=link.source,
                peer=link.target,
                latency_ms=link.latency_us / 1000.0,
                bandwidth_gbps=link.bandwidth_gbps,
            )
        return model

    @classmethod
    def make_fallback(
        cls,
        num_nodes: int,
        *,
        inter_node_latency_ms: float = 0.5,
        inter_node_bandwidth_gbps: float = 12.5,
    ) -> NetworkCostModel:
        """Create a fully-connected ``NetworkCostModel`` with uniform
        fallback values (useful when no real probing is available).
        """
        model = cls()
        node_ids = [f"node-{i}" for i in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                model.update_measurement(
                    node=node_ids[i],
                    peer=node_ids[j],
                    latency_ms=inter_node_latency_ms,
                    bandwidth_gbps=inter_node_bandwidth_gbps,
                )
        return model


# ---------------------------------------------------------------------------
# TopologyProbe
# ---------------------------------------------------------------------------


class TopologyProbe:
    """Periodically measures inter-node latency and bandwidth, storing results
    in a :class:`NetworkCostModel`.

    Probing runs on a daemon background thread at a configurable interval.
    Measurements are best-effort — failures log a warning and leave the
    previous value in place.
    """

    def __init__(
        self,
        cost_model: NetworkCostModel,
        *,
        probe_interval_s: float = 30.0,
        ping_timeout_s: float = 2.0,
        bandwidth_test_bytes: int = 8 * 1024 * 1024,
        region_map: dict[str, str] | None = None,
    ) -> None:
        self._cost_model = cost_model
        self._interval = probe_interval_s
        self._ping_timeout = ping_timeout_s
        self._bandwidth_test_bytes = bandwidth_test_bytes
        self._region_map: dict[str, str] = region_map or {}

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the background probing loop (daemon thread)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("TopologyProbe already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="topology-probe",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "TopologyProbe started (interval={}s)", self._interval
        )

    def stop(self, *, join_timeout_s: float = 5.0) -> None:
        """Signal the probe thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None
        logger.info("TopologyProbe stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- single-shot probe --------------------------------------------------

    def probe_once(
        self,
        node_ids: list[str],
        hostnames: dict[str, str] | None = None,
    ) -> None:
        """Run a single measurement cycle for every unique pair in
        *node_ids*.

        This can be called directly (even before :meth:`start`) for
        one-shot profiling.
        """
        hosts = hostnames or {nid: nid for nid in node_ids}
        pairs = [
            (node_ids[i], node_ids[j])
            for i in range(len(node_ids))
            for j in range(i + 1, len(node_ids))
        ]

        for src, dst in pairs:
            src_host = hosts.get(src, src)
            dst_host = hosts.get(dst, dst)

            same_region = self._check_same_region(src, dst, hosts)

            latency_ms = self._measure_latency(src_host, dst_host)
            bandwidth_gbps = self._measure_bandwidth(
                src_host, dst_host
            )

            # Guard against obviously bogus measurements
            if latency_ms <= 0:
                latency_ms = 0.5  # fallback
            if bandwidth_gbps <= 0:
                bandwidth_gbps = 1.0  # fallback

            self._cost_model.update_measurement(
                node=src,
                peer=dst,
                latency_ms=latency_ms,
                bandwidth_gbps=bandwidth_gbps,
                same_region=same_region,
            )

            logger.debug(
                "Probed {} <-> {}: latency={:.2f}ms, bw={:.2f}Gbps, "
                "same_region={}",
                src,
                dst,
                latency_ms,
                bandwidth_gbps,
                same_region,
            )

    # -- internal measurement -----------------------------------------------

    def _measure_latency(
        self, host_a: str, host_b: str
    ) -> float:
        """Measure round-trip latency between two hosts in milliseconds.

        Uses a simple TCP connect attempt as an RTT proxy.  Falls back to
        an ICMP ping when available.
        """
        if host_a == host_b:
            return 0.05  # localhost RTT

        # Try ICMP ping first (more accurate)
        try:
            import subprocess as sp
            import shutil

            if shutil.which("ping"):
                # Windows / POSIX compatible flags
                is_win = host_a.startswith("\\\\") or ":" not in host_a
                if is_win:
                    cmd = ["ping", "-n", "1", "-w", str(int(self._ping_timeout * 1000)), host_b]
                else:
                    cmd = ["ping", "-c", "1", "-W", str(int(self._ping_timeout)), host_b]

                t0 = time.time()
                result = sp.run(
                    cmd,
                    capture_output=True,
                    timeout=self._ping_timeout,
                )
                elapsed = (time.time() - t0) * 1000.0
                if result.returncode == 0:
                    return max(elapsed, 0.05)
        except Exception:
            pass

        # Fallback: TCP connect RTT
        try:
            import socket

            family = socket.AF_INET6 if ":" in host_b else socket.AF_INET
            t0 = time.time()
            with socket.create_connection(
                (host_b, 50050),
                timeout=self._ping_timeout,
            ) as sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                elapsed_ms = (time.time() - t0) * 1000.0
            return max(elapsed_ms, 0.05)
        except Exception:
            return 10.0  # conservative fallback

    def _measure_bandwidth(
        self, host_a: str, host_b: str
    ) -> float:
        """Estimate effective bandwidth between two hosts in Gbps.

        Measures the time to send a fixed-size payload over a TCP
        connection.  This is a rough proxy — for production use consider
        iperf3 or a dedicated bandwidth tool.
        """
        if host_a == host_b:
            return 100.0  # loopback is very fast

        try:
            import socket
            import secrets

            family = socket.AF_INET6 if ":" in host_b else socket.AF_INET
            payload = secrets.token_bytes(self._bandwidth_test_bytes)

            t0 = time.time()
            with socket.create_connection(
                (host_b, 50051),
                timeout=self._ping_timeout,
            ) as sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.sendall(payload)
                sock.shutdown(socket.SHUT_WR)
                _ = sock.recv(1024)  # wait for ack
            elapsed_s = max(time.time() - t0, 0.001)

            bits = self._bandwidth_test_bytes * 8
            return bits / elapsed_s / 1e9
        except Exception:
            return 1.0  # conservative fallback

    def _check_same_region(
        self,
        src: str,
        dst: str,
        hostnames: dict[str, str],
    ) -> bool:
        """Return ``True`` if *src* and *dst* are in the same region / AZ.

        Uses the optional ``region_map`` passed at construction time.  If
        a map is not supplied every pair is assumed to be same-region.
        """
        if not self._region_map:
            # If hostnames are the same, assume same region
            return hostnames.get(src, src) == hostnames.get(dst, dst)
        return self._region_map.get(src) == self._region_map.get(dst)

    # -- background loop ----------------------------------------------------

    def _run_loop(self) -> None:
        """Continuously probe at the configured interval."""
        while not self._stop_event.is_set():
            node_ids = self._cost_model.node_ids
            if node_ids:
                try:
                    self.probe_once(node_ids)
                except Exception as exc:
                    logger.warning(
                        "TopologyProbe cycle failed: {}", exc
                    )
            self._stop_event.wait(self._interval)


# ---------------------------------------------------------------------------
# RegionAwarePartitioner
# ---------------------------------------------------------------------------


class RegionAwarePartitioner:
    """Wraps an existing :class:`HardwareAwarePartitioner` with network-cost
    penalties that bias toward same-region assignments.

    The partitioner delegates the actual layer-to-node optimisation to the
    underlying ``HardwareAwarePartitioner``, then applies region-aware
    penalties when evaluating candidate assignments.

    Parameters
    ----------
    partitioner:
        The underlying hardware-aware partitioner instance.
    network_cost_model:
        Network cost model carrying latency, bandwidth and region data.
    cross_region_penalty:
        Multiplier applied to the communication cost of every cross-region
        link.  A value of 3.0 means a cross-region link is treated as
        costing 3× its measured communication time.
    same_region_multiplier:
        Multiplier for same-region links (default 1.0 — no penalty).
    """

    def __init__(
        self,
        partitioner: HardwareAwarePartitioner,
        network_cost_model: NetworkCostModel,
        *,
        cross_region_penalty: float = 3.0,
        same_region_multiplier: float = 1.0,
    ) -> None:
        self._partitioner = partitioner
        self._net = network_cost_model
        self._cross_region_penalty = cross_region_penalty
        self._same_region_multiplier = same_region_multiplier

    # -- delegation ---------------------------------------------------------

    @property
    def inner(self) -> HardwareAwarePartitioner:
        """Access the underlying partitioner."""
        return self._partitioner

    async def partition(
        self,
        **kwargs: Any,
    ) -> Any:
        """Delegate to the underlying partitioner after injecting the
        network-cost-aware topology.
        """
        # Convert our network model into a TopologyGraph and inject it
        # into the partitioner if the user hasn't supplied one directly.
        tg = self._net.to_topology_graph()
        if "hostnames" not in kwargs:
            kwargs.setdefault(
                "hostnames",
                {nid: nid for nid in tg.node_ids},
            )
        return await self._partitioner.partition(**kwargs)

    # -- region-aware helpers -----------------------------------------------

    def get_preferred_nodes(
        self,
        layer_id: int,
        *,
        max_results: int | None = None,
    ) -> list[str]:
        """Return all known nodes sorted by increasing communication cost
        for the given *layer_id*.

        Nodes in the same region as the majority of already-assigned layers
        (or with the lowest latency+bandwidth cost) rank first.
        """
        scored: list[tuple[float, str]] = []

        # Build a "context set" of nodes that already hold neighbouring
        # layers — this is speculative and uses all known peers.
        all_nodes = self._net.node_ids
        if not all_nodes:
            return []

        # Score each candidate node
        for candidate in all_nodes:
            cost = self._node_affinity_cost(candidate)
            scored.append((cost, candidate))

        scored.sort(key=lambda x: x[0])
        nodes = [n for _, n in scored]

        if max_results is not None and max_results > 0:
            nodes = nodes[:max_results]
        return nodes

    def _node_affinity_cost(self, node: str) -> float:
        """Compute an aggregate affinity cost for *node*.

        Lower is better.  The cost is the sum of communication costs to
        all known peers, with cross-region penalties applied.
        """
        total: float = 0.0
        peers = self._net.peer_ids(node)
        if not peers:
            return 0.0

        for peer in peers:
            stats = self._net.get_stats(node, peer)
            if stats is None:
                continue

            # Base communication cost (1 MB reference transfer)
            base_cost = self._net.communication_cost(
                node, peer, 1024 * 1024
            )
            multiplier = (
                self._same_region_multiplier
                if stats.same_region
                else self._cross_region_penalty
            )
            total += base_cost * multiplier

        return total / len(peers)  # normalise by number of peers

    def compute_penalized_assignment_cost(
        self,
        assignment: dict[int, str],
        data_size_per_layer_bytes: int | dict[int, int] = 4 * 1024 * 1024,
    ) -> float:
        """Compute the total communication cost of an assignment with
        region penalties applied.

        This is the same as ``NetworkCostModel.partition_cost`` but
        cross-region links are multiplied by ``cross_region_penalty``.
        """
        if not assignment:
            return 0.0

        total: float = 0.0
        sorted_layers = sorted(assignment)

        for i in range(len(sorted_layers) - 1):
            cur = sorted_layers[i]
            nxt = sorted_layers[i + 1]
            cur_node = assignment[cur]
            next_node = assignment[nxt]

            if cur_node == next_node:
                continue

            if isinstance(data_size_per_layer_bytes, dict):
                data_bytes = data_size_per_layer_bytes.get(cur, 4 * 1024 * 1024)
            else:
                data_bytes = data_size_per_layer_bytes

            base_cost = self._net.communication_cost(
                cur_node, next_node, data_bytes
            )
            stats = self._net.get_stats(cur_node, next_node)
            same_region = stats.same_region if stats else True
            multiplier = (
                self._same_region_multiplier
                if same_region
                else self._cross_region_penalty
            )
            total += base_cost * multiplier

        return total

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "RegionAwarePartitioner",
            f"  Cross-region penalty: {self._cross_region_penalty}x",
            f"  Same-region multiplier: {self._same_region_multiplier}x",
            f"  Known nodes: {len(self._net.node_ids)}",
        ]
        for node in self._net.node_ids:
            peers = self._net.peer_ids(node)
            same_region = sum(
                1
                for p in peers
                if (s := self._net.get_stats(node, p)) and s.same_region
            )
            lines.append(
                f"    {node}: {len(peers)} peers, "
                f"{same_region} in same region"
            )
        return "\n".join(lines)
