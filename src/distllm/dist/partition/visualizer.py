"""Cluster topology visualizer.

Rich terminal visualization of cluster topology, GPU profiles,
partition assignments, and link bandwidths.

Typical usage::

    viz = ClusterVisualizer()
    viz.print_topology(topology, gpu_profiles)
    viz.print_partition(solution, layer_weights, gpu_profiles)
    json_str = viz.to_json(topology, solution, gpu_profiles)
"""

from __future__ import annotations

import json
from typing import Any

from distllm.dist.partition.cost_model import NodeCost
from distllm.dist.partition.optimizer import PartitionSolution
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import TopologyGraph


def _fmt_bytes(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / (1024**3):.1f}GB"
    if b >= 1024 ** 2:
        return f"{b / (1024**2):.0f}MB"
    return f"{b}B"


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - len(s))


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


class ClusterVisualizer:
    """Terminal visualization of cluster topology and partitions."""

    def print_topology(
        self,
        topology: TopologyGraph,
        gpu_profiles: dict[str, GPUProfile] | list[GPUProfile] | None = None,
    ) -> str:
        """Print cluster topology as a formatted table.

        Args:
            topology: Cluster topology graph.
            gpu_profiles: Optional GPU profiles for detail.

        Returns:
            Formatted string (also printed to stdout).
        """
        profiles = self._normalize_profiles(gpu_profiles)
        lines: list[str] = []

        total_gpus = topology.total_gpus()
        lines.append(f"Cluster Topology: {len(topology.node_ids)} nodes, {total_gpus} GPUs")
        lines.append("")

        # Node table
        header = f"  {_pad('Node', 16)} {_pad('GPU', 12)} {_pad('VRAM', 8)} {_pad('TFLOPS', 8)} {_pad('BW (Gbps)', 10)}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for nid in topology.node_ids:
            gpu_count = topology.gpu_counts.get(nid, 1)
            prof = profiles.get(nid)
            if prof:
                gpu_name = _pad(prof.name, 12)
                vram = _pad(_fmt_bytes(prof.total_memory_bytes), 8)
                tflops = _pad(f"{prof.compute_tflops:.0f}", 8)
                bw = _pad(f"{prof.memory_bandwidth_gbps:.0f}", 10)
            else:
                gpu_name = _pad("unknown", 12)
                vram = _pad("?", 8)
                tflops = _pad("?", 8)
                bw = _pad("?", 10)

            lines.append(f"  {_pad(nid, 16)} {gpu_name} {vram} {tflops} {bw}")
            if gpu_count > 1:
                for g in range(1, gpu_count):
                    lines.append(f"  {_pad(f'{nid}/gpu{g}', 16)} {gpu_name} {vram} {tflops} {bw}")

        # Link table
        if topology.links:
            lines.append("")
            lines.append("  Links:")
            for link in topology.links:
                link_type = "NVLink" if link.is_nvlink else ("IB" if link.is_infiniband else "Eth")
                lines.append(
                    f"    {link.source} <-> {link.target}: "
                    f"{link.bandwidth_gbps:.1f} Gbps, {link.latency_us:.0f}us ({link_type})"
                )

        output = "\n".join(lines)
        print(output)
        return output

    def print_partition(
        self,
        solution: PartitionSolution,
        layer_weights: list[LayerWeights] | None = None,
        gpu_profiles: dict[str, GPUProfile] | list[GPUProfile] | None = None,
        node_costs: list[NodeCost] | None = None,
    ) -> str:
        """Print partition solution as a formatted table.

        Args:
            solution: Partition solution.
            layer_weights: Optional layer weights for type info.
            gpu_profiles: Optional GPU profiles for VRAM info.
            node_costs: Optional per-node cost breakdowns.

        Returns:
            Formatted string (also printed to stdout).
        """
        profiles = self._normalize_profiles(gpu_profiles)
        lines: list[str] = []

        lines.append(
            f"Partition: {solution.num_nodes} nodes, "
            f"{solution.coverage[1] - solution.coverage[0]} layers, "
            f"max latency {solution.max_node_time_ms:.1f}ms"
        )
        lines.append("")

        # Assignment table
        header = (
            f"  {_pad('Node', 16)} {_pad('Layers', 12)} {_pad('Count', 6)} "
            f"{_pad('Time', 10)} {_pad('Memory', 12)} {_pad('Util', 22)} Status"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for i, pt in enumerate(solution.points):
            prof = profiles.get(pt.node_id)
            layer_range = f"[{pt.start_layer}, {pt.end_layer})"
            layer_count = pt.end_layer - pt.start_layer
            time_str = f"{pt.estimated_time_ms:.1f}ms"

            mem_str = "?"
            util_str = ""
            status = "OK"

            if node_costs and i < len(node_costs):
                nc = node_costs[i]
                mem_str = f"{_fmt_bytes(nc.memory_bytes)}/{_fmt_bytes(nc.memory_available_bytes)}"
                util_pct = nc.memory_utilization
                util_str = f"{_bar(util_pct)} {util_pct:.0%}"
                if not nc.fits_in_memory:
                    status = "OOM!"
            elif prof:
                mem_str = f"?/{_fmt_bytes(prof.total_memory_bytes)}"

            lines.append(
                f"  {_pad(pt.node_id, 16)} {_pad(layer_range, 12)} {_pad(str(layer_count), 6)} "
                f"{_pad(time_str, 10)} {_pad(mem_str, 12)} {_pad(util_str, 22)} {status}"
            )

        # Summary
        lines.append("")
        lines.append(f"  Max node time: {solution.max_node_time_ms:.1f}ms")
        lines.append(f"  Pipeline latency: {solution.pipeline_latency_ms:.1f}ms")
        lines.append(f"  Throughput: {solution.estimated_throughput_tok_s:.0f} tok/s")
        if solution.num_oom_nodes > 0:
            lines.append(f"  OOM nodes: {solution.num_oom_nodes}")
        if solution.explanation:
            lines.append(f"  Strategy: {solution.explanation}")

        output = "\n".join(lines)
        print(output)
        return output

    def print_comparison(
        self, comparison: dict[str, Any],
    ) -> str:
        """Print strategy comparison as a formatted table."""
        lines: list[str] = []
        lines.append("Strategy Comparison:")
        lines.append("")
        lines.append(f"  {_pad('Strategy', 20)} {_pad('Latency', 12)} {_pad('Throughput', 12)}")
        lines.append("  " + "-" * 44)

        for strategy, metrics in comparison.items():
            if isinstance(metrics, dict):
                lat = metrics.get("max_latency_ms", "N/A")
                tp = metrics.get("throughput", "N/A")
                lat_str = f"{lat}ms" if isinstance(lat, (int, float)) else str(lat)
                tp_str = f"{tp} tok/s" if isinstance(tp, (int, float)) else str(tp)
                lines.append(f"  {_pad(strategy, 20)} {_pad(lat_str, 12)} {_pad(tp_str, 12)}")
            else:
                lines.append(f"  {_pad(strategy, 20)} {metrics}")

        output = "\n".join(lines)
        print(output)
        return output

    def to_json(
        self,
        topology: TopologyGraph | None = None,
        solution: PartitionSolution | None = None,
        gpu_profiles: dict[str, GPUProfile] | list[GPUProfile] | None = None,
        node_costs: list[NodeCost] | None = None,
    ) -> str:
        """Export topology/partition as JSON."""
        data: dict[str, Any] = {}

        if topology:
            data["topology"] = topology.to_dict()

        if solution:
            data["solution"] = {
                "num_nodes": solution.num_nodes,
                "coverage": solution.coverage,
                "max_node_time_ms": solution.max_node_time_ms,
                "pipeline_latency_ms": solution.pipeline_latency_ms,
                "throughput_tok_s": solution.estimated_throughput_tok_s,
                "oom_nodes": solution.num_oom_nodes,
                "explanation": solution.explanation,
                "assignments": [
                    {
                        "node_id": p.node_id,
                        "start_layer": p.start_layer,
                        "end_layer": p.end_layer,
                        "estimated_time_ms": p.estimated_time_ms,
                    }
                    for p in solution.points
                ],
            }

        if node_costs:
            data["node_costs"] = [
                {
                    "node_id": nc.node_id,
                    "layers": f"[{nc.start_layer}, {nc.end_layer})",
                    "compute_ms": nc.compute_time_ms,
                    "comm_ms": nc.communication_time_ms,
                    "total_ms": nc.total_time_ms,
                    "memory_bytes": nc.memory_bytes,
                    "memory_available_bytes": nc.memory_available_bytes,
                    "fits_in_memory": nc.fits_in_memory,
                    "utilization": nc.memory_utilization,
                }
                for nc in node_costs
            ]

        profiles = self._normalize_profiles(gpu_profiles)
        if profiles:
            data["gpu_profiles"] = {
                nid: {
                    "name": p.name,
                    "total_memory_bytes": p.total_memory_bytes,
                    "compute_tflops": p.compute_tflops,
                    "memory_bandwidth_gbps": p.memory_bandwidth_gbps,
                    "sm_count": p.sm_count,
                }
                for nid, p in profiles.items()
            }

        return json.dumps(data, indent=2)

    def _normalize_profiles(
        self, profiles: dict[str, GPUProfile] | list[GPUProfile] | None,
    ) -> dict[str, GPUProfile]:
        if profiles is None:
            return {}
        if isinstance(profiles, list):
            return {str(p.gpu_id): p for p in profiles}
        return profiles
