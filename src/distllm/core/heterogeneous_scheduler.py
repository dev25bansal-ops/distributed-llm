"""Heterogeneous pipeline scheduler — assigns model layers across mixed GPU types.

Supports clusters with NVIDIA + AMD + Intel + Apple devices in a single
pipeline. Uses each device's VRAM, compute capacity, and memory bandwidth
to allocate layers proportionally, then orders nodes by throughput to
minimize pipeline bubble.

Also provides disaggregated prefill/decode routing:
- Prefill (compute-bound) → high-TFLOPS nodes
- Decode (memory-bound) → high-bandwidth nodes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from distllm.constants import DeviceFamily, DEVICE_FAMILY
from distllm.core.device_registry import DeviceInfo, detect_all_devices


@dataclass
class HeterogeneousNode:
    """A node in the heterogeneous cluster with its device capabilities."""
    node_id: str
    host: str
    port: int
    device_info: DeviceInfo
    start_layer: int = 0
    end_layer: int = 0
    weight_score: float = 0.0
    throughput_score: float = 0.0


@dataclass
class HeterogeneousCluster:
    """The full cluster description for heterogeneous scheduling."""
    nodes: list[HeterogeneousNode] = field(default_factory=list)
    total_layers: int = 0
    hidden_size: int = 4096
    dtype_bytes: int = 2

    @property
    def device_types(self) -> set[str]:
        return {n.device_info.device_type for n in self.nodes}

    @property
    def device_families(self) -> set[DeviceFamily]:
        return {n.device_info.device_family for n in self.nodes}

    @property
    def is_heterogeneous(self) -> bool:
        """True if cluster has more than one device family."""
        return len(self.device_families) > 1


def build_heterogeneous_cluster(
    node_configs: list[dict[str, Any]],
    total_layers: int = 32,
    hidden_size: int = 4096,
) -> HeterogeneousCluster:
    """Build a HeterogeneousCluster from node configs with auto-detected devices.

    Each config dict should have:
        node_id, host, port (required)
        device_type (optional, auto-detected if not set)
    """
    devices = detect_all_devices()
    device_map: dict[int, DeviceInfo] = {d.device_id: d for d in devices}

    cluster = HeterogeneousCluster(
        total_layers=total_layers,
        hidden_size=hidden_size,
    )

    for cfg in node_configs:
        device_type = cfg.get("device_type", "auto")
        if device_type == "auto":
            from distllm.core.device_registry import detect_platform
            device_type = detect_platform()

        dev_info = DeviceInfo(
            device_type=device_type,
            device_family=DEVICE_FAMILY.get(device_type, DeviceFamily.UNKNOWN),
            device_id=cfg.get("device_id", 0),
            name=cfg.get("gpu_name", device_type.upper()),
            total_memory_bytes=cfg.get("total_memory", 8 * 1024**3),
        )

        node = HeterogeneousNode(
            node_id=cfg["node_id"],
            host=cfg["host"],
            port=cfg["port"],
            device_info=dev_info,
        )
        cluster.nodes.append(node)

    return cluster


def compute_node_weights(cluster: HeterogeneousCluster) -> list[HeterogeneousNode]:
    """Compute weight and throughput scores for each node.

    Weight score: based on VRAM (proportional capacity for model weights).
    Throughput score: based on FP16 TFLOPS and memory bandwidth.
    """
    for node in cluster.nodes:
        di = node.device_info
        mem_gb = di.total_memory_bytes / (1024**3)
        tflops = di.tflops_fp16 or 1.0
        bw = di.memory_bandwidth_gbps or 20.0

        # Weight score: VRAM capacity (linear)
        node.weight_score = max(0.1, mem_gb)

        # Throughput score: combined compute and memory bandwidth
        node.throughput_score = tflops * 0.6 + (bw / 100.0) * 0.4

    return cluster.nodes


def assign_layers_proportional(
    cluster: HeterogeneousCluster,
) -> HeterogeneousCluster:
    """Assign layers to nodes proportional to their weight scores.

    Each node gets a number of consecutive layers proportional to
    its weight_score / total_weight, ensuring contiguous layer ranges
    for pipeline parallelism.
    """
    if not cluster.nodes:
        return cluster

    compute_node_weights(cluster)

    total_weight = sum(n.weight_score for n in cluster.nodes)
    if total_weight <= 0:
        total_weight = len(cluster.nodes)

    assigned = 0
    for i, node in enumerate(cluster.nodes):
        is_last = i == len(cluster.nodes) - 1
        if is_last:
            node.start_layer = assigned
            node.end_layer = cluster.total_layers - 1
        else:
            proportion = node.weight_score / total_weight
            layers_for_node = max(1, int(cluster.total_layers * proportion))
            node.start_layer = assigned
            node.end_layer = min(assigned + layers_for_node - 1, cluster.total_layers - 1)
            assigned = node.end_layer + 1

    return cluster


def order_nodes_by_throughput(
    cluster: HeterogeneousCluster,
) -> HeterogeneousCluster:
    """Order nodes by throughput (fastest first) to minimize pipeline bubble.

    In a heterogeneous pipeline, the slowest node determines overall
    throughput. Ordering by throughput helps balance.
    """
    cluster.nodes.sort(key=lambda n: n.throughput_score, reverse=True)
    return cluster


def get_device_compatibility_map() -> dict[str, list[str]]:
    """Return which device types can communicate efficiently with each other.

    Returns dict mapping source device type to list of compatible peer types.
    """
    return {
        "cuda": ["cuda", "cpu"],
        "rocm": ["rocm", "cpu"],
        "mps": ["mps", "cpu"],
        "xpu": ["xpu", "cpu"],
        "cpu": ["cuda", "rocm", "mps", "xpu", "cpu"],
    }


def estimate_heterogeneous_throughput(
    cluster: HeterogeneousCluster,
) -> float:
    """Estimate overall pipeline throughput in tokens/second.

    Accounts for the slowest node and network transfer overhead
    between different device families.
    """
    if not cluster.nodes:
        return 0.0

    min_throughput = min(n.throughput_score for n in cluster.nodes)

    cross_family_penalty = 1.0
    families = list(cluster.device_families)
    if len(families) > 1:
        cross_family_penalty = 0.85  # 15% penalty for cross-platform transfers

    node_count = len(cluster.nodes)
    pipeline_efficiency = 1.0 - (0.05 * (node_count - 1))

    return min_throughput * cross_family_penalty * pipeline_efficiency


def schedule_heterogeneous_pipeline(
    node_configs: list[dict[str, Any]],
    total_layers: int,
    hidden_size: int = 4096,
) -> list[dict[str, Any]]:
    """Full heterogeneous pipeline scheduling pipeline.

    Args:
        node_configs: List of dicts with node_id, host, port, optional device_type.
        total_layers: Total number of layers in the model.
        hidden_size: Model hidden dimension.

    Returns:
        List of node assignments with start_layer, end_layer, layers_device info.
    """
    cluster = build_heterogeneous_cluster(node_configs, total_layers, hidden_size)

    if not cluster.is_heterogeneous:
        compute_node_weights(cluster)
        assign_layers_proportional(cluster)
    else:
        logger.info(
            f"Heterogeneous cluster detected: {[f.value for f in cluster.device_families]}"
        )
        order_nodes_by_throughput(cluster)
        assign_layers_proportional(cluster)

    throughput = estimate_heterogeneous_throughput(cluster)
    logger.info(f"Estimated heterogeneous throughput: {throughput:.1f} tokens/s")

    assignments = []
    for node in cluster.nodes:
        di = node.device_info
        assignments.append({
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "start_layer": node.start_layer,
            "end_layer": node.end_layer,
            "device_type": di.device_type,
            "device_family": di.device_family.value,
            "gpu_name": di.name,
            "total_memory_gb": round(di.total_memory_bytes / (1024**3), 1),
            "throughput_score": round(node.throughput_score, 2),
        })

    return assignments


class NodeRole(str, Enum):
    """Node role for prefill-decode disaggregation."""
    AUTO = "auto"
    PREFILL = "prefill"
    DECODE = "decode"


def assign_prefill_decode_roles(cluster: HeterogeneousCluster) -> dict[str, NodeRole]:
    """Assign prefill/decode roles to nodes based on device characteristics.

    Prefill is compute-bound (favors high TFLOPS).
    Decode is memory-bound (favors high memory bandwidth).

    When nodes are roughly equivalent, the first half are assigned prefill
    and the second half decode to balance throughput.
    """
    if not cluster.nodes:
        return {}

    compute_node_weights(cluster)

    tflops_list = [(n.node_id, n.device_info.tflops_fp16 or 1.0, n.device_info.memory_bandwidth_gbps or 20.0)
                   for n in cluster.nodes]

    tflops_ratios = []
    for _, tflops, bw in tflops_list:
        ratio = tflops / max(bw, 1.0)
        tflops_ratios.append(ratio)

    mean_ratio = sum(tflops_ratios) / max(len(tflops_ratios), 1)
    threshold = mean_ratio * 1.1

    roles: dict[str, NodeRole] = {}
    prefills = 0
    decodes = 0

    for i, (nid, tflops, bw) in enumerate(tflops_list):
        ratio = tflops_ratios[i]
        if ratio > threshold and tflops > 50:
            roles[nid] = NodeRole.PREFILL
            prefills += 1
        elif ratio < mean_ratio * 0.9 and bw > 100:
            roles[nid] = NodeRole.DECODE
            decodes += 1
        else:
            roles[nid] = NodeRole.AUTO

    if prefills == 0 and decodes == 0:
        mid = len(cluster.nodes) // 2
        for i, n in enumerate(cluster.nodes):
            roles[n.node_id] = NodeRole.PREFILL if i < mid else NodeRole.DECODE

    return roles


@dataclass
class PrefillDecodeRoute:
    """Routing decision for a single request step."""
    node_id: str
    is_prefill: bool
    kv_transfer_required: bool = False
    source_node_id: str = ""


class PrefillDecodeRouter:
    """Routes requests to prefill or decode nodes and manages KV cache transfer.

    In a disaggregated setup:
    - Prefill phase (full prompt) → one or more prefill nodes
    - Decode phase (single-token steps) → decode nodes
    - KV cache is transferred from prefill to decode after the first step
    """

    def __init__(
        self,
        prefill_node_ids: list[str],
        decode_node_ids: list[str],
        auto_node_ids: list[str] | None = None,
    ):
        self._prefill_nodes = list(prefill_node_ids)
        self._decode_nodes = list(decode_node_ids)
        self._auto_nodes = list(auto_node_ids or [])
        self._all_nodes = prefill_node_ids + decode_node_ids + self._auto_nodes
        self._next_prefill: int = 0
        self._next_decode: int = 0
        self._kv_sources: dict[str, str] = {}

    @property
    def is_disaggregated(self) -> bool:
        """True when dedicated prefill *and* decode node pools exist."""
        return bool(self._prefill_nodes) and bool(self._decode_nodes)

    def route(self, is_prefill_step: bool, request_id: str = "") -> PrefillDecodeRoute:
        """Route a request step to the appropriate node.

        Args:
            is_prefill_step: True for the full-prompt step, False for decode steps.
            request_id: Optional request ID for KV transfer tracking.

        Returns:
            A ``PrefillDecodeRoute`` with the target node and transfer metadata.
        """
        if not self.is_disaggregated:
            nid = self._all_nodes[0] if self._all_nodes else ""
            return PrefillDecodeRoute(node_id=nid, is_prefill=is_prefill_step)

        if is_prefill_step:
            nid = self._next_prefill_node()
            return PrefillDecodeRoute(
                node_id=nid, is_prefill=True, kv_transfer_required=False,
            )

        nid = self._next_decode_node()
        source = self._kv_sources.get(request_id, "")
        return PrefillDecodeRoute(
            node_id=nid, is_prefill=False,
            kv_transfer_required=bool(source and source != nid),
            source_node_id=source,
        )

    def record_prefill_node(self, request_id: str, prefill_node_id: str) -> None:
        """Record which prefill node handled a request (for KV transfer routing)."""
        self._kv_sources[request_id] = prefill_node_id

    def _next_prefill_node(self) -> str:
        if not self._prefill_nodes:
            return self._all_nodes[0] if self._all_nodes else ""
        nid = self._prefill_nodes[self._next_prefill % len(self._prefill_nodes)]
        self._next_prefill += 1
        return nid

    def _next_decode_node(self) -> str:
        if not self._decode_nodes:
            return self._all_nodes[0] if self._all_nodes else ""
        nid = self._decode_nodes[self._next_decode % len(self._decode_nodes)]
        self._next_decode += 1
        return nid

    def cleanup_request(self, request_id: str) -> None:
        """Remove KV transfer tracking for a completed request."""
        self._kv_sources.pop(request_id, None)

    @classmethod
    def from_node_roles(cls, roles: dict[str, NodeRole]) -> PrefillDecodeRouter:
        """Build a router from a role assignment dict."""
        prefills = [nid for nid, r in roles.items() if r == NodeRole.PREFILL]
        decodes = [nid for nid, r in roles.items() if r == NodeRole.DECODE]
        autos = [nid for nid, r in roles.items() if r == NodeRole.AUTO]
        return cls(prefill_node_ids=prefills, decode_node_ids=decodes, auto_node_ids=autos)


def build_disaggregated_pipeline_plan(
    node_configs: list[dict[str, Any]],
    total_layers: int,
    hidden_size: int = 4096,
    force_disagg: bool = False,
) -> dict[str, Any]:
    """Build a full disaggregated pipeline plan.

    Returns a dict with:
        - ``roles``: node_id → NodeRole mapping
        - ``layer_assignments``: list of node assignments with start/end layers
        - ``router``: configured PrefillDecodeRouter
        - ``is_disaggregated``: whether dedicated prefill/decode pools exist
    """
    cluster = build_heterogeneous_cluster(node_configs, total_layers, hidden_size)
    compute_node_weights(cluster)
    assign_layers_proportional(cluster)

    roles = assign_prefill_decode_roles(cluster)
    router = PrefillDecodeRouter.from_node_roles(roles)

    assignments = []
    for node in cluster.nodes:
        assignments.append({
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "start_layer": node.start_layer,
            "end_layer": node.end_layer,
            "device_type": node.device_info.device_type,
            "device_family": node.device_info.device_family.value,
            "role": roles.get(node.node_id, NodeRole.AUTO).value,
            "throughput_score": round(node.throughput_score, 2),
        })

    return {
        "roles": roles,
        "layer_assignments": assignments,
        "router": router,
        "is_disaggregated": router.is_disaggregated,
    }


@dataclass
class KVTransferRecord:
    """Tracks a KV cache transfer between prefill and decode nodes."""
    request_id: str
    source_node_id: str
    dest_node_id: str
    kv_data: Any = None
    transfer_bytes: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    status: str = "pending"  # pending, transferring, completed, failed


class KVTransferManager:
    """Manages KV cache transfers between prefill and decode nodes.

    In a disaggregated pipeline, after the prefill phase the KV cache
    must be transferred to the decode node. This manager handles the
    async transfer, tracks in-flight transfers, and provides fallback
    when transfer fails.
    """

    def __init__(self, pipeline_orchestrator: Any = None):
        self._pipeline = pipeline_orchestrator
        self._transfers: dict[str, KVTransferRecord] = {}
        self._lock = __import__("threading").Lock()

    def initiate_transfer(
        self,
        request_id: str,
        source_node_id: str,
        dest_node_id: str,
        kv_data: Any = None,
    ) -> KVTransferRecord:
        """Initiate a KV cache transfer from prefill to decode node."""
        record = KVTransferRecord(
            request_id=request_id,
            source_node_id=source_node_id,
            dest_node_id=dest_node_id,
            kv_data=kv_data,
            started_at=time.time(),
            status="transferring",
        )
        with self._lock:
            self._transfers[request_id] = record

        # Estimate transfer size
        if kv_data is not None:
            try:
                import sys
                record.transfer_bytes = sys.getsizeof(kv_data)
            except Exception:
                record.transfer_bytes = 0

        logger.debug(
            f"KV transfer initiated: {request_id} "
            f"{source_node_id} -> {dest_node_id} "
            f"({record.transfer_bytes} bytes)"
        )
        return record

    def complete_transfer(self, request_id: str, success: bool = True) -> None:
        """Mark a transfer as completed or failed."""
        with self._lock:
            record = self._transfers.get(request_id)
            if record:
                record.completed_at = time.time()
                record.status = "completed" if success else "failed"
                if record.status == "completed":
                    elapsed_ms = (record.completed_at - record.started_at) * 1000
                    logger.debug(f"KV transfer completed: {request_id} in {elapsed_ms:.1f}ms")

    def get_transfer(self, request_id: str) -> KVTransferRecord | None:
        """Get the transfer record for a request."""
        with self._lock:
            return self._transfers.get(request_id)

    def cleanup_request(self, request_id: str) -> None:
        """Remove transfer tracking for a completed request."""
        with self._lock:
            self._transfers.pop(request_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Get transfer statistics."""
        with self._lock:
            records = list(self._transfers.values())
        completed = [r for r in records if r.status == "completed"]
        failed = [r for r in records if r.status == "failed"]
        in_flight = [r for r in records if r.status == "transferring"]

        avg_latency = 0.0
        if completed:
            latencies = [(r.completed_at - r.started_at) * 1000 for r in completed]
            avg_latency = sum(latencies) / len(latencies)

        return {
            "total_transfers": len(records),
            "completed": len(completed),
            "failed": len(failed),
            "in_flight": len(in_flight),
            "avg_latency_ms": round(avg_latency, 2),
            "total_bytes": sum(r.transfer_bytes for r in records),
        }


class DisaggregatedPipelineExecutor:
    """Executes requests through a disaggregated prefill/decode pipeline.

    Orchestrates the full flow:
    1. Route prefill step to a prefill node
    2. Transfer KV cache from prefill to decode node
    3. Route decode steps to decode nodes
    4. Return the final result

    Integrates with PrefillDecodeRouter for node selection and
    KVTransferManager for cache transfers.
    """

    def __init__(
        self,
        router: PrefillDecodeRouter,
        pipeline_orchestrator: Any = None,
        transfer_manager: KVTransferManager | None = None,
    ):
        self._router = router
        self._pipeline = pipeline_orchestrator
        self._kv_transfer = transfer_manager or KVTransferManager(pipeline_orchestrator)
        self._stats = {
            "total_requests": 0,
            "prefill_requests": 0,
            "decode_requests": 0,
            "kv_transfers": 0,
            "transfer_failures": 0,
        }

    @property
    def is_disaggregated(self) -> bool:
        return self._router.is_disaggregated

    def route_prefill(self, request_id: str) -> PrefillDecodeRoute:
        """Route the prefill phase of a request to a prefill node."""
        route = self._router.route(is_prefill_step=True, request_id=request_id)
        self._router.record_prefill_node(request_id, route.node_id)
        self._stats["prefill_requests"] += 1
        self._stats["total_requests"] += 1
        return route

    def route_decode(self, request_id: str) -> PrefillDecodeRoute:
        """Route decode steps to a decode node, initiating KV transfer if needed."""
        route = self._router.route(is_prefill_step=False, request_id=request_id)
        self._stats["decode_requests"] += 1

        if route.kv_transfer_required:
            self._stats["kv_transfers"] += 1
            self._kv_transfer.initiate_transfer(
                request_id=request_id,
                source_node_id=route.source_node_id,
                dest_node_id=route.node_id,
            )

        return route

    def complete_request(self, request_id: str) -> None:
        """Clean up after a request completes."""
        self._router.cleanup_request(request_id)
        self._kv_transfer.cleanup_request(request_id)

    def execute_step(
        self,
        request_id: str,
        input_ids: Any,
        is_prefill: bool,
        node_kv_caches: dict | None = None,
    ) -> tuple[str, Any]:
        """Execute a single step through the disaggregated pipeline.

        Args:
            request_id: Request identifier.
            input_ids: Input token IDs for this step.
            is_prefill: True for the initial prefill step, False for decode.
            node_kv_caches: Existing KV caches for this request.

        Returns:
            (node_id, logits) tuple.
        """
        if is_prefill:
            route = self.route_prefill(request_id)
        else:
            route = self.route_decode(request_id)

        if self._pipeline is not None:
            try:
                logits = self._pipeline.run_pipeline(
                    input_ids,
                    node_kv_caches or {},
                    request_id=request_id,
                )
                return route.node_id, logits
            except Exception as e:
                logger.error(f"Pipeline step failed for {request_id}: {e}")
                raise

        return route.node_id, None

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            **self._stats,
            "is_disaggregated": self.is_disaggregated,
            "kv_transfer_stats": self._kv_transfer.get_stats(),
        }
