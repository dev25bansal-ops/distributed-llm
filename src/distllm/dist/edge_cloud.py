"""Edge-to-Cloud Continuum — seamless inference across any device.

Unifies device discovery, capability detection, and transport selection
so that inference works transparently across phones, laptops, desktops,
and cloud GPUs.

Features:
- Automatic device type detection (phone, tablet, laptop, desktop, cloud)
- Capability-aware layer assignment (assigns fewer layers to weaker devices)
- Transport auto-selection (NCCL for local GPUs, gRPC for LAN, QUIC for WAN,
  WebRTC for browsers/mobile)
- Unified discovery combining mDNS (LAN), federation (WAN), and cloud registry
- Adaptive quality scaling (reduces precision/activations on weaker devices)
"""

from __future__ import annotations

import enum
import platform
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


class DeviceType(enum.Enum):
    """Device type classification."""
    PHONE = "phone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    CLOUD_VM = "cloud_vm"
    EDGE_SERVER = "edge_server"
    UNKNOWN = "unknown"


class TransportType(enum.Enum):
    """Transport protocol selection."""
    NCCL = "nccl"          # Same-machine multi-GPU (GPU direct)
    GRPC = "grpc"          # LAN / reliable TCP
    QUIC = "quic"          # WAN / high-latency links
    WEBRTC = "webrtc"      # Browser / mobile / NAT traversal
    DIRECT = "direct"      # In-process (same device)


class NetworkTier(enum.Enum):
    """Network proximity tier."""
    LOCAL = "local"         # Same machine (< 1ms)
    LAN = "lan"             # Same network (< 10ms)
    WAN = "wan"             # Internet (< 200ms)
    REMOTE = "remote"       # High latency (> 200ms)


@dataclass
class DeviceCapabilities:
    """Detected capabilities of a device."""
    device_id: str
    device_type: DeviceType = DeviceType.UNKNOWN
    network_tier: NetworkTier = NetworkTier.REMOTE

    # Hardware
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    gpu_name: str = ""
    cpu_cores: int = 0
    ram_bytes: int = 0
    has_mps: bool = False          # Apple Metal Performance Shaders
    has_cuda: bool = False
    has_rocm: bool = False
    has_xpu: bool = False          # Intel XPU

    # Network
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    supports_quic: bool = False
    supports_webrtc: bool = False
    supports_nccl: bool = False

    # Inference capacity
    max_layers: int = 0            # Max model layers this device can handle
    recommended_dtype: str = "float16"
    recommended_quantization: str | None = None
    max_batch_size: int = 1
    supports_speculative: bool = False

    # Metadata
    os: str = ""
    python_version: str = ""
    torch_version: str = ""
    last_seen: float = field(default_factory=time.time)
    trust_level: float = 0.5       # 0.0-1.0

    @property
    def is_edge(self) -> bool:
        return self.device_type in (DeviceType.PHONE, DeviceType.TABLET)

    @property
    def is_cloud(self) -> bool:
        return self.device_type == DeviceType.CLOUD_VM

    @property
    def effective_tflops(self) -> float:
        """Estimate effective TFLOPS based on device type and hardware."""
        if self.has_cuda and self.gpu_memory_bytes > 0:
            # Rough estimate based on GPU memory (higher mem usually = faster)
            if self.gpu_memory_bytes >= 80 * 1024**3:  # 80GB+ (A100/H100)
                return 300.0
            if self.gpu_memory_bytes >= 24 * 1024**3:  # 24GB+ (RTX 4090, A5000)
                return 100.0
            if self.gpu_memory_bytes >= 8 * 1024**3:   # 8GB+ (RTX 3070, etc.)
                return 30.0
            return 10.0
        if self.has_mps:
            return 15.0  # Apple Silicon
        return float(self.cpu_cores) * 0.5  # CPU fallback


def detect_device_capabilities() -> DeviceCapabilities:
    """Auto-detect the capabilities of the current device.

    Returns a DeviceCapabilities object with all detected hardware
    and network information.
    """
    import os

    caps = DeviceCapabilities(device_id=_generate_device_id())
    caps.os = platform.system().lower()
    caps.python_version = platform.python_version()

    # Detect device type
    caps.device_type = _detect_device_type()

    # Detect CPU and RAM
    caps.cpu_cores = os.cpu_count() or 1
    try:
        import psutil
        caps.ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        caps.ram_bytes = 8 * 1024**3  # Default 8GB

    # Detect GPU
    _detect_gpu(caps)

    # Set inference capacity based on hardware
    _set_inference_capacity(caps)

    logger.info(
        f"Detected device: {caps.device_type.value} | "
        f"GPU: {caps.gpu_count}x {caps.gpu_name or 'none'} | "
        f"RAM: {caps.ram_bytes / 1024**3:.1f}GB | "
        f"Max layers: {caps.max_layers}"
    )
    return caps


def _generate_device_id() -> str:
    """Generate a stable device identifier."""
    import hashlib
    import os
    raw = f"{platform.node()}-{platform.machine()}-{os.getpid()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _detect_device_type() -> DeviceType:
    """Classify the current device type."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Check for cloud VM indicators
    import os
    if any(os.path.exists(p) for p in [
        "/etc/cloud", "/run/cloud-init", "/var/lib/cloud"
    ]):
        return DeviceType.CLOUD_VM

    # Check for container
    if os.path.exists("/.dockerenv") or os.environ.get("KUBERNETES_SERVICE_HOST"):
        return DeviceType.CLOUD_VM

    # macOS detection
    if system == "darwin":
        if machine == "arm64":
            # Apple Silicon — could be laptop or desktop
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True, text=True, timeout=5,
                )
                model = result.stdout.strip().lower()
                if "macbook" in model:
                    return DeviceType.LAPTOP
                return DeviceType.DESKTOP
            except Exception:
                return DeviceType.LAPTOP
        return DeviceType.DESKTOP

    # Windows/Linux laptop detection (heuristic: battery present)
    if system in ("windows", "linux"):
        try:
            import psutil
            if hasattr(psutil, "sensors_battery") and psutil.sensors_battery():
                return DeviceType.LAPTOP
        except Exception:
            pass
        return DeviceType.DESKTOP

    return DeviceType.UNKNOWN


def _detect_gpu(caps: DeviceCapabilities) -> None:
    """Detect GPU hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            caps.has_cuda = True
            caps.gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            caps.gpu_memory_bytes = int(props.total_memory)
            caps.gpu_name = props.name
            caps.supports_nccl = caps.gpu_count > 1
            try:
                caps.torch_version = torch.__version__
            except Exception:
                pass
            return
    except ImportError:
        pass

    # Check for Apple MPS
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            caps.has_mps = True
            caps.gpu_count = 1
            caps.gpu_name = "Apple Silicon GPU"
            # Estimate unified memory
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5,
                )
                caps.gpu_memory_bytes = int(result.stdout.strip())
            except Exception:
                caps.gpu_memory_bytes = 16 * 1024**3
            return
    except ImportError:
        pass

    # Check for Intel XPU
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            caps.has_xpu = True
            caps.gpu_count = torch.xpu.device_count()
            caps.gpu_name = "Intel XPU"
            caps.gpu_memory_bytes = 16 * 1024**3  # Estimate
            return
    except (ImportError, AttributeError):
        pass


def _set_inference_capacity(caps: DeviceCapabilities) -> None:
    """Set recommended inference parameters based on detected hardware."""
    if caps.gpu_memory_bytes >= 80 * 1024**3:
        # High-end GPU (A100/H100)
        caps.max_layers = 999  # Can handle full model
        caps.recommended_dtype = "bfloat16"
        caps.max_batch_size = 32
        caps.supports_speculative = True
    elif caps.gpu_memory_bytes >= 24 * 1024**3:
        # Mid-range GPU (RTX 4090, A5000)
        caps.max_layers = 40
        caps.recommended_dtype = "float16"
        caps.max_batch_size = 8
        caps.supports_speculative = True
    elif caps.gpu_memory_bytes >= 8 * 1024**3:
        # Entry GPU (RTX 3070, etc.)
        caps.max_layers = 16
        caps.recommended_dtype = "float16"
        caps.recommended_quantization = "bnb_4bit"
        caps.max_batch_size = 4
    elif caps.has_mps:
        # Apple Silicon
        caps.max_layers = 20
        caps.recommended_dtype = "float16"
        caps.max_batch_size = 4
    elif caps.ram_bytes >= 16 * 1024**3:
        # CPU with decent RAM
        caps.max_layers = 8
        caps.recommended_dtype = "float32"
        caps.recommended_quantization = "bnb_4bit"
        caps.max_batch_size = 1
    else:
        # Low-end device
        caps.max_layers = 4
        caps.recommended_dtype = "float32"
        caps.recommended_quantization = "bnb_4bit"
        caps.max_batch_size = 1


def select_transport(
    local_caps: DeviceCapabilities,
    remote_caps: DeviceCapabilities,
) -> TransportType:
    """Select the optimal transport protocol between two devices.

    Decision matrix:
    - Same machine + CUDA → NCCL (GPU direct)
    - LAN + reliable → gRPC
    - WAN + supports QUIC → QUIC
    - Browser/mobile → WebRTC
    - Fallback → gRPC
    """
    # Same machine
    if local_caps.network_tier == NetworkTier.LOCAL:
        if local_caps.supports_nccl and remote_caps.supports_nccl:
            return TransportType.NCCL
        return TransportType.DIRECT

    # LAN
    if local_caps.network_tier == NetworkTier.LAN:
        return TransportType.GRPC

    # WAN
    if local_caps.network_tier == NetworkTier.WAN:
        if local_caps.supports_quic and remote_caps.supports_quic:
            return TransportType.QUIC
        return TransportType.GRPC

    # Remote / browser / mobile
    if remote_caps.is_edge or remote_caps.supports_webrtc:
        return TransportType.WEBRTC

    return TransportType.GRPC


def assign_layers_for_continuum(
    devices: list[DeviceCapabilities],
    total_layers: int,
) -> dict[str, list[int]]:
    """Assign model layers across devices proportionally to their capability.

    Stronger devices get more layers. Edge devices get fewer layers.
    Cloud GPUs with huge VRAM may get the entire model.

    Args:
        devices: List of device capabilities.
        total_layers: Total number of model layers.

    Returns:
        Dict mapping device_id -> list of layer indices.
    """
    if not devices:
        return {}

    if len(devices) == 1:
        return {devices[0].device_id: list(range(total_layers))}

    # Score each device by effective capacity
    scores: list[float] = []
    for dev in devices:
        score = dev.effective_tflops
        # Penalize edge devices (they should get fewer layers)
        if dev.is_edge:
            score *= 0.3
        # Penalize high-latency devices
        if dev.latency_ms > 100:
            score *= 0.5
        # Bonus for trust level
        score *= (0.5 + dev.trust_level * 0.5)
        scores.append(max(score, 0.1))

    total_score = sum(scores)
    assignments: dict[str, list[int]] = {}
    layer_idx = 0

    for i, dev in enumerate(devices):
        if i == len(devices) - 1:
            # Last device gets remaining layers
            n_layers = total_layers - layer_idx
        else:
            n_layers = max(1, int(round(total_layers * scores[i] / total_score)))
            n_layers = min(n_layers, total_layers - layer_idx)

        assignments[dev.device_id] = list(range(layer_idx, layer_idx + n_layers))
        layer_idx += n_layers

    # Ensure first device gets layer 0 and last gets final layer
    first_id = devices[0].device_id
    last_id = devices[-1].device_id
    if 0 not in assignments.get(first_id, []):
        assignments[first_id] = [0] + assignments.get(first_id, [])
    if (total_layers - 1) not in assignments.get(last_id, []):
        assignments[last_id] = assignments.get(last_id, []) + [total_layers - 1]

    return assignments


@dataclass
class ContinuumNode:
    """A node in the edge-to-cloud continuum."""
    device_id: str
    capabilities: DeviceCapabilities
    transport: TransportType = TransportType.GRPC
    assigned_layers: list[int] = field(default_factory=list)
    host: str = ""
    port: int = 0
    status: str = "discovered"  # discovered, connecting, ready, busy, offline
    last_heartbeat: float = 0.0
    cumulative_tokens: int = 0
    total_requests: int = 0


class EdgeCloudContinuum:
    """Manages the edge-to-cloud inference continuum.

    Coordinates discovery, capability detection, transport selection,
    and layer assignment across all available devices.
    """

    def __init__(self, total_layers: int = 0):
        self.total_layers = total_layers
        self._nodes: dict[str, ContinuumNode] = {}
        self._local_caps: DeviceCapabilities | None = None
        self._lock = __import__("threading").Lock()

    def initialize(self) -> DeviceCapabilities:
        """Initialize the continuum with local device detection."""
        self._local_caps = detect_device_capabilities()
        return self._local_caps

    def register_node(
        self,
        device_id: str,
        capabilities: DeviceCapabilities,
        host: str = "",
        port: int = 0,
    ) -> ContinuumNode:
        """Register a discovered node."""
        with self._lock:
            if self._local_caps:
                transport = select_transport(self._local_caps, capabilities)
            else:
                transport = TransportType.GRPC

            node = ContinuumNode(
                device_id=device_id,
                capabilities=capabilities,
                transport=transport,
                host=host,
                port=port,
                status="discovered",
                last_heartbeat=time.time(),
            )
            self._nodes[device_id] = node

            logger.info(
                f"Registered continuum node: {device_id} "
                f"({capabilities.device_type.value}, {transport.value})"
            )
            return node

    def remove_node(self, device_id: str) -> None:
        """Remove a node from the continuum."""
        with self._lock:
            self._nodes.pop(device_id, None)

    def rebalance_layers(self) -> dict[str, list[int]]:
        """Rebalance layer assignments across all available nodes."""
        with self._lock:
            devices = [
                n.capabilities for n in self._nodes.values()
                if n.status in ("discovered", "ready")
            ]
            if not devices:
                return {}

            assignments = assign_layers_for_continuum(devices, self.total_layers)

            # Update node assignments
            for device_id, layers in assignments.items():
                if device_id in self._nodes:
                    self._nodes[device_id].assigned_layers = layers
                    self._nodes[device_id].status = "ready"

            return assignments

    def get_node_for_layer(self, layer_idx: int) -> ContinuumNode | None:
        """Find the node responsible for a given layer."""
        with self._lock:
            for node in self._nodes.values():
                if layer_idx in node.assigned_layers:
                    return node
            return None

    def get_nodes(self) -> list[ContinuumNode]:
        """Get all registered nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_stats(self) -> dict[str, Any]:
        """Get continuum statistics."""
        with self._lock:
            nodes = list(self._nodes.values())
            return {
                "total_nodes": len(nodes),
                "ready_nodes": sum(1 for n in nodes if n.status == "ready"),
                "device_types": {
                    t.value: sum(1 for n in nodes if n.capabilities.device_type == t)
                    for t in DeviceType
                    if any(n.capabilities.device_type == t for n in nodes)
                },
                "transports": {
                    t.value: sum(1 for n in nodes if n.transport == t)
                    for t in TransportType
                    if any(n.transport == t for n in nodes)
                },
                "total_layers_assigned": sum(len(n.assigned_layers) for n in nodes),
                "total_layers": self.total_layers,
                "total_tokens_served": sum(n.cumulative_tokens for n in nodes),
            }

    def heartbeat(self, device_id: str) -> bool:
        """Update heartbeat for a node. Returns False if node unknown."""
        with self._lock:
            node = self._nodes.get(device_id)
            if not node:
                return False
            node.last_heartbeat = time.time()
            return True

    def detect_stale_nodes(self, timeout_seconds: float = 30.0) -> list[str]:
        """Find nodes that haven't sent a heartbeat recently."""
        with self._lock:
            now = time.time()
            stale = []
            for did, node in self._nodes.items():
                if now - node.last_heartbeat > timeout_seconds:
                    stale.append(did)
                    node.status = "offline"
            return stale
