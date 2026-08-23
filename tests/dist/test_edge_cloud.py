"""Tests for the edge_cloud module -- edge-to-cloud continuum.

Zero mocks -- uses only real objects from the module.
Deterministic: no GPU required, no network, no timing assertions.
"""

from __future__ import annotations

from distllm.dist.edge_cloud import (
    ContinuumNode,
    DeviceCapabilities,
    DeviceType,
    EdgeCloudContinuum,
    NetworkTier,
    TransportType,
    assign_layers_for_continuum,
    detect_device_capabilities,
    select_transport,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestDeviceType:
    """DeviceType enum values and completeness."""

    def test_values(self) -> None:
        assert DeviceType.PHONE.value == "phone"
        assert DeviceType.TABLET.value == "tablet"
        assert DeviceType.LAPTOP.value == "laptop"
        assert DeviceType.DESKTOP.value == "desktop"
        assert DeviceType.CLOUD_VM.value == "cloud_vm"
        assert DeviceType.EDGE_SERVER.value == "edge_server"
        assert DeviceType.UNKNOWN.value == "unknown"

    def test_all_members_present(self) -> None:
        expected = {
            DeviceType.PHONE,
            DeviceType.TABLET,
            DeviceType.LAPTOP,
            DeviceType.DESKTOP,
            DeviceType.CLOUD_VM,
            DeviceType.EDGE_SERVER,
            DeviceType.UNKNOWN,
        }
        assert set(DeviceType) == expected


class TestTransportType:
    """TransportType enum values and completeness."""

    def test_values(self) -> None:
        assert TransportType.NCCL.value == "nccl"
        assert TransportType.GRPC.value == "grpc"
        assert TransportType.QUIC.value == "quic"
        assert TransportType.WEBRTC.value == "webrtc"
        assert TransportType.DIRECT.value == "direct"

    def test_all_members_present(self) -> None:
        expected = {
            TransportType.NCCL,
            TransportType.GRPC,
            TransportType.QUIC,
            TransportType.WEBRTC,
            TransportType.DIRECT,
        }
        assert set(TransportType) == expected


class TestNetworkTier:
    """NetworkTier enum values and completeness."""

    def test_values(self) -> None:
        assert NetworkTier.LOCAL.value == "local"
        assert NetworkTier.LAN.value == "lan"
        assert NetworkTier.WAN.value == "wan"
        assert NetworkTier.REMOTE.value == "remote"

    def test_all_members_present(self) -> None:
        expected = {
            NetworkTier.LOCAL,
            NetworkTier.LAN,
            NetworkTier.WAN,
            NetworkTier.REMOTE,
        }
        assert set(NetworkTier) == expected


# ---------------------------------------------------------------------------
# DeviceCapabilities
# ---------------------------------------------------------------------------


class TestDeviceCapabilities:
    """DeviceCapabilities dataclass construction, properties, edge cases."""

    def test_default_construction(self) -> None:
        caps = DeviceCapabilities(device_id="d1")
        assert caps.device_id == "d1"
        assert caps.device_type == DeviceType.UNKNOWN
        assert caps.network_tier == NetworkTier.REMOTE
        assert caps.gpu_count == 0
        assert caps.gpu_memory_bytes == 0
        assert caps.gpu_name == ""
        assert caps.cpu_cores == 0
        assert caps.ram_bytes == 0
        assert caps.has_mps is False
        assert caps.has_cuda is False
        assert caps.has_rocm is False
        assert caps.has_xpu is False
        assert caps.bandwidth_mbps == 0.0
        assert caps.latency_ms == 0.0
        assert caps.supports_quic is False
        assert caps.supports_webrtc is False
        assert caps.supports_nccl is False
        assert caps.max_layers == 0
        assert caps.recommended_dtype == "float16"
        assert caps.recommended_quantization is None
        assert caps.max_batch_size == 1
        assert caps.supports_speculative is False
        assert caps.os == ""
        assert caps.python_version == ""
        assert caps.torch_version == ""
        assert caps.last_seen > 0
        assert caps.trust_level == 0.5

    def test_construction_with_all_fields(self) -> None:
        caps = DeviceCapabilities(
            device_id="d2",
            device_type=DeviceType.CLOUD_VM,
            network_tier=NetworkTier.LAN,
            gpu_count=4,
            gpu_memory_bytes=80 * 1024**3,
            gpu_name="H100",
            cpu_cores=64,
            ram_bytes=512 * 1024**3,
            has_cuda=True,
            supports_nccl=True,
            max_layers=999,
            recommended_dtype="bfloat16",
            max_batch_size=32,
            os="linux",
        )
        assert caps.device_id == "d2"
        assert caps.device_type == DeviceType.CLOUD_VM
        assert caps.gpu_count == 4
        assert caps.has_cuda is True

    # --- is_edge property ---

    def test_is_edge_phone(self) -> None:
        caps = DeviceCapabilities(device_id="p1", device_type=DeviceType.PHONE)
        assert caps.is_edge is True

    def test_is_edge_tablet(self) -> None:
        caps = DeviceCapabilities(device_id="t1", device_type=DeviceType.TABLET)
        assert caps.is_edge is True

    def test_is_edge_false_for_desktop(self) -> None:
        caps = DeviceCapabilities(device_id="d1", device_type=DeviceType.DESKTOP)
        assert caps.is_edge is False

    def test_is_edge_false_for_cloud(self) -> None:
        caps = DeviceCapabilities(device_id="c1", device_type=DeviceType.CLOUD_VM)
        assert caps.is_edge is False

    # --- is_cloud property ---

    def test_is_cloud_true(self) -> None:
        caps = DeviceCapabilities(device_id="c1", device_type=DeviceType.CLOUD_VM)
        assert caps.is_cloud is True

    def test_is_cloud_false(self) -> None:
        caps = DeviceCapabilities(device_id="d1", device_type=DeviceType.DESKTOP)
        assert caps.is_cloud is False

    # --- effective_tflops ---

    def test_effective_tflops_high_end_cuda(self) -> None:
        caps = DeviceCapabilities(
            device_id="h1", has_cuda=True, gpu_memory_bytes=80 * 1024**3 + 1,
        )
        assert caps.effective_tflops == 300.0

    def test_effective_tflops_mid_range_cuda(self) -> None:
        caps = DeviceCapabilities(
            device_id="m1", has_cuda=True, gpu_memory_bytes=24 * 1024**3,
        )
        assert caps.effective_tflops == 100.0

    def test_effective_tflops_entry_cuda(self) -> None:
        caps = DeviceCapabilities(
            device_id="e1", has_cuda=True, gpu_memory_bytes=8 * 1024**3,
        )
        assert caps.effective_tflops == 30.0

    def test_effective_tflops_low_cuda(self) -> None:
        caps = DeviceCapabilities(
            device_id="l1", has_cuda=True, gpu_memory_bytes=1 * 1024**3,
        )
        assert caps.effective_tflops == 10.0

    def test_effective_tflops_exact_boundary_80gb(self) -> None:
        caps = DeviceCapabilities(
            device_id="b1", has_cuda=True, gpu_memory_bytes=80 * 1024**3,
        )
        assert caps.effective_tflops == 300.0

    def test_effective_tflops_exact_boundary_24gb(self) -> None:
        caps = DeviceCapabilities(
            device_id="b2", has_cuda=True, gpu_memory_bytes=24 * 1024**3,
        )
        assert caps.effective_tflops == 100.0

    def test_effective_tflops_exact_boundary_8gb(self) -> None:
        caps = DeviceCapabilities(
            device_id="b3", has_cuda=True, gpu_memory_bytes=8 * 1024**3,
        )
        assert caps.effective_tflops == 30.0

    def test_effective_tflops_mps(self) -> None:
        caps = DeviceCapabilities(device_id="m1", has_mps=True)
        assert caps.effective_tflops == 15.0

    def test_effective_tflops_mps_overrides_cpu(self) -> None:
        caps = DeviceCapabilities(
            device_id="m2", has_mps=True, cpu_cores=128,
        )
        assert caps.effective_tflops == 15.0

    def test_effective_tflops_cpu_fallback(self) -> None:
        caps = DeviceCapabilities(device_id="cpu1", cpu_cores=8)
        assert caps.effective_tflops == 4.0  # 8 * 0.5

    def test_effective_tflops_cpu_zero_cores(self) -> None:
        caps = DeviceCapabilities(device_id="cpu0")
        assert caps.effective_tflops == 0.0


# ---------------------------------------------------------------------------
# select_transport
# ---------------------------------------------------------------------------


class TestSelectTransport:
    """select_transport -- decision matrix."""

    @staticmethod
    def _caps(
        device_id: str,
        network_tier: NetworkTier = NetworkTier.LAN,
        supports_nccl: bool = False,
        supports_quic: bool = False,
        supports_webrtc: bool = False,
        device_type: DeviceType = DeviceType.DESKTOP,
    ) -> DeviceCapabilities:
        return DeviceCapabilities(
            device_id=device_id,
            network_tier=network_tier,
            supports_nccl=supports_nccl,
            supports_quic=supports_quic,
            supports_webrtc=supports_webrtc,
            device_type=device_type,
        )

    def test_same_machine_both_nccl(self) -> None:
        local = self._caps("local", NetworkTier.LOCAL, supports_nccl=True)
        remote = self._caps("remote", supports_nccl=True)
        assert select_transport(local, remote) == TransportType.NCCL

    def test_same_machine_one_no_nccl(self) -> None:
        local = self._caps("local", NetworkTier.LOCAL, supports_nccl=False)
        remote = self._caps("remote", supports_nccl=True)
        assert select_transport(local, remote) == TransportType.DIRECT

    def test_same_machine_no_nccl(self) -> None:
        local = self._caps("local", NetworkTier.LOCAL)
        remote = self._caps("remote")
        assert select_transport(local, remote) == TransportType.DIRECT

    def test_lan(self) -> None:
        local = self._caps("local", NetworkTier.LAN)
        remote = self._caps("remote")
        assert select_transport(local, remote) == TransportType.GRPC

    def test_wan_both_quic(self) -> None:
        local = self._caps("local", NetworkTier.WAN, supports_quic=True)
        remote = self._caps("remote", supports_quic=True)
        assert select_transport(local, remote) == TransportType.QUIC

    def test_wan_only_local_quic(self) -> None:
        local = self._caps("local", NetworkTier.WAN, supports_quic=True)
        remote = self._caps("remote", supports_quic=False)
        assert select_transport(local, remote) == TransportType.GRPC

    def test_wan_only_remote_quic(self) -> None:
        local = self._caps("local", NetworkTier.WAN, supports_quic=False)
        remote = self._caps("remote", supports_quic=True)
        assert select_transport(local, remote) == TransportType.GRPC

    def test_remote_edge_device(self) -> None:
        local = self._caps("local", NetworkTier.REMOTE)
        remote = self._caps("remote", device_type=DeviceType.PHONE)
        assert select_transport(local, remote) == TransportType.WEBRTC

    def test_remote_remote_webrtc(self) -> None:
        local = self._caps("local", NetworkTier.REMOTE)
        remote = self._caps("remote", supports_webrtc=True)
        assert select_transport(local, remote) == TransportType.WEBRTC

    def test_remote_fallback_grpc(self) -> None:
        local = self._caps("local", NetworkTier.REMOTE)
        remote = self._caps("remote")
        assert select_transport(local, remote) == TransportType.GRPC


# ---------------------------------------------------------------------------
# assign_layers_for_continuum
# ---------------------------------------------------------------------------


class TestAssignLayersForContinuum:
    """assign_layers_for_continuum -- proportional layer distribution."""

    @staticmethod
    def _dev(
        device_id: str,
        tflops: float = 100.0,
        is_edge: bool = False,
        latency_ms: float = 0.0,
        trust_level: float = 0.5,
    ) -> DeviceCapabilities:
        """Create a DeviceCapabilities with deterministic effective_tflops.

        effective_tflops = cpu_cores * 0.5, so we set cpu_cores = tflops * 2.
        """
        dev = DeviceCapabilities(device_id=device_id, cpu_cores=int(tflops * 2))
        if is_edge:
            dev.device_type = DeviceType.PHONE
        dev.latency_ms = latency_ms
        dev.trust_level = trust_level
        return dev

    def test_empty_devices(self) -> None:
        assert assign_layers_for_continuum([], 10) == {}

    def test_single_device(self) -> None:
        dev = self._dev("only")
        result = assign_layers_for_continuum([dev], 10)
        assert result == {"only": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    def test_single_device_zero_layers(self) -> None:
        dev = self._dev("only")
        result = assign_layers_for_continuum([dev], 0)
        assert result == {"only": []}

    def test_two_equal_devices(self) -> None:
        d1 = self._dev("d1", tflops=100.0)
        d2 = self._dev("d2", tflops=100.0)
        result = assign_layers_for_continuum([d1, d2], 10)
        assert result["d1"] == [0, 1, 2, 3, 4]
        assert result["d2"] == [5, 6, 7, 8, 9]

    def test_stronger_device_gets_more_layers(self) -> None:
        strong = self._dev("strong", tflops=300.0)
        weak = self._dev("weak", tflops=10.0)
        result = assign_layers_for_continuum([strong, weak], 20)
        assert len(result["strong"]) > len(result["weak"])
        # All layers from 0..19 are assigned
        all_assigned = result["strong"] + result["weak"]
        assert all_assigned == list(range(20))

    def test_edge_device_penalty(self) -> None:
        desktop = self._dev("desktop", tflops=200.0, is_edge=False)
        phone = self._dev("phone", tflops=200.0, is_edge=True)
        result = assign_layers_for_continuum([desktop, phone], 16)
        assert len(result["desktop"]) > len(result["phone"])

    def test_high_latency_penalty(self) -> None:
        fast = self._dev("fast", tflops=100.0, latency_ms=1)
        slow = self._dev("slow", tflops=100.0, latency_ms=200)
        result = assign_layers_for_continuum([fast, slow], 16)
        assert len(result["fast"]) > len(result["slow"])

    def test_low_trust_penalty(self) -> None:
        trusted = self._dev("trusted", tflops=100.0, trust_level=1.0)
        untrusted = self._dev("untrusted", tflops=100.0, trust_level=0.0)
        result = assign_layers_for_continuum([trusted, untrusted], 16)
        assert len(result["trusted"]) > len(result["untrusted"])

    def test_first_device_gets_layer_zero(self) -> None:
        d1 = self._dev("d1", tflops=10.0)
        d2 = self._dev("d2", tflops=10.0)
        result = assign_layers_for_continuum([d1, d2], 2)
        assert result["d1"][0] == 0

    def test_last_device_gets_final_layer(self) -> None:
        d1 = self._dev("d1", tflops=10.0)
        d2 = self._dev("d2", tflops=10.0)
        result = assign_layers_for_continuum([d1, d2], 2)
        assert result["d2"][-1] == 1

    def test_min_score_floor(self) -> None:
        """Devices with zero effective_tflops still get at least 0.1 score."""
        d1 = self._dev("d1", tflops=0.0)
        d2 = self._dev("d2", tflops=0.0)
        # Both have score=0.1 (the floor), so they split evenly
        result = assign_layers_for_continuum([d1, d2], 4)
        all_layers = result["d1"] + result["d2"]
        assert all_layers == list(range(4))

    def test_more_devices_than_layers(self) -> None:
        """When devices outnumber layers, some devices may get no layers.

        The enforcement step ensures first/last still get their critical
        layers, potentially causing overlap.
        """
        devices = [self._dev(f"d{i}") for i in range(5)]
        result = assign_layers_for_continuum(devices, 2)
        # Every device_id shows up in the result
        assert len(result) == 5
        # First device has layer 0, last device has layer 1
        assert 0 in result["d0"]
        assert 1 in result["d4"]


# ---------------------------------------------------------------------------
# ContinuumNode
# ---------------------------------------------------------------------------


class TestContinuumNode:
    """ContinuumNode dataclass."""

    def test_minimal_construction(self) -> None:
        caps = DeviceCapabilities(device_id="n1")
        node = ContinuumNode(device_id="n1", capabilities=caps)
        assert node.device_id == "n1"
        assert node.capabilities is caps
        assert node.transport == TransportType.GRPC
        assert node.assigned_layers == []
        assert node.host == ""
        assert node.port == 0
        assert node.status == "discovered"
        assert node.last_heartbeat == 0.0
        assert node.cumulative_tokens == 0
        assert node.total_requests == 0

    def test_construction_all_fields(self) -> None:
        caps = DeviceCapabilities(device_id="n2")
        node = ContinuumNode(
            device_id="n2",
            capabilities=caps,
            transport=TransportType.QUIC,
            assigned_layers=[0, 1, 2],
            host="10.0.0.1",
            port=8080,
            status="ready",
            last_heartbeat=1234.0,
            cumulative_tokens=500,
            total_requests=10,
        )
        assert node.transport == TransportType.QUIC
        assert node.assigned_layers == [0, 1, 2]
        assert node.host == "10.0.0.1"
        assert node.port == 8080
        assert node.status == "ready"
        assert node.last_heartbeat == 1234.0
        assert node.cumulative_tokens == 500
        assert node.total_requests == 10

    def test_assigned_layers_is_distinct_per_node(self) -> None:
        """Each node gets its own list, not a shared default."""
        caps = DeviceCapabilities(device_id="n1")
        n1 = ContinuumNode(device_id="n1", capabilities=caps)
        n2 = ContinuumNode(device_id="n2", capabilities=caps)
        n1.assigned_layers.append(99)
        assert n2.assigned_layers == []  # not mutated


# ---------------------------------------------------------------------------
# EdgeCloudContinuum
# ---------------------------------------------------------------------------


class TestEdgeCloudContinuum:
    """Integration-level tests for the continuum manager."""

    def test_initialize(self) -> None:
        """Initialize performs real device detection without crashing."""
        continuum = EdgeCloudContinuum(total_layers=32)
        caps = continuum.initialize()
        assert isinstance(caps, DeviceCapabilities)
        assert len(caps.device_id) == 16  # SHA-256 hex digest[:16]
        assert caps.os != ""
        assert caps.python_version != ""
        assert caps.cpu_cores > 0
        assert caps.ram_bytes > 0

    def test_register_and_remove_node(self) -> None:
        continuum = EdgeCloudContinuum()
        continuum.initialize()
        caps = DeviceCapabilities(
            device_id="worker-1",
            device_type=DeviceType.DESKTOP,
            network_tier=NetworkTier.LAN,
            cpu_cores=8,
        )
        node = continuum.register_node(
            device_id="worker-1", capabilities=caps, host="192.168.1.10", port=50051,
        )
        assert isinstance(node, ContinuumNode)
        assert node.device_id == "worker-1"
        assert node.host == "192.168.1.10"
        assert node.port == 50051
        assert node.status == "discovered"
        assert node.last_heartbeat > 0

        nodes = continuum.get_nodes()
        assert len(nodes) == 1
        assert nodes[0].device_id == "worker-1"

        continuum.remove_node("worker-1")
        assert continuum.get_nodes() == []

    def test_register_node_without_initialize(self) -> None:
        """register_node should work even without calling initialize()."""
        continuum = EdgeCloudContinuum()
        caps = DeviceCapabilities(device_id="r1")
        node = continuum.register_node("r1", caps)
        assert node.transport == TransportType.GRPC  # fallback when _local_caps is None

    def test_remove_nonexistent_node(self) -> None:
        continuum = EdgeCloudContinuum()
        continuum.remove_node("ghost")  # should not raise

    def test_rebalance_layers(self) -> None:
        continuum = EdgeCloudContinuum(total_layers=16)
        continuum.initialize()

        caps_a = DeviceCapabilities(
            device_id="a", cpu_cores=80, device_type=DeviceType.DESKTOP,
        )
        caps_b = DeviceCapabilities(
            device_id="b", cpu_cores=40, device_type=DeviceType.LAPTOP,
        )
        continuum.register_node("a", caps_a)
        continuum.register_node("b", caps_b)

        assignments = continuum.rebalance_layers()
        # Both nodes should have assignments covering all 16 layers
        assert "a" in assignments
        assert "b" in assignments
        all_layers = assignments["a"] + assignments["b"]
        assert all_layers == list(range(16))
        # Stronger node gets more layers
        assert len(assignments["a"]) >= len(assignments["b"])
        # Nodes status updated to "ready"
        for node in continuum.get_nodes():
            assert node.status == "ready"

    def test_rebalance_layers_empty(self) -> None:
        """rebalance_layers returns empty dict when no nodes are available."""
        continuum = EdgeCloudContinuum(total_layers=16)
        assert continuum.rebalance_layers() == {}

    def test_get_node_for_layer(self) -> None:
        continuum = EdgeCloudContinuum(total_layers=8)
        continuum.initialize()
        caps_a = DeviceCapabilities(device_id="a", cpu_cores=80)
        caps_b = DeviceCapabilities(device_id="b", cpu_cores=80)
        continuum.register_node("a", caps_a)
        continuum.register_node("b", caps_b)
        continuum.rebalance_layers()

        node0 = continuum.get_node_for_layer(0)
        node7 = continuum.get_node_for_layer(7)
        assert node0 is not None
        assert node7 is not None
        assert node0.device_id == "a"
        assert node7.device_id == "b"

    def test_get_node_for_layer_unknown(self) -> None:
        continuum = EdgeCloudContinuum()
        node = continuum.get_node_for_layer(999)
        assert node is None

    def test_get_stats(self) -> None:
        continuum = EdgeCloudContinuum(total_layers=8)
        continuum.initialize()
        caps_a = DeviceCapabilities(
            device_id="a", cpu_cores=80, device_type=DeviceType.DESKTOP,
        )
        caps_b = DeviceCapabilities(
            device_id="b", cpu_cores=40, device_type=DeviceType.LAPTOP,
        )
        continuum.register_node("a", caps_a)
        continuum.register_node("b", caps_b)
        continuum.rebalance_layers()

        stats = continuum.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["ready_nodes"] == 2
        assert stats["total_layers"] == 8
        assert stats["total_layers_assigned"] == 8
        assert stats["total_tokens_served"] == 0
        assert "desktop" in stats["device_types"]
        assert "laptop" in stats["device_types"]
        assert stats["device_types"]["desktop"] == 1
        assert stats["device_types"]["laptop"] == 1

    def test_get_stats_no_nodes(self) -> None:
        continuum = EdgeCloudContinuum()
        stats = continuum.get_stats()
        assert stats["total_nodes"] == 0
        assert stats["ready_nodes"] == 0
        assert stats["device_types"] == {}
        assert stats["transports"] == {}
        assert stats["total_layers_assigned"] == 0

    def test_heartbeat_known_node(self) -> None:
        continuum = EdgeCloudContinuum()
        continuum.initialize()
        caps = DeviceCapabilities(device_id="alive")
        continuum.register_node("alive", caps)
        assert continuum.heartbeat("alive") is True

    def test_heartbeat_unknown_node(self) -> None:
        continuum = EdgeCloudContinuum()
        assert continuum.heartbeat("ghost") is False

    def test_detect_stale_nodes(self) -> None:
        """Nodes with old last_heartbeat are detected as stale."""
        continuum = EdgeCloudContinuum()
        continuum.initialize()
        # Register a node normally (last_heartbeat = now)
        caps = DeviceCapabilities(device_id="fresh")
        continuum.register_node("fresh", caps)

        # Inject a stale node directly (bypass register_node which sets time.time())
        stale_caps = DeviceCapabilities(device_id="stale")
        continuum._nodes["stale"] = ContinuumNode(
            device_id="stale",
            capabilities=stale_caps,
            last_heartbeat=0.0,  # epoch — definitely stale
        )

        stale_ids = continuum.detect_stale_nodes(timeout_seconds=1.0)
        assert "stale" in stale_ids
        assert "fresh" not in stale_ids
        # Stale node's status should be set to "offline"
        stale_node = continuum._nodes["stale"]
        assert stale_node.status == "offline"

    def test_detect_stale_nodes_no_stale(self) -> None:
        continuum = EdgeCloudContinuum()
        continuum.initialize()
        caps = DeviceCapabilities(device_id="fresh")
        continuum.register_node("fresh", caps)
        stale = continuum.detect_stale_nodes(timeout_seconds=0)
        # register_node sets last_heartbeat=time.time(), so timeout=0
        # may or may not catch it. Instead use a large timeout.
        stale = continuum.detect_stale_nodes(timeout_seconds=3600)
        assert stale == []

    def test_detect_stale_nodes_empty(self) -> None:
        continuum = EdgeCloudContinuum()
        assert continuum.detect_stale_nodes() == []


# ---------------------------------------------------------------------------
# detect_device_capabilities (standalone function)
# ---------------------------------------------------------------------------


class TestDetectDeviceCapabilities:
    """Real device detection — runs on the actual host without GPU required."""

    def test_detects_basic_info(self) -> None:
        caps = detect_device_capabilities()
        assert isinstance(caps, DeviceCapabilities)
        assert len(caps.device_id) == 16
        assert caps.os != ""
        assert caps.python_version != ""
        assert caps.cpu_cores > 0
        assert caps.ram_bytes > 0

    def test_detects_some_device_type(self) -> None:
        caps = detect_device_capabilities()
        assert caps.device_type in DeviceType
