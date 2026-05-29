"""Tests for TopologyGraph and TopologyProber."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from distllm.dist.partition.topology import LinkProfile, TopologyGraph, TopologyProber


class TestTopologyGraph:
    def test_get_bandwidth(self, two_node_topology):
        assert two_node_topology.get_bandwidth("gpu-0", "gpu-1") == 25.0

    def test_get_bandwidth_fallback(self, two_node_topology):
        assert two_node_topology.get_bandwidth("gpu-0", "unknown") == 1.0

    def test_get_latency(self, two_node_topology):
        assert two_node_topology.get_latency("gpu-0", "gpu-1") == 500.0

    def test_get_latency_fallback(self, two_node_topology):
        assert two_node_topology.get_latency("a", "b") == 1000.0

    def test_total_gpus(self, two_node_topology):
        assert two_node_topology.total_gpus() == 2

    def test_to_dict(self, two_node_topology):
        d = two_node_topology.to_dict()
        assert d["total_gpus"] == 2
        assert len(d["nodes"]) == 2
        assert len(d["links"]) == 1

    def test_bandwidth_symmetric(self, two_node_topology):
        assert two_node_topology.get_bandwidth("gpu-0", "gpu-1") == two_node_topology.get_bandwidth("gpu-1", "gpu-0")


class TestTopologyProber:
    def test_fallback_topology(self):
        topo = TopologyProber.make_fallback_topology(3, 2)
        assert len(topo.node_ids) == 3
        assert topo.total_gpus() == 6
        assert len(topo.links) == 3

    def test_fallback_topology_links(self):
        topo = TopologyProber.make_fallback_topology(2)
        assert len(topo.links) == 1
        assert topo.links[0].bandwidth_gbps == 12.5

    @patch.object(TopologyProber, "_detect_nvlink", return_value=True)
    def test_local_topology_nvlink(self, _):
        prober = TopologyProber()
        links = prober.probe_local_topology(4)
        assert len(links) == 6
        assert all(l.is_nvlink for l in links)

    @patch.object(TopologyProber, "_detect_nvlink", return_value=False)
    def test_local_topology_no_nvlink(self, _):
        prober = TopologyProber()
        links = prober.probe_local_topology(3)
        assert len(links) == 3
        assert all(not l.is_nvlink for l in links)


class TestLinkProfile:
    def test_defaults(self):
        link = LinkProfile(source="a", target="b")
        assert link.bandwidth_gbps == 12.5
        assert link.latency_us == 100.0
        assert not link.is_nvlink

    def test_custom(self):
        link = LinkProfile(source="a", target="b", bandwidth_gbps=600.0, latency_us=5.0, is_nvlink=True)
        assert link.bandwidth_gbps == 600.0
        assert link.is_nvlink
