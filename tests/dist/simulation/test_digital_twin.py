"""Tests for the digital twin simulator."""
import time
from distllm.dist.simulation.digital_twin import SimClusterNode, SimRequest, DigitalTwin, WhatIfEngine, SimulationResult


class TestSimClusterNode:
    def test_node_creation(self):
        node = SimClusterNode(node_id="node-0", gpu_type="A100", gpu_count=4, region="us-west1", hourly_cost=15.0)
        assert node.node_id == "node-0"
        assert node.gpu_type == "A100"

    def test_node_with_layers(self):
        node = SimClusterNode(node_id="node-1", gpu_type="H100", gpu_count=8, region="us-east1", hourly_cost=18.0, layers=(0, 40))
        assert node.layers == (0, 40)


class TestSimRequest:
    def test_request_creation(self):
        req = SimRequest(prompt="Hello", prompt_length=100, max_tokens=50, model="llama-70b", arrival_time=time.time())
        assert req.prompt_length == 100


class TestDigitalTwin:
    def test_add_nodes(self):
        twin = DigitalTwin()
        twin.add_nodes(count=2, gpu_type="H100", region="us-east1")
        assert len(twin._nodes) == 2

    def test_run_simulation(self):
        twin = DigitalTwin()
        twin.add_nodes(count=2, gpu_type="A100", region="us-central1")
        result = twin.run_simulation(duration_s=10)
        assert isinstance(result, SimulationResult)


class TestWhatIfEngine:
    def test_query(self):
        twin = DigitalTwin()
        twin.add_nodes(count=1, gpu_type="A100", region="us-west1")
        engine = WhatIfEngine(twin)
        result = engine.query({"add_nodes": (2, "H100", "us-east1")})
        assert isinstance(result, SimulationResult)
