"""Real tests for pipeline/orchestrator — PipelineOrchestrator."""
from __future__ import annotations


class TestPipelineNode:
    def test_pipeline_node_init(self):
        from distllm.dist.pipeline.orchestrator import PipelineNode

        node = PipelineNode(
            node_id="n1", host="localhost", port=50051,
            start_layer=0, end_layer=7,
        )
        assert node.node_id == "n1"
        assert node.start_layer == 0
        assert node.end_layer == 7


class TestPipelineOrchestrator:
    def test_orchestrator_init(self):
        from distllm.dist.pipeline.orchestrator import PipelineOrchestrator

        po = PipelineOrchestrator()
        assert po is not None

    def test_register_node(self):
        from distllm.dist.pipeline.orchestrator import PipelineOrchestrator

        po = PipelineOrchestrator()
        po.register_node("n1", "localhost", 50051, 0, 7)
        po.register_node("n2", "localhost", 50052, 8, 15)
        assert len(po.nodes) == 2
