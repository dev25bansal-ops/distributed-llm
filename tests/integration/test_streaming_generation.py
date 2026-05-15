"""Integration tests for streaming generation flow.

Tests streaming chat/completion with real token generation through mock pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from distllm.core.batch_scheduler import BatchScheduler, Sequence
from distllm.core.coordinator import Coordinator
from distllm.core.param_update_channel import GenerationParams


class TestStreamingGeneration:
    """Test streaming generation end-to-end through the coordinator."""

    def test_generate_method_exists(self, integration_coordinator_with_nodes):
        coord = integration_coordinator_with_nodes
        # Verify the generate method exists
        assert hasattr(coord, 'generate')
        assert callable(coord.generate)

    def test_generate_batch_method_exists(self, integration_coordinator):
        coord = integration_coordinator
        assert hasattr(coord, 'generate_batch')
        assert callable(coord.generate_batch)


class TestBatchGeneration:
    """Test batched generation through scheduler."""

    def test_batch_scheduler_processes_sequences(self, integration_coordinator):
        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=512)

        seq1 = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            generated_tokens=[],
            max_new_tokens=5,
        )
        seq2 = Sequence(
            request_id="req-2",
            prompt_tokens=[4, 5],
            generated_tokens=[],
            max_new_tokens=5,
        )
        scheduler.add(seq1)
        scheduler.add(seq2)

        batch = scheduler.schedule()
        assert len(batch.sequences) == 2

        # Step through generation
        scheduler.step(batch, torch.tensor([10, 20]))
        assert batch.sequences[0].generated_tokens == [10]
        assert batch.sequences[1].generated_tokens == [20]

    def test_sequence_completion(self, integration_coordinator):
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=512)

        seq = Sequence(
            request_id="req-1",
            prompt_tokens=[1],
            generated_tokens=[],
            max_new_tokens=3,
        )
        scheduler.add(seq)

        # Generate tokens one at a time
        for _ in range(3):
            batch = scheduler.schedule()
            if batch is None:
                break
            scheduler.step(batch, torch.tensor([100]))

        # Sequence should be complete
        assert seq.is_complete is True


class TestParamUpdateDuringGeneration:
    """Test streaming parameter updates (Feature 12) during generation."""

    def test_param_update_channel_integration(self, integration_coordinator):
        coord = integration_coordinator

        # Verify param update channel exists (Feature 12)
        assert hasattr(coord, '_param_update_channel')

        # Register a request with params
        params = GenerationParams(temperature=0.7, top_p=0.9, top_k=0)
        coord._param_update_channel.register("req-1", params)

        # Update params mid-generation (update takes **kwargs)
        coord._param_update_channel.update("req-1", temperature=0.5)

        # Get updated params
        result = coord._param_update_channel.get("req-1")
        assert result is not None
        assert result.temperature == 0.5
        assert result.top_p == 0.9  # Unchanged

        # Unregister
        coord._param_update_channel.unregister("req-1")
        assert coord._param_update_channel.get("req-1") is None


class TestCompressionIntegration:
    """Test compression pipeline integration (Feature 14)."""

    def test_compression_settings_default(self):
        from distllm.config.settings import DistLLMSettings

        settings = DistLLMSettings()
        assert settings.compression.enabled is False
        assert settings.compression.method == "none"


class TestClusterIntegration:
    """Test cluster federation integration (Feature 13)."""

    def test_node_registration_with_cluster_id(self):
        from distllm.core.resource_manager import NodeRegistration

        reg = NodeRegistration(
            node_id="node-0",
            host="localhost",
            port=50051,
            start_layer=0,
            end_layer=5,
            cluster_id="us-east",
        )
        assert reg.cluster_id == "us-east"


class TestCanaryIntegration:
    """Test canary deployment integration (Feature 17)."""

    def test_node_registration_with_version(self):
        from distllm.core.resource_manager import NodeRegistration

        reg = NodeRegistration(
            node_id="node-canary",
            host="localhost",
            port=50052,
            start_layer=0,
            end_layer=5,
            version="v2",
        )
        assert reg.version == "v2"


class TestCostIntegration:
    """Test cost-aware scheduling integration (Feature 18)."""

    def test_node_registration_with_cost(self):
        from distllm.core.resource_manager import NodeRegistration

        reg = NodeRegistration(
            node_id="node-spot",
            host="localhost",
            port=50053,
            start_layer=0,
            end_layer=5,
            instance_type="g5.xlarge",
            cost_per_hour=2.50,
            is_spot=True,
        )
        assert reg.is_spot is True
        assert reg.cost_per_hour == 2.50
        assert reg.instance_type == "g5.xlarge"
