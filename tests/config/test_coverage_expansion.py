"""Coverage expansion — edge cases for modules near target."""
import pytest
from pydantic import ValidationError
from distllm.config._model import QuantizationSettings, SpeculativeSettings, LoRASettings, MoESettings, MultiModelSettings, CompressionSettings, AdaptiveCompressionSettings, EmbeddingSettings, PromptTemplateSettings
from distllm.config._network import WideAreaSettings, RouteRuleSettings, ChatRouterSettings, TLSSettings
from distllm.config._cache import GossipSettings, PredictiveCacheSettings, DefragmentationSettings
from distllm.config._parallelism import HybridParallelSettings, ZeroCopySettings, PartitioningSettings, RebalancerSettings, PrioritySettings, DisaggSettings, NodeRole, NodeSettings
from distllm.config._performance import CudaGraphSettings, CompileSettings, AdaptivePrecisionSettings, SelfOptimizingSettings
from distllm.config._hardware import HardwareSettings
from distllm.config._backends import VLLMSettings, LlamacppSettings
from distllm.config._generation import GenerationSettings
from distllm.config._observability import ChaosSettings
from distllm.config._application import AgentSettings, RAGSettings, PluginSettings
from distllm.config._deployment import CanarySettings, CostSettings, TenantSettings, VersionSettings, RolloutStageModel


class TestModelEdge:
    def test_quant_methods(self):
        for m in ("none", "bnb_4bit", "gptq", "awq", "fp8"):
            assert QuantizationSettings(method=m).method == m

    def test_spec_medusa(self):
        assert SpeculativeSettings(method="medusa", medusa_num_heads=4).medusa_num_heads == 4

    def test_lora(self):
        assert len(LoRASettings(enabled=True, adapters={"l1": "p"}).adapters) == 1

    def test_moe(self):
        assert MoESettings(enabled=True, num_experts=16).num_experts == 16

    def test_multi_model(self):
        m = MultiModelSettings(models={"llama": "meta-llama/Llama-2-7b"}, default_model="llama")
        assert m.default_model == "llama"

    def test_compression(self):
        assert CompressionSettings(enabled=True, target_bits=4).target_bits == 4

    def test_embedding(self):
        assert EmbeddingSettings(normalize=False).normalize is False


class TestNetworkEdge:
    def test_wide_area(self):
        assert WideAreaSettings(enabled=True, p2p_forwarding=True).p2p_forwarding is True

    def test_route_rule(self):
        r = RouteRuleSettings(name="r1", match_type="keyword", match="llama", target_model="m1")
        assert r.priority >= 0

    def test_chat_router(self):
        r = RouteRuleSettings(name="r1", match_type="keyword", match="test", target_model="m1")
        assert len(ChatRouterSettings(enabled=True, routes=[r]).routes) == 1

    def test_tls_mutual(self):
        t = TLSSettings(enabled=True, cert_file="/tmp/cert.pem", key_file="/tmp/key.pem", ca_cert_file="/tmp/ca.pem", require_client_cert=True)
        assert t.require_client_cert is True


class TestCacheEdge:
    def test_gossip(self):
        assert GossipSettings(enabled=True, interval=5.0).interval == 5.0

    def test_defrag_invalid_raises(self):
        with pytest.raises(ValidationError):
            DefragmentationSettings(policy="invalid")


class TestParallelEdge:
    def test_hybrid(self):
        assert HybridParallelSettings(enabled=True, tp_enabled=True).tp_enabled is True

    def test_zero_copy(self):
        assert ZeroCopySettings(enabled=True, prefer_rdma=True).prefer_rdma is True

    def test_partitioning(self):
        assert PartitioningSettings(strategy="gpu_aware").strategy == "gpu_aware"

    def test_rebalancer(self):
        assert RebalancerSettings(enabled=True, check_interval=5.0).check_interval == 5.0

    def test_priority(self):
        assert PrioritySettings(enabled=True, num_levels=5).num_levels == 5

    def test_disagg(self):
        assert DisaggSettings(enabled=True).enabled is True

    def test_node_role(self):
        assert NodeRole.PREFILL.value == "prefill"

    def test_node_with_role(self):
        n = NodeSettings(node_id="n1", host="localhost", port=50051, start_layer=0, end_layer=10, role=NodeRole.PREFILL)
        assert n.role == NodeRole.PREFILL


class TestOtherEdge:
    def test_cuda_graph(self):
        assert CudaGraphSettings(enabled=True, batch_sizes=[1, 2]).batch_sizes == [1, 2]

    def test_compile(self):
        assert CompileSettings(enabled=True, fullgraph=True).fullgraph is True

    def test_hardware_rocm(self):
        assert HardwareSettings(device_type="rocm").device_type == "rocm"

    def test_vllm(self):
        assert VLLMSettings(enabled=True, tensor_parallel_size=4).tensor_parallel_size == 4

    def test_llamacpp(self):
        assert LlamacppSettings(enabled=True, model_path="/models/llama.gguf").model_path == "/models/llama.gguf"

    def test_generation_top_k(self):
        assert GenerationSettings(top_k=50).top_k == 50

    def test_chaos(self):
        c = ChaosSettings(enabled=True, allowed_scenarios=["kill_node"])
        assert "kill_node" in c.allowed_scenarios

    def test_agent(self):
        assert AgentSettings(enabled=True, max_iterations=15).max_iterations == 15

    def test_rag(self):
        assert RAGSettings(enabled=True, chunk_size=256).chunk_size == 256

    def test_plugin(self):
        p = PluginSettings(enabled=True, plugins=[{"module": "distllm.plugins.test"}])
        assert p.plugins[0]["module"] == "distllm.plugins.test"

    def test_canary(self):
        stages = [RolloutStageModel(weight_pct=100, analysis_duration_s=60)]
        c = CanarySettings(enabled=True, stable_version="v1", canary_version="v2", stages=stages)
        assert c.stages[0].weight_pct == 100

    def test_cost(self):
        assert CostSettings(enabled=True, budget_per_hour=50.0).budget_per_hour == 50.0

    def test_tenant(self):
        assert TenantSettings(enabled=True, default_tier="pro").default_tier == "pro"

    def test_version_shadow(self):
        assert VersionSettings(enabled=True, shadow_enabled=True).shadow_enabled is True
