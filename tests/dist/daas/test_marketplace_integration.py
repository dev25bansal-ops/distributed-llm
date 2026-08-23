"""Tests for DaaS marketplace integration."""
from distllm.dist.daas.marketplace_integration import DaaSProviderInfo, DaaSConsumer, MarketplaceIntegration


class TestDaaSProviderInfo:
    def test_provider_info_creation(self):
        info = DaaSProviderInfo(provider_id="test-provider", host="localhost", port=9000, hardware="cuda:0",
                                model_name="SmolLM-135M", price_per_token=0.001)
        assert info.provider_id == "test-provider"
        assert info.hardware == "cuda:0"
        assert info.price_per_token == 0.001

    def test_provider_info_defaults(self):
        info = DaaSProviderInfo(provider_id="default-test", host="localhost", port=9001)
        assert info.model_name == ""
        assert info.price_per_token == 0.0


class TestDaaSConsumer:
    def test_query_marketplace(self):
        mgr = MarketplaceIntegration()
        mgr.register_provider(DaaSProviderInfo("p1", "host1", 9000, price_per_token=0.001))
        mgr.register_provider(DaaSProviderInfo("p2", "host2", 9001, price_per_token=0.005))
        consumer = DaaSConsumer(mgr)
        result = consumer.query_marketplace(max_price=0.002, min_quality=0, max_latency_ms=9999)
        assert len(result) >= 1
        assert any(p.provider_id == "p1" for p in result)


class TestMarketplaceIntegration:
    def test_register_and_discover(self):
        mgr = MarketplaceIntegration()
        mgr.register_provider(DaaSProviderInfo("test", "localhost", 9000))
        discovered = mgr.discover_providers()
        assert any(p.provider_id == "test" for p in discovered)

    def test_heartbeat(self):
        mgr = MarketplaceIntegration()
        mgr.register_provider(DaaSProviderInfo("alive", "host", 9000))
        assert mgr.heartbeat("alive") is True
        assert mgr.heartbeat("nonexistent") is False
