"""Tests for CarbonMigrationEngine and related types.

Uses the import-helper pattern to load modules.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/carbon_migration.py")
CarbonMigrationEngine = _mod.CarbonMigrationEngine
CarbonIntensityClient = _mod.CarbonIntensityClient
MigrationEvent = _mod.MigrationEvent
SLATier = _mod.SLATier
SLA_TIERS = _mod.SLA_TIERS
get_sla_tier = _mod.get_sla_tier
list_sla_tiers = _mod.list_sla_tiers


class TestMigrationEvent:
    def test_defaults(self):
        ev = MigrationEvent("us-east-1", "us-west-1", 400.0, 200.0, 200.0)
        assert ev.from_region == "us-east-1"
        assert ev.to_region == "us-west-1"
        assert ev.from_carbon == 400.0
        assert ev.to_carbon == 200.0
        assert ev.carbon_saved == 200.0
        assert ev.request_ids == []
        assert ev.success is False
        assert ev.duration_ms == 0.0
        assert ev.timestamp > 0


class TestCarbonIntensityClient:
    def test_init_default(self):
        client = CarbonIntensityClient()
        # No API key set, falls back to static
        assert client._api_key == ""

    def test_init_with_key(self):
        client = CarbonIntensityClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_get_intensity_fallback_static(self):
        """Without API key, falls back to _REGIONAL_CARBON_INTENSITY."""
        client = CarbonIntensityClient(api_key="")
        val = client.get_intensity("us-east-1")
        assert val > 0
        assert isinstance(val, (int, float))

    def test_get_intensity_caches(self):
        client = CarbonIntensityClient(api_key="")
        val1 = client.get_intensity("us-east-1")
        val2 = client.get_intensity("us-east-1")
        assert val1 == val2
        assert "us-east-1" in client._cache

    def test_get_intensity_unknown_zone(self):
        client = CarbonIntensityClient(api_key="")
        val = client.get_intensity("nonexistent-zone")
        assert val == 0.0

    def test_get_all_intensities(self):
        client = CarbonIntensityClient(api_key="")
        zones = ["us-east-1", "us-west-1"]
        result = client.get_all_intensities(zones)
        assert set(result.keys()) == {"us-east-1", "us-west-1"}
        for v in result.values():
            assert v > 0


class TestCarbonMigrationEngine:
    def test_init_defaults(self):
        engine = CarbonMigrationEngine()
        assert engine._threshold == 400.0
        assert engine._check_interval == 300.0
        assert engine._cooldown == 900.0
        assert engine._min_savings_pct == 0.2
        assert engine._active_region == ""
        assert engine._active_requests == []
        assert engine._running is False

    def test_init_with_custom_values(self):
        engine = CarbonMigrationEngine(
            threshold=500.0,
            check_interval_s=60.0,
            migration_cooldown_s=300.0,
            min_savings_pct=10.0,
        )
        assert engine._threshold == 500.0
        assert engine._check_interval == 60.0
        assert engine._cooldown == 300.0
        assert engine._min_savings_pct == 0.1

    def test_set_active_region(self):
        engine = CarbonMigrationEngine()
        engine.set_active_region("us-east-1", ["req-1", "req-2"])
        assert engine._active_region == "us-east-1"
        assert engine._active_requests == ["req-1", "req-2"]

    def test_set_active_region_no_requests(self):
        engine = CarbonMigrationEngine()
        engine.set_active_region("us-east-1")
        assert engine._active_region == "us-east-1"
        # _active_requests should not be overwritten when None
        assert engine._active_requests == []

    def test_set_migration_callback(self):
        engine = CarbonMigrationEngine()

        def callback(a, b, c):
            return True

        engine.set_migration_callback(callback)
        assert engine._migration_callback is callback

    def test_start_and_stop(self):
        engine = CarbonMigrationEngine(check_interval_s=0.01)
        engine.set_active_region("us-east-1")
        engine.start()
        assert engine._running is True
        assert engine._thread is not None
        assert engine._thread.is_alive()
        engine.stop()
        assert engine._running is False

    def test_start_idempotent(self):
        engine = CarbonMigrationEngine()
        engine.start()
        thread_id = id(engine._thread)
        engine.start()  # second start should not create new thread
        assert id(engine._thread) == thread_id
        engine.stop()

    def test_get_events_empty(self):
        engine = CarbonMigrationEngine()
        events = engine.get_events()
        assert events == []


class TestCarbonMigrationEngineCheck:
    """Tests for _check_and_migrate with controlled carbon provider."""

    def test_no_active_region_skips(self):
        engine = CarbonMigrationEngine(check_interval_s=0.01)
        engine._check_and_migrate()  # should not raise
        assert engine._events == []

    def test_below_threshold_skips(self):
        """If carbon intensity is below threshold, no migration."""
        provider = CarbonIntensityClient()
        engine = CarbonMigrationEngine(
            carbon_provider=provider,
            threshold=1000.0,  # high threshold
        )
        engine.set_active_region("us-east-1")
        engine._check_and_migrate()
        assert engine._events == []

    def test_no_better_region_skips(self):
        """If no region is sufficiently cleaner, no migration."""
        provider = CarbonIntensityClient()
        engine = CarbonMigrationEngine(
            carbon_provider=provider,
            threshold=1.0,  # trigger check
            min_savings_pct=100.0,  # impossible savings
        )
        engine.set_active_region("us-east-1")
        engine._check_and_migrate()
        assert engine._events == []

    def test_triggers_migration_event(self):
        """When current region exceeds threshold and a cleaner region exists."""
        class MockProvider:
            def get_intensity(self, zone):
                if zone == "us-east-1":
                    return 500.0  # above threshold
                return 0.0

            def get_all_intensities(self, zones):
                return {"us-west-1": 100.0, "eu-west-1": 50.0}

        engine = CarbonMigrationEngine(
            carbon_provider=MockProvider(),
            threshold=400.0,
            min_savings_pct=20.0,
        )
        engine.set_active_region("us-east-1", ["req-1"])
        engine._check_and_migrate()
        assert len(engine._events) == 1
        ev = engine._events[0]
        assert ev.from_region == "us-east-1"
        assert ev.to_region == "eu-west-1"
        assert ev.from_carbon == 500.0
        assert ev.success is False  # no callback set

    def test_migration_callback_updates_region(self):
        """Successful migration callback updates active region."""
        class MockProvider:
            def get_intensity(self, zone):
                if zone == "us-east-1":
                    return 500.0
                return 0.0

            def get_all_intensities(self, zones):
                return {"us-west-1": 100.0}

        engine = CarbonMigrationEngine(
            carbon_provider=MockProvider(),
            threshold=400.0,
            min_savings_pct=20.0,
        )
        engine.set_active_region("us-east-1", ["req-1"])

        def migrate(from_region, to_region, request_ids):
            return True

        engine.set_migration_callback(migrate)
        engine._check_and_migrate()
        assert engine._active_region == "us-west-1"
        assert len(engine._events) == 1
        assert engine._events[0].success is True

    def test_cooldown_prevents_migration(self):
        class MockProvider:
            def get_intensity(self, zone):
                return 500.0
            def get_all_intensities(self, zones):
                return {"us-west-1": 100.0}

        engine = CarbonMigrationEngine(
            carbon_provider=MockProvider(),
            threshold=400.0,
            min_savings_pct=20.0,
            migration_cooldown_s=3600.0,
        )
        engine.set_active_region("us-east-1")
        engine._last_migration = 9999999999.0  # far in future
        engine._check_and_migrate()
        # Cooldown should prevent migration
        assert engine._events == []


class TestSLATier:
    def test_constructor_defaults(self):
        tier = SLATier(name="Test")
        assert tier.name == "Test"
        assert tier.prefer_spot is True
        assert tier.allow_on_demand is True
        assert tier.max_latency_ms == 200.0
        assert tier.carbon_weight == 0.3

    def test_to_routing_kwargs(self):
        tier = SLATier("Custom", prefer_spot=False, max_latency_ms=100.0,
                       max_carbon_intensity=400.0, max_price_per_hour=20.0,
                       carbon_weight=0.5)
        kwargs = tier.to_routing_kwargs()
        assert kwargs["prefer_spot"] is False
        assert kwargs["max_latency_ms"] == 100.0
        assert kwargs["max_carbon_intensity"] == 400.0
        assert kwargs["max_price"] == 20.0
        assert kwargs["carbon_weight"] == 0.5

    def test_predefined_critical(self):
        tier = SLA_TIERS["critical"]
        assert tier.name == "Critical"
        assert tier.prefer_spot is False
        assert tier.carbon_weight == 0.0

    def test_predefined_batch(self):
        tier = SLA_TIERS["batch"]
        assert tier.name == "Batch"
        assert tier.prefer_spot is True
        assert tier.allow_on_demand is False
        assert tier.carbon_weight == 0.5

    def test_predefined_green(self):
        tier = SLA_TIERS["green"]
        assert tier.max_carbon_intensity == 200.0
        assert tier.carbon_weight == 1.0

    def test_get_sla_tier_exists(self):
        tier = get_sla_tier("standard")
        assert tier.name == "Standard"

    def test_get_sla_tier_fallback(self):
        tier = get_sla_tier("nonexistent")
        assert tier.name == "Standard"  # falls back to standard

    def test_list_sla_tiers(self):
        tiers = list_sla_tiers()
        names = [t["name"] for t in tiers]
        assert "Critical" in names
        assert "Standard" in names
        assert "Batch" in names
        assert "Green" in names
