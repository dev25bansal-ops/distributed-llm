"""Tests for StructuredOutputConfig and SchemaConfig."""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/structured_output/config.py")
StructuredOutputConfig = _mod.StructuredOutputConfig
SchemaConfig = _mod.SchemaConfig


class TestSchemaConfig:
    """Construction and defaults for SchemaConfig."""

    def test_defaults(self):
        cfg = SchemaConfig()
        assert cfg.resolve_refs is True
        assert cfg.allow_additional_properties is False

    def test_custom_values(self):
        cfg = SchemaConfig(resolve_refs=False, allow_additional_properties=True)
        assert cfg.resolve_refs is False
        assert cfg.allow_additional_properties is True

    def test_partial_custom(self):
        cfg = SchemaConfig(resolve_refs=False)
        assert cfg.resolve_refs is False
        assert cfg.allow_additional_properties is False  # still default


class TestStructuredOutputConfig:
    """Construction, defaults, and customisation for StructuredOutputConfig."""

    def test_defaults(self):
        cfg = StructuredOutputConfig()
        assert cfg.enabled is True
        assert cfg.default_mode == "json_schema"
        assert cfg.max_repair_attempts == 3
        assert cfg.validate_output is True
        assert cfg.repair_output is True
        assert cfg.streaming_buffer_size == 50
        assert cfg.response_format is None
        assert isinstance(cfg.schema_config, SchemaConfig)
        assert cfg.schema_config.resolve_refs is True

    def test_custom_values(self):
        cfg = StructuredOutputConfig(
            enabled=False,
            default_mode="regex",
            max_repair_attempts=5,
            validate_output=False,
            repair_output=False,
            streaming_buffer_size=100,
            response_format={"type": "json_object"},
            schema_config=SchemaConfig(resolve_refs=False),
        )
        assert cfg.enabled is False
        assert cfg.default_mode == "regex"
        assert cfg.max_repair_attempts == 5
        assert cfg.validate_output is False
        assert cfg.repair_output is False
        assert cfg.streaming_buffer_size == 100
        assert cfg.response_format == {"type": "json_object"}
        assert cfg.schema_config.resolve_refs is False

    def test_partial_custom(self):
        """Only override a subset of fields."""
        cfg = StructuredOutputConfig(enabled=False)
        assert cfg.enabled is False
        assert cfg.default_mode == "json_schema"  # unchanged default

    def test_response_format_none_by_default(self):
        cfg = StructuredOutputConfig()
        assert cfg.response_format is None

    def test_response_format_custom(self):
        cfg = StructuredOutputConfig(response_format={"type": "json_schema", "schema": {"type": "object"}})
        assert cfg.response_format == {"type": "json_schema", "schema": {"type": "object"}}

    def test_schema_config_default_factory(self):
        """Each instance gets its own SchemaConfig."""
        cfg1 = StructuredOutputConfig()
        cfg2 = StructuredOutputConfig()
        assert cfg1.schema_config is not cfg2.schema_config

    def test_schema_config_custom_factory(self):
        sc = SchemaConfig(resolve_refs=False, allow_additional_properties=True)
        cfg = StructuredOutputConfig(schema_config=sc)
        assert cfg.schema_config is sc

    def test_all_boolean_flags_false(self):
        cfg = StructuredOutputConfig(
            enabled=False,
            validate_output=False,
            repair_output=False,
        )
        assert cfg.enabled is False
        assert cfg.validate_output is False
        assert cfg.repair_output is False

    def test_streaming_buffer_size_edge(self):
        """Zero and negative buffer sizes are accepted (no validation in dataclass)."""
        cfg0 = StructuredOutputConfig(streaming_buffer_size=0)
        assert cfg0.streaming_buffer_size == 0

        cfg_neg = StructuredOutputConfig(streaming_buffer_size=-1)
        assert cfg_neg.streaming_buffer_size == -1
