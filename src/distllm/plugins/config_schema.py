"""Plugin configuration schema validation.

Provides JSON Schema-based validation for plugin settings, with default
config generation and type coercion.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


# Minimal JSON Schema validator — covers the subset used by plugin configs
_TYPE_VALIDATORS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _validate_type(value: Any, schema_type: str) -> bool:
    """Check if value matches the JSON Schema type."""
    validator = _TYPE_VALIDATORS.get(schema_type)
    if validator is None:
        return True  # Unknown types pass
    return validator(value)


def _validate_property(value: Any, prop_schema: dict[str, Any], path: str) -> list[str]:
    """Validate a single property against its schema."""
    errors: list[str] = []
    schema_type = prop_schema.get("type")

    if schema_type and not _validate_type(value, schema_type):
        errors.append(f"{path}: expected type '{schema_type}', got '{type(value).__name__}'")
        return errors

    # Enum check
    if "enum" in prop_schema and value not in prop_schema["enum"]:
        errors.append(f"{path}: value '{value}' not in allowed values {prop_schema['enum']}")

    # Range checks for numbers
    if schema_type in ("integer", "number"):
        if "minimum" in prop_schema and value < prop_schema["minimum"]:
            errors.append(f"{path}: value {value} below minimum {prop_schema['minimum']}")
        if "maximum" in prop_schema and value > prop_schema["maximum"]:
            errors.append(f"{path}: value {value} above maximum {prop_schema['maximum']}")

    # String constraints
    if schema_type == "string":
        if "minLength" in prop_schema and len(value) < prop_schema["minLength"]:
            errors.append(f"{path}: string too short (min {prop_schema['minLength']})")
        if "maxLength" in prop_schema and len(value) > prop_schema["maxLength"]:
            errors.append(f"{path}: string too long (max {prop_schema['maxLength']})")
        if "pattern" in prop_schema:
            import re
            if not re.search(prop_schema["pattern"], value):
                errors.append(f"{path}: value does not match pattern '{prop_schema['pattern']}'")

    # Array constraints
    if schema_type == "array":
        if "minItems" in prop_schema and len(value) < prop_schema["minItems"]:
            errors.append(f"{path}: array too short (min {prop_schema['minItems']} items)")
        if "maxItems" in prop_schema and len(value) > prop_schema["maxItems"]:
            errors.append(f"{path}: array too long (max {prop_schema['maxItems']} items)")

    return errors


class PluginConfigValidator:
    """Validates plugin configuration against a JSON Schema."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}

    def register_schema(self, plugin_name: str, schema: dict[str, Any]) -> None:
        """Register a JSON Schema for a plugin's configuration."""
        self._schemas[plugin_name] = schema
        logger.debug(f"Registered config schema for plugin '{plugin_name}'")

    def validate_config(self, plugin_name: str, config: dict[str, Any]) -> list[str]:
        """Validate a config dict against the registered schema.

        Returns a list of validation errors (empty if valid).
        """
        schema = self._schemas.get(plugin_name)
        if schema is None:
            return [f"No config schema registered for plugin '{plugin_name}'"]

        errors: list[str] = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field_name in required:
            if field_name not in config:
                errors.append(f"{plugin_name}.{field_name}: required field missing")

        # Validate present fields
        for key, value in config.items():
            if key in properties:
                path = f"{plugin_name}.{key}"
                errors.extend(_validate_property(value, properties[key], path))

        # Reject unknown fields if additionalProperties is false
        if not schema.get("additionalProperties", True):
            for key in config:
                if key not in properties:
                    errors.append(f"{plugin_name}.{key}: unknown field not allowed")

        return errors

    def get_default_config(self, plugin_name: str) -> dict[str, Any]:
        """Generate a default config from the schema's default values."""
        schema = self._schemas.get(plugin_name)
        if schema is None:
            return {}

        defaults = {}
        for key, prop_schema in schema.get("properties", {}).items():
            if "default" in prop_schema:
                defaults[key] = prop_schema["default"]

        return defaults

    def load_schema_from_file(self, plugin_name: str, file_path: str) -> None:
        """Load a JSON Schema from a file."""
        import json
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")

        with open(path) as f:
            schema = json.load(f)

        self.register_schema(plugin_name, schema)
