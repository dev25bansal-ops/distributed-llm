"""Configuration for structured output generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchemaConfig:
    """Configuration for JSON schema handling."""

    resolve_refs: bool = True
    allow_additional_properties: bool = False


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output generation.

    Attributes:
        enabled: Whether structured output is enabled.
        default_mode: Default constraint mode (json_schema, json_object, regex).
        max_repair_attempts: Maximum number of repair attempts.
        validate_output: Whether to validate generated output.
        repair_output: Whether to attempt repair on invalid output.
        streaming_buffer_size: Minimum characters before flushing stream buffer.
        schema_config: Schema handling configuration.
    """

    enabled: bool = True
    default_mode: str = "json_schema"
    max_repair_attempts: int = 3
    validate_output: bool = True
    repair_output: bool = True
    streaming_buffer_size: int = 50
    response_format: dict | None = None
    schema_config: SchemaConfig = field(default_factory=SchemaConfig)
