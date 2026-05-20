"""Configuration for the structured output engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SchemaConfig(BaseModel):
    """Configuration for JSON schema handling."""

    resolve_refs: bool = True
    max_nesting_depth: int = 10
    strict_validation: bool = True
    allow_additional_properties: bool = False


class GrammarConfig(BaseModel):
    """Configuration for grammar-constrained decoding."""

    use_dfa: bool = True
    start_rule: str = "root"
    max_grammar_recursion: int = 10


class StreamingConfig(BaseModel):
    """Configuration for streaming structured output."""

    partial_parsing: bool = True
    yield_partial_on_flush: bool = True
    flush_complete_pairs: bool = True
    min_chars_for_partial: int = 10


class ValidationConfig(BaseModel):
    """Configuration for output validation."""

    validate_output: bool = True
    repair_attempts: int = 2
    max_repair_tokens: int = 100
    strict_type_checking: bool = True


class StructuredOutputConfig(BaseModel):
    """Top-level configuration for the structured output engine.

    Controls how structured (JSON, grammar, regex) output is generated,
    validated, and streamed.
    """

    enabled: bool = True
    default_mode: Literal[
        "json_object", "json_schema", "grammar", "regex", "pydantic"
    ] = "json_schema"
    max_retries: int = 3

    schema_config: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    grammar: GrammarConfig = Field(default_factory=GrammarConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
