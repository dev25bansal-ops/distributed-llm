"""Structured output engine for constrained generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.core.structured_output.config import StructuredOutputConfig
from distllm.core.structured_output.validator import SchemaValidator, OutputRepairer, ValidationResult, RepairResult


@dataclass
class GenerationResult:
    """Result of structured output generation."""

    text: str
    data: Any = None
    valid: bool = False
    constraint: Any = None
    validation_result: ValidationResult | None = None
    repair_result: RepairResult | None = None
    token_count: int = 0


class StructuredOutputEngine:
    """Engine for generating and validating structured output.

    Usage::

        engine = StructuredOutputEngine()
        result = engine.validate('{"a": 1}', response_format={"type": "json_object"})
        assert result.valid
    """

    def __init__(self, config: StructuredOutputConfig | None = None) -> None:
        self._config = config or StructuredOutputConfig()
        self._validator = SchemaValidator()
        self._repairer = OutputRepairer(max_attempts=self._config.max_repair_attempts)

    def validate(self, text: str, response_format: dict | None = None) -> ValidationResult:
        """Validate output text against a response format.

        Args:
            text: The text to validate.
            response_format: Response format dict (e.g., {"type": "json_object"}).

        Returns:
            ValidationResult with valid flag and error messages.
        """
        if not text:
            return ValidationResult(valid=False, errors=["Empty output"])

        fmt_type = (response_format or {}).get("type", "")

        if fmt_type in ("json_object", "json_schema"):
            try:
                data = json.loads(text)
                schema = (response_format or {}).get("schema")
                if schema:
                    return self._validator.validate(data, schema)
                return ValidationResult(valid=True, data=data)
            except json.JSONDecodeError as e:
                return ValidationResult(valid=False, errors=[f"Invalid JSON: {e}"])

        # For other types, consider any non-empty text as valid
        return ValidationResult(valid=True, data=text)

    def repair(self, text: str, response_format: dict | None = None) -> RepairResult:
        """Attempt to repair invalid output.

        Args:
            text: The text to repair.
            response_format: Response format dict.

        Returns:
            RepairResult with success flag and repaired text.
        """
        schema = (response_format or {}).get("schema")
        return self._repairer.repair(text, schema)
