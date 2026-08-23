"""Schema validation and output repair for structured output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Draft 2020-12 validation backing.
#
# The JSON-schema validator used to be backed by ``jsonschema``'s
# ``Draft202012Validator`` (the E3 fix upgraded the schema dialect from
# Draft 7 to 2020-12).  ``SchemaValidator`` above intentionally performs
# lightweight field-level checks instead of a full jsonschema pass, so the
# ``jsonschema`` library remains an *optional* dependency: importing this
# module must not hard-require it.
#
# ``Draft202012Validator`` is still re-exported for callers (and regression
# tests) that exercise 2020-12-specific keywords directly.  When
# ``jsonschema`` is not installed the eager-import fallback is never
# reached; ``Draft202012Validator`` stays ``None`` and this module imports
# cleanly.
try:
    from jsonschema import Draft202012Validator  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - jsonschema absent
    Draft202012Validator = None  # type: ignore[assignment]


@dataclass
class ValidationResult:
    """Result of validating output against a schema."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    data: Any = None


@dataclass
class RepairResult:
    """Result of attempting to repair invalid output."""

    success: bool
    repaired_text: str = ""
    original_text: str = ""
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


class SchemaValidator:
    """Validates data against JSON schemas.

    Usage::

        validator = SchemaValidator()
        result = validator.validate({"x": 1}, schema={"type": "object"})
        assert result.valid
    """

    def validate(self, data: Any, schema: dict | None = None) -> ValidationResult:
        """Validate data against a schema.

        Args:
            data: The data to validate.
            schema: JSON schema dict. If None, any data is valid.

        Returns:
            ValidationResult with valid flag and error messages.
        """
        if schema is None:
            return ValidationResult(valid=True, data=data)

        errors = []

        # Type checking
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(data, expected_type):
                errors.append(f"Expected type '{expected_type}', got '{type(data).__name__}'")
                return ValidationResult(valid=False, errors=errors)

        # Required fields
        required = schema.get("required", [])
        if isinstance(data, dict) and required:
            for field_name in required:
                if field_name not in data:
                    errors.append(f"Missing required field: '{field_name}'")

        # Properties validation
        properties = schema.get("properties", {})
        if isinstance(data, dict) and properties:
            for key, prop_schema in properties.items():
                if key in data:
                    prop_result = self.validate(data[key], prop_schema)
                    if not prop_result.valid:
                        errors.extend(prop_result.errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            data=data,
        )

    def _check_type(self, data: Any, expected_type: str) -> bool:
        """Check if data matches the expected JSON type."""
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(data, expected)


class OutputRepairer:
    """Attempts to repair invalid structured output.

    Usage::

        repairer = OutputRepairer(max_attempts=3)
        result = repairer.repair('{"key": "value"')
        assert result.success
    """

    def __init__(self, max_attempts: int = 3) -> None:
        self._max_attempts = max_attempts

    def repair(self, text: str, schema: dict | None = None) -> RepairResult:
        """Attempt to repair invalid JSON output.

        Args:
            text: The text to repair.
            schema: Optional JSON schema for validation.

        Returns:
            RepairResult with success flag and repaired text.
        """
        # Try direct parse first
        try:
            data = json.loads(text)
            return RepairResult(
                success=True,
                repaired_text=text,
                original_text=text,
                attempts=0,
            )
        except json.JSONDecodeError:
            pass

        # Attempt repairs
        repaired = text
        for attempt in range(self._max_attempts):
            repaired = self._try_repair(repaired)
            try:
                data = json.loads(repaired)
                return RepairResult(
                    success=True,
                    repaired_text=repaired,
                    original_text=text,
                    attempts=attempt + 1,
                )
            except json.JSONDecodeError:
                continue

        return RepairResult(
            success=False,
            repaired_text=repaired,
            original_text=text,
            attempts=self._max_attempts,
            errors=["Failed to repair JSON output"],
        )

    def _try_repair(self, text: str) -> str:
        """Try a single repair attempt."""
        # Close unclosed strings
        if text.count('"') % 2 != 0:
            text += '"'

        # Close unclosed objects/arrays
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        # Remove trailing comma if present
        text = text.rstrip()
        if text.endswith(","):
            text = text[:-1]

        # Close unclosed structures
        text += "}" * max(0, open_braces)
        text += "]" * max(0, open_brackets)

        return text
