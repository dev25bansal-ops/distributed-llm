"""Output validation and repair for structured generation.

Validates generated output against JSON schemas and attempts
to repair common issues like truncation, extra commas, and
missing quotes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result of validating a structured output.

    Attributes:
        valid: Whether the output is valid.
        data: Parsed JSON data (if valid).
        errors: List of validation error messages.
        schema: The schema used for validation.
    """
    valid: bool = True
    data: Any = None
    errors: list[str] = field(default_factory=list)
    schema: dict | None = None

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class RepairResult:
    """Result of attempting to repair an invalid output.

    Attributes:
        success: Whether repair was successful.
        repaired_text: The repaired output text.
        data: Parsed repaired data (if valid JSON).
        strategy: Name of the repair strategy that succeeded.
        attempts: Number of repair attempts made.
    """
    success: bool = False
    repaired_text: str = ""
    data: Any = None
    strategy: str = ""
    attempts: int = 0


class SchemaValidator:
    """Validates structured output against a JSON schema.

    Performs:
    - JSON parse validation
    - Type checking for all fields
    - Required property presence
    - Enum value validation
    - String pattern validation
    - Numeric range validation
    - Array item type validation
    """

    def __init__(self, strict: bool = True):
        self._strict = strict

    def validate(self, output: Any, schema: dict | None = None) -> ValidationResult:
        """Validate a structured output against an optional schema.

        Args:
            output: Parsed JSON data or string to validate.
            schema: JSON Schema dict (optional).

        Returns:
            ValidationResult with errors if invalid.
        """
        errors: list[str] = []

        if schema is None:
            return ValidationResult(valid=True, data=output)

        if not schema:
            return ValidationResult(valid=True, data=output)

        data = output
        errors = self._validate_against_schema(data, schema, "$")

        if errors:
            return ValidationResult(
                valid=False,
                data=data,
                errors=errors,
                schema=schema,
            )

        return ValidationResult(valid=True, data=data, schema=schema)

    def _validate_against_schema(
        self, data: Any, schema: dict, path: str = "$"
    ) -> list[str]:
        """Recursively validate data against a schema node."""
        errors: list[str] = []

        if not schema:
            return errors

        schema_type = schema.get("type")
        resolved = self._resolve_ref(schema)

        if resolved:
            return self._validate_against_schema(data, resolved, path)

        # Handle composition
        if "allOf" in schema:
            for sub in schema["allOf"]:
                errors.extend(self._validate_against_schema(data, sub, path))
            return errors

        if "anyOf" in schema:
            sub_errors = []
            for sub in schema["anyOf"]:
                sub_err = self._validate_against_schema(data, sub, path)
                if not sub_err:
                    return errors
                sub_errors.append(sub_err)
            errors.append(f"{path}: no anyOf branch matched")
            return errors

        if "oneOf" in schema:
            matched = 0
            for sub in schema["oneOf"]:
                sub_err = self._validate_against_schema(data, sub, path)
                if not sub_err:
                    matched += 1
            if matched != 1:
                errors.append(f"{path}: oneOf matched {matched} branches (expected 1)")
            return errors

        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: value {json.dumps(data)} not in enum {schema['enum']}")
            return errors

        if "const" in schema and data != schema["const"]:
            errors.append(f"{path}: value {data!r} != const {schema['const']!r}")

        if schema_type is None:
            if "properties" in schema or "required" in schema:
                schema_type = "object"
            else:
                return errors

        # Type checking
        if schema_type == "object":
            errors.extend(self._validate_object(data, schema, path))
        elif schema_type == "array":
            errors.extend(self._validate_array(data, schema, path))
        elif schema_type == "string":
            errors.extend(self._validate_string(data, schema, path))
        elif schema_type == "integer":
            errors.extend(self._validate_integer(data, schema, path))
        elif schema_type == "number":
            errors.extend(self._validate_number(data, schema, path))
        elif schema_type == "boolean":
            if not isinstance(data, bool):
                errors.append(f"{path}: expected boolean, got {type(data).__name__}")
        elif schema_type == "null":
            if data is not None:
                errors.append(f"{path}: expected null, got {data!r}")

        return errors

    def _resolve_ref(self, schema: dict) -> dict | None:
        if "$ref" not in schema:
            return None
        ref_path = schema["$ref"].lstrip("#/")
        return None

    def _validate_object(self, data: Any, schema: dict, path: str) -> list[str]:
        errors: list[str] = []
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return errors

        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        for req in required:
            if req not in data:
                errors.append(f"{path}.{req}: missing required property")

        for key, value in data.items():
            if key in props:
                prop_path = f"{path}.{key}"
                errors.extend(self._validate_against_schema(value, props[key], prop_path))

        if not schema.get("additionalProperties", True) and self._strict:
            allowed = set(props.keys()) | set(required)
            extra = set(data.keys()) - allowed
            for key in extra:
                errors.append(f"{path}.{key}: unexpected property")

        return errors

    def _validate_array(self, data: Any, schema: dict, path: str) -> list[str]:
        errors: list[str] = []
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return errors

        items_schema = schema.get("items", {})
        prefix_items = schema.get("prefixItems", [])
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", -1)

        if len(data) < min_items:
            errors.append(f"{path}: too few items ({len(data)} < {min_items})")
        if max_items > 0 and len(data) > max_items:
            errors.append(f"{path}: too many items ({len(data)} > {max_items})")

        for i, item in enumerate(data):
            item_path = f"{path}[{i}]"
            if i < len(prefix_items):
                errors.extend(self._validate_against_schema(item, prefix_items[i], item_path))
            elif items_schema:
                errors.extend(self._validate_against_schema(item, items_schema, item_path))

        return errors

    def _validate_string(self, data: Any, schema: dict, path: str) -> list[str]:
        errors: list[str] = []
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return errors

        min_len = schema.get("minLength", 0)
        max_len = schema.get("maxLength", -1)
        pattern = schema.get("pattern", "")

        if len(data) < min_len:
            errors.append(f"{path}: string too short ({len(data)} < {min_len})")
        if max_len > 0 and len(data) > max_len:
            errors.append(f"{path}: string too long ({len(data)} > {max_len})")
        if pattern and not re.match(pattern, data):
            errors.append(f"{path}: string does not match pattern {pattern!r}")

        return errors

    def _validate_integer(self, data: Any, schema: dict, path: str) -> list[str]:
        errors: list[str] = []
        if isinstance(data, bool) or not isinstance(data, int):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
            return errors

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} < minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} > maximum {maximum}")

        return errors

    def _validate_number(self, data: Any, schema: dict, path: str) -> list[str]:
        errors: list[str] = []
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return errors

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} < minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} > maximum {maximum}")

        return errors


class OutputRepairer:
    """Attempts to repair invalid structured output.

    Repair strategies (applied in order):
    1. Basic JSON structure fixes (brackets, commas, quotes)
    2. Truncated content completion
    3. Last-resort extraction of any valid JSON
    """

    def __init__(self, max_attempts: int = 2):
        self._max_attempts = max_attempts

    def repair(self, text: str) -> RepairResult:
        """Attempt to repair invalid JSON text.

        Args:
            text: The text to repair.

        Returns:
            RepairResult with repaired text.
        """
        if not text or not text.strip():
            return RepairResult(success=False, repaired_text=text)

        attempts = 0

        for strategy, repair_fn in [
            ("direct_parse", self._try_parse_direct),
            ("close_all", self._close_strings_and_brackets),
            ("close_brackets", self._close_brackets),
            ("fix_common", self._fix_common_issues),
            ("extract_json", self._extract_json),
            ("complete_truncated", self._complete_truncated),
        ]:
            attempts += 1
            result = repair_fn(text)
            if result is not None:
                return RepairResult(
                    success=True,
                    repaired_text=result,
                    data=json.loads(result),
                    strategy=strategy,
                    attempts=attempts,
                )

        return RepairResult(
            success=False,
            repaired_text=text,
            attempts=attempts,
        )

    def _try_parse_direct(self, text: str) -> str | None:
        """Try direct JSON parse."""
        stripped = text.strip()
        if not stripped:
            return None
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            return None

    def _close_strings_and_brackets(self, text: str) -> str | None:
        """Close unclosed strings and brackets in one pass."""
        repaired = list(text)
        stack = []
        in_string = False
        escaped = False

        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        if in_string:
            repaired.append('"')
        for b in reversed(stack):
            repaired.append("}" if b == "{" else "]")

        result = "".join(repaired)
        try:
            json.loads(result)
            return result
        except json.JSONDecodeError:
            return None

    def _close_brackets(self, text: str) -> str | None:
        """Close unclosed brackets and braces."""
        stack = []
        in_string = False
        escaped = False

        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        closing = "".join("}" if b == "{" else "]" for b in reversed(stack))
        repaired = text + closing
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None

    def _fix_common_issues(self, text: str) -> str | None:
        """Fix common JSON formatting issues."""
        repaired = text

        # Remove trailing comma before closing bracket
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # Replace single quotes with double quotes (simple cases)
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)

        # Unquote unquoted keys (simple identifier pattern)
        repaired = re.sub(
            r"(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)",
            r'\1"\2"\3',
            repaired,
        )

        # Complete partial string values
        in_string = False
        escaped = False
        for i, ch in enumerate(repaired):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
        if in_string:
            repaired = repaired + '"'

        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None

    def _extract_json(self, text: str) -> str | None:
        """Extract the first valid JSON object or array from text."""
        for start_char, end_char, closer in [("{", "}", "}"), ("[", "]", "]")]:
            start = text.find(start_char)
            if start < 0:
                continue

            depth = 0
            in_str = False
            escaped = False
            best = -1

            for i in range(start, len(text)):
                ch = text[i]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == start_char:
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        best = i + 1
                        break

            if best > start:
                candidate = text[start:best]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return None

    def _complete_truncated(self, text: str) -> str | None:
        """Attempt to repair severely truncated JSON."""
        import re

        # Find JSON-like content
        obj_match = re.search(r"\{.*", text, re.DOTALL)
        arr_match = re.search(r"\[.*", text, re.DOTALL)

        best = None
        if obj_match:
            candidate = self._close_brackets(obj_match.group())
            if candidate:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    candidate = self._fix_common_issues(candidate)
                    if candidate:
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            pass

        if arr_match:
            candidate = self._close_brackets(arr_match.group())
            if candidate:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return None
