"""Structured output engine for constrained generation."""

from __future__ import annotations

import json
import re
from collections import deque
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


@dataclass
class RepairTrajectory:
    """Record of a single repair attempt for offline learning/metrics."""

    original_output: str = ""
    schema: dict | None = None
    strategy: str = ""
    attempt_number: int = 0
    success: bool = False
    repaired_output: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class RepairConfig:
    """Configuration for the self-healing repair orchestrator."""

    max_repair_attempts: int = 3
    repair_strategies: list[str] = field(default_factory=lambda: ["heuristic", "truncate", "regenerate"])
    log_repair_trajectories: bool = True
    max_trajectory_buffer: int = 1000


class StructuredOutputEngine:
    """Engine for generating and validating structured output.

    Usage::

        engine = StructuredOutputEngine()
        result = engine.validate('{"a": 1}', response_format={"type": "json_object"})
        assert result.valid
    """

    def __init__(
        self,
        config: StructuredOutputConfig | None = None,
        repair_config: RepairConfig | None = None,
    ) -> None:
        self._config = config or StructuredOutputConfig()
        self._repair_config = repair_config or RepairConfig()
        self._validator = SchemaValidator()
        self._repairer = OutputRepairer(max_attempts=self._config.max_repair_attempts)
        self._valid_prefix: str = ""
        self._repair_trajectories: deque[RepairTrajectory] = deque(
            maxlen=self._repair_config.max_trajectory_buffer
        )
        self._repair_attempts: int = 0
        self._successful_repairs: int = 0

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

    # ── Self-healing repair orchestrator ────────────────────────────────

    def validate_token(self, token: str, prefix_so_far: str | None = None, schema: dict | None = None) -> bool:
        """Check whether *token* is a valid next character at the current JS state.

        Uses a lightweight character-level state machine tracking JSON
        structure (object/array/string/number progress), the same model as
        ``JSONSchemaConstraint``.  ``prefix_so_far`` (optional) advances the
        state machine before checking the token; without it the engine's last
        valid prefix is used.

        Args:
            token: Candidate next character.
            prefix_so_far: Optional generated-so-far text to derive the state.
            schema: Optional JSON schema (accepted for parity; JSON-syntax
                validity is what the state machine enforces).

        Returns:
            True if *token* may appear next in valid JSON.
        """
        if token == "":
            return True

        prefix = prefix_so_far if prefix_so_far is not None else self._valid_prefix
        state = "object_start"
        stack: list[str] = []
        in_string = False
        escape_next = False
        in_number = False

        for ch in prefix:
            if escape_next:
                escape_next = False
                continue
            if in_string:
                if ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                    state = "after_key" if state in ("object_start", "after_open_brace", "after_comma") else "after_value"
                continue
            if in_number:
                if ch in "0123456789.eE+-":
                    continue
                in_number = False
                state = "after_value" if ch in ",}]\n\r\t " else "after_colon"
            if ch == '"':
                in_string = True
                state = "in_key" if state in ("object_start", "after_open_brace", "after_comma") else "in_value"
                continue
            if ch == "{":
                stack.append("object")
                state = "after_open_brace"
                continue
            if ch == "[":
                stack.append("array")
                state = "array_start"
                continue
            if ch == "}":
                if stack:
                    stack.pop()
                state = "after_value"
                continue
            if ch == "]":
                if stack:
                    stack.pop()
                state = "after_array_value"
                continue
            if ch == ":":
                state = "after_colon"
                continue
            if ch == ",":
                state = "after_comma" if state in ("after_value", "after_array_value") else "after_comma"
                continue
            if ch in "0123456789-":
                in_number = True
                state = "in_number"
                continue
            if ch in "tfn":
                state = "after_value"
                continue

        valid: set[str]
        if in_string:
            valid = set('"\\') | set('"\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\b\t\n\u000b\f\r\u000e\u000f\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f !') 
        elif in_number:
            valid = set("0123456789.eE+-") | {",", "}", "]", " ", "\t", "\n", "\r"}
        elif state in ("object_start", "after_open_brace", "after_key", "after_comma"):
            if state == "object_start":
                valid = {'"', "}"}
            elif state == "after_open_brace":
                valid = {'"', "}"}
            elif state == "after_key":
                valid = {":"}
            else:  # after_comma
                valid = {'"'}
        elif state == "after_colon":
            valid = {'"', "{", "[", "t", "f", "n", "-", *map(str, range(10))}
        elif state == "after_value":
            valid = {",", "}"} if stack and stack[-1] == "object" else {",", "]"}
        elif state == "after_array_value":
            valid = {",", "]"}
        elif state == "array_start":
            valid = {"]", '"', "{", "[", "t", "f", "n", "-", *map(str, range(10))}
        else:
            valid = set()

        return token in valid

    def get_valid_prefix(self) -> str:
        """Return the last known-good prefix."""
        return self._valid_prefix

    def set_valid_prefix(self, prefix: str) -> None:
        """Update the last known-good prefix (called on each valid token)."""
        self._valid_prefix = prefix

    def _set_valid_prefix(self, prefix: str) -> None:
        """Alias of set_valid_prefix for internal/tests use."""
        self._valid_prefix = prefix

    def repair_output(self, invalid_output: str, schema: dict | None = None) -> str:
        """Attempt to repair an invalid output.

        Already-valid JSON succeeds on the fast path (no strategy loop, no
        rate counters).  Otherwise each configured strategy is tried in order;
        the repaired output is returned, or the original if all fail.
        """
        if self._is_valid_json(invalid_output, schema):
            return invalid_output

        result = invalid_output
        for strategy in self._repair_config.repair_strategies:
            if strategy == "heuristic":
                result = self._repair_heuristic(invalid_output)
            elif strategy == "truncate":
                result = self._repair_truncate(invalid_output)
            elif strategy == "regenerate":
                result = self._repair_regenerate()
            else:
                logger.warning(f"Unknown repair strategy: {strategy}")
                continue

            if result and self._is_valid_json(result):
                self._repair_attempts += 1
                self._successful_repairs += 1
                if self._repair_config.log_repair_trajectories:
                    self._repair_trajectories.append(
                        RepairTrajectory(
                            original_output=invalid_output,
                            schema=schema,
                            strategy=strategy,
                            attempt_number=self._repair_attempts,
                            success=True,
                            repaired_output=result,
                        )
                    )
                return result

        self._repair_attempts += 1
        if self._repair_config.log_repair_trajectories:
            self._repair_trajectories.append(
                RepairTrajectory(
                    original_output=invalid_output,
                    schema=schema,
                    strategy="",
                    attempt_number=self._repair_attempts,
                    success=False,
                    repaired_output=invalid_output,
                )
            )
        return invalid_output

    def learn_from_repair(self, trajectory: RepairTrajectory) -> None:
        """Store a repair trajectory for offline training."""
        if self._repair_config.log_repair_trajectories:
            self._repair_trajectories.append(trajectory)

    @property
    def trajectories(self) -> list[RepairTrajectory]:
        """Return a copy of all stored repair trajectories."""
        return list(self._repair_trajectories)

    def clear_trajectories(self) -> None:
        """Clear all stored repair trajectories and reset rate counters."""
        self._repair_trajectories.clear()
        self._repair_attempts = 0
        self._successful_repairs = 0

    @property
    def repair_rate(self) -> float:
        """Ratio of successful repairs to total repair attempts."""
        if self._repair_attempts == 0:
            return 0.0
        return self._successful_repairs / self._repair_attempts

    # ── Repair strategies ───────────────────────────────────────────────

    def _repair_heuristic(self, text: str) -> str:
        """Apply heuristic JSON fixes — insert missing brackets, fix quotes."""
        text = text.strip()
        # Strip trailing commas before closing brackets
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Balance brackets
        openers = text.count("{") + text.count("[")
        closers = text.count("}") + text.count("]")
        for _ in range(openers - closers):
            text += "}"
        # Balance parentheses
        if text.count('"') % 2 != 0:
            text += '"'
        return text

    def _repair_truncate(self, text: str) -> str:
        """Truncate to the last valid JSON prefix."""
        for i in range(len(text), 0, -1):
            candidate = text[:i]
            if self._is_valid_json(candidate):
                return candidate
        return ""

    def _repair_regenerate(self) -> str:
        """Regenerate from the last valid prefix, repairing it if needed."""
        valid = self.get_valid_prefix()
        if not valid:
            return ""
        if self._is_valid_json(valid):
            return valid
        return self._repair_heuristic(valid)

    @staticmethod
    def _is_valid_json(text: str, schema: dict | None = None) -> bool:
        """Return True if *text* is valid JSON and matches the schema type."""
        if text is None or not text.strip():
            return False
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return False
        if schema is None:
            return True
        return SchemaValidator()._check_type(data, schema.get("type"))

    def __repr__(self) -> str:
        return (
            f"StructuredOutputEngine(repair_config={self._repair_config!r}, "
            f"trajectories={len(self._repair_trajectories)})"
        )
