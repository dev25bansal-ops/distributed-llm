"""StructuredOutputEngine - main orchestrator for constrained generation.

Integrates JSON schema conversion, GBNF grammar compilation,
schema-aware constrained decoding, streaming structured output,
and output validation/repair into a single cohesive engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from loguru import logger

from distllm.core.structured_output.config import StructuredOutputConfig
from distllm.core.structured_output.schema import SchemaConverter
from distllm.core.structured_output.validator import (
    OutputRepairer,
    SchemaValidator,
    ValidationResult,
)


@dataclass
class GenerationResult:
    """Result of a structured generation call.

    Attributes:
        text: Raw generated text.
        data: Parsed structured data (if JSON mode).
        valid: Whether the output passed validation.
        constraint: The constraint used during generation (if any).
        validation_result: Detailed validation info.
        repair_result: Repair info if repair was needed.
        token_count: Number of tokens generated.
    """
    text: str = ""
    data: Any = None
    valid: bool = True
    constraint: Any = None
    validation_result: ValidationResult | None = None
    repair_result: Any = None
    token_count: int = 0


class StructuredOutputEngine:
    """Orchestrates structured generation with schema enforcement.

    Integrates with:
    - SchemaConstrainedDecoder for token-level logit masking
    - SchemaConverter for JSON schema to GBNF conversion
    - SchemaValidator for output validation
    - OutputRepairer for output repair

    Usage:
        engine = StructuredOutputEngine()
        constraint = engine.build_constraint(
            {"type": "json_schema", "schema": my_schema},
            tokenizer,
        )
        result = engine.generate(
            token_generator_fn=generate_tokens,
            prompt="Return JSON",
            response_format={"type": "json_schema", "schema": my_schema},
        )
    """

    def __init__(self, config: StructuredOutputConfig | None = None):
        self._config = config or StructuredOutputConfig()
        self._schema_converter = SchemaConverter(
            resolve_refs=self._config.schema_config.resolve_refs,
            max_depth=self._config.schema_config.max_nesting_depth,
            allow_additional=self._config.schema_config.allow_additional_properties,
        )
        self._validator = SchemaValidator(strict=self._config.validation.strict_type_checking)
        self._repairer = OutputRepairer(max_attempts=self._config.validation.repair_attempts)

    # ------------------------------------------------------------------
    # Constraint building
    # ------------------------------------------------------------------

    def build_constraint(
        self,
        response_format: dict,
        tokenizer: Any = None,
    ) -> Any:
        """Build a constrained decoding constraint from a response format.

        Supports the same formats as SchemaConstrainedDecoder:
        - json_object: generic JSON
        - json_schema: JSON schema constrained
        - grammar: GBNF grammar constrained
        - regex: regex pattern constrained
        - pydantic: Pydantic model constrained

        Args:
            response_format: Dict with 'type' and optional 'schema'/'grammar'/'pattern'.
            tokenizer: Tokenizer for constraint creation.

        Returns:
            ConstrainedConstraint or None if no constraint needed.
        """
        fmt_type = response_format.get("type", "")

        if fmt_type == "json_schema" and self._config.grammar.use_dfa and tokenizer:
            schema = response_format.get("schema", {})
            if schema:
                try:
                    grammar = self._schema_converter.convert(schema)
                    gbnf_text = str(grammar)
                    return self._build_grammar_constraint(gbnf_text, tokenizer)
                except Exception as e:
                    logger.debug(f"GBNF conversion failed, falling back: {e}")

        return self._build_fallback_constraint(response_format, tokenizer)

    def _build_grammar_constraint(self, grammar_text: str, tokenizer: Any) -> Any:
        """Build a grammar constraint from GBNF text."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder

        decoder = SchemaConstrainedDecoder(tokenizer)
        return decoder.grammar(grammar_text, self._config.grammar.start_rule)

    def _build_fallback_constraint(self, response_format: dict, tokenizer: Any) -> Any:
        """Build a constraint using the standard SchemaConstrainedDecoder."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder

        if tokenizer is None:
            return None

        decoder = SchemaConstrainedDecoder(tokenizer)
        fmt_type = response_format.get("type", "")

        if fmt_type == "json_object":
            schema = response_format.get("schema", {})
            return decoder.json_schema(schema)

        if fmt_type == "json_schema":
            schema = response_format.get("schema", {})
            return decoder.json_schema(schema)

        if fmt_type == "grammar":
            grammar_text = response_format.get("grammar", "")
            start_rule = response_format.get("start_rule", self._config.grammar.start_rule)
            if grammar_text:
                return decoder.grammar(grammar_text, start_rule)
            return None

        if fmt_type == "regex":
            pattern = response_format.get("pattern", "")
            if pattern:
                return decoder.regex(pattern)
            return None

        if fmt_type == "pydantic":
            model = response_format.get("model")
            if model:
                return decoder.pydantic(model)
            return None

        logger.debug(f"Unknown response_format type: {fmt_type}")
        return None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        token_generator_fn: Any,
        prompt: str,
        response_format: dict | None = None,
        stream: bool = False,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate structured output with schema enforcement.

        Args:
            token_generator_fn: Callable that accepts a ConstrainedConstraint
                and kwargs, returns generated text or async generator.
            prompt: Input prompt.
            response_format: Response format specification.
            stream: Whether to use streaming.
            tokenizer: Tokenizer for constraint building.
            **kwargs: Additional arguments passed to token_generator_fn.

        Returns:
            GenerationResult with parsed output.
        """
        constraint = None
        if response_format:
            constraint = self.build_constraint(response_format, tokenizer)

        kwargs["constraint"] = constraint
        result_text = token_generator_fn(prompt, **kwargs)

        data = None
        validation = None
        valid = True

        if response_format and self._is_json_mode(response_format):
            data, validation, valid = self._process_json_result(
                result_text, response_format
            )

        return GenerationResult(
            text=result_text,
            data=data,
            valid=valid,
            constraint=constraint,
            validation_result=validation,
        )

    def generate_with_repair(
        self,
        token_generator_fn: Any,
        prompt: str,
        response_format: dict | None = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with automatic retry and repair on validation failure.

        Attempts up to max_retries times, applying repair strategies
        between attempts.

        Args:
            Same as generate().

        Returns:
            GenerationResult with repair info.
        """
        max_retries = self._config.max_retries
        last_text = ""

        for attempt in range(max_retries + 1):
            kwargs.pop("constraint", None)

            if attempt == 0 and response_format:
                constraint = self.build_constraint(response_format, tokenizer)
                kwargs["constraint"] = constraint

            last_text = token_generator_fn(prompt, **kwargs)

            if not response_format or not self._is_json_mode(response_format):
                return GenerationResult(text=last_text, valid=True)

            data, validation, valid = self._process_json_result(
                last_text, response_format
            )

            if valid:
                return GenerationResult(
                    text=last_text,
                    data=data,
                    valid=True,
                    constraint=kwargs.get("constraint"),
                    validation_result=validation,
                )

            repair_result = self.repair(last_text, response_format)
            if repair_result.success:
                return GenerationResult(
                    text=repair_result.repaired_text,
                    data=repair_result.data,
                    valid=True,
                    repair_result=repair_result,
                )

            logger.debug(
                f"Attempt {attempt + 1}/{max_retries + 1} failed, retrying..."
            )

        return GenerationResult(text=last_text, valid=False)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_structured(
        self,
        token_stream: AsyncIterator[str],
        response_format: dict | None = None,
    ) -> AsyncIterator[Any]:
        """Stream structured output with partial parsing.

        Wraps a token stream and yields StructuredStreamChunks with
        incremental JSON parsing.

        Args:
            token_stream: Async iterator of token strings.
            response_format: Response format for detection of JSON mode.

        Yields:
            StructuredStreamChunk from streaming.py.
        """
        from distllm.core.structured_output.streaming import StructuredStreamHandler

        handler = StructuredStreamHandler(
            enable_partial_parsing=(
                self._config.streaming.partial_parsing
                and response_format is not None
            ),
            flush_complete_pairs=self._config.streaming.flush_complete_pairs,
            min_chars=self._config.streaming.min_chars_for_partial,
        )

        async for chunk in handler.process_buffered(token_stream):
            yield chunk

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        output: str,
        response_format: dict | None = None,
    ) -> ValidationResult:
        """Validate structured output against a schema.

        Args:
            output: Raw output text.
            response_format: Response format with optional 'schema'.

        Returns:
            ValidationResult.
        """
        if not response_format or not self._is_json_mode(response_format):
            return ValidationResult(valid=True)

        schema = response_format.get("schema", {}) if response_format else {}

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                errors=[f"JSON parse error: {e}"],
                schema=schema,
            )

        return self._validator.validate(data, schema)

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair(
        self,
        output: str,
        response_format: dict | None = None,
    ) -> Any:
        """Attempt to repair invalid structured output.

        Args:
            output: Raw output text to repair.
            response_format: Response format (for schema context).

        Returns:
            RepairResult.
        """
        return self._repairer.repair(output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_json_mode(self, response_format: dict) -> bool:
        fmt_type = response_format.get("type", "")
        return fmt_type in ("json_object", "json_schema", "pydantic")

    def _process_json_result(
        self,
        result_text: str,
        response_format: dict,
    ) -> tuple[Any, ValidationResult | None, bool]:
        """Parse and validate JSON output.

        Returns:
            Tuple of (data, validation_result, is_valid).
        """
        schema = response_format.get("schema", {})

        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            return None, None, False

        valid = True
        validation = None

        if self._config.validation.validate_output and schema:
            validation = self._validator.validate(data, schema)
            valid = validation.valid

        return data, validation, valid
