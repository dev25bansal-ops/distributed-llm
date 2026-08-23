"""Text completion service -- encapsulates /v1/completions business logic.

Usage::

    from distllm.api.services.completion_service import CompletionService

    service = CompletionService(coordinator)
    constraint = service.build_constraint(response_format)
    result = await service.complete(prompt, max_tokens=512, temperature=0.7, top_p=0.9)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from distllm.core.structured_output import JSONSchemaConstraint, StructuredOutputEngine


class CompletionService:
    """Encapsulates text completion business logic.

    The constructor takes a *coordinator* (not importing from ``api_state``).
    Each method maps to a distinct phase of the ``/v1/completions`` flow.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator
        self._so_engine = StructuredOutputEngine()

    # -- structured output constraint ----------------------------------------

    def build_constraint(
        self,
        response_format: dict | None,
    ) -> JSONSchemaConstraint | None:
        """Build a ``JSONSchemaConstraint`` from a *response_format* dict.

        Args:
            response_format: The response format dict (e.g.
                ``{"type": "json_object"}``).

        Returns:
            A ``JSONSchemaConstraint`` instance or ``None`` when no
            constraint is applicable.
        """
        if not response_format:
            return None
        fmt_type = response_format.get("type", "")
        if fmt_type in ("json_object", "json_schema"):
            return JSONSchemaConstraint.from_response_format(response_format)
        return None

    # -- generation -----------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: dict | None = None,
        constraint: JSONSchemaConstraint | None = None,
        user_id: str = "default",
    ) -> str:
        """Generate a completion via the coordinator.

        Delegates to ``coordinator.generate()`` inside
        ``asyncio.to_thread`` since the coordinator may block on GPU
        synchronisation.

        Args:
            prompt: The formatted prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            response_format: Structured output format (passed through
                to the coordinator for any server-side handling).
            constraint: Token-level JSON constraint for logit masking.
            user_id: User / tenant identifier.

        Returns:
            The generated text.
        """
        result = await asyncio.to_thread(
            self._coordinator.generate,
            prompt,
            max_tokens,
            temperature,
            top_p,
            user_id=user_id,
            response_format=response_format,
            constraint=constraint,
        )
        return result

    # -- full completion flow with validation ---------------------------------

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_format: dict | None = None,
        user_id: str = "default",
    ) -> str:
        """Full completion flow: constraint building, generation, and validation.

        When *strict* mode is enabled (``response_format["strict"]`` is
        truthy) the generated output is validated against the schema with
        up to 3 retries, appending corrective feedback on each failure.

        Args:
            prompt: The formatted prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            response_format: Structured output format.
            user_id: User / tenant identifier.

        Returns:
            The (possibly corrected) generated text.
        """
        constraint = self.build_constraint(response_format)

        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            constraint=constraint,
            user_id=user_id,
        )

        if response_format and response_format.get("strict", False):
            result = await self._validate_with_retry(
                result=result,
                response_format=response_format,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                constraint=constraint,
                user_id=user_id,
            )

        return result

    # -- structured output validation with retry ------------------------------

    async def _validate_with_retry(
        self,
        result: str,
        response_format: dict,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        constraint: JSONSchemaConstraint | None,
        user_id: str,
    ) -> str:
        """Validate *result* against *response_format* with up to 3 retries.

        On each retry the schema validation errors are appended to the
        prompt as corrective feedback so the model can self-correct.

        Args:
            result: The generated text to validate.
            response_format: The response format dict.
            prompt: The original prompt (extended with feedback on
                each retry).
            max_tokens: Maximum tokens for the retry generation.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            constraint: Token-level JSON constraint.
            user_id: User / tenant identifier.

        Returns:
            The (possibly corrected) generated text.
        """
        max_retries = 3
        current = result
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            validation = self._so_engine.validate(current, response_format)
            if validation.valid:
                return current

            if attempt >= max_retries:
                logger.warning(
                    "Structured output validation failed after %d retries",
                    max_retries,
                )
                return current

            logger.info(
                "Structured output retry %d/%d: %s",
                attempt + 1,
                max_retries,
                validation.errors,
            )

            current_prompt += (
                "\n\nYour previous response was not valid JSON matching the "
                "required schema.  Schema validation errors:\n"
                f"{json.dumps(validation.errors)}\n"
                "Please correct your response and try again."
            )

            current = await asyncio.to_thread(
                self._coordinator.generate,
                current_prompt,
                max_tokens,
                temperature,
                top_p,
                user_id=user_id,
                response_format=response_format,
                constraint=constraint,
            )

        return current
