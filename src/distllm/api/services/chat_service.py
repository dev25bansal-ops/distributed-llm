"""Chat completion service -- encapsulates all business logic from routes/chat.py.

Usage::

    from distllm.api.services.chat_service import ChatService

    service = ChatService(coordinator)
    resolved = service.resolve_model(body, request)
    prompt = service.build_prompt(body.messages, has_images, vlm_pipeline)
    ...
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger

from distllm.api.tool_calling import ToolCallingEngine
from distllm.core.structured_output import JSONSchemaConstraint, StructuredOutputEngine
from distllm.prompts.engine import TemplateEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content_items: Any) -> str:
    """Extract text from multi-modal content list (list of dicts or objects)."""
    if isinstance(content_items, str):
        return content_items
    if isinstance(content_items, list):
        parts: list[str] = []
        for item in content_items:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
            elif hasattr(item, "type"):
                if item.type == "text":
                    parts.append(item.text or "")
                elif item.type == "image_url":
                    parts.append("[image]")
        return " ".join(parts)
    return str(content_items)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ChatService:
    """Encapsulates all chat completion business logic.

    The constructor takes a *coordinator* (not importing from ``api_state``).
    Each method maps to a distinct phase of the ``/v1/chat/completions`` flow.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator
        self._so_engine = StructuredOutputEngine()

    # -- model resolution ---------------------------------------------------

    def resolve_model(self, body: Any, request_state: Any) -> str:
        """Resolve the effective model name.

        Steps performed in order:

        1.  **Admin bypass** -- if the ``x-model-router`` header equals
            ``"bypass"`` the resolved model is taken from the
            ``x-model-router-model`` header.  Only callers with
            ``api_key_role == "admin"`` may use this path.

        2.  **Model router** -- content-based route via
            ``coordinator._model_router``.

        3.  **Registry validation** -- the resolved name is checked against
            ``coordinator.list_models()``.

        4.  **Adapter validation** -- if ``body.adapter`` is set it must exist
            in the adapter manager.

        Args:
            body: The chat completion request body (duck-typed, expected to
                have ``.model``, ``.messages``, ``.adapter`` attributes).
            request_state: The incoming request object (duck-typed, expected
                to have ``.headers`` for header access and ``.state`` for
                properties such as ``api_key_role``).

        Returns:
            The resolved model name.

        Raises:
            PermissionError: If the caller lacks the admin role for the
                bypass path.
            LookupError: If the resolved model or requested adapter is
                not registered.
        """
        coord = self._coordinator

        # -- 1. Admin bypass via headers ------------------------------------
        router_override = (request_state.headers.get("x-model-router", "") if hasattr(request_state, "headers") else "").lower()
        override_model = request_state.headers.get("x-model-router-model", "") if hasattr(request_state, "headers") else ""
        resolved = body.model

        if router_override == "bypass" and override_model:
            caller_role = getattr(request_state.state, "api_key_role", None) if hasattr(request_state, "state") else None
            if caller_role != "admin":
                raise PermissionError("Model router bypass requires admin role")
            resolved = override_model
            routing_metrics = getattr(coord, "_routing_metrics", None)
            if routing_metrics is not None:
                routing_metrics.record_bypass()
            return resolved

        # -- 2. Model router (content-based) --------------------------------
        model_router = getattr(coord, "_model_router", None)
        if model_router is not None and body.model:
            if model_router.is_hybrid_model(body.model):
                first_content = _extract_text(body.messages[0].content) if body.messages else ""
                resolved = model_router.resolve(first_content)
            elif body.model in ("distributed-llm", ""):
                first_content = _extract_text(body.messages[0].content) if body.messages else ""
                available = coord.list_models()
                routed = model_router.resolve(
                    first_content,
                    available_models=available,
                )
                if routed:
                    resolved = routed

        # -- 3. Validate resolved model against registry --------------------
        if hasattr(coord, "list_models") and resolved not in ("distributed-llm", ""):
            available = coord.list_models()
            if resolved not in available:
                raise LookupError(
                    f"Model '{resolved}' not found. Available: {available}"
                )

        # -- 4. Adapter validation ------------------------------------------
        adapter_mgr = getattr(coord, "adapter_manager", None)
        if body.adapter is not None and adapter_mgr is not None:
            if body.adapter not in adapter_mgr.list_adapters():
                raise LookupError(
                    f"Adapter '{body.adapter}' not found. "
                    f"Available: {adapter_mgr.list_adapters()}"
                )

        return resolved

    # -- prompt building ----------------------------------------------------

    def build_prompt(
        self,
        messages: Any,
        has_images: bool,
        vlm_pipeline: Any,
        template_engine: TemplateEngine | None = None,
    ) -> str:
        """Build the generation prompt from *messages*.

        *   **Multi-modal** -- delegates to the VLM pipeline for image
            encoding and token injection.
        *   **Text-only** -- uses ``TemplateEngine`` with
            ``tokenizer.apply_chat_template()`` fallback.

        Args:
            messages: List of message objects (duck-typed with ``.role``
                and ``.content`` attributes).
            has_images: ``True`` when at least one message holds an image.
            vlm_pipeline: The VLM pipeline instance (or ``None``).
            template_engine: Optional pre-configured ``TemplateEngine``.
                Created fresh when not supplied.

        Returns:
            The formatted prompt string.
        """
        if has_images and vlm_pipeline is not None:
            vlm_messages = [m.model_dump() for m in messages]
            text_prompt, images = vlm_pipeline.parse_messages(vlm_messages)
            image_embeddings = vlm_pipeline.encode_images_to_embeddings(images)
            prompt, _ = vlm_pipeline.build_prompt_with_images(
                text_prompt,
                image_embeddings,
            )
            return prompt

        # -- Text-only path -------------------------------------------------
        engine = template_engine or TemplateEngine()
        tokenizer = getattr(self._coordinator, "tokenizer", None)
        if tokenizer is not None:
            engine.set_tokenizer(tokenizer)

        messages_dicts: list[dict[str, str]] = [
            {
                "role": m.role,
                "content": m.content
                if isinstance(m.content, str)
                else _extract_text(m.content),
            }
            for m in messages
        ]

        try:
            prompt = engine.apply(messages_dicts, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(
                f"{m.role}: {m.content}"
                if isinstance(m.content, str)
                else f"{m.role}: {_extract_text(m.content)}"
                for m in messages
            )

        return prompt

    # -- structured output constraint ---------------------------------------

    def build_constraint(
        self,
        response_format: dict | None,
        tokenizer: Any,
    ) -> JSONSchemaConstraint | None:
        """Build a ``JSONSchemaConstraint`` from a *response_format* dict.

        Args:
            response_format: The response format dict (e.g.
                ``{"type": "json_object"}``).
            tokenizer: The model tokenizer used for character-level
                logit masking.

        Returns:
            A ``JSONSchemaConstraint`` instance or ``None`` when no
            constraint is applicable.
        """
        if not response_format:
            return None
        fmt_type = response_format.get("type", "")
        if fmt_type in ("json_object", "json_schema"):
            return JSONSchemaConstraint.from_response_format(
                response_format,
                tokenizer=tokenizer,
            )
        return None

    # -- tool setup ---------------------------------------------------------

    def setup_tools(
        self,
        tools: list[dict] | None,
        functions: list[dict] | None,
        function_call: str | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        """Initialise and return the tool-calling infrastructure.

        Args:
            tools: OpenAI-format tool definitions.
            functions: Deprecated function definitions (used only when
                *tools* is ``None``).
            function_call: Deprecated function-call control.
            tool_choice: Tool-choice value (``"none"``, ``"auto"``,
                ``"required"``, or a dict).

        Returns:
            A dict with keys:

            * ``tool_engine`` (:class:`ToolCallingEngine`) -- the
              initialised tool-calling engine.
            * ``tool_schemas`` (list[dict]) -- parsed tool/function
              schemas.
            * ``tool_calls_list`` (list[dict]) -- empty list, populated
              later after generation.
            * ``tool_choice_value`` (str | dict | None) -- the effective
              tool-choice value.
        """
        tool_engine = ToolCallingEngine()
        tool_calls_list: list[dict] = []
        tool_choice_value = tool_choice
        tool_schemas: list[dict] = []

        if tools:
            tool_schemas = tool_engine.parse_schemas(tools)
            tool_engine.register_tools_from_schemas(tools)

        # Handle deprecated ``functions`` parameter
        if functions and not tools:
            func_tools: list[dict] = []
            for func_def in functions:
                func_tools.append({
                    "type": "function",
                    "function": func_def,
                })
            tool_schemas = tool_engine.parse_schemas(func_tools)
            if function_call:
                tool_choice_value = function_call

        return {
            "tool_engine": tool_engine,
            "tool_schemas": tool_schemas,
            "tool_calls_list": tool_calls_list,
            "tool_choice_value": tool_choice_value,
        }

    # -- generation ---------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        user_id: str | None = None,
        response_format: dict | None = None,
        constraint: JSONSchemaConstraint | None = None,
        cache_breakpoints: list[int] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        logit_bias: dict[str, float] | None = None,
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
            user_id: User / tenant identifier.
            response_format: Structured output format (passed through
                to the coordinator for any server-side handling).
            constraint: Token-level JSON constraint for logit masking.
            cache_breakpoints: Indices of cache-control breakpoint messages.
            presence_penalty: Penalize new tokens based on presence in text.
            frequency_penalty: Penalize tokens based on frequency in text.
            seed: Random seed for deterministic sampling.
            stop: Stop sequences to halt generation.
            logit_bias: Modify likelihood of specified tokens.

        Returns:
            The generated text.
        """
        # Build kwargs dict with optional sampling parameters so they are
        # passed through even when the coordinator does not yet support them
        # natively (unknown kwargs are silently ignored by asyncio.to_thread).
        kwargs = dict(
            user_id=user_id,
            response_format=response_format,
            constraint=constraint,
            cache_breakpoints=cache_breakpoints,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            stop=stop,
            logit_bias=logit_bias,
        )
        # Remove None values so the coordinator only receives explicitly set params
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        result = await asyncio.to_thread(
            self._coordinator.generate,
            prompt,
            max_tokens,
            temperature,
            top_p,
            **kwargs,
        )
        return result

    # -- structured output validation with retry ----------------------------

    async def validate_structured_output(
        self,
        generated: str,
        response_format: dict | None,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        user_id: str | None = None,
        constraint: JSONSchemaConstraint | None = None,
        cache_breakpoints: list[int] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        logit_bias: dict[str, float] | None = None,
    ) -> str:
        """Validate *generated* text against *response_format* with retries.

        When *strict* mode is enabled (``response_format["strict"]`` is
        truthy) the method retries up to 3 times.  On each retry the
        schema validation errors are appended to the prompt as a
        corrective feedback message so the model can self-correct.

        Args:
            generated: The generated text to validate.
            response_format: The response format dict.
            prompt: The original prompt (extended with feedback on
                each retry).
            max_tokens: Maximum tokens for the retry generation.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            user_id: User / tenant identifier.
            constraint: Token-level JSON constraint.
            cache_breakpoints: Cache-control breakpoint indices.
            presence_penalty: Penalize new tokens based on presence in text.
            frequency_penalty: Penalize tokens based on frequency in text.
            seed: Random seed for deterministic sampling.
            stop: Stop sequences to halt generation.
            logit_bias: Modify likelihood of specified tokens.

        Returns:
            The (possibly corrected) generated text.
        """
        if not response_format or not generated:
            return generated

        strict = response_format.get("strict", False)
        max_retries = 3 if strict else 0
        current = generated
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            validation = self._so_engine.validate(current, response_format)
            if validation.valid:
                return current

            if attempt >= max_retries:
                logger.warning(
                    "Structured output validation failed after %d retries: %s",
                    max_retries,
                    validation.errors,
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

            retry_kwargs = dict(
                user_id=user_id,
                response_format=response_format,
                constraint=constraint,
                cache_breakpoints=cache_breakpoints,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
                stop=stop,
                logit_bias=logit_bias,
            )
            retry_kwargs = {k: v for k, v in retry_kwargs.items() if v is not None}
            result = await asyncio.to_thread(
                self._coordinator.generate,
                current_prompt,
                max_tokens,
                temperature,
                top_p,
                **retry_kwargs,
            )
            current = result

        return current

    # -- tool detection & execution -----------------------------------------

    async def detect_and_execute_tools(
        self,
        result: str,
        tool_engine: ToolCallingEngine,
        tool_choice_value: str | dict | None,
        body: Any,
    ) -> tuple[str, str, list[dict] | None]:
        """Detect tool calls in *result*, enforce choice, and execute.

        When tools are invoked and return results this method performs a
        second generation pass with the tool results injected into the
        conversation context (a "tool use loop").

        Args:
            result: The generated text to inspect for tool calls.
            tool_engine: The ``ToolCallingEngine`` returned by
                :meth:`setup_tools`.
            tool_choice_value: Tool-choice value (``"none"``, ``"auto"``,
                ``"required"``, or a ``{"type": "function", ...}`` dict).
            body: The chat completion request body (duck-typed).

        Returns:
            A 3-tuple ``(generated, finish_reason, assistant_tool_calls)``:

            * **generated** -- the final generated text.
            * **finish_reason** -- ``"stop"`` or ``"tool_calls"``.
            * **assistant_tool_calls** -- list of tool-call dicts for
              the response, or ``None``.
        """
        # Reconstruct tool_schemas from body for the detection check and
        # second-pass tool prompt.
        tool_schemas: list[dict] = []
        if body.tools:
            tool_schemas = tool_engine.parse_schemas(body.tools)
        elif body.functions:
            func_tools = [
                {"type": "function", "function": f}
                for f in body.functions
            ]
            tool_schemas = tool_engine.parse_schemas(func_tools)

        generated = result
        finish_reason = "stop"
        assistant_tool_calls: list[dict] | None = None

        if not tool_schemas or not tool_engine.has_tool_calls(generated):
            return (generated, finish_reason, assistant_tool_calls)

        # -- Extract and enforce --------------------------------------------
        tool_calls_list = tool_engine.extract_tool_calls(generated)
        tool_calls_list = tool_engine.enforce_tool_choice(
            tool_choice_value,
            tool_calls_list,
            parallel=getattr(body, "parallel_tool_calls", True),
        )

        if not tool_calls_list:
            return (generated, finish_reason, assistant_tool_calls)

        finish_reason = "tool_calls"

        # -- Execute tool calls ---------------------------------------------
        tool_results = await asyncio.to_thread(
            tool_engine.execute_tool_calls,
            tool_calls_list,
        )

        # Build assistant tool-call response entries
        assistant_tool_calls = [
            {
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": json.dumps(
                        tc.get("function", {}).get("arguments", {})
                    ),
                },
            }
            for tc in tool_calls_list
        ]

        # -- Second generation pass with tool results -----------------------
        if tool_results and tool_engine.should_continue_after_tool_calls(
            tool_calls_list, tool_results,
        ):
            messages_list = [
                m.model_dump(exclude_none=True) for m in body.messages
            ]
            new_messages = tool_engine.inject_tool_results(
                messages_list,
                generated,
                tool_calls_list,
                tool_results,
            )

            final_prompt, _ = tool_engine.build_tool_prompt(
                tool_schemas,
                new_messages,
                tool_choice="none",
            )

            tokenizer = getattr(self._coordinator, "tokenizer", None)
            remaining = (
                body.max_tokens - len(tokenizer.encode(generated))
                if tokenizer
                else body.max_tokens
            )

            final_result = await asyncio.to_thread(
                self._coordinator.generate,
                final_prompt,
                max(1, remaining),
                getattr(body, "temperature", 0.7),
                getattr(body, "top_p", 0.9),
            )

            generated = (
                final_result[len(final_prompt):]
                if final_result.startswith(final_prompt)
                else final_result
            )
            finish_reason = "stop"

        return (generated, finish_reason, assistant_tool_calls)

    # -- scheduling metadata ------------------------------------------------

    def build_scheduling_meta(
        self,
        body: Any,
        request_state: Any,
    ) -> dict[str, Any]:
        """Compute scheduling hints and store them on the coordinator.

        Merges top-level fields (``body.priority``, ``body.max_latency_ms``)
        with the ``body.scheduling`` object when present.  When the
        coordinator exposes a ``_pending_scheduling_hints`` dict a hint
        entry is created and the returned ``effective_user_id`` includes
        the hint reference.

        Args:
            body: The chat completion request body (duck-typed).
            request_state: The incoming request object (duck-typed,
                expected to have a ``.state`` attribute for tenant info).

        Returns:
            A dict with keys:

            * ``effective_priority`` (int)
            * ``effective_max_latency`` (float | None)
            * ``effective_user_id`` (str)
            * ``scheduling_meta`` (dict) -- advanced scheduling hints.
        """
        coord = self._coordinator

        effective_priority = body.priority
        effective_max_latency = body.max_latency_ms
        scheduling_meta: dict[str, Any] = {}

        scheduling: Any = getattr(body, "scheduling", None)
        if scheduling is not None:
            if scheduling.priority is not None:
                effective_priority = scheduling.priority
            if scheduling.max_latency_ms is not None:
                effective_max_latency = scheduling.max_latency_ms
            scheduling_meta = {
                "preemptible": scheduling.preemptible,
                "estimated_output_tokens": scheduling.estimated_output_tokens,
                "scheduling_group": scheduling.scheduling_group,
                "cost_limit": scheduling.cost_limit,
            }

        hints_store = getattr(coord, "_pending_scheduling_hints", None)
        if hints_store is not None:
            hint_id = uuid.uuid4().hex[:8]
            hints_store[hint_id] = {
                "priority": effective_priority,
                "max_latency_ms": effective_max_latency,
                **scheduling_meta,
            }
            tenant = getattr(request_state.state, "tenant", "default") if hasattr(request_state, "state") else "default"
            effective_user_id = f"{tenant}:{hint_id}"
        else:
            tenant = getattr(request_state.state, "tenant", "default") if hasattr(request_state, "state") else "default"
            effective_user_id = tenant

        return {
            "effective_priority": effective_priority,
            "effective_max_latency": effective_max_latency,
            "effective_user_id": effective_user_id,
            "scheduling_meta": scheduling_meta,
        }
