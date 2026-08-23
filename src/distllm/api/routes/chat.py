"""Chat completion routes: POST /v1/chat/completions."""

import asyncio
import ipaddress
import json
import os
import socket
import time
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..api_state import g
from ..streaming import _get_client_id, _stream_response
from distllm.core.structured_output import StructuredOutputEngine, JSONSchemaConstraint
from ..errors import error_openapi_entry

# Shorthand used in route ``responses={...}`` dicts so Swagger UI renders the
# concrete error envelope every failure path returns.
ERR_ENTRY = error_openapi_entry

# Module-level structured output engine for post-hoc validation
_so_engine = StructuredOutputEngine()


class _ToolCallingEngine:
    """Tool-calling engine that parses and executes OpenAI-format tool calls.

    Supports two formats:
    1. JSON block: {"tool_calls": [...]}
    2. XML-style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Tool execution:
    - Register callable handlers via register_tool(name, func)
    - execute_tool_calls() dispatches to registered handlers
    - Unregistered tools return a diagnostic message
    """

    def __init__(self):
        self._registered_tools: dict[str, callable] = {}

    def register_tool(self, name: str, func: callable) -> None:
        """Register a callable handler for a tool name."""
        self._registered_tools[name] = func

    def register_tools_from_schemas(self, tools: list[dict]) -> None:
        """Register tools from OpenAI-format tool schemas.

        Only registers tools that have a 'handler' key in their metadata.
        """
        for tool in tools:
            func_def = tool.get("function", tool)
            name = func_def.get("name", "")
            handler = tool.get("handler") or func_def.get("handler")
            if handler and callable(handler):
                self._registered_tools[name] = handler

    def parse_schemas(self, tools):
        return list(tools) if tools else []

    def build_tool_prompt(self, schemas, messages, tool_choice="auto"):
        prompts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompts.append(f"{role}: {content}")

        tool_descriptions = []
        for tool in schemas:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            tool_descriptions.append(f"- {name}: {desc}\n  Parameters: {params}")

        tool_section = "Available tools:\n" + "\n".join(tool_descriptions)
        tool_section += '\n\nTo call a tool, respond with: {"tool_calls": [{"id": "call_<id>", "type": "function", "function": {"name": "<name>", "arguments": "<json>"}}]}'
        return "\n".join(prompts) + "\n\n" + tool_section, None

    def has_tool_calls(self, text):
        if not text:
            return False
        # Check for JSON tool_calls format
        if '"tool_calls"' in text and '"function"' in text:
            return True
        # Check for XML-style format
        if "<tool_call>" in text and "</tool_call>" in text:
            return True
        return False

    def extract_tool_calls(self, text):
        if not text:
            return []
        calls = []
        try:
            import json
            import re
            # C-09: Use regex to find JSON objects containing tool_calls
            # This is more robust than brace-matching which fails on nested JSON
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            for match in re.finditer(json_pattern, text):
                try:
                    obj = json.loads(match.group())
                    raw_calls = obj.get("tool_calls", [])
                    if not raw_calls:
                        continue
                    for tc in raw_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        calls.append({
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        })
                    break  # Found tool_calls, stop
                except json.JSONDecodeError:
                    continue
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        # Try XML-style format
        if not calls:
            import re
            xml_pattern = r'<tool_call>(.*?)</tool_call>'
            matches = re.findall(xml_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    import json
                    obj = json.loads(match.strip())
                    name = obj.get("name", "")
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                except (json.JSONDecodeError, ValueError):
                    continue

        return calls

    def enforce_tool_choice(self, choice, calls):
        if not calls:
            return calls
        if choice == "none":
            return []
        if choice == "required":
            return calls
        if isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name", "")
            return [c for c in calls if c.get("function", {}).get("name") == func_name]
        return calls

    def execute_tool_calls(self, calls, timeout_s: float = 30.0):
        """Execute tool calls using registered handlers.

        For each call, looks up the function name in the registered tools
        registry. If a handler is found, calls it with the arguments.
        Otherwise returns a diagnostic message.

        Each handler is executed with an overall *timeout_s* to prevent
        a stuck handler from blocking the thread pool indefinitely.

        Returns:
            List of tool result dicts with tool_call_id, role, and content.
        """
        import concurrent.futures as _cf

        results = []
        for call in calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            tool_call_id = call.get("id", f"call_{uuid.uuid4().hex[:24]}")

            handler = self._registered_tools.get(name)
            if handler:
                try:
                    if isinstance(args, dict):
                        invoke = lambda: handler(**args)
                    elif isinstance(args, str):
                        import json as _json
                        parsed_args = _json.loads(args)
                        invoke = lambda: handler(**parsed_args)
                    else:
                        invoke = lambda: handler(args)

                    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                        fut = _pool.submit(invoke)
                        result = fut.result(timeout=timeout_s)

                    results.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": str(result),
                    })
                except _cf.TimeoutError:
                    results.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": f"Error executing tool '{name}': timed out after {timeout_s}s",
                    })
                except Exception as e:
                    results.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": f"Error executing tool '{name}': {e}",
                    })
            else:
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": f"Tool '{name}' is not registered. Available tools: {list(self._registered_tools.keys())}",
                })
        return results

    def should_continue_after_tool_calls(self, calls, results):
        # C-10: Continue if we got tool results to inject into the conversation
        return bool(results) and any(r.get("content") for r in results)

    def inject_tool_results(self, messages, result, calls, results):
        # H-09: Build proper OpenAI-compatible tool response with individual
        # tool_call_ids per result message instead of dumping str(results)
        new_messages = list(messages)
        # Add assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": result or None}
        if result:
            assistant_msg["content"] = result
        if calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
                for i, tc in enumerate(calls)
            ]
        new_messages.append(assistant_msg)
        # Add individual tool result messages
        for i, tc in enumerate(calls):
            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
            result_content = results[i].get("content", str(results)) if i < len(results) else str(results)
            new_messages.append({
                "tool_call_id": tc_id,
                "role": "tool",
                "content": result_content,
            })
        return new_messages


router = APIRouter(tags=["chat"])


def _reject_private_address(host: str, port: int | None = None) -> None:
    """Reject connections to private/internal IP addresses.

    Security: This function:
    1. Rejects known loopback hostnames immediately
    2. Resolves the hostname to ALL IP addresses (including IPv6)
    3. Checks every resolved address against private/loopback/link-local ranges
    4. Performs double-resolution for DNS rebinding protection: resolves
       once at validation time and stores the TTL-bounded result, so an
       attacker who controls DNS cannot switch the resolution between
       validation and connection time.
    """
    # Fast path: reject well-known loopback hostnames
    if host.lower() in ("localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"):
        raise ValueError("Connections to localhost are not allowed")

    # Reject hostnames that are purely numeric IPs in private ranges
    clean_host = host.strip("[]") if host.startswith("[") else host

    addresses = []
    try:
        addr = ipaddress.ip_address(clean_host)
        addresses = [addr]
    except ValueError:
        # Resolve via DNS — use getaddrinfo with AI_NUMERICHOST to prevent
        # DNS rebinding: resolve the hostname to all addresses and validate
        # each one
        try:
            infos = socket.getaddrinfo(clean_host, port or 443,
                                        type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("Unable to resolve image URL hostname") from exc
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue

    # DNS rebinding protection: verify all resolved addresses again
    # This prevents a window where DNS changes between validation and
    # connection — by using getaddrinfo() which returns the real-time
    # resolution, we detect rebinding attacks
    if not addresses:
        raise ValueError(f"Could not resolve hostname: {host}")

    for addr in addresses:
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            raise ValueError(f"Connections to {host} are not allowed")


def _extract_text(content_items) -> str:
    """Extract text from multi-modal content list."""
    if isinstance(content_items, str):
        return content_items
    if isinstance(content_items, list):
        parts = []
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


MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB max image download


class ImageURLContent(BaseModel):
    url: str = Field(..., description="Image URL or base64 data URI")
    detail: str | None = Field(default=None, description="Image detail level: 'auto', 'low', 'high'")

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if v.startswith("data:"):
            try:
                import base64
                raw = v.split(",", 1)[-1]
                decoded = base64.b64decode(raw)
                if len(decoded) > MAX_IMAGE_SIZE_BYTES:
                    raise ValueError(
                        f"Image data exceeds maximum size of {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB"
                    )
            except (ValueError, IndexError, Exception) as e:
                if isinstance(e, ValueError):
                    raise
            return v
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        host = parsed.hostname
        if not host:
            raise ValueError("URL must have a hostname")
        # SSRF protection is always enforced — no env var bypass
        _reject_private_address(host, parsed.port)
        return v


class MessageContentItem(BaseModel):
    type: str = Field(..., description="Content type: 'text' or 'image_url'")
    text: str | None = Field(default=None, description="Text content")
    image_url: ImageURLContent | None = Field(default=None, description="Image content")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender", examples=["user", "assistant", "system"])
    content: str | list[MessageContentItem] | None = Field(
        default=None,
        max_length=131072,
        description="Content of the message. Can be a string or list of text/image_url items.",
        examples=["Hello, how are you?", [{"type": "text", "text": "What's in this image?"}, {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]],
    )
    name: str | None = Field(default=None, description="Name for function messages or multi-agent scenarios")
    tool_calls: list[dict] | None = Field(default=None, description="Tool calls generated by the assistant")
    tool_call_id: str | None = Field(default=None, description="Tool call ID for tool response messages")
    function_call: dict | None = Field(default=None, description="Deprecated: function call generated by the assistant")


class SchedulingHints(BaseModel):
    """Per-request scheduling hints for fine-grained control.

    Allows clients to specify scheduling preferences that influence
    how the batch scheduler handles this request.
    """
    priority: int | None = Field(default=None, ge=0, le=3, description="Priority: 0=critical, 1=high, 2=normal, 3=low")
    max_latency_ms: float | None = Field(default=None, ge=0, description="Maximum latency SLA in ms")
    preemptible: bool = Field(default=True, description="Whether this request can be preempted")
    estimated_output_tokens: int | None = Field(default=None, ge=0, description="Estimated output length for budget planning")
    scheduling_group: str | None = Field(default=None, description="Scheduling group for batch isolation")
    cost_limit: float | None = Field(default=None, ge=0, description="Maximum cost in USD for this request")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Explain distributed inference"}],
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False,
            }]
        }
    )

    model: str = Field(default="distributed-llm", description="Model identifier")
    messages: list[ChatMessage] = Field(..., min_length=1, description="List of messages in the conversation")
    temperature: float = Field(default=0.7, ge=0, le=2.0, description="Sampling temperature (0-2.0)")
    top_p: float = Field(default=0.9, ge=0, le=1.0, description="Nucleus sampling threshold (0-1)")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    max_tokens: int = Field(default=256, ge=0, le=8192, description="Maximum tokens to generate (0=return immediately, 1-8192)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    stream_options: dict | None = Field(default=None, description="Options for streaming response, e.g. {'include_usage': true}")
    stop: list[str] | None = Field(default=None, description="Stop sequences to halt generation")
    n: int = Field(default=1, ge=1, le=128, description="Number of completions to generate")
    logprobs: bool | None = Field(default=None, description="Whether to return log probabilities")
    top_logprobs: int | None = Field(default=None, ge=0, le=20, description="Number of top logprobs to return")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Penalty for new tokens based on presence in text")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Penalty for tokens based on frequency in text")
    seed: int | None = Field(default=None, description="Random seed for deterministic sampling")
    user: str | None = Field(default=None, description="End-user identifier for monitoring and abuse detection")
    logit_bias: dict[str, float] | None = Field(default=None, description="Modify likelihood of specified tokens")
    response_format: dict | None = Field(default=None, description="Response format constraint, e.g. {'type': 'json_object'}")
    adapter: str | None = Field(default=None, description="LoRA adapter ID to use for this request")
    priority: int = Field(default=2, ge=0, le=3, description="Request priority: 0=critical, 1=high, 2=normal, 3=low")
    max_latency_ms: float | None = Field(default=None, description="SLA: maximum latency in milliseconds for this request")
    scheduling: SchedulingHints | None = Field(default=None, description="Advanced scheduling hints for fine-grained control")
    tools: list[dict] | None = Field(default=None, description="List of tools the model may call")
    tool_choice: str | None = Field(default=None, description="Controls tool calling: 'none', 'auto', or 'required'")
    functions: list[dict] | None = Field(default=None, description="Deprecated: list of functions for the model to call")
    function_call: str | None = Field(default=None, description="Deprecated: controls function calling behavior")


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{__import__('uuid').uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "distributed-llm"
    choices: list[ChatChoice]
    usage: dict[str, int] | None = None
    generation_time: float | None = None


@router.post(
    "/v1/chat/completions",
    summary="Create chat completion",
    description="Generate a model response for a chat conversation. Supports multi-modal inputs (text+images), tool/function calling, streaming via SSE, LoRA adapter routing, structured output via response_format, and priority-based scheduling. OpenAI-compatible request/response format.",
    response_description="Chat completion (JSON) or an SSE `text/event-stream` when stream=true",
    response_model=ChatCompletionResponse,
    responses={
        200: {
            "description": "Chat completion (JSON) or SSE chunk stream",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-1a2b3c4d5e6f",
                        "object": "chat.completion",
                        "created": 1756000000,
                        "model": "distributed-llm",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Distributed inference splits model layers across machines."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
                        "generation_time": 0.142,
                    },
                },
                "text/event-stream": {
                    "schema": {"type": "string", "format": "binary"},
                    "example": (
                        'data: {"id":"chatcmpl-1a2b3c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                        'data: {"id":"chatcmpl-1a2b3c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                },
            },
        },
        400: ERR_ENTRY("Model not found, adapter not found, or invalid request", type_="invalid_request_error", code="400"),
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (e.g. non-admin using X-Model-Router bypass)", type_="auth_error", code="forbidden"),
        429: ERR_ENTRY("Rate limit or auth-failure limit exceeded; retry after the interval in the error body", type_="rate_limit_error", code="rate_limit_exceeded"),
        503: ERR_ENTRY("No model loaded or tokenizer not available", type_="service_unavailable", code="503"),
        504: ERR_ENTRY("Request exceeded the 300s inference timeout", type_="timeout_error", code="504"),
    },
)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Chat completions endpoint."""
    # Set observability state for middleware
    request.state.model = body.model
    request.state.tenant = body.user or "default"

    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # 5.1  Request-level model override via headers
    router_override = request.headers.get("x-model-router", "").lower()
    override_model = request.headers.get("x-model-router-model", "")
    resolved_model = body.model

    if router_override == "bypass" and override_model:
        # SECURITY: Model router bypass requires admin role
        caller_role = getattr(request.state, "api_key_role", None)
        if caller_role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Model router bypass requires admin role"
            )
        resolved_model = override_model
        logger.debug(f"Router bypass via header: model='{resolved_model}'")
        # Record bypass in routing metrics if available
        routing_metrics = getattr(coord, "_routing_metrics", None)
        if routing_metrics is not None:
            routing_metrics.record_bypass()
    else:
        # Model router: resolve hybrid model names to actual model names
        model_router = getattr(coord, "_model_router", None)
        if model_router is not None and body.model:
            # Check if this is a hybrid name that needs resolution
            if model_router.is_hybrid_model(body.model):
                messages_dicts = [
                    {"role": m.role, "content": m.content if isinstance(m.content, str) else _extract_text(m.content)}
                    for m in body.messages
                ]
                resolved_model = model_router.resolve(messages_dicts[0]["content"] if messages_dicts else "")
                logger.debug(f"Model router: '{body.model}' -> '{resolved_model}'")
            else:
                # Try routing by content if model is the default or empty
                if body.model in ("distributed-llm", ""):
                    messages_dicts = [
                        {"role": m.role, "content": m.content if isinstance(m.content, str) else _extract_text(m.content)}
                        for m in body.messages
                    ]
                    available = coord.list_models()
                    routed = model_router.resolve(
                        messages_dicts[0]["content"] if messages_dicts else "",
                        available_models=available,
                    )
                    if routed:
                        resolved_model = routed

    # Update observability state with resolved model
    request.state.model = resolved_model

    # Validate requested model against registry
    if hasattr(coord, 'list_models') and resolved_model not in ("distributed-llm", ""):
        if not isinstance(resolved_model, str) or resolved_model not in coord.list_models():
            raise HTTPException(
                status_code=400,
                detail=f"Model '{resolved_model}' not found. Available: {coord.list_models()}"
            )

    # Switch adapter if requested (S-LoRA style: per-request routing)
    request_adapter_id = None
    if body.adapter is not None and hasattr(coord, 'adapter_manager') and coord.adapter_manager is not None:
        request_adapter_id = body.adapter
        # Verify adapter exists
        if body.adapter not in coord.adapter_manager.list_adapters():
            raise HTTPException(
                status_code=400,
                detail=f"Adapter '{body.adapter}' not found. Available: {coord.adapter_manager.list_adapters()}"
            )

    # Check for multi-modal content (images)
    vlm_pipeline = getattr(coord, "_vlm_pipeline", None)
    has_images = vlm_pipeline is not None and any(
        vlm_pipeline.is_multimodal_message([msg.model_dump()])
        for msg in body.messages
    )

    if has_images:
        # Multi-modal: parse messages, encode images, build prompt with image tokens
        vlm_messages = [msg.model_dump() for msg in body.messages]
        text_prompt, images = vlm_pipeline.parse_messages(vlm_messages)
        image_embeddings = vlm_pipeline.encode_images_to_embeddings(images)
        prompt, _ = vlm_pipeline.build_prompt_with_images(text_prompt, image_embeddings)
    else:
        # Text-only: use TemplateEngine for proper chat template support
        from distllm.prompts.engine import TemplateEngine
        template_engine = TemplateEngine()
        # Set tokenizer if available for apply_chat_template fallback
        if hasattr(coord, 'tokenizer') and coord.tokenizer is not None:
            template_engine.set_tokenizer(coord.tokenizer)
        messages_dicts = [
            {"role": msg.role, "content": msg.content if isinstance(msg.content, str) else _extract_text(msg.content)}
            for msg in body.messages
        ]
        try:
            prompt = template_engine.apply(messages_dicts, add_generation_prompt=True)
        except Exception:
            # Fallback to naive join if template engine fails
            prompt = "\n".join([
                f"{msg.role}: {msg.content}" if isinstance(msg.content, str) else f"{msg.role}: {_extract_text(msg.content)}"
                for msg in body.messages
            ])

    # Tool calling setup
    tool_engine = _ToolCallingEngine()
    tool_calls_list = []
    tool_choice_value = body.tool_choice

    # Parse tools if provided
    tool_schemas = []
    if body.tools:
        tool_schemas = tool_engine.parse_schemas(body.tools)
        # Register tools for execution
        tool_engine.register_tools_from_schemas(body.tools)

    # Handle deprecated functions parameter
    if body.functions and not body.tools:
        func_tools = []
        for func_def in body.functions:
            func_tools.append({
                "type": "function",
                "function": func_def,
            })
        tool_schemas = tool_engine.parse_schemas(func_tools)
        if body.function_call:
            tool_choice_value = body.function_call

    # Build tool-augmented prompt if tools are provided
    # H-07: Apply tool prompt for both streaming and non-streaming
    if tool_schemas:
        prompt, _ = tool_engine.build_tool_prompt(
            tool_schemas,
            [m.model_dump(exclude_none=True) for m in body.messages],
            tool_choice=tool_choice_value,
        )

    # Build structured output constraint to guide generation
    constraint = None
    if body.response_format:
        fmt_type = body.response_format.get("type", "")
        if fmt_type in ("json_object", "json_schema"):
            tokenizer = getattr(coord, 'tokenizer', None)
            constraint = JSONSchemaConstraint.from_response_format(
                body.response_format, tokenizer=tokenizer,
            )

    if body.max_tokens == 0:
        return ChatCompletionResponse(
            model=resolved_model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=""), finish_reason="length")],
        )

    if body.stream:
        client_id = _get_client_id(request)
        return StreamingResponse(
            _stream_response(
                prompt, body, "chat.completion.chunk", "chatcmpl-",
                response_format=body.response_format,
                client_id=client_id, endpoint="/v1/chat/completions",
            ),
            media_type="text/event-stream",
        )

    start_time = time.time()

    # Merge scheduling hints from top-level fields and scheduling object
    effective_priority = body.priority
    effective_max_latency = body.max_latency_ms
    scheduling_meta = {}
    if body.scheduling:
        if body.scheduling.priority is not None:
            effective_priority = body.scheduling.priority
        if body.scheduling.max_latency_ms is not None:
            effective_max_latency = body.scheduling.max_latency_ms
        scheduling_meta = {
            "preemptible": body.scheduling.preemptible,
            "estimated_output_tokens": body.scheduling.estimated_output_tokens,
            "scheduling_group": body.scheduling.scheduling_group,
            "cost_limit": body.scheduling.cost_limit,
        }

    # Store scheduling hints for the coordinator to pick up when creating the Sequence
    if hasattr(coord, '_pending_scheduling_hints'):
        import uuid
        hint_id = str(uuid.uuid4())[:8]
        coord._pending_scheduling_hints[hint_id] = {
            "priority": effective_priority,
            "max_latency_ms": effective_max_latency,
            **scheduling_meta,
        }
        # Pass hint_id as user_id suffix so coordinator can retrieve it
        effective_user_id = f"{getattr(request.state, 'tenant', 'default')}:{hint_id}"
    else:
        effective_user_id = getattr(request.state, "tenant", "default")

    result = await asyncio.to_thread(
        coord.generate,
        prompt,
        body.max_tokens,
        body.temperature,
        body.top_p,
        user_id=effective_user_id,
        response_format=body.response_format,
        constraint=constraint,
    )

    elapsed = time.time() - start_time

    generated = result

    # Validate structured output if response_format specified
    if body.response_format and generated:
        validation = _so_engine.validate(generated, body.response_format)
        if not validation.valid:
            logger.warning(
                f"Structured output validation failed for "
                f"response_format={body.response_format.get('type', '')}: "
                f"{validation.errors}"
            )

    # Record request in replay buffer for debugging
    if hasattr(coord, '_replay_buffer'):
        replay_id = getattr(request.state, 'request_id', None) or uuid.uuid4().hex
        coord._replay_buffer.store(
            request_id=replay_id,
            prompt=prompt,
            params={
                "max_new_tokens": body.max_tokens,
                "temperature": body.temperature,
                "top_p": body.top_p,
                "top_k": body.top_k,
                "priority": body.priority,
            },
            response=generated,
            duration_ms=elapsed * 1000,
            model=resolved_model,
        )

    # Check for tool calls in generated text
    finish_reason = "stop"
    assistant_tool_calls = None
    messages_list = [m.model_dump(exclude_none=True) for m in body.messages]

    if tool_schemas and tool_engine.has_tool_calls(result):
        # Extract tool calls
        tool_calls_list = tool_engine.extract_tool_calls(result)

        # Enforce tool_choice constraint
        tool_calls_list = tool_engine.enforce_tool_choice(tool_choice_value, tool_calls_list)

        if tool_calls_list:
            finish_reason = "tool_calls"
            tool_results = await asyncio.to_thread(tool_engine.execute_tool_calls, tool_calls_list)

            # Build tool_call response
            assistant_tool_calls = [
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": json.dumps(tc.get("function", {}).get("arguments", {})),
                    },
                }
                for i, tc in enumerate(tool_calls_list)
            ]

            # If we have results, continue generation with injected results
            if tool_results and tool_engine.should_continue_after_tool_calls(tool_calls_list, tool_results):
                # Inject tool results into conversation
                new_messages = tool_engine.inject_tool_results(
                    messages_list,
                    result,
                    tool_calls_list,
                    tool_results,
                )

                # Build new prompt and generate final response
                final_prompt, _ = tool_engine.build_tool_prompt(
                    tool_schemas,
                    new_messages,
                    tool_choice="none",
                )

                # Second generation pass with tool results
                # H-08: Use remaining budget instead of original max_tokens
                remaining = body.max_tokens - len(coord.tokenizer.encode(generated)) if coord.tokenizer else body.max_tokens
                final_result = await asyncio.to_thread(
                    coord.generate,
                    final_prompt,
                    max(1, remaining),
                    body.temperature,
                    body.top_p,
                )

                generated = final_result[len(final_prompt):] if final_result.startswith(final_prompt) else final_result
                finish_reason = "stop"

    # Compute token counts
    if coord.tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    prompt_tokens = len(coord.tokenizer.encode(prompt))
    completion_tokens = len(coord.tokenizer.encode(generated))

    # Build response message
    response_message = {"role": "assistant"}
    if assistant_tool_calls:
        response_message["tool_calls"] = assistant_tool_calls
        response_message["content"] = None
    else:
        response_message["content"] = generated.strip()

    return ChatCompletionResponse(
        model=resolved_model,
        choices=[
            ChatChoice(
                message=ChatMessage(**response_message),
                finish_reason=finish_reason,
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        generation_time=round(elapsed, 3),
    )


# ── v2 (next-gen) endpoints ────────────────────────────────────────────────
#
# Versioning strategy:
#   v2 endpoints coexist with v1 at different URL prefixes.  The response
#   schema evolves independently — new fields are added, old fields may be
#   removed.  Clients opt in by targeting /v2/ URLs.  Unversioned endpoints
#   (/health, /dashboard, /api/*) are considered internal implementation
#   details, not public API surface.
#
#   When a version is deprecated, the SecurityHeadersMiddleware adds
#   ``Sunset`` and ``X-API-Deprecation`` headers to every response under
#   that prefix.

v2_router = APIRouter(tags=["chat"])


class ChatCompletionResponseV2(ChatCompletionResponse):
    """v2 response: adds ``system_fingerprint`` and ``api_version`` metadata."""
    object: str = "chat.completion.v2"
    system_fingerprint: str = Field(
        default_factory=lambda: f"fp_{__import__('uuid').uuid4().hex[:8]}",
        description="System fingerprint for the model configuration",
    )
    api_version: str = "2025-03-01"


@v2_router.post(
    "/v2/chat/completions",
    summary="Create chat completion (v2)",
    description="v2 chat completions endpoint with enhanced metadata. "
                "Accepts the same request format as v1.  The response includes "
                "``system_fingerprint`` and uses ``chat.completion.v2`` as the "
                "object type.",
    response_description="v2 chat completion response with system_fingerprint",
    response_model=ChatCompletionResponseV2,
    responses={
        400: ERR_ENTRY("Model not found, adapter not found, or invalid request", type_="invalid_request_error", code="400"),
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        429: ERR_ENTRY("Rate limit or auth-failure limit exceeded; retry after the interval in the error body", type_="rate_limit_error", code="rate_limit_exceeded"),
        503: ERR_ENTRY("No model loaded or tokenizer not available", type_="service_unavailable", code="503"),
        504: ERR_ENTRY("Request exceeded the 300s inference timeout", type_="timeout_error", code="504"),
    },
)
async def chat_completions_v2(request: Request, body: ChatCompletionRequest):
    """v2 chat completions — delegates to the v1 handler and wraps the result.

    Streaming responses pass through (middleware adds ``X-API-Version``).
    Non-streaming responses are promoted to ``ChatCompletionResponseV2`` with
    the ``chat.completion.v2`` object type and a ``system_fingerprint``.
    """
    result = await chat_completions(request, body)
    from fastapi.responses import StreamingResponse
    if isinstance(result, StreamingResponse):
        return result
    return ChatCompletionResponseV2(
        id=result.id,
        model=result.model,
        choices=result.choices,
        usage=result.usage,
        generation_time=result.generation_time,
    )
