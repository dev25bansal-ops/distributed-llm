"""MCP Server exposing DistLLM as Model Context Protocol tools and resources.

Uses FastMCP from the ``mcp`` package when available, with a graceful fallback
to a plain stdio JSON-RPC implementation when ``mcp`` is not installed.

Tools
-----
- chat_completion  — send a chat completion request
- complete         — send a text completion request
- embed            — generate an embedding vector
- list_models      — list available models

Resources
---------
- distllm://health  — cluster health status (JSON)
- distllm://models  — model list (JSON)

Prompts
-------
- summarize          — given a text, returns a summarization prompt template
- analyze_sentiment  — given a text, returns a sentiment-analysis prompt template
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

from distllm_sdk.client import DistLLMClient

logger = logging.getLogger("distllm-mcp")

# ---------------------------------------------------------------------------
# Try the official FastMCP first
# ---------------------------------------------------------------------------

_HAS_FASTMCP: bool
try:
    from mcp.server.fastmcp import FastMCP as _FastMCP
    from mcp.types import GetPromptResult, PromptMessage, TextContent
    _HAS_FASTMCP = True
except ImportError:  # pragma: no cover
    _HAS_FASTMCP = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_base_url() -> str:
    return os.environ.get("DISTLLM_BASE_URL", "http://localhost:8000")


def _default_api_key() -> str | None:
    return os.environ.get("DISTLLM_API_KEY", None)


def _ensure_async_client(
    base_url: str | None = None,
    api_key: str | None = None,
) -> DistLLMClient:
    """Return an async DistLLMClient, constructing one if necessary."""
    return DistLLMClient(
        base_url=base_url or _default_base_url(),
        api_key=api_key or _default_api_key(),
    )


# ===================================================================
# FastMCP-based server (preferred)
# ===================================================================

class DistLLMMCPServer:
    """Expose DistLLM as MCP tools, resources and prompts over stdio.

    Parameters
    ----------
    base_url:
        Base URL of the DistLLM API (default from ``DISTLLM_BASE_URL`` env).
    api_key:
        Optional API key (default from ``DISTLLM_API_KEY`` env).
    name:
        Server name advertised in the MCP handshake.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        name: str = "distllm-mcp",
    ) -> None:
        self._base_url = base_url or _default_base_url()
        self._api_key = api_key or _default_api_key()
        self._name = name

        # Client is created lazily so it picks up the correct event loop.
        self._client: DistLLMClient | None = None

    # -- lazy client property ------------------------------------------------

    @property
    def client(self) -> DistLLMClient:
        if self._client is None:
            self._client = _ensure_async_client(self._base_url, self._api_key)
        return self._client

    # -- public entry-point ---------------------------------------------------

    def run(self) -> None:
        """Start the MCP server (blocks until shutdown)."""
        if _HAS_FASTMCP:
            self._run_fastmcp()
        else:
            self._run_stdio_jsonrpc()

    # ==================================================================
    # FastMCP implementation
    # ==================================================================

    def _run_fastmcp(self) -> None:
        """Start via FastMCP (``mcp`` package) on stdio transport."""
        mcp = _FastMCP(
            name=self._name,
            instructions="MCP server for the DistLLM distributed LLM cluster.",
            debug=os.environ.get("DISTLLM_MCP_DEBUG", "").lower() in ("1", "true"),
        )

        # -- Tools -----------------------------------------------------------

        @mcp.tool(
            name="chat_completion",
            description="Send a chat completion request to the DistLLM cluster.",
        )
        async def chat_completion(
            messages: list[dict[str, str]],
            model: str = "distributed-llm",
            temperature: float = 0.7,
            max_tokens: int = 256,
        ) -> str:
            """Generate a chat completion response.

            Parameters
            ----------
            messages:
                List of message dicts, each with ``role`` and ``content`` keys.
                Example: ``[{"role": "user", "content": "Hello"}]``.
            model:
                Model identifier to use for generation.
            temperature:
                Sampling temperature (0.0 -- 2.0).
            max_tokens:
                Maximum number of tokens to generate.
            """
            resp = await self.client.chat_completions(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _chat_response_to_str(resp)

        @mcp.tool(
            name="complete",
            description="Send a text completion request to the DistLLM cluster.",
        )
        async def complete(
            prompt: str,
            model: str = "distributed-llm",
            max_tokens: int = 256,
        ) -> str:
            """Generate a text completion.

            Parameters
            ----------
            prompt:
                Input text to complete.
            model:
                Model identifier.
            max_tokens:
                Maximum number of tokens to generate.
            """
            resp = await self.client.completions(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
            )
            return _completion_response_to_str(resp)

        @mcp.tool(
            name="embed",
            description="Generate an embedding vector for the given input text.",
        )
        async def embed(
            input: str,
            model: str = "distributed-llm",
        ) -> str:
            """Return embedding vector as a JSON string.

            Parameters
            ----------
            input:
                Input text to embed.
            model:
                Model identifier.
            """
            resp = await self.client.embeddings(input=input, model=model)
            return _embedding_response_to_str(resp)

        @mcp.tool(
            name="list_models",
            description="List models available on the DistLLM cluster.",
        )
        async def list_models_tool() -> str:
            """Return available models as a JSON string."""
            resp = await self.client.list_models()
            return _model_list_to_str(resp)

        # -- Resources -------------------------------------------------------

        @mcp.resource(
            uri="distllm://health",
            name="Cluster Health",
            description="Current health status of the DistLLM cluster.",
            mime_type="application/json",
        )
        async def health_resource() -> str:
            data = await self.client.health_check()
            return json.dumps(data, indent=2, default=str)

        @mcp.resource(
            uri="distllm://models",
            name="Model List",
            description="List of available models on the DistLLM cluster.",
            mime_type="application/json",
        )
        async def models_resource() -> str:
            resp = await self.client.list_models()
            return _model_list_to_str(resp)

        # -- Prompts ---------------------------------------------------------

        @mcp.prompt(
            name="summarize",
            description="Create a summarization prompt for the given text.",
        )
        def summarize_prompt(text: str) -> GetPromptResult:
            """Return a prompt template that asks the model to summarise *text*."""
            return GetPromptResult(
                description=f"Summarize the provided text",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"Please provide a concise summary of the following text:\n\n{text}",
                        ),
                    ),
                ],
            )

        @mcp.prompt(
            name="analyze_sentiment",
            description="Create a sentiment-analysis prompt for the given text.",
        )
        def analyze_sentiment_prompt(text: str) -> GetPromptResult:
            """Return a prompt template that asks the model to analyze sentiment."""
            return GetPromptResult(
                description=f"Analyze the sentiment of the provided text",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=(
                                f"Analyze the sentiment of the following text. "
                                f"Respond with one of: positive, negative, or neutral, "
                                f"and include a brief explanation.\n\n{text}"
                            ),
                        ),
                    ),
                ],
            )

        # Start the server over stdio (blocking).
        mcp.run(transport="stdio")

    # ==================================================================
    # Fallback: basic stdio JSON-RPC implementation
    # ==================================================================

    def _run_stdio_jsonrpc(self) -> None:  # pragma: no cover
        """Fallback server that speaks JSON-RPC 2.0 over stdin/stdout.

        This is a minimal implementation that covers the same tools, resources
        and prompts as the FastMCP version.  It is used only when the ``mcp``
        package is not installed.
        """
        import traceback

        _log = logger.getChild("stdio")

        # -- request / response helpers -------------------------------------

        def _write(msg: dict[str, Any]) -> None:
            line = json.dumps(msg, ensure_ascii=False)
            sys.stdout.write(f"Content-Length: {len(line)}\r\n\r\n{line}")
            sys.stdout.flush()

        def _error(id: Any, code: int, message: str, data: Any = None) -> dict:
            err: dict[str, Any] = {"code": code, "message": message}
            if data is not None:
                err["data"] = data
            return {"jsonrpc": "2.0", "id": id, "error": err}

        # -- async event loop -----------------------------------------------

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _dispatch(method: str, params: dict[str, Any]) -> Any:
            """Route an MCP method to the corresponding handler."""
            handlers: dict[str, Any] = {
                # Lifecycle
                "initialize": lambda p: {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self._name, "version": "1.0.0"},
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {},
                    },
                },
                "ping": lambda p: {},
                # Tools
                "tools/call": self._handle_tool_call,
                "tools/list": lambda p: {
                    "tools": [
                        {
                            "name": "chat_completion",
                            "description": "Send a chat completion request",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "role": {"type": "string"},
                                                "content": {"type": "string"},
                                            },
                                        },
                                        "description": "Conversation messages",
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "Model identifier",
                                        "default": "distributed-llm",
                                    },
                                    "temperature": {
                                        "type": "number",
                                        "description": "Sampling temperature",
                                        "default": 0.7,
                                    },
                                    "max_tokens": {
                                        "type": "integer",
                                        "description": "Max tokens to generate",
                                        "default": 256,
                                    },
                                },
                                "required": ["messages"],
                            },
                        },
                        {
                            "name": "complete",
                            "description": "Send a text completion request",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "Input text to complete",
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "Model identifier",
                                        "default": "distributed-llm",
                                    },
                                    "max_tokens": {
                                        "type": "integer",
                                        "description": "Max tokens to generate",
                                        "default": 256,
                                    },
                                },
                                "required": ["prompt"],
                            },
                        },
                        {
                            "name": "embed",
                            "description": "Generate an embedding vector",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "input": {
                                        "type": "string",
                                        "description": "Input text to embed",
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "Model identifier",
                                        "default": "distributed-llm",
                                    },
                                },
                                "required": ["input"],
                            },
                        },
                        {
                            "name": "list_models",
                            "description": "List available models",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    ],
                },
                # Resources
                "resources/list": lambda p: {
                    "resources": [
                        {
                            "uri": "distllm://health",
                            "name": "Cluster Health",
                            "description": "Current health status of the DistLLM cluster",
                            "mimeType": "application/json",
                        },
                        {
                            "uri": "distllm://models",
                            "name": "Model List",
                            "description": "List of available models on the DistLLM cluster",
                            "mimeType": "application/json",
                        },
                    ],
                },
                "resources/read": self._handle_resource_read,
                # Prompts
                "prompts/list": lambda p: {
                    "prompts": [
                        {
                            "name": "summarize",
                            "description": "Create a summarization prompt for the given text",
                            "arguments": [
                                {
                                    "name": "text",
                                    "description": "The text to summarize",
                                    "required": True,
                                },
                            ],
                        },
                        {
                            "name": "analyze_sentiment",
                            "description": "Create a sentiment-analysis prompt for the given text",
                            "arguments": [
                                {
                                    "name": "text",
                                    "description": "The text to analyze",
                                    "required": True,
                                },
                            ],
                        },
                    ],
                },
                "prompts/get": self._handle_prompt_get,
            }
            handler = handlers.get(method)
            if handler is None:
                raise ValueError(f"Unknown method: {method}")
            result = handler(params)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        async def _process_request(req: dict[str, Any]) -> dict[str, Any]:
            req_id = req.get("id")
            method = req.get("method", "")
            params = req.get("params", {})
            try:
                result = await _dispatch(method, params)
                return {"jsonrpc": "2.0", "id": req_id, "result": result}
            except Exception as exc:
                _log.exception("Error handling %s", method)
                return _error(req_id, -32603, str(exc), traceback.format_exc())

        # -- main loop -------------------------------------------------------

        async def _serve_stdio() -> None:
            buffer = ""
            content_length = 0
            headers_done = False

            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            while True:
                line = await reader.readline()
                if not line:
                    break  # EOF
                text = line.decode("utf-8", errors="replace")
                if not headers_done:
                    text = text.strip()
                    if not text:  # empty line marks end of headers
                        headers_done = True
                    elif text.lower().startswith("content-length:"):
                        content_length = int(text.split(":", 1)[1].strip())
                else:
                    buffer += text
                    if len(buffer) >= content_length:
                        req = json.loads(buffer[:content_length])
                        resp = await _process_request(req)
                        _write(resp)
                        buffer = ""
                        content_length = 0
                        headers_done = False

        try:
            loop.run_until_complete(_serve_stdio())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    # --- tool/resource/prompt handlers shared between transports ------------

    async def _handle_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        args = params.get("arguments", {})

        if name == "chat_completion":
            resp = await self.client.chat_completions(
                messages=args.get("messages", []),
                model=args.get("model", "distributed-llm"),
                temperature=args.get("temperature", 0.7),
                max_tokens=args.get("max_tokens", 256),
            )
            return {"content": [{"type": "text", "text": _chat_response_to_str(resp)}]}
        elif name == "complete":
            resp = await self.client.completions(
                prompt=args.get("prompt", ""),
                model=args.get("model", "distributed-llm"),
                max_tokens=args.get("max_tokens", 256),
            )
            return {"content": [{"type": "text", "text": _completion_response_to_str(resp)}]}
        elif name == "embed":
            resp = await self.client.embeddings(
                input=args.get("input", ""),
                model=args.get("model", "distributed-llm"),
            )
            return {"content": [{"type": "text", "text": _embedding_response_to_str(resp)}]}
        elif name == "list_models":
            resp = await self.client.list_models()
            return {"content": [{"type": "text", "text": _model_list_to_str(resp)}]}
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _handle_resource_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri", "")
        if uri == "distllm://health":
            data = await self.client.health_check()
            text = json.dumps(data, indent=2, default=str)
        elif uri == "distllm://models":
            resp = await self.client.list_models()
            text = _model_list_to_str(resp)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": text}]}

    async def _handle_prompt_get(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        args = params.get("arguments", {})
        text = args.get("text", "")

        if name == "summarize":
            return {
                "description": "Summarize the provided text",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Please provide a concise summary of the following text:\n\n{text}",
                        },
                    },
                ],
            }
        elif name == "analyze_sentiment":
            return {
                "description": "Analyze the sentiment of the provided text",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": (
                                f"Analyze the sentiment of the following text. "
                                f"Respond with one of: positive, negative, or neutral, "
                                f"and include a brief explanation.\n\n{text}"
                            ),
                        },
                    },
                ],
            }
        else:
            raise ValueError(f"Unknown prompt: {name}")


# ===================================================================
# Serialization helpers
# ===================================================================


def _chat_response_to_str(resp: Any) -> str:
    """Convert a ChatCompletionResponse to a plain-text string."""
    lines: list[str] = []
    for choice in resp.choices:
        if choice.message:
            lines.append(f"{choice.message.role}: {choice.message.content}")
        if choice.finish_reason:
            lines.append(f"[finish_reason: {choice.finish_reason}]")
    return "\n".join(lines) if lines else json.dumps(resp.model_dump(), default=str)


def _completion_response_to_str(resp: Any) -> str:
    """Convert a CompletionResponse to a plain-text string."""
    lines: list[str] = []
    for choice in resp.choices:
        lines.append(choice.text)
        if choice.finish_reason:
            lines.append(f"[finish_reason: {choice.finish_reason}]")
    return "\n".join(lines) if lines else json.dumps(resp.model_dump(), default=str)


def _embedding_response_to_str(resp: Any) -> str:
    """Convert an EmbeddingResponse to a JSON string."""
    obj = {
        "model": resp.model,
        "data": [
            {"index": d.index, "embedding": d.embedding} for d in resp.data
        ],
    }
    return json.dumps(obj, default=str)


def _model_list_to_str(resp: Any) -> str:
    """Convert a ModelList to a JSON string."""
    obj = {
        "models": [
            {"id": m.id, "owned_by": m.owned_by} for m in resp.data
        ],
    }
    return json.dumps(obj, indent=2, default=str)


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    """Run the MCP server from the command line.

    Usage::

        python -m distllm_sdk.mcp_server
    """
    logging.basicConfig(
        level=os.environ.get("DISTLLM_MCP_LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s [%(name)s] %(message)s",
    )
    server = DistLLMMCPServer()
    server.run()


if __name__ == "__main__":
    main()
