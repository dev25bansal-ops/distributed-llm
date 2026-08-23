"""Tool-calling routes: ``GET /v1/tools`` and ``POST /v1/tools/{name}``.

The framework tool adapters (LangChain, LlamaIndex, CrewAI, Agno) discover tools
via ``GET /v1/tools`` and invoke them via ``POST /v1/tools/{name}``.  These
endpoints previously did not exist, so every adapter silently fell back to
defaults and every tool call returned an HTTP 404.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..api_state import g

router = APIRouter(tags=["tools"])

DEFAULT_TOOLS: list[dict[str, str]] = [
    {
        "name": "distllm_chat",
        "description": "Generate a chat completion. Input: messages (list of {role, content})",
    },
    {
        "name": "distllm_complete",
        "description": "Generate a text completion. Input: prompt (string)",
    },
    {
        "name": "distllm_embed",
        "description": "Generate embeddings. Input: input (string or list of strings)",
    },
]


@router.get("/v1/tools")
async def list_tools() -> dict[str, Any]:
    """Return the discoverable tool definitions."""
    return {"data": DEFAULT_TOOLS}


@router.post("/v1/tools/{name}")
async def call_tool(name: str, request: Request) -> dict[str, Any]:
    """Dispatch a tool call to the coordinator."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    body = await request.json()
    parameters: dict[str, Any] = (body or {}).get("parameters", {}) or {}

    if name in ("distllm_complete", "distllm_chat"):
        if name == "distllm_chat":
            messages = parameters.get("messages", [])
            prompt = ""
            if isinstance(messages, str):
                prompt = messages
            else:
                for msg in reversed(messages or []):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break
        else:
            prompt = parameters.get("prompt", "")
        if not prompt:
            return {"result": None, "error": "missing prompt/messages"}
        text = await asyncio.to_thread(
            coord.generate,
            prompt=prompt,
            max_new_tokens=int(parameters.get("max_tokens", 128)),
            temperature=float(parameters.get("temperature", 0.7)),
        )
        return {"result": text}

    if name == "distllm_embed":
        inputs = parameters.get("input", parameters.get("inputs", ""))
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            return {"result": None, "error": "missing input"}
        embed_loader = getattr(coord, "_embedding_loader", None)
        if embed_loader is not None and getattr(embed_loader, "embedding_model", None) is not None:
            emb = embed_loader.encode(list(inputs), normalize=True)
            if hasattr(emb, "tolist"):
                return {"result": emb.tolist()}
            return {"result": emb}
        return {"result": None, "error": "embeddings not available (no embedding model loaded)"}

    return {"result": None, "error": f"unknown tool: {name}"}