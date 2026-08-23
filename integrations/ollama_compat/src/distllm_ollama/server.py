"""Ollama-compatible API proxy for DistLLM.

Exposes the Ollama API surface so any Ollama client (including the
``ollama`` Python package, Open WebUI's Ollama mode, and CLI tools)
can talk to a DistLLM cluster without modification.

Usage::

    # As a module
    from distllm_ollama import create_app
    app = create_app(distllm_base="http://localhost:8000")

    # As a CLI
    distllm-ollama-proxy --distllm-url http://localhost:8000 --port 11434
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("distllm_ollama")

_DEFAULT_DISTLLM_URL = "http://localhost:8000"


def create_app(
    distllm_base: str = _DEFAULT_DISTLLM_URL,
    api_key: str | None = None,
) -> FastAPI:
    """Create a FastAPI app that translates Ollama API → DistLLM API.

    Args:
        distllm_base: Base URL of the live DistLLM server.
        api_key: Bearer key for the DistLLM server.  Read from
            ``DISTLLM_API_KEY`` when omitted.  Without it every upstream
            call is rejected 401 (auth is always required server-side),
            which previously surfaced as silent EMPTY responses.
    """
    import os as _os

    app = FastAPI(title="DistLLM Ollama Proxy", version="0.1.0")
    key = api_key or _os.environ.get("DISTLLM_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = httpx.AsyncClient(base_url=distllm_base, timeout=120.0, headers=headers)

    # ------------------------------------------------------------------
    # GET /api/tags — list models
    # ------------------------------------------------------------------

    @app.get("/api/tags")
    async def list_models():
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return {
            "models": [
                {
                    "name": m["id"],
                    "model": m["id"],
                    "modified_at": "",
                    "size": 0,
                    "digest": "",
                    "details": {
                        "format": "gguf",
                        "family": "distllm",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                }
                for m in data
            ]
        }

    # ------------------------------------------------------------------
    # POST /api/generate — text completion
    # ------------------------------------------------------------------

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        model = body.get("model", "distributed-llm")
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": body.get("options", {}).get("temperature", 0.7),
            "top_p": body.get("options", {}).get("top_p", 0.9),
            "max_tokens": body.get("options", {}).get("num_predict", 256),
            "stream": stream,
        }

        if stream:
            return StreamingResponse(
                _stream_generate(client, payload, model),
                media_type="application/x-ndjson",
            )

        resp = await client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("text", "")
        return {
            "model": model,
            "created_at": "",
            "response": text,
            "done": True,
            "context": [],
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens", 0),
            "eval_count": data.get("usage", {}).get("completion_tokens", 0),
        }

    # ------------------------------------------------------------------
    # POST /api/chat — chat completion
    # ------------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        model = body.get("model", "distributed-llm")
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": body.get("options", {}).get("temperature", 0.7),
            "top_p": body.get("options", {}).get("top_p", 0.9),
            "max_tokens": body.get("options", {}).get("num_predict", 256),
            "stream": stream,
        }

        if stream:
            return StreamingResponse(
                _stream_chat(client, payload, model),
                media_type="application/x-ndjson",
            )

        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "model": model,
            "created_at": "",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens", 0),
            "eval_count": data.get("usage", {}).get("completion_tokens", 0),
        }

    # ------------------------------------------------------------------
    # POST /api/embeddings — embeddings
    # ------------------------------------------------------------------

    @app.post("/api/embeddings")
    async def embeddings(request: Request):
        body = await request.json()
        model = body.get("model", "distributed-llm")
        prompt = body.get("prompt", "")

        resp = await client.post(
            "/v1/embeddings",
            json={"model": model, "input": [prompt]},
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("data", [{}])[0].get("embedding", [])
        return {"embedding": embedding}

    # ------------------------------------------------------------------
    # GET /api/version
    # ------------------------------------------------------------------

    @app.get("/api/version")
    async def version():
        try:
            resp = await client.get("/health")
            return {"version": resp.json().get("version", "0.0.0")}
        except Exception:
            return {"version": "unknown"}

    return app


# ------------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------------


async def _stream_generate(
    client: httpx.AsyncClient, payload: dict, model: str
) -> AsyncIterator[str]:
    payload["stream"] = True
    async with client.stream("POST", "/v1/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                yield json.dumps({"model": model, "done": True}) + "\n"
                break
            try:
                data = json.loads(data_str)
                text = data.get("choices", [{}])[0].get("text", "")
                yield json.dumps(
                    {"model": model, "response": text, "done": False}
                ) + "\n"
            except (json.JSONDecodeError, KeyError):
                continue


async def _stream_chat(
    client: httpx.AsyncClient, payload: dict, model: str
) -> AsyncIterator[str]:
    payload["stream"] = True
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                yield json.dumps({"model": model, "done": True}) + "\n"
                break
            try:
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield json.dumps(
                        {
                            "model": model,
                            "message": {"role": "assistant", "content": content},
                            "done": False,
                        }
                    ) + "\n"
            except (json.JSONDecodeError, KeyError):
                continue


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DistLLM Ollama Compatibility Proxy")
    parser.add_argument(
        "--distllm-url",
        default=_DEFAULT_DISTLLM_URL,
        help="DistLLM coordinator URL (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port to listen on (default: 11434, same as Ollama)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    import uvicorn

    app = create_app(distllm_base=args.distllm_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
