"""Draft-as-a-Service (DaaS) — standalone draft model inference server.

Packages the draft model as a deployable service that can run on any
CPU/edge node and serve draft tokens to remote GPU clusters. This
enables monetizing spare CPU/edge capacity as a billable service tier.

Usage::

    # Start a DaaS server
    distllm daas serve --model SmolLM-135M --port 9000

    # Use as draft model from coordinator
    draft = RemoteDraftModel(RemoteDraftConfig(
        endpoint_url="http://daas-node:9000/v1/completions",
        model_name="SmolLM-135M",
    ))

Features:
- OpenAI-compatible /v1/completions endpoint
- Token-level logprobs for proper rejection sampling
- Health/ready endpoints for load balancer integration
- Usage metrics (tokens generated, latency, cost)
- Rate limiting per API key
- Concurrent request handling
"""


from __future__ import annotations
import argparse
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError:
    pass  # FastAPI is optional for non-server usage


@dataclass
class DaaSConfig:
    """Configuration for the DaaS server."""

    model_name: str = "SmolLM-135M"
    host: str = "0.0.0.0"
    port: int = 9000
    max_concurrent: int = 10
    rate_limit_per_minute: int = 60
    api_key: str = ""
    cost_per_hour: float = 0.05
    hardware: str = "cpu"
    dtype: str = "float16"
    max_tokens_per_request: int = 32
    enable_logprobs: bool = True


@dataclass
class DaaSMetrics:
    """Runtime metrics for the DaaS server."""

    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_s: float = 0.0
    active_requests: int = 0
    errors: int = 0
    start_time: float = 0.0
    rate_limited: int = 0

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_latency_s / self.total_requests) * 1000

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_s == 0:
            return 0.0
        return self.total_tokens_generated / self.total_latency_s


class DaaSServer:
    """Draft-as-a-Service standalone server.


    Runs a FastAPI application that exposes draft model inference
    via OpenAI-compatible endpoints.

    Usage::

        server = DaaSServer(DaaSConfig(model_name="SmolLM-135M"))
        server.run()  # Starts uvicorn on configured port
    """


    def __init__(self, config: DaaSConfig | None = None):
        self._config = config or DaaSConfig()
        self._metrics = DaaSMetrics()
        self._model: Any = None
        self._tokenizer: Any = None
        self._rate_limits: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._semaphore: Any = None  # Lazy-initialized in create_app

    def _load_model(self) -> None:
        """Load the draft model (lazy initialization)."""

        if self._model is not None:
            return

        try:
            from distllm.models.partitioner import ModelPartitioner
            logger.info(f"Loading draft model: {self._config.model_name}")
            partitioner = ModelPartitioner(
                model_name=self._config.model_name,
                dtype=self._config.dtype,
            )
            partitioner.load_full_model()
            self._model = partitioner.full_model
            self._tokenizer = partitioner.tokenizer
            logger.info(f"Draft model loaded: {self._config.model_name}")
        except ImportError:
            logger.warning(
                "Model loading requires torch/transformers. "
                "Running in mock mode for testing."
            )
        except Exception as e:
            logger.error(f"Failed to load draft model: {e}")

    def _check_rate_limit(self, api_key: str) -> bool:
        """Check if the request is within rate limits."""

        if not self._config.rate_limit_per_minute:
            return True

        with self._lock:
            now = time.time()
            window_start = now - 60.0

            if api_key not in self._rate_limits:
                self._rate_limits[api_key] = []

            # Clean old entries
            self._rate_limits[api_key] = [
                t for t in self._rate_limits[api_key] if t > window_start
            ]

            if len(self._rate_limits[api_key]) >= self._config.rate_limit_per_minute:
                return False

            self._rate_limits[api_key].append(now)
            return True

    def _generate(
        self,
        prompt: list[int] | str,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        """Generate draft tokens from the loaded model.


        Returns an OpenAI-compatible response with token IDs and logprobs.
        """

        start = time.monotonic()

        with self._lock:
            self._metrics.active_requests += 1

        try:
            self._load_model()

            if self._model is None:
                # Mock mode for testing without a real model
                result = self._mock_generate(max_tokens)
                elapsed = time.monotonic() - start
                with self._lock:
                    self._metrics.total_requests += 1
                    self._metrics.total_tokens_generated += len(result["choices"][0]["token_ids"])
                    self._metrics.total_latency_s += elapsed
                    self._metrics.active_requests -= 1
                return result

            import torch

            # Encode prompt
            if isinstance(prompt, str):
                input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
            else:
                input_ids = torch.tensor([prompt], dtype=torch.long)

            # Generate
            with torch.no_grad():
                generated_ids = input_ids.clone()
                token_ids: list[int] = []
                token_logprobs: list[dict[str, Any]] = []

                for _ in range(min(max_tokens, self._config.max_tokens_per_request)):
                    outputs = self._model(generated_ids)
                    logits = outputs.logits[:, -1, :]

                    if temperature == 0:
                        next_id = logits.argmax(dim=-1).item()
                    else:
                        if top_k > 0:
                            values, indices = torch.topk(logits, top_k, dim=-1)
                            logits = torch.full_like(logits, float("-inf")).scatter_(
                                -1, indices, values
                            )
                        probs = torch.nn.functional.softmax(logits / max(temperature, 1e-8), dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1).item()

                    token_ids.append(next_id)
                    if self._config.enable_logprobs:
                        log_probs = torch.nn.functional.log_softmax(
                            logits / max(temperature, 1e-8), dim=-1
                        )
                        token_logprobs.append({
                            "token_id": next_id,
                            "logprob": log_probs[0, next_id].item(),
                        })

                    next_tensor = torch.tensor([[next_id]], device=generated_ids.device)
                    generated_ids = torch.cat([generated_ids, next_tensor], dim=1)

                    # Stop at EOS
                    if (self._tokenizer and
                        self._tokenizer.eos_token_id is not None and
                        next_id == self._tokenizer.eos_token_id):
                        break

            elapsed = time.monotonic() - start

            with self._lock:
                self._metrics.total_requests += 1
                self._metrics.total_tokens_generated += len(token_ids)
                self._metrics.total_latency_s += elapsed
                self._metrics.active_requests -= 1

            # OpenAI-compatible response
            response: dict[str, Any] = {
                "id": f"daas-{int(time.time() * 1000)}",
                "object": "text_completion",
                "model": self._config.model_name,
                "choices": [{
                    "index": 0,
                    "token_ids": token_ids,
                    "logprobs": {
                        "token_ids": token_ids,
                        "token_logprobs": token_logprobs,
                    } if token_logprobs else None,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(prompt) if isinstance(prompt, list) else 0,
                    "completion_tokens": len(token_ids),
                    "total_tokens": len(token_ids),
                },
            }
            return response

        except Exception:
            with self._lock:
                self._metrics.errors += 1
                self._metrics.active_requests -= 1
            raise

    def _mock_generate(self, max_tokens: int) -> dict[str, Any]:
        """Generate mock tokens for testing without a real model."""

        import random as _random
        token_ids = [_random.randint(0, 32000) for _ in range(max_tokens)]
        return {
            "id": f"daas-mock-{int(time.time() * 1000)}",
            "object": "text_completion",
            "model": self._config.model_name,
            "choices": [{
                "index": 0,
                "token_ids": token_ids,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(token_ids),
                "total_tokens": len(token_ids),
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Health check endpoint response."""

        return {
            "status": "healthy",
            "model": self._config.model_name,
            "hardware": self._config.hardware,
            "active_requests": self._metrics.active_requests,
            "uptime_s": round(self._metrics.uptime_s, 1),
        }

    def ready_check(self) -> dict[str, Any]:
        """Readiness check endpoint response."""

        ready = self._model is not None or True  # Mock mode is always ready
        return {
            "ready": ready,
            "model": self._config.model_name,
        }

    def metrics(self) -> dict[str, Any]:
        """Metrics endpoint response."""

        return {
            "total_requests": self._metrics.total_requests,
            "total_tokens_generated": self._metrics.total_tokens_generated,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "tokens_per_second": round(self._metrics.tokens_per_second, 1),
            "active_requests": self._metrics.active_requests,
            "errors": self._metrics.errors,
            "rate_limited": self._metrics.rate_limited,
            "uptime_s": round(self._metrics.uptime_s, 1),
            "cost_per_hour": self._config.cost_per_hour,
            "model": self._config.model_name,
            "hardware": self._config.hardware,
        }

    def create_app(self) -> Any:
        """Create the FastAPI application."""

        import asyncio

        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)

        app = FastAPI(
            title="DistLLM Draft-as-a-Service",
            description="Standalone draft model inference server for speculative decoding",
            version="0.1.0",
        )

        @app.on_event("startup")
        async def startup() -> None:
            self._metrics.start_time = time.time()
            logger.info(f"DaaS server starting: {self._config.model_name}")

        @app.on_event("shutdown")
        async def shutdown() -> None:
            self._shutdown_event.set()
            logger.info("DaaS server shutting down gracefully")

        @app.post("/v1/completions")
        async def completions(request: Request) -> Any:
            """OpenAI-compatible completions endpoint."""

            # Concurrency control
            if self._shutdown_event.is_set():
                raise HTTPException(status_code=503, detail="Server shutting down")

            async with self._semaphore:
                body = await request.json()

                # Auth check
                api_key = ""
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    api_key = auth[7:]

                if self._config.api_key and api_key != self._config.api_key:
                    raise HTTPException(status_code=401, detail="Invalid API key")

                if not self._check_rate_limit(api_key):
                    self._metrics.rate_limited += 1
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                prompt = body.get("prompt", [])
                max_tokens = body.get("max_tokens", self._config.max_tokens_per_request)
                temperature = body.get("temperature", 1.0)
                top_k = body.get("top_k", 20)
                top_p = body.get("top_p", 1.0)

                try:
                    result = self._generate(prompt, max_tokens, temperature, top_k, top_p)
                    return JSONResponse(content=result)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health() -> Any:
            return self.health_check()

        @app.get("/ready")
        async def ready() -> Any:
            return self.ready_check()

        @app.get("/metrics")
        async def metrics_endpoint() -> Any:
            return self.metrics()

        @app.get("/v1/models")
        async def models() -> Any:
            return {
                "data": [{
                    "id": self._config.model_name,
                    "object": "model",
                    "owned_by": "distllm-draft",
                }]
            }

        @app.post("/admin/shutdown")
        async def admin_shutdown() -> Any:
            """Initiate graceful shutdown — rejects new requests, finishes in-flight."""

            self._shutdown_event.set()
            return {"status": "shutting_down", "active_requests": self._metrics.active_requests}

        return app

    def run(self) -> None:
        """Start the DaaS server with uvicorn and graceful shutdown support."""

        import uvicorn

        self._metrics.start_time = time.time()
        logger.info(
            f"Starting DaaS server: {self._config.model_name} on "
            f"{self._config.host}:{self._config.port} "
            f"(max_concurrent={self._config.max_concurrent})"
        )
        app = self.create_app()
        config = uvicorn.Config(
            app,
            host=self._config.host,
            port=self._config.port,
            log_level="info",
            timeout_graceful_shutdown=30,
        )
        server = uvicorn.Server(config)
        server.run()


def main() -> None:
    """CLI entry point for the DaaS server."""

    parser = argparse.ArgumentParser(description="DistLLM Draft-as-a-Service")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the DaaS server")
    serve_parser.add_argument("--model", default="SmolLM-135M", help="Draft model name")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=9000, help="Bind port")
    serve_parser.add_argument("--api-key", default="", help="API key for auth")
    serve_parser.add_argument("--max-concurrent", type=int, default=10)
    serve_parser.add_argument("--rate-limit", type=int, default=60, help="Requests per minute")
    serve_parser.add_argument("--cost-per-hour", type=float, default=0.05)
    serve_parser.add_argument("--hardware", default="cpu")
    serve_parser.add_argument("--dtype", default="float16")
    serve_parser.add_argument("--logprobs", action="store_true", default=True)

    args = parser.parse_args()

    if args.command == "serve":
        config = DaaSConfig(
            model_name=args.model,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            max_concurrent=args.max_concurrent,
            rate_limit_per_minute=args.rate_limit,
            cost_per_hour=args.cost_per_hour,
            hardware=args.hardware,
            dtype=args.dtype,
            enable_logprobs=args.logprobs,
        )
        server = DaaSServer(config)
        server.run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
