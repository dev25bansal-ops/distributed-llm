# DistLLM SDK

Python client SDK for the Distributed LLM API.

## Installation

```bash
pip install distllm-sdk
```

## Quick Start

```python
from distllm_sdk import DistLLMClient

async with DistLLMClient(base_url="http://localhost:8000") as client:
    response = await client.chat_completions(
        messages=[{"role": "user", "content": "Hello!"}],
        model="distributed-llm",
    )
    print(response.choices[0].message.content)
```

## Features

- **Async & Sync clients** — `DistLLMClient` (async) and `DistLLMClientSync` (sync)
- **OpenAI-compatible API** — chat completions, streaming, embeddings, batch, audio, images, moderations, files, fine-tuning
- **Automatic retry** — configurable exponential backoff with jitter
- **Circuit breaker** — prevents cascading failures when the backend is unhealthy
- **Usage tracking** — per-call and aggregate token/latency/cost statistics
- **Typed responses** — all API responses are typed dataclasses
- **Connection pooling** — configurable httpx connection pool

## Documentation

See the [DistLLM documentation](https://github.com/distributed-llm/distributed-llm) for full API reference.

## License

Apache 2.0
