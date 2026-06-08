# distllm-litellm

Use [DistLLM](https://github.com/distributed-llm/distributed-llm) as a backend in [LiteLLM](https://github.com/BerriAI/litellm).

## Installation

```bash
pip install distllm-litellm
```

## Quick Start

```python
import litellm
from distllm_litellm import get_distllm_custom_llm

# Register the provider (call once at startup)
get_distllm_custom_llm()

# Use DistLLM via LiteLLM
response = litellm.completion(
    model="distllm/distributed-llm",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="http://localhost:8000/v1",
)
print(response.choices[0].message.content)
```

## Async

```python
import asyncio
import litellm

async def main():
    response = await litellm.acompletion(
        model="distllm/distributed-llm",
        messages=[{"role": "user", "content": "Hello!"}],
        api_base="http://localhost:8000/v1",
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Embeddings

```python
response = litellm.embedding(
    model="distllm/bge-large",
    input=["Hello world", "Goodbye"],
    api_base="http://localhost:8000/v1",
)
```

## LiteLLM Proxy

Add to your LiteLLM proxy config:

```yaml
model_list:
  - model_name: distllm-chat
    litellm_params:
      model: distllm/distributed-llm
      api_base: http://distllm:8000/v1
  - model_name: distllm-embed
    litellm_params:
      model: distllm/bge-large
      api_base: http://distllm:8000/v1
```
