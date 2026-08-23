# one-api Provider for DistLLM

[one-api](https://github.com/songquanpeng/one-api) is an open-source,
multi-provider API gateway for LLMs. This module registers DistLLM as
a supported provider in one-api, enabling users to route requests through
one-api's management layer (rate limiting, key management, logging).

Since DistLLM exposes an OpenAI-compatible API, this provider simply
configures the base URL and model prefix mapping.

## Usage

### 1. Build the provider module

```bash
pip install -e integrations/one-api
```

### 2. Register in one-api's config

```python
from distllm_one_api import DistLLMProviderConfig

config = DistLLMProviderConfig(
    name="distllm",
    base_url="http://localhost:8000/v1",
    api_key="your-distllm-api-key",
    models=["distributed-llm", "llama-3.1-8b"],
)
# Apply config to one-api
config.apply()
```

### 3. Configure via Environment Variables

```bash
export DISTLLM_ONEAPI_BASE_URL=http://localhost:8000/v1
export DISTLLM_ONEAPI_API_KEY=your-key
export DISTLLM_ONEAPI_MODELS=distributed-llm,llama-3.1-8b
```

### 4. Use with one-api

Once registered, any one-api client can use DistLLM:

```bash
curl http://one-api:8080/v1/chat/completions \
  -H "Authorization: Bearer one-api-token" \
  -d '{"model": "distllm-distributed-llm", "messages": [{"role": "user", "content": "hello"}]}'
```
