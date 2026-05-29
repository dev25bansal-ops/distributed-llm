# Dify + DistLLM Integration

Connect [Dify](https://dify.ai) to DistLLM as a custom model provider.

## Prerequisites

- Running DistLLM coordinator with API server (default: `http://localhost:8000`)
- Dify installed and running

## Configuration

### Step 1: Navigate to Model Provider Settings

In Dify, go to **Settings → Model Provider → Add Model**.

### Step 2: Register Custom Model Provider

| Field | Value |
|-------|-------|
| **Provider Type** | OpenAI API Compatible |
| **Model Name** | `distributed-llm` |
| **API Base URL** | `http://host.docker.internal:8000/v1` (Docker Dify) or `http://localhost:8000/v1` |
| **API Key** | Any non-empty string |
| **Model Type** | LLM |

### Step 3: Additional Models (Optional)

For text completion (if needed):
- Add as a separate "OpenAI API Compatible" model with `/v1/completions` endpoint

For embeddings:
- Add as "OpenAI API Compatible" embedding model at `http://localhost:8000/v1`

## Docker Compose Example

```yaml
services:
  distllm:
    image: distributed-llm:latest
    ports:
      - "8000:8000"
    networks:
      - dify-net

  dify:
    image: langgenius/dify:latest
    ports:
      - "3000:3000"
    environment:
      - CUSTOM_MODEL_PROVIDER=openai_api_compatible
      - CUSTOM_MODEL_NAME=distributed-llm
      - CUSTOM_MODEL_ENDPOINT=http://distllm:8000/v1
    networks:
      - dify-net

networks:
  dify-net:
```

## Troubleshooting

- **401 Unauthorized**: DistLLM does not require auth; use any non-empty API key
- **Model not loading**: Check DistLLM health at `http://localhost:8000/health`
- **Dify in Docker**: Use `http://host.docker.internal:8000/v1` for host DistLLM
- **Embeddings not working**: Verify DistLLM embedding endpoint at `/v1/embeddings`
