# SDK Reference

Complete API reference for the DistLLM SDK in Python, JavaScript/TypeScript, Go, and Rust.

---

## Python SDK

### Installation

```bash
pip install distllm-sdk
```

### Quick Start

```python
from distllm_sdk import DistLLMClient

# Async client
async with DistLLMClient(base_url="http://localhost:8000") as client:
    response = await client.chat_completions(
        messages=[{"role": "user", "content": "Hello!"}],
        model="distributed-llm",
    )
    print(response.choices[0].message.content)

# Sync client
from distllm_sdk import DistLLMClientSync

with DistLLMClientSync(base_url="http://localhost:8000") as client:
    response = client.chat_completions(
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

### Streaming

```python
async for chunk in client.chat_completions_stream(
    messages=[{"role": "user", "content": "Write a poem"}],
    max_tokens=256,
):
    print(chunk, end="", flush=True)
```

### Configuration

```python
from distllm_sdk import DistLLMClient, RetryConfig, PoolConfig

# With API key (production)
client = DistLLMClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    timeout=120.0,
    retry=RetryConfig(max_retries=3, initial_delay=1.0),
    pool=PoolConfig(max_connections=100),
)

# Without API key (development with --no-auth)
client = DistLLMClient(
    base_url="http://localhost:8000",
    api_key="not-needed",
)
```

### Error Handling

```python
from distllm_sdk.errors import AuthenticationError, RateLimitError, TimeoutError

try:
    response = await client.chat_completions(messages=[...])
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except TimeoutError:
    print("Request timed out")
```

### Usage Stats

```python
print(client.stats.total_calls)
print(client.stats.tokens_per_second)
print(client.stats.avg_latency)
print(client.stats.estimate_cost())
```

---

## JavaScript/TypeScript SDK

### Installation

```bash
npm install distllm-sdk
```

### Quick Start

```typescript
import { DistLLMClient } from 'distllm-sdk';

const client = new DistLLMClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'sk-your-key',
});

const response = await client.chat.completions.create({
  model: 'distributed-llm',
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(response.choices[0].message.content);
```

### Streaming

```typescript
const stream = client.chat.stream({
  model: 'distributed-llm',
  messages: [{ role: 'user', content: 'Write a poem' }],
  max_tokens: 256,
});

for await (const chunk of stream) {
  process.stdout.write(chunk);
}
```

---

## Go SDK

### Installation

```bash
go get github.com/distributed-llm/distributed-llm/sdk/go
```

### Quick Start

```go
package main

import (
    "context"
    "fmt"
    distllm "github.com/distributed-llm/distributed-llm/sdk/go"
)

func main() {
    client := distllm.NewClient("http://localhost:8000", "sk-your-key")
    
    resp, err := client.ChatCompletion(context.Background(), &distllm.ChatRequest{
        Model: "distributed-llm",
        Messages: []distllm.Message{
            distllm.Message{Role: "user", Content: "Hello!"},
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Choices[0].Message.Content)
}
```

### Streaming

```go
ch, errCh := client.ChatCompletionStream(ctx, &distllm.ChatRequest{
    Model: "distributed-llm",
    Messages: []distllm.Message{{Role: "user", Content: "Write a poem"}},
    MaxTokens: 256,
})

for chunk := range ch {
    fmt.Print(chunk)
}
if err := <-errCh; err != nil {
    fmt.Println("Error:", err)
}
```

---

## Rust SDK

### Installation

```toml
[dependencies]
distllm-sdk = "0.4"
tokio = { version = "1", features = ["full"] }
```

### Quick Start

```rust
use distllm_sdk::{Client, ChatRequest, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://localhost:8000", "sk-your-key");
    
    let response = client.chat_completion(&ChatRequest {
        model: "distributed-llm".to_string(),
        messages: vec![Message::user("Hello!")],
        ..Default::default()
    }).await?;
    
    println!("{}", response.choices[0].message.as_ref().unwrap().content);
    Ok(())
}
```

---

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"distributed-llm"` | Model name |
| `temperature` | float | `0.7` | Sampling temperature (0-2) |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `max_tokens` | int | `256` | Maximum tokens to generate |
| `stream` | bool | `false` | Enable streaming |
| `adapter` | string | `null` | LoRA adapter name |
| `response_format` | object | `null` | Output format (`json_object`, `json_schema`) |

## Error Codes

| Code | HTTP | Description | Docs |
|------|------|-------------|------|
| `AUTH_ERROR` | 401 | Invalid API key | [Troubleshooting](https://distllm.dev/docs/troubleshooting#4-api-errors) |
| `RATE_LIMIT` | 429 | Rate limit exceeded | [Troubleshooting](https://distllm.dev/docs/troubleshooting#4-api-errors) |
| `MODEL_NOT_FOUND` | 404 | Model not loaded | [Troubleshooting](https://distllm.dev/docs/troubleshooting#2-model-loading-failures) |
| `TIMEOUT` | 504 | Request timed out | [Troubleshooting](https://distllm.dev/docs/troubleshooting#5-performance-issues) |
| `NODE_UNREACHABLE` | 503 | Worker node down | [Troubleshooting](https://distllm.dev/docs/troubleshooting#3-distributed-pipeline-errors) |
| `OOM_ERROR` | 500 | GPU out of memory | [Troubleshooting](https://distllm.dev/docs/troubleshooting#6-gpu-issues) |
