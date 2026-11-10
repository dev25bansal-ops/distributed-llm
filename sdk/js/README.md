# distllm-sdk (JavaScript / TypeScript)

Client for the DistLLM OpenAI-compatible inference API. Works in Node.js >= 18
and browsers (uses the global `fetch`).

## Install

Not published to npm yet — `npm view distllm-sdk` returns 404. Install this
directory directly, or `npm pack` it and install the tarball:

```bash
npm install /path/to/distributed-llm/sdk/js
```

## Quick start

```ts
import { DistLLMClient } from 'distllm-sdk';

const client = new DistLLMClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'sk-...',          // optional
  timeout: 120_000,          // ms
  maxRetries: 3,
});

const response = await client.chat.completions.create({
  model: 'distributed-llm',
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(response.choices[0].message.content);
```

## Streaming

```ts
for await (const delta of client.chat.completions.stream({
  model: 'distributed-llm',
  messages: [{ role: 'user', content: 'Count to five.' }],
})) {
  process.stdout.write(delta);   // each item is a string
}
```

## API surface

This client implements a **common core**. Anything not listed here is not
implemented — see `sdk/MATRIX.md` for the full cross-language matrix.

| Method | Endpoint | Returns |
|---|---|---|
| `client.chat.completions.create(req)` | `POST /v1/chat/completions` | `ChatCompletionResponse` |
| `client.chat.completions.stream(req)` | `POST /v1/chat/completions` (SSE) | `AsyncGenerator<string>` |
| `client.completions.create(req)` | `POST /v1/completions` | `CompletionResponse` |
| `client.embeddings.create(req)` | `POST /v1/embeddings` | `EmbeddingResponse` |
| `client.models.list()` | `GET /v1/models` | `ModelList` |

Not implemented: batch, files, fine-tuning, moderations, audio, images,
`health_check`, and `completions_stream`. Use the Python SDK for those today.

## Timeout semantics

`timeout` (milliseconds) behaves differently for streaming and non-streaming
requests:

- **Non-streaming:** bounds the whole exchange — connect, response headers, and
  body read.
- **Streaming (SSE):** bounds connect + response headers, then acts as an
  **idle** timeout that re-arms between chunks. A stream that keeps producing
  chunks may legally run longer than `timeout`; a stream that goes silent for
  longer is aborted. The abort surfaces as a `DistLLMTimeoutError`.

## Errors

- `DistLLMApiError` — non-2xx response; carries `statusCode` and `errorType`.
  4xx responses are **not** retried.
- `DistLLMTimeoutError` — the configured timeout elapsed (see above).

5xx and network errors are retried with exponential backoff up to `maxRetries`.

## License

Apache-2.0