/**
 * Live-call smoke tests for the DistLLM JS SDK.
 *
 * Each suite spins an in-process mock OpenAI-compatible HTTP server
 * (node:http, ephemeral port) serving canned /v1/models,
 * /v1/chat/completions and /health responses, then asserts that
 * DistLLMClient parses them correctly.
 *
 * Run: npm test   (= vitest run)
 */
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterAll, afterEach, describe, expect, it } from 'vitest';
import { DistLLMClient } from '../src/index';

const MODELS_BODY = {
  object: 'list',
  data: [
    { id: 'distributed-llm', object: 'model', created: 1700000000, owned_by: 'distllm' },
    { id: 'tiny-stories-1m', object: 'model', created: 1700000001, owned_by: 'distllm' },
  ],
};

const CHAT_BODY = {
  id: 'chatcmpl-smoke-001',
  object: 'chat.completion',
  created: 1700000100,
  model: 'distributed-llm',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'Hello from the mock cluster!' },
      finish_reason: 'stop',
    },
  ],
  usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
};

const COMPLETIONS_BODY = {
  id: 'cmpl-smoke-001',
  object: 'text_completion',
  created: 1700000200,
  model: 'distributed-llm',
  choices: [{ index: 0, text: ' Once upon a time.', finish_reason: 'stop' }],
  usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
};

const EMBEDDINGS_BODY = {
  object: 'list',
  model: 'distributed-llm',
  data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
  usage: { prompt_tokens: 2, completion_tokens: 0, total_tokens: 2 },
};

const HEALTH_BODY = { status: 'ok', model: 'distributed-llm', nodes: 2, uptime: 1234.5 };

interface RecordedRequest {
  method?: string;
  path?: string;
  auth?: string;
  body?: unknown;
}

/** Start an ephemeral-port mock OpenAI-compatible server; record requests. */
async function startMockServer(): Promise<{ server: Server; url: string; requests: RecordedRequest[] }> {
  const requests: RecordedRequest[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      requests.push({
        method: req.method,
        path: req.url,
        auth: req.headers.authorization,
        body: chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : undefined,
      });

      const sendJson = (payload: unknown) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(payload));
      };

      if (req.method === 'GET' && req.url === '/v1/models') return sendJson(MODELS_BODY);
      if (req.method === 'POST' && req.url === '/v1/chat/completions') {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        if (body?.stream) {
          res.writeHead(200, { 'Content-Type': 'text/event-stream' });
          res.write('data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n');
          res.write('data: {"choices":[{"index":0,"delta":{"content":"lo!"}}]}\n\n');
          res.write('data: [DONE]\n\n');
          res.end();
          return;
        }
        return sendJson(CHAT_BODY);
      }
      if (req.method === 'POST' && req.url === '/v1/completions') return sendJson(COMPLETIONS_BODY);
      if (req.method === 'POST' && req.url === '/v1/embeddings') return sendJson(EMBEDDINGS_BODY);
      if (req.method === 'GET' && req.url === '/health') return sendJson(HEALTH_BODY);

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: `no route ${req.url}`, type: 'invalid_request_error' } }));
    });
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return { server, url: `http://127.0.0.1:${port}`, requests };
}

describe('DistLLMClient live-call smoke (mock server)', () => {
  let server: Server;
  let baseUrl: string;
  let requests: RecordedRequest[];
  let client: DistLLMClient;

  afterEach(async () => {
    await new Promise<void>(resolve => server.close(() => resolve()));
  });
  afterAll(async () => {
    // no-op placeholder to keep vitest happy about open handles
  });

  it('lists models and parses the payload', async () => {
    ({ server, url: baseUrl, requests } = await startMockServer());
    client = new DistLLMClient({ baseUrl, apiKey: 'sk-test-key', maxRetries: 0 });

    const models = await client.models.list();
    expect(models.data.map(m => m.id)).toEqual(['distributed-llm', 'tiny-stories-1m']);
    expect(models.data[0].owned_by).toBe('distllm');
    expect(requests[0].method).toBe('GET');
    expect(requests[0].path).toBe('/v1/models');
    expect(requests[0].auth).toBe('Bearer sk-test-key');
  });

  it('creates a chat completion and parses choices + usage', async () => {
    ({ server, url: baseUrl, requests } = await startMockServer());
    client = new DistLLMClient({ baseUrl, apiKey: 'sk-test-key', maxRetries: 0 });

    const resp = await client.chat.completions.create({
      model: 'distributed-llm',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    expect(resp.id).toBe('chatcmpl-smoke-001');
    expect(resp.model).toBe('distributed-llm');
    expect(resp.choices[0]?.message?.role).toBe('assistant');
    expect(resp.choices[0]?.message?.content).toBe('Hello from the mock cluster!');
    expect(resp.choices[0]?.finish_reason).toBe('stop');
    expect(resp.usage?.total_tokens).toBe(13);

    const sent = requests[0];
    expect(sent.method).toBe('POST');
    expect(sent.path).toBe('/v1/chat/completions');
    expect((sent.body as Record<string, unknown>).messages).toEqual([
      { role: 'user', content: 'Hi' },
    ]);
  });

  it('streams chat deltas over SSE until [DONE]', async () => {
    ({ server, url: baseUrl } = await startMockServer());
    client = new DistLLMClient({ baseUrl, maxRetries: 0 });

    const deltas: string[] = [];
    for await (const delta of client.chat.completions.stream({
      model: 'distributed-llm',
      messages: [{ role: 'user', content: 'Hi' }],
    })) {
      deltas.push(delta);
    }
    expect(deltas).toEqual(['Hel', 'lo!']);
  });

  it('creates text completions', async () => {
    ({ server, url: baseUrl } = await startMockServer());
    client = new DistLLMClient({ baseUrl, maxRetries: 0 });

    const resp = await client.completions.create({ model: 'distributed-llm', prompt: 'Tell me a story' });
    expect(resp.id).toBe('cmpl-smoke-001');
    expect(resp.choices[0]?.text).toBe(' Once upon a time.');
    expect(resp.choices[0]?.finish_reason).toBe('stop');
  });

  it('creates embeddings', async () => {
    ({ server, url: baseUrl } = await startMockServer());
    client = new DistLLMClient({ baseUrl, maxRetries: 0 });

    const resp = await client.embeddings.create({ input: 'hello' });
    expect(resp.model).toBe('distributed-llm');
    expect(resp.data[0]?.embedding).toEqual([0.1, 0.2, 0.3]);
  });

  it('maps HTTP errors to DistLLMApiError and does not retry 4xx', async () => {
    ({ server, url: baseUrl, requests } = await startMockServer());
    // A base URL with an unknown prefix makes every request hit the 404 branch.
    const badClient = new DistLLMClient({ baseUrl: `${baseUrl}/nope`, apiKey: 'sk-test-key', maxRetries: 3 });

    await expect(badClient.models.list()).rejects.toMatchObject({
      name: 'DistLLMApiError',
      statusCode: 404,
    });
    expect(requests).toHaveLength(1); // 4xx must not be retried
  });
});
