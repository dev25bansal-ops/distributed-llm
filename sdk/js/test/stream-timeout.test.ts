/**
 * Streaming/non-streaming timeout semantics for DistLLMClient.
 *
 * Contract under test (WAVE2-PLAN item 37):
 *   - Streaming (SSE) requests: the client timeout bounds CONNECTION
 *     ESTABLISHMENT + RESPONSE HEADERS, then an IDLE TIMER re-arms between
 *     chunks. A stream that keeps trickling bytes may legally run longer
 *     than the timeout; a stream that goes silent for `timeout` ms is dead.
 *   - Non-streaming requests: the timeout bounds the WHOLE exchange
 *     (connect + headers + body), not just the pre-header phase.
 *
 * All servers are ephemeral in-process node:http instances; all timeouts
 * are short (100-250ms) via the injectable `timeout` client option.
 */
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';
import { DistLLMClient, DistLLMTimeoutError } from '../src/index';

const SSE_HEADERS = { 'Content-Type': 'text/event-stream' };

const sseChunk = (content: string) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content } }] })}\n\n`;

let server: Server | undefined;

async function startServer(handler: (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => void): Promise<string> {
  server = createServer(handler);
  await new Promise<void>(resolve => server!.listen(0, '127.0.0.1', resolve));
  const { port } = server!.address() as AddressInfo;
  return `http://127.0.0.1:${port}`;
}

/** Tear down aggressively: kill keep-alive + in-flight sockets so vitest exits cleanly. */
afterEach(async () => {
  if (!server) return;
  const s = server as unknown as {
    closeAllConnections?: () => void;
    closeIdleConnections?: () => void;
  };
  s.closeAllConnections?.();
  s.closeIdleConnections?.();
  await new Promise<void>(resolve => server!.close(() => resolve()));
  server = undefined;
});

/** Collect stream deltas, capturing whatever error ends the iteration. */
async function collectStream(
  client: DistLLMClient,
): Promise<{ deltas: string[]; error: unknown }> {
  const deltas: string[] = [];
  try {
    for await (const delta of client.chat.completions.stream({
      model: 'distributed-llm',
      messages: [{ role: 'user', content: 'Hi' }],
    })) {
      deltas.push(delta);
    }
    return { deltas, error: undefined };
  } catch (error) {
    return { deltas, error };
  }
}

describe('streaming timeout semantics (W2-37)', () => {
  it('completes a slow-trickle stream whose TOTAL duration exceeds the timeout', async () => {
    const baseUrl = await startServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        res.writeHead(200, SSE_HEADERS);
        res.flushHeaders();
        const pieces = ['a', 'b', 'c', 'd'];
        let i = 0;
        const tick = () => {
          if (i < pieces.length) {
            res.write(sseChunk(pieces[i++]));
            setTimeout(tick, 90); // 4 x 90ms = ~360ms total >> timeout below
          } else {
            res.write('data: [DONE]\n\n');
            res.end();
          }
        };
        tick();
      });
    });

    // Timeout far SHORTER than the stream's total runtime: the stream must
    // survive because it never goes idle for longer than the timeout.
    const client = new DistLLMClient({ baseUrl, timeout: 150, maxRetries: 0 });

    const started = Date.now();
    const { deltas, error } = await collectStream(client);
    const elapsed = Date.now() - started;

    expect(error).toBeUndefined();
    expect(deltas).toEqual(['a', 'b', 'c', 'd']);
    expect(elapsed).toBeGreaterThanOrEqual(280); // really did outlive the 150ms timeout
  });

  it('fails a streaming request at connect-timeout when the server never responds', async () => {
    // Accepts the TCP connection, parses the request, then goes silent
    // before sending ANY headers: fetch() stays pending until we abort.
    const baseUrl = await startServer(() => {
      /* deliberately never respond */
    });

    const client = new DistLLMClient({ baseUrl, timeout: 200, maxRetries: 0 });

    const started = Date.now();
    const { error } = await collectStream(client);
    const elapsed = Date.now() - started;

    expect(error).toBeInstanceOf(DistLLMTimeoutError);
    expect((error as Error).message).toMatch(/timed out/i);
    expect(elapsed).toBeLessThan(2000); // failed fast, not hung
  });

  it('fails a stream that sends headers then goes silent (idle timeout)', async () => {
    const baseUrl = await startServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        res.writeHead(200, SSE_HEADERS);
        res.flushHeaders();
        // headers flushed, then NEVER write a chunk and NEVER end
      });
    });

    const client = new DistLLMClient({ baseUrl, timeout: 200, maxRetries: 0 });

    const { error } = await collectStream(client);

    expect(error).toBeInstanceOf(DistLLMTimeoutError);
    expect((error as Error).message).toMatch(/idle/i);
  });

  it('fails a stream that stalls MID-GENERATION after delivering a first chunk', async () => {
    const baseUrl = await startServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        res.writeHead(200, SSE_HEADERS);
        res.flushHeaders();
        res.write(sseChunk('Hel'));
        // first delta delivered, then eternal silence
      });
    });

    const client = new DistLLMClient({ baseUrl, timeout: 200, maxRetries: 0 });

    const { deltas, error } = await collectStream(client);

    expect(deltas).toEqual(['Hel']); // first chunk made it to the consumer
    expect(error).toBeInstanceOf(DistLLMTimeoutError);
    expect((error as Error).message).toMatch(/idle/i);
  });

  it('bounds a NON-streaming request end-to-end, including a slow body', async () => {
    const baseUrl = await startServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        // Headers immediately, but body withheld past the client timeout.
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.flushHeaders();
        setTimeout(() => {
          res.end(
            JSON.stringify({
              id: 'chatcmpl-slow-body',
              model: 'distributed-llm',
              created: 1700000100,
              choices: [
                { index: 0, message: { role: 'assistant', content: 'late' }, finish_reason: 'stop' },
              ],
            }),
          );
        }, 600);
      });
    });

    const client = new DistLLMClient({ baseUrl, timeout: 150, maxRetries: 0 });

    const started = Date.now();
    await expect(client.chat.completions.create({
      model: 'distributed-llm',
      messages: [{ role: 'user', content: 'Hi' }],
    })).rejects.toBeInstanceOf(DistLLMTimeoutError);
    const elapsed = Date.now() - started;

    expect(elapsed).toBeLessThan(550); // aborted near the timeout, not at the server's 600ms leisure
  });
});
