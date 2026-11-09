/**
 * DistLLM JavaScript/TypeScript SDK
 *
 * OpenAI-compatible client for distributed LLM inference.
 *
 * Usage:
 *   import { DistLLMClient } from 'distllm-sdk';
 *
 *   const client = new DistLLMClient({ baseUrl: 'http://localhost:8000' });
 *   const response = await client.chat.completions.create({
 *     model: 'distributed-llm',
 *     messages: [{ role: 'user', content: 'Hello!' }],
 *   });
 *   console.log(response.choices[0].message.content);
 */
/**
 * Thrown when a request exceeds `timeout`, or a stream exceeds the idle
 * gap between chunks. Distinguishable from network failures via `name`.
 */
export class DistLLMTimeoutError extends Error {
    constructor(message) {
        super(message);
        this.name = 'DistLLMTimeoutError';
    }
}
/**
 * DistLLM client for Node.js and browsers.
 */
export class DistLLMClient {
    baseUrl;
    apiKey;
    timeout;
    maxRetries;
    defaultHeaders;
    chat;
    completions;
    embeddings;
    models;
    constructor(options = {}) {
        this.baseUrl = (options.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
        this.apiKey = options.apiKey || '';
        this.timeout = options.timeout || 120_000;
        this.maxRetries = options.maxRetries || 3;
        this.defaultHeaders = options.headers || {};
        this.chat = { completions: new ChatCompletions(this) };
        this.completions = new Completions(this);
        this.embeddings = new Embeddings(this);
        this.models = new Models(this);
    }
    /** @internal */
    async request(method, path, body) {
        const url = `${this.baseUrl}${path}`;
        const headers = {
            'Content-Type': 'application/json',
            ...this.defaultHeaders,
        };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        let lastError = null;
        for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
            // The timeout bounds the WHOLE exchange (connect + headers + body),
            // not just the pre-header phase.
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(new DistLLMTimeoutError(`Request to ${path} timed out after ${this.timeout}ms`)), this.timeout);
            try {
                const response = await fetch(url, {
                    method,
                    headers,
                    body: body ? JSON.stringify(body) : undefined,
                    signal: controller.signal,
                });
                if (!response.ok) {
                    const errorBody = await response.json().catch(() => ({}));
                    const error = errorBody;
                    throw new DistLLMApiError(error?.error?.message || `HTTP ${response.status}`, response.status, error?.error?.type || 'api_error');
                }
                return (await response.json());
            }
            catch (error) {
                lastError = error;
                if (error instanceof DistLLMApiError && error.statusCode < 500) {
                    throw error;
                }
                // Abortions we scheduled ourselves are timeouts, not transient
                // network faults: surface them instead of retrying.
                if (error instanceof DistLLMTimeoutError ||
                    controller.signal.aborted) {
                    throw error instanceof DistLLMTimeoutError
                        ? error
                        : new DistLLMTimeoutError(`Request to ${path} timed out after ${this.timeout}ms`);
                }
                if (attempt < this.maxRetries) {
                    await new Promise(r => setTimeout(r, Math.min(1000 * 2 ** attempt, 30_000)));
                }
            }
            finally {
                clearTimeout(timeoutId);
            }
        }
        throw lastError;
    }
    /**
     * Streaming SSE request.
     *
     * Timeout semantics differ from {@link DistLLMClient.request}: the
     * configured timeout bounds CONNECTION + RESPONSE HEADERS only. After
     * headers arrive, the deadline is replaced by an IDLE TIMER that re-arms
     * before every chunk read — long-lived generations that keep trickling
     * data are never killed mid-stream, but a silent/dead server cannot hang
     * the consumer forever.
     *
     * @internal
     */
    async *stream(method, path, body) {
        const url = `${this.baseUrl}${path}`;
        const headers = {
            'Content-Type': 'application/json',
            ...this.defaultHeaders,
        };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        const controller = new AbortController();
        // Phase 1: bound connection establishment + time-to-first-byte.
        let timerId = setTimeout(() => controller.abort(), this.timeout);
        try {
            const response = await fetch(url, {
                method,
                headers,
                body: JSON.stringify(body),
                signal: controller.signal,
            });
            if (!response.ok) {
                throw new DistLLMApiError(`HTTP ${response.status}`, response.status);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            // Phase 2: swap the whole-stream deadline for an idle timer that
            // re-arms around every read. A productive stream runs indefinitely;
            // silence longer than `timeout` kills it.
            clearTimeout(timerId);
            timerId = setTimeout(() => controller.abort(new DistLLMTimeoutError(`Stream idle for over ${this.timeout}ms`)), this.timeout);
            while (true) {
                let chunk;
                try {
                    chunk = await reader.read();
                }
                catch (error) {
                    if (controller.signal.aborted) {
                        throw new DistLLMTimeoutError(`Stream idle: no data received for ${this.timeout}ms`);
                    }
                    throw error;
                }
                const { done, value } = chunk;
                if (done)
                    break;
                // Got bytes — re-arm the idle window for the next gap.
                clearTimeout(timerId);
                timerId = setTimeout(() => controller.abort(new DistLLMTimeoutError(`Stream idle for over ${this.timeout}ms`)), this.timeout);
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (!line.startsWith('data: '))
                        continue;
                    const data = line.slice(6).trim();
                    if (data === '[DONE]')
                        return;
                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices?.[0]?.delta?.content;
                        if (content)
                            yield content;
                    }
                    catch { }
                }
            }
        }
        catch (error) {
            // The connect-phase abort carries no reason (older runtimes drop it),
            // so map any post-abort failure to the connect-timeout variant.
            if (!(error instanceof DistLLMApiError) &&
                !(error instanceof DistLLMTimeoutError) &&
                controller.signal.aborted) {
                throw new DistLLMTimeoutError(`Connection timed out after ${this.timeout}ms waiting for response headers from ${path}`);
            }
            throw error;
        }
        finally {
            clearTimeout(timerId);
        }
    }
}
class ChatCompletions {
    client;
    constructor(client) {
        this.client = client;
    }
    async create(request) {
        return this.client.request('POST', '/v1/chat/completions', request);
    }
    async *stream(request) {
        yield* this.client.stream('POST', '/v1/chat/completions', { ...request, stream: true });
    }
}
class Completions {
    client;
    constructor(client) {
        this.client = client;
    }
    async create(request) {
        return this.client.request('POST', '/v1/completions', request);
    }
}
class Embeddings {
    client;
    constructor(client) {
        this.client = client;
    }
    async create(request) {
        return this.client.request('POST', '/v1/embeddings', request);
    }
}
class Models {
    client;
    constructor(client) {
        this.client = client;
    }
    async list() {
        return this.client.request('GET', '/v1/models');
    }
}
export class DistLLMApiError extends Error {
    statusCode;
    errorType;
    constructor(message, statusCode, errorType = 'api_error') {
        super(message);
        this.statusCode = statusCode;
        this.errorType = errorType;
        this.name = 'DistLLMApiError';
    }
}
