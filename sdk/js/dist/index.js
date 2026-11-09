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
            let timedOut = false;
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => {
                    timedOut = true;
                    controller.abort();
                }, this.timeout);
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
                    // The timeout stays armed through the body read. Clearing it as soon
                    // as headers arrived meant a server that sent headers and then
                    // stalled hung the caller forever, and a slow body was never bounded.
                    return (await response.json());
                }
                finally {
                    clearTimeout(timeoutId);
                }
            }
            catch (error) {
                // A timeout is not retryable and must be distinguishable from a
                // network error, so convert the abort before the retry logic runs.
                if (timedOut) {
                    throw new DistLLMTimeoutError(`Request timed out after ${this.timeout}ms`, this.timeout);
                }
                lastError = error;
                if (error instanceof DistLLMApiError && error.statusCode < 500) {
                    throw error;
                }
                if (attempt < this.maxRetries) {
                    await new Promise(r => setTimeout(r, Math.min(1000 * 2 ** attempt, 30_000)));
                }
            }
        }
        throw lastError;
    }
    /** @internal */
    async *stream(method, path, body) {
        const url = `${this.baseUrl}${path}`;
        const headers = {
            'Content-Type': 'application/json',
            ...this.defaultHeaders,
        };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        // Streaming had no timeout at all: a server that accepted the connection
        // and then stopped sending hung the caller forever. The timer below covers
        // both phases -- it aborts if the response headers do not arrive within
        // `timeout`, and it is re-armed on every chunk so a stream that goes
        // silent mid-generation is also bounded.
        const controller = new AbortController();
        let timedOut = false;
        let timeoutId;
        const arm = () => {
            if (timeoutId !== undefined)
                clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                timedOut = true;
                controller.abort();
            }, this.timeout);
        };
        try {
            arm();
            let response;
            try {
                response = await fetch(url, {
                    method,
                    headers,
                    body: JSON.stringify(body),
                    signal: controller.signal,
                });
            }
            catch (connectError) {
                if (timedOut) {
                    throw new DistLLMTimeoutError(`Request timed out after ${this.timeout}ms waiting for response headers`, this.timeout);
                }
                throw connectError;
            }
            if (!response.ok) {
                throw new DistLLMApiError(`HTTP ${response.status}`, response.status);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                let chunk;
                try {
                    arm();
                    chunk = await reader.read();
                }
                catch (readError) {
                    if (timedOut) {
                        throw new DistLLMTimeoutError(`Stream idle for ${this.timeout}ms: no data received from the server`, this.timeout);
                    }
                    throw readError;
                }
                const { done, value } = chunk;
                if (done)
                    break;
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
        finally {
            if (timeoutId !== undefined)
                clearTimeout(timeoutId);
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
/**
 * Raised when a request exceeds the configured timeout.
 *
 * The client aborts with an AbortController; without this class the abort
 * surfaced as a bare DOMException, so callers could not tell a slow server
 * apart from a network failure, and the streaming tests that assert on this
 * type could never pass.
 */
export class DistLLMTimeoutError extends Error {
    timeoutMs;
    constructor(message = 'Request timed out', timeoutMs) {
        super(message);
        this.timeoutMs = timeoutMs;
        this.name = 'DistLLMTimeoutError';
    }
}
