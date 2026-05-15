/**
 * k6 load test for Distributed LLM API
 *
 * Targets /v1/chat/completions with various load profiles.
 * Run: k6 run script.js
 * Run with config: k6 run --config config.js smoke
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// --- Configuration ---

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || '';
const MODEL = __ENV.MODEL || 'distributed-llm';

// Prompts of varying lengths for realistic load testing
const PROMPTS = [
  'Explain the concept of pipeline parallelism in large language model inference.',
  'Write a short Python function that implements a consistent hash ring with virtual nodes.',
  'What are the key differences between data parallelism, tensor parallelism, and pipeline parallelism for distributed training?',
  'Describe the architecture of the Transformer model, including self-attention, multi-head attention, and positional encoding.',
  'How does KV caching improve the efficiency of autoregressive text generation? Explain with time complexity analysis.',
];

function buildHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (API_KEY) {
    headers['Authorization'] = `Bearer ${API_KEY}`;
  }
  return headers;
}

function randomPrompt() {
  return PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
}

// --- Scenarios ---

/**
 * Non-streaming chat completion
 */
function chatCompletion() {
  const payload = JSON.stringify({
    model: MODEL,
    messages: [
      { role: 'system', content: 'You are a helpful AI assistant.' },
      { role: 'user', content: randomPrompt() },
    ],
    max_tokens: parseInt(__ENV.MAX_TOKENS || '128'),
    temperature: parseFloat(__ENV.TEMPERATURE || '0.7'),
    stream: false,
  });

  const params = { headers: buildHeaders() };
  const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has choices': (r) => {
      if (r.status !== 200) return true;
      const body = r.json();
      return body.choices && body.choices.length > 0;
    },
    'response time < 30s': (r) => r.timings.duration < 30000,
  });

  return res;
}

/**
 * Streaming chat completion (SSE)
 */
function streamingChatCompletion() {
  const payload = JSON.stringify({
    model: MODEL,
    messages: [
      { role: 'system', content: 'You are a helpful AI assistant.' },
      { role: 'user', content: randomPrompt() },
    ],
    max_tokens: parseInt(__ENV.MAX_TOKENS || '64'),
    temperature: parseFloat(__ENV.TEMPERATURE || '0.7'),
    stream: true,
  });

  const params = { headers: buildHeaders() };
  const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'content-type is SSE': (r) =>
      r.headers['Content-Type'] === 'text/event-stream' ||
      r.status !== 200,
  });

  return res;
}

/**
 * Health check endpoint
 */
function healthCheck() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'health status 200': (r) => r.status === 200,
    'health has status': (r) => {
      if (r.status !== 200) return true;
      const body = r.json();
      return body.status !== undefined;
    },
  });
  return res;
}

/**
 * Metrics endpoint
 */
function metricsCheck() {
  const res = http.get(`${BASE_URL}/metrics`);
  check(res, {
    'metrics status 200': (r) => r.status === 200,
  });
  return res;
}

// --- Export default (single scenario) ---

export default function () {
  const mode = __ENV.SCENARIO || 'chat';

  if (mode === 'chat') {
    chatCompletion();
  } else if (mode === 'stream') {
    streamingChatCompletion();
  } else if (mode === 'health') {
    healthCheck();
  } else if (mode === 'metrics') {
    metricsCheck();
  } else if (mode === 'mixed') {
    // 70% chat, 20% stream, 10% health
    const r = Math.random();
    if (r < 0.7) {
      chatCompletion();
    } else if (r < 0.9) {
      streamingChatCompletion();
    } else {
      healthCheck();
    }
  }

  sleep(parseFloat(__ENV.THINK_TIME || '1'));
}

// --- Options (overridden by config files) ---

export const options = {
  scenarios: {
    chat_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 5 },
        { duration: '1m', target: 5 },
        { duration: '30s', target: 0 },
      ],
      exec: 'default',
      env: { SCENARIO: 'mixed' },
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<30000'],
    http_req_failed: ['rate<0.1'],
    http_reqs: ['rate>1'],
  },
};

// --- Custom summary ---

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: '  ', enableColors: true }),
    'tests/load/results/summary.json': JSON.stringify(data, null, 2),
  };
}
