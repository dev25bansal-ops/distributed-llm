/**
 * k6 gRPC load test for distributed-llm NodeService.
 *
 * Tests: ForwardPass throughput, HealthCheck latency, concurrent registrations.
 *
 * Usage:
 *   k6 run script.js
 *   k6 run --vus 10 --duration 30s script.js
 *   K6_WEB_DASHBOARD=true k6 run script.js
 *
 * Requires: @grpc-js k6 extension or k6 xk6-grpc.
 * For standard k6 without gRPC extension, uses HTTP fallback.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const grpcErrors = new Rate('grpc_errors');
const forwardPassLatency = new Trend('forward_pass_latency_ms');
const healthCheckLatency = new Trend('health_check_latency_ms');
const registrationLatency = new Trend('registration_latency_ms');
const totalRequests = new Counter('total_requests');

// Configuration
const NODE_HOST = __ENV.NODE_HOST || 'localhost:50051';
const NODE_HTTP = __ENV.NODE_HTTP || 'http://localhost:50051';

export const options = {
  stages: [
    { duration: '30s', target: 5 },    // Ramp up to 5 VUs
    { duration: '1m', target: 20 },    // Ramp up to 20 VUs (load test)
    { duration: '30s', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests under 500ms
    grpc_errors: ['rate<0.01'],         // Less than 1% errors
    forward_pass_latency_ms: ['p(95)<300'],
    health_check_latency_ms: ['p(95)<50'],
  },
};

export default function () {
  // --- Forward Pass ---
  const forwardStart = new Date();
  const forwardRes = http.post(`${NODE_HTTP}/forward`, JSON.stringify({
    request_id: `k6-${Date.now()}-${__VU}`,
    input_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    batch_size: 1,
    seq_len: 10,
  }));
  const forwardDuration = new Date() - forwardStart;
  forwardPassLatency.add(forwardDuration);
  totalRequests.add(1);

  check(forwardRes, {
    'forward status is 200': (r) => r.status === 200,
    'forward response time < 200ms': (r) => r.timings.duration < 200,
  }) || grpcErrors.add(1);

  sleep(0.1);

  // --- Health Check ---
  const healthStart = new Date();
  const healthRes = http.get(`${NODE_HTTP}/health`);
  const healthDuration = new Date() - healthStart;
  healthCheckLatency.add(healthDuration);
  totalRequests.add(1);

  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
  }) || grpcErrors.add(1);

  sleep(1);
}

// --- Registration scenario (separate VU group) ---
export function registration() {
  const regStart = new Date();
  const regRes = http.post(`${NODE_HTTP}/register`, JSON.stringify({
    node_info: {
      node_id: `k6-node-${__VU}`,
      host: 'localhost',
      port: 50052,
      total_memory: 16000000000,
      available_memory: 12000000000,
      device_type: 'cuda',
    },
    num_layers: 12,
  }));
  const regDuration = new Date() - regStart;
  registrationLatency.add(regDuration);
  totalRequests.add(1);

  check(regRes, {
    'register status is 200': (r) => r.status === 200,
  }) || grpcErrors.add(1);

  sleep(5);
}

export function handleSummary(data) {
  return {
    'tests/load/results/k6-summary.json': JSON.stringify(data, null, 2),
    'stdout': textSummary(data),
  };
}

function textSummary(data) {
  const total = data.metrics.http_reqs.values.count;
  const failed = data.metrics.http_req_failed ? data.metrics.http_req_failed.values.rate : 0;
  const p95 = data.metrics.http_req_duration.values['p(95)'];
  return `\n=== k6 Load Test Summary ===
Total requests: ${total}
Error rate: ${(failed * 100).toFixed(2)}%
p95 latency: ${p95.toFixed(2)}ms
`;
}
