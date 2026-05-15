/**
 * k6 gRPC load test presets for distributed-llm.
 *
 * Presets:
 *   - smoke: 2 VUs, 30s — basic sanity check
 *   - load: ramp to 20 VUs, 2m — normal load
 *   - stress: ramp to 100 VUs, 3m — stress test
 *
 * Usage:
 *   k6 run --config config.js smoke
 *   k6 run --config config.js load
 *   k6 run --config config.js stress
 */

export const options = {
  scenarios: {
    smoke: {
      executor: 'constant-vus',
      vus: 2,
      duration: '30s',
      gracefulStop: '5s',
    },
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '1m', target: 20 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 25 },
        { duration: '1m', target: 50 },
        { duration: '1m', target: 100 },
        { duration: '30s', target: 0 },
      ],
      gracefulRampDown: '15s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.05'],
  },
};
