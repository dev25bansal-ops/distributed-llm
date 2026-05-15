/**
 * k6 configuration presets for Distributed LLM load testing
 *
 * Usage:
 *   k6 run --config tests/load/k6/config.js smoke
 *   k6 run --config tests/load/k6/config.js load
 *   k6 run --config tests/load/k6/config.js stress
 *   k6 run --config tests/load/k6/config.js soak
 *
 * Or set env vars:
 *   BASE_URL=http://your-server:8000 k6 run tests/load/k6/script.js --config tests/load/k6/config.js load
 */

const configs = {
  /**
   * Smoke test — quick sanity check that the API responds
   * 1-2 VUs for 1 minute
   */
  smoke: {
    scenarios: {
      smoke: {
        executor: 'constant-vus',
        vus: 2,
        duration: '1m',
        env: { SCENARIO: 'chat', MAX_TOKENS: '32' },
      },
    },
    thresholds: {
      http_req_duration: ['p(95)<30000'],
      http_req_failed: ['rate<0.05'],
    },
  },

  /**
   * Load test — simulate normal production traffic
   * Ramp up to 10 concurrent users, sustain, ramp down
   */
  load: {
    scenarios: {
      load_chat: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
          { duration: '1m', target: 5 },    // ramp up
          { duration: '3m', target: 5 },    // steady state
          { duration: '1m', target: 10 },   // increase load
          { duration: '3m', target: 10 },   // steady state
          { duration: '1m', target: 0 },    // ramp down
        ],
        env: { SCENARIO: 'mixed', MAX_TOKENS: '128' },
        gracefulRampDown: '30s',
      },
    },
    thresholds: {
      http_req_duration: ['p(50)<10000', 'p(95)<30000', 'p(99)<45000'],
      http_req_failed: ['rate<0.05'],
    },
  },

  /**
   * Stress test — push beyond normal capacity to find breaking point
   * Ramp up to 50 concurrent users
   */
  stress: {
    scenarios: {
      stress_chat: {
        executor: 'ramping-vus',
        startVUs: 0,
        stages: [
          { duration: '2m', target: 10 },
          { duration: '2m', target: 20 },
          { duration: '2m', target: 30 },
          { duration: '2m', target: 50 },
          { duration: '5m', target: 50 },
          { duration: '1m', target: 0 },
        ],
        env: { SCENARIO: 'chat', MAX_TOKENS: '256' },
        gracefulRampDown: '30s',
      },
    },
    thresholds: {
      http_req_duration: ['p(95)<60000'],
      http_req_failed: ['rate<0.25'],
    },
  },

  /**
   * Soak test — long-running stability test (4+ hours)
   * Sustained moderate load to detect memory leaks, resource exhaustion
   */
  soak: {
    scenarios: {
      soak_chat: {
        executor: 'constant-vus',
        vus: 5,
        duration: '4h',
        env: { SCENARIO: 'mixed', MAX_TOKENS: '128' },
      },
    },
    thresholds: {
      http_req_duration: ['p(95)<30000'],
      http_req_failed: ['rate<0.05'],
    },
  },
};

// Select config via positional argument or SCENARIO_TYPE env var
const scenarioType = __ENV.SCENARIO_TYPE || (__ENV.__ITER === undefined ? '' : '');

// When run as --config, k6 merges this with script.js options
// The script's default options act as fallback
export const options = {
  // Merge with base options from script.js
  ...options,
  // Override with the desired profile
  // Use the config that matches SCENARIO_TYPE, defaulting to 'load'
  ...(configs[scenarioType] || configs.load),
};

// If you want to use a specific config, set SCENARIO_TYPE env var:
// SCENARIO_TYPE=stress k6 run tests/load/k6/script.js --config tests/load/k6/config.js
