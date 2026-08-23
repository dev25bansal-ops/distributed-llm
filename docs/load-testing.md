# Load Testing

How to load-test the DistLLM API server, and the current baseline numbers.

## Tooling

| File | Purpose |
|------|---------|
| `scripts/locustfile.py` | Locust user hitting `POST /v1/chat/completions` (non-streaming). Every request carries a unique payload so the dedup middleware never collapses concurrent requests into cached responses. |
| `scripts/load_test_runner.py` | All-in-one harness: starts the real FastAPI app under uvicorn **in-process** with a deterministic mock coordinator (generation stubbed), runs locust headless, prints req/s + latency percentiles. |

## Reproducing

```bash
# Full harness (mock backend, no model needed):
python scripts/load_test_runner.py --users 10 --runtime 60s

# Or against a REAL server with real inference:
export API_KEY=<your key>
locust -f scripts/locustfile.py --headless -u 10 -r 2 \
    --run-time 60s --host http://127.0.0.1:8000
```

Tunables via env: `DISTLLM_LOADTEST_MODEL`, `DISTLLM_LOADTEST_MAX_TOKENS`.

## Baseline (2026-08-24)

Environment: Windows 11 dev laptop, Python 3.14, single uvicorn worker,
in-process server with full middleware stack active (auth, CSRF, rate limit,
prompt-injection scan, circuit breaker, tracing, audit-log plugin) and a
zero-computation mock backend. Profile: 10 users, no think time, 60 s,
non-streaming chat completions.

| Metric | Value |
|--------|-------|
| Total requests | 274 |
| Failures | 0 (0.00%) |
| Throughput | **4.5 req/s** |
| Latency median | 2,200 ms |
| Latency avg | 2,160 ms |
| Latency p95 | 2,500 ms |
| Latency p99 | 3,100 ms |
| Latency max | 3,795 ms |

A 2-user smoke run of the same harness: 4.0 req/s at 430 ms median — i.e.
throughput barely scales from 2 → 10 users while latency grows ~5x.

### Honest limitations — read before quoting these numbers

1. **This is a serving-layer benchmark, not a model benchmark.** Generation is
   stubbed (`coord.generate` returns instantly), so every millisecond here is
   HTTP + middleware + serialization overhead. Real inference will add
   model time on top.
2. **Single process, localhost.** No network hop, no multi-worker uvicorn, no
   TLS. Production deployments front this differently.
3. **The flat throughput between 2 and 10 users (~4–4.5 req/s) with rising
   latency indicates request serialization in the pipeline** — most plausibly
   synchronous per-request audit/trace log writes to the console and shared
   state in the middleware chain. This is the top follow-up: profile the
   middleware stack (candidates: `AuditLogPlugin.on_response` console I/O,
   `TracingMiddleware` span export) before drawing any capacity conclusions.
4. Numbers were captured on a laptop with variable background load; treat them
   as order-of-magnitude baselines for regression comparison, not absolute
   capacity figures.

## Follow-ups

1. Profile and fix the serialization bottleneck above; re-run this suite and
   update the table (the runner is deterministic enough for A/B comparison).
2. Add a streaming variant to the locustfile once SSE chunk timing can be
   recorded per-event (locust measures full-response time today).
3. Run the same profile against a real model to publish combined
   serving+inference numbers for the benchmarks page.
