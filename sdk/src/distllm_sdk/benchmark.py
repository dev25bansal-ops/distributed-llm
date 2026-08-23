"""Benchmark suite for the DistLLM SDK.

Measures latency, throughput, tokens-per-second, and cost across
one or more models.  Results are returned as structured dataclasses
and can be compared, saved to JSON, or printed as tables.

Usage::

    from distllm_sdk.benchmark import BenchmarkSuite, BenchmarkConfig

    suite = BenchmarkSuite(base_url="http://localhost:8000")
    result = suite.run_chat(
        model="llama-3-70b",
        prompts=["Hello!", "What is AI?", "Explain quantum computing"],
        num_runs=10,
    )
    print(result.summary())
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        model: Model name tested.
        endpoint: API endpoint used (e.g. ``/v1/chat/completions``).
        num_requests: Total requests fired.
        latency_ms: List of per-request latencies in milliseconds.
        tokens_per_request: List of output tokens per request.
        errors: Number of failed requests.
        total_time_seconds: Wall-clock time for the entire run.
        cost_usd: Estimated total cost (uses SDK cost headers when
            available, otherwise estimates from token counts).
        timestamp: ISO-8601 timestamp of the run.
        run_id: Unique identifier for this run.
    """
    model: str
    endpoint: str
    num_requests: int
    latency_ms: list[float] = field(default_factory=list)
    tokens_per_request: list[int] = field(default_factory=list)
    errors: int = 0
    total_time_seconds: float = 0.0
    cost_usd: float = 0.0
    timestamp: str = ""
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return _percentile(sorted(self.latency_ms), 50) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return _percentile(sorted(self.latency_ms), 95) if self.latency_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return _percentile(sorted(self.latency_ms), 99) if self.latency_ms else 0.0

    @property
    def min_latency_ms(self) -> float:
        return min(self.latency_ms) if self.latency_ms else 0.0

    @property
    def max_latency_ms(self) -> float:
        return max(self.latency_ms) if self.latency_ms else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        total_tokens = sum(self.tokens_per_request)
        total_latency_s = sum(self.latency_ms) / 1000.0 if self.latency_ms else 1.0
        return total_tokens / total_latency_s if total_latency_s > 0 else 0.0

    @property
    def throughput_rps(self) -> float:
        """Requests per second over the entire run."""
        return self.num_requests / self.total_time_seconds if self.total_time_seconds > 0 else 0.0

    @property
    def success_rate(self) -> float:
        if self.num_requests == 0:
            return 0.0
        return (self.num_requests - self.errors) / self.num_requests * 100.0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"Benchmark[{self.run_id}] {self.model} @ {self.endpoint}\n"
            f"  Requests: {self.num_requests} ({self.errors} errors, {self.success_rate:.1f}% success)\n"
            f"  Latency:  avg={self.avg_latency_ms:.1f}ms  p50={self.p50_latency_ms:.1f}ms  "
            f"p95={self.p95_latency_ms:.1f}ms  p99={self.p99_latency_ms:.1f}ms  "
            f"min={self.min_latency_ms:.1f}ms  max={self.max_latency_ms:.1f}ms\n"
            f"  Throughput: {self.throughput_rps:.1f} req/s  {self.avg_tokens_per_second:.1f} tok/s\n"
            f"  Cost: ${self.cost_usd:.6f}  Duration: {self.total_time_seconds:.1f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        d = asdict(self)
        d["avg_latency_ms"] = self.avg_latency_ms
        d["p50_latency_ms"] = self.p50_latency_ms
        d["p95_latency_ms"] = self.p95_latency_ms
        d["p99_latency_ms"] = self.p99_latency_ms
        d["avg_tokens_per_second"] = self.avg_tokens_per_second
        d["throughput_rps"] = self.throughput_rps
        d["success_rate"] = self.success_rate
        return d


@dataclass
class BenchmarkComparison:
    """Comparison of benchmark results across multiple models or configs."""
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def winner(self, metric: str = "avg_latency_ms") -> BenchmarkResult | None:
        """Return the best result for a given metric (lower is better)."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: getattr(r, metric, float("inf")))

    def summary(self) -> str:
        lines = ["Benchmark Comparison"]
        lines.append("=" * 72)
        for r in self.results:
            lines.append(
                f"  {r.model:30s}  avg={r.avg_latency_ms:>8.1f}ms  "
                f"p95={r.p95_latency_ms:>8.1f}ms  "
                f"{r.avg_tokens_per_second:>6.1f} tok/s  "
                f"${r.cost_usd:<8.6f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_json(self, filepath: str | None = None) -> str:
        data = [r.to_dict() for r in self.results]
        dumped = json.dumps(data, indent=2, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(dumped)
        return dumped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: int) -> float:
    """Compute the *p*-th percentile of a sorted list."""
    if not sorted_data:
        return 0.0
    k = (p / 100.0) * (len(sorted_data) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    return sorted_data[-1]


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Runs configurable benchmarks against the DistLLM API.

    Args:
        base_url: DistLLM coordinator URL.
        api_key: Optional API key.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def run_chat(
        self,
        model: str,
        prompts: list[str],
        num_runs: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 100,
        concurrency: int = 1,
    ) -> BenchmarkResult:
        """Benchmark chat completions.

        Args:
            model: Model name.
            prompts: List of prompt strings to cycle through.
            num_runs: Total requests to fire.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Max tokens per response.
            concurrency: Number of concurrent requests (1 = sequential).

        Returns:
            BenchmarkResult with latency and throughput stats.
        """
        import httpx

        result = BenchmarkResult(model=model, endpoint="/v1/chat/completions", num_requests=num_runs)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start = time.time()

        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            for i in range(num_runs):
                prompt = prompts[i % len(prompts)]
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                t0 = time.time()
                try:
                    resp = client.post("/v1/chat/completions", json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    elapsed = (time.time() - t0) * 1000
                    result.latency_ms.append(elapsed)

                    usage = data.get("usage", {})
                    tokens = usage.get("completion_tokens", 0) or usage.get("total_tokens", 0)
                    result.tokens_per_request.append(tokens)

                    # Cost from response header
                    cost_str = resp.headers.get("x-distllm-cost", "")
                    if cost_str:
                        try:
                            result.cost_usd += float(cost_str)
                        except ValueError:
                            pass
                except Exception:
                    result.errors += 1
                    elapsed = (time.time() - t0) * 1000
                    result.latency_ms.append(elapsed)
                    result.tokens_per_request.append(0)

        result.total_time_seconds = time.time() - start
        return result

    async def run_chat_async(
        self,
        model: str,
        prompts: list[str],
        num_runs: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 100,
        concurrency: int = 5,
    ) -> BenchmarkResult:
        """Async benchmark with configurable concurrency.

        Uses ``asyncio.Semaphore`` to limit concurrent requests.
        """
        import asyncio
        import httpx

        result = BenchmarkResult(model=model, endpoint="/v1/chat/completions", num_requests=num_runs)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        sem = asyncio.Semaphore(concurrency)

        start = time.time()

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout) as client:

            async def _run_one(prompt: str) -> None:
                async with sem:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    t0 = time.time()
                    try:
                        resp = await client.post("/v1/chat/completions", json=payload, headers=headers)
                        resp.raise_for_status()
                        data = resp.json()
                        elapsed = (time.time() - t0) * 1000
                        result.latency_ms.append(elapsed)
                        usage = data.get("usage", {})
                        tokens = usage.get("completion_tokens", 0) or usage.get("total_tokens", 0)
                        result.tokens_per_request.append(tokens)
                        cost_str = resp.headers.get("x-distllm-cost", "")
                        if cost_str:
                            try:
                                result.cost_usd += float(cost_str)
                            except ValueError:
                                pass
                    except Exception:
                        result.errors += 1
                        result.latency_ms.append((time.time() - t0) * 1000)
                        result.tokens_per_request.append(0)

            tasks = [_run_one(prompts[i % len(prompts)]) for i in range(num_runs)]
            await asyncio.gather(*tasks)

        result.total_time_seconds = time.time() - start
        return result

    def run_embeddings(
        self,
        model: str,
        texts: list[str],
        num_runs: int = 10,
    ) -> BenchmarkResult:
        """Benchmark embeddings."""
        import httpx

        result = BenchmarkResult(model=model, endpoint="/v1/embeddings", num_requests=num_runs)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start = time.time()

        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            for i in range(num_runs):
                text = texts[i % len(texts)]
                payload = {"model": model, "input": text}
                t0 = time.time()
                try:
                    resp = client.post("/v1/embeddings", json=payload, headers=headers)
                    resp.raise_for_status()
                    elapsed = (time.time() - t0) * 1000
                    result.latency_ms.append(elapsed)
                    result.tokens_per_request.append(0)
                except Exception:
                    result.errors += 1
                    result.latency_ms.append((time.time() - t0) * 1000)
                    result.tokens_per_request.append(0)

        result.total_time_seconds = time.time() - start
        return result

    def compare(
        self,
        models: list[str],
        prompts: list[str],
        num_runs: int = 5,
        **kwargs: Any,
    ) -> BenchmarkComparison:
        """Run the same benchmark across multiple models and compare."""
        comparison = BenchmarkComparison()
        for model in models:
            result = self.run_chat(model=model, prompts=prompts, num_runs=num_runs, **kwargs)
            comparison.add(result)
        return comparison

    async def compare_async(
        self,
        models: list[str],
        prompts: list[str],
        num_runs: int = 5,
        **kwargs: Any,
    ) -> BenchmarkComparison:
        """Async version of ``compare``."""
        import asyncio
        results = await asyncio.gather(*[
            self.run_chat_async(model=model, prompts=prompts, num_runs=num_runs, **kwargs)
            for model in models
        ])
        comparison = BenchmarkComparison()
        for r in results:
            comparison.add(r)
        return comparison
