#!/usr/bin/env python3
"""GPU memory profiler for Distributed LLM.

Tracks GPU memory over time during repeated generation cycles to detect leaks.
Works on both GPU (via pynvml) and CPU (via psutil/tracemalloc).

Usage:
    # Profile a running DistLLM server
    python tests/profiling/memory_profiler.py --url http://localhost:8000

    # Profile local model loading
    python tests/profiling/memory_profiler.py --model meta-llama/Llama-3.2-1B --local

    # Profile with specific prompts
    python tests/profiling/memory_profiler.py --url http://localhost:8000 --prompts tests/profiling/prompts.txt

    # Output to file
    python tests/profiling/memory_profiler.py --url http://localhost:8000 --output memory_profile.json
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# --- Memory Sampling ---


class GPUMemorySampler:
    """Sample GPU memory usage via pynvml (NVIDIA)."""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._initialized = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self._initialized = True
        except Exception:
            pass

    def sample(self) -> Dict[str, float]:
        """Sample current GPU memory usage."""
        if not self._initialized:
            return {"error": "pynvml not available"}

        import pynvml

        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return {
                "used_mb": info.used / (1024 * 1024),
                "total_mb": info.total / (1024 * 1024),
                "free_mb": info.free / (1024 * 1024),
                "used_percent": info.used / info.total * 100,
                "temperature_c": temp,
            }
        except Exception as e:
            return {"error": str(e)}

    def shutdown(self):
        if self._initialized:
            import pynvml

            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class CPUMemorySampler:
    """Sample process memory usage via tracemalloc/psutil."""

    def sample(self) -> Dict[str, float]:
        """Sample current process memory usage."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            return {
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except ImportError:
            # Fallback: read /proc/self/status on Linux
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            return {"rss_mb": rss_kb / 1024}
            except Exception:
                pass
            return {"error": "psutil not available"}


@dataclass
class MemorySnapshot:
    """A single memory measurement."""

    timestamp: float
    elapsed_s: float
    iteration: int
    phase: str  # "idle", "loading", "generating", "complete"
    prompt: str = ""
    generated_tokens: int = 0
    memory: Dict[str, float] = field(default_factory=dict)
    fragmentation_ratio: Optional[float] = None


@dataclass
class ProfileReport:
    """Complete profiling report."""

    model: str
    start_time: str
    end_time: str
    total_duration_s: float
    num_iterations: int
    snapshots: List[MemorySnapshot]
    leak_analysis: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


# --- Leak Detection ---


def detect_memory_leak(snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
    """Analyze memory snapshots for monotonic growth indicating a leak.

    Uses linear regression slope and compares idle-state memory before/after.
    """
    if len(snapshots) < 3:
        return {"status": "insufficient_data", "min_samples": 3, "actual": len(snapshots)}

    # Extract memory values (prefer used_mb for GPU, rss_mb for CPU)
    values = []
    for s in snapshots:
        mem = s.memory
        val = mem.get("used_mb", mem.get("rss_mb"))
        if val is not None:
            values.append((s.elapsed_s, val))

    if len(values) < 3:
        return {"status": "insufficient_memory_data"}

    # Simple linear regression
    n = len(values)
    sum_x = sum(v[0] for v in values)
    sum_y = sum(v[1] for v in values)
    sum_xy = sum(v[0] * v[1] for v in values)
    sum_x2 = sum(v[0] ** 2 for v in values)

    denom = n * sum_x2 - sum_x**2
    if denom == 0:
        return {"status": "constant_timestamps"}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((v[1] - y_mean) ** 2 for v in values)
    ss_res = sum((v[1] - (slope * v[0] + intercept)) ** 2 for v in values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Compare first and last idle measurements
    idle_snapshots = [s for s in snapshots if s.phase == "idle"]
    if len(idle_snapshots) >= 2:
        first_idle = idle_snapshots[0].memory.get(
            "used_mb", idle_snapshots[0].memory.get("rss_mb", 0)
        )
        last_idle = idle_snapshots[-1].memory.get(
            "used_mb", idle_snapshots[-1].memory.get("rss_mb", 0)
        )
        delta = last_idle - first_idle
        delta_percent = (delta / first_idle * 100) if first_idle > 0 else 0
    else:
        first_val = values[0][1]
        last_val = values[-1][1]
        delta = last_val - first_val
        delta_percent = (delta / first_val * 100) if first_val > 0 else 0

    # Determine leak status
    # Consider it a leak if: positive slope + high R² + growth > 1%
    is_leak = slope > 0 and r_squared > 0.7 and delta_percent > 1.0

    return {
        "status": "leak_detected" if is_leak else "no_leak_detected",
        "slope_mb_per_sec": round(slope, 4),
        "r_squared": round(r_squared, 4),
        "first_value_mb": round(values[0][1], 2),
        "last_value_mb": round(values[-1][1], 2),
        "delta_mb": round(delta, 2),
        "delta_percent": round(delta_percent, 2),
        "threshold": "slope > 0, R² > 0.7, growth > 1%",
    }


# --- Profiler ---


class MemoryProfiler:
    """Profile memory usage during LLM generation."""

    def __init__(
        self,
        model: str,
        url: Optional[str] = None,
        local: bool = False,
        gpu_id: int = 0,
        defrag_stats_url: Optional[str] = None,
    ):
        self.model = model
        self.url = url
        self.local = local
        self.gpu_sampler = GPUMemorySampler(gpu_id)
        self.cpu_sampler = CPUMemorySampler()
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()
        self._defrag_stats_url = defrag_stats_url or (
            f"{url}/v1/defrag/stats" if url else None
        )
        self._last_frag_ratio: Optional[float] = None

    def _sample_memory(self) -> Dict[str, float]:
        """Sample both GPU and CPU memory."""
        gpu = self.gpu_sampler.sample()
        cpu = self.cpu_sampler.sample()
        return {**gpu, **cpu}

    def _snapshot(self, iteration: int, phase: str, prompt: str = "", generated_tokens: int = 0):
        """Take a memory snapshot."""
        elapsed = time.time() - self.start_time
        mem = self._sample_memory()
        self.snapshots.append(
            MemorySnapshot(
                timestamp=time.time(),
                elapsed_s=round(elapsed, 3),
                iteration=iteration,
                phase=phase,
                prompt=prompt[:100] if prompt else "",
                generated_tokens=generated_tokens,
                memory=mem,
                fragmentation_ratio=self._last_frag_ratio,
            )
        )

    async def _fetch_frag_ratio(self, client: httpx.AsyncClient) -> Optional[float]:
        """Fetch fragmentation ratio from the defrag stats endpoint."""
        if not self._defrag_stats_url:
            return None
        try:
            resp = await client.get(self._defrag_stats_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("fragmentation_ratio")
        except Exception:
            pass
        return None

    async def profile_remote(
        self,
        prompts: List[str],
        num_cycles: int = 5,
        idle_interval: float = 2.0,
    ) -> ProfileReport:
        """Profile a remote DistLLM server."""
        self.start_time = time.time()
        client = httpx.AsyncClient(timeout=60)

        # Initial idle snapshot (with defrag metrics)
        self._last_frag_ratio = await self._fetch_frag_ratio(client)
        self._snapshot(0, "idle")

        for cycle in range(num_cycles):
            for i, prompt in enumerate(prompts):
                iteration = cycle * len(prompts) + i + 1

                # Pre-generation snapshot (with defrag metrics)
                self._last_frag_ratio = await self._fetch_frag_ratio(client)
                self._snapshot(iteration, "idle", prompt=prompt)
                await asyncio.sleep(idle_interval)

                # Generation
                self._snapshot(iteration, "generating", prompt=prompt)

                try:
                    response = await client.post(
                        f"{self.url}/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 128,
                            "temperature": 0.0,
                            "stream": True,
                        },
                    )
                    response.raise_for_status()

                    # Count generated tokens from SSE stream
                    tokens = 0
                    for line in response.iter_lines():
                        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                            try:
                                data = json.loads(line[6:])
                                choice = data.get("choices", [{}])[0]
                                if choice.get("delta", {}).get("content"):
                                    tokens += 1
                            except json.JSONDecodeError:
                                continue

                    self._snapshot(iteration, "complete", prompt=prompt, generated_tokens=tokens)

                except Exception as e:
                    print(f"  Error on iteration {iteration}: {e}")
                    self._snapshot(iteration, "complete", prompt=prompt)

                await asyncio.sleep(idle_interval)

        # Final idle snapshot (with defrag metrics)
        await asyncio.sleep(idle_interval * 2)
        self._last_frag_ratio = await self._fetch_frag_ratio(client)
        self._snapshot(num_cycles * len(prompts) + 1, "idle")

        await client.aclose()
        return self._build_report()

    async def profile_local(
        self,
        prompts: List[str],
        num_cycles: int = 3,
    ) -> ProfileReport:
        """Profile local model loading and generation."""
        self.start_time = time.time()

        # Initial snapshot
        self._snapshot(0, "idle")

        # Model loading
        self._snapshot(1, "loading")
        try:
            from distllm.core.coordinator import Coordinator

            coord = Coordinator(model_name=self.model, dtype="float16")
            coord.load_local_model()
            self._snapshot(1, "complete")
        except ImportError:
            print("DistLLM not available for local profiling")
            return self._build_report()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._snapshot(1, "complete")
            return self._build_report()

        # Generation cycles
        for cycle in range(num_cycles):
            for i, prompt in enumerate(prompts):
                iteration = cycle * len(prompts) + i + 2

                self._snapshot(iteration, "generating", prompt=prompt)

                try:
                    result = coord.generate(prompt, max_new_tokens=64)
                    tokens = len(coord.tokenizer.encode(result))
                    self._snapshot(iteration, "complete", prompt=prompt, generated_tokens=tokens)
                except Exception as e:
                    print(f"  Error on iteration {iteration}: {e}")
                    self._snapshot(iteration, "complete", prompt=prompt)

                time.sleep(1)

        # Final snapshot
        self._snapshot(num_cycles * len(prompts) + 2, "idle")
        return self._build_report()

    def _build_report(self) -> ProfileReport:
        """Build the final profiling report."""
        leak_analysis = detect_memory_leak(self.snapshots)

        # Summary stats
        mem_values = []
        for s in self.snapshots:
            val = s.memory.get("used_mb", s.memory.get("rss_mb"))
            if val is not None:
                mem_values.append(val)

        if mem_values:
            summary = {
                "min_memory_mb": round(min(mem_values), 2),
                "max_memory_mb": round(max(mem_values), 2),
                "avg_memory_mb": round(sum(mem_values) / len(mem_values), 2),
                "memory_range_mb": round(max(mem_values) - min(mem_values), 2),
            }
        else:
            summary = {}

        return ProfileReport(
            model=self.model,
            start_time=datetime.fromtimestamp(self.start_time).isoformat(),
            end_time=datetime.fromtimestamp(time.time()).isoformat(),
            total_duration_s=round(time.time() - self.start_time, 2),
            num_iterations=len(self.snapshots),
            snapshots=self.snapshots,
            leak_analysis=leak_analysis,
            summary=summary,
        )


# --- Output ---


def print_report(report: ProfileReport):
    """Print a human-readable memory profile report."""
    print("\n" + "=" * 70)
    print("GPU MEMORY PROFILE REPORT")
    print("=" * 70)

    print(f"\nModel: {report.model}")
    print(f"Duration: {report.total_duration_s:.1f}s")
    print(f"Iterations: {report.num_iterations}")

    print("\n--- Memory Summary ---")
    for key, val in report.summary.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {val}")

    print("\n--- Leak Analysis ---")
    analysis = report.leak_analysis
    status = analysis.get("status", "unknown")
    print(f"  Status: {status}")
    if "slope_mb_per_sec" in analysis:
        print(f"  Growth rate: {analysis['slope_mb_per_sec']} MB/s")
        print(f"  R²: {analysis['r_squared']}")
        print(f"  Delta: {analysis['delta_mb']} MB ({analysis['delta_percent']}%)")

    # Per-iteration breakdown
    print("\n--- Per-Iteration Memory ---")
    print(f"  {'Iter':>4} | {'Phase':<12} | {'Tokens':>6} | {'Mem (MB)':>10} | {'Frag%':>7} | {'Elapsed':>8}")
    print(f"  {'-'*4}-+{'-'*14}-+{'-'*8}-+{'-'*12}-+{'-'*9}-+{'-'*10}")
    for s in report.snapshots:
        mem = s.memory.get("used_mb", s.memory.get("rss_mb", 0))
        frag = f"{s.fragmentation_ratio*100:>5.1f}%" if s.fragmentation_ratio is not None else "  N/A "
        print(
            f"  {s.iteration:>4} | {s.phase:<12} | {s.generated_tokens:>6} | {mem:>10.1f} | {frag:>7} | {s.elapsed_s:>7.1f}s"
        )

    print("\n" + "=" * 70)


# --- Main ---

DEFAULT_PROFILE_PROMPTS = [
    "What is pipeline parallelism?",
    "Explain KV caching in LLM inference.",
    "How does speculative decoding work?",
    "Compare tensor parallelism and pipeline parallelism.",
    "What are the benefits of batched inference?",
]


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Profiler for DistLLM")
    parser.add_argument("--model", default="distributed-llm", help="Model name")
    parser.add_argument("--url", help="URL of running DistLLM server")
    parser.add_argument("--local", action="store_true", help="Profile local model loading")
    parser.add_argument("--prompts", type=str, help="Path to prompts file")
    parser.add_argument("--cycles", type=int, default=5, help="Number of generation cycles")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--output", type=str, help="Output file path (JSON)")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Idle interval between requests (seconds)"
    )
    parser.add_argument(
        "--defrag-url", type=str, help="Defrag stats URL (defaults to <url>/v1/defrag/stats)"
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompts and Path(args.prompts).exists():
        prompts = Path(args.prompts).read_text().strip().split("\n")
    else:
        prompts = DEFAULT_PROFILE_PROMPTS

    profiler = MemoryProfiler(
        model=args.model,
        url=args.url,
        local=args.local,
        gpu_id=args.gpu,
        defrag_stats_url=args.defrag_url,
    )

    if args.url:
        print(f"Profiling remote server at {args.url}...")
        report = asyncio.run(
            profiler.profile_remote(prompts, num_cycles=args.cycles, idle_interval=args.interval)
        )
    elif args.local:
        print(f"Profiling local model loading for {args.model}...")
        report = asyncio.run(profiler.profile_local(prompts, num_cycles=args.cycles))
    else:
        print("Specify --url <server_url> or --local for profiling")
        return

    print_report(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
