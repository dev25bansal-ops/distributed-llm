"""Self-Optimizing Engine: continuous profiling and auto-tuning.

Continuously profiles every operation (attention, MLP, communication),
auto-tunes batch size, KV cache quantization, and speculative decoding
parameters. Learns the optimal configuration per model per hardware setup.

Reduces manual tuning from hours to zero by:
- Profiling each operation in real-time
- Building a performance model for the hardware
- Searching for optimal parameters via Bayesian-like optimization
- Persisting learned configs for reuse

Integrates with: coordinator, batch_scheduler, speculative_decoder, prefix_cache
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


# ---------------------------------------------------------------------------
# Profiled Operation Types
# ---------------------------------------------------------------------------

class OpType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    COMMUNICATION = "communication"
    EMBEDDING = "embedding"
    NORM = "norm"
    KV_CACHE = "kv_cache"
    SPECULATIVE_DECODE = "speculative_decode"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class OpSample:
    """A single profiling sample for an operation."""
    op_type: OpType
    duration_ms: float
    input_size: int            # Total input tokens
    hidden_dim: int = 0
    batch_size: int = 1
    seq_len: int = 0
    precision: str = "fp16"
    timestamp: float = 0.0


@dataclass
class OpProfile:
    """Aggregated profile for an operation type."""
    op_type: OpType
    samples: list[OpSample] = field(default_factory=list)
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    throughput_tok_s: float = 0.0
    last_updated: float = 0.0


# ---------------------------------------------------------------------------
# Tunable Parameters
# ---------------------------------------------------------------------------

@dataclass
class TunableParams:
    """All parameters the self-optimizing engine can tune."""
    batch_size: int = 1
    kv_cache_quant_bits: int = 16       # 16, 8, or 4
    speculative_decoding_enabled: bool = False
    speculative_decoding_k: int = 3      # Number of draft tokens
    speculative_decoding_alpha: float = 0.6  # Acceptance threshold
    chunked_prefill_enabled: bool = True
    chunk_size: int = 512
    max_seq_len: int = 4096
    prefix_caching_enabled: bool = True
    paged_attention_block_size: int = 16
    communication_compression: bool = True
    flash_attention_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TunableParams:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Performance Model
# ---------------------------------------------------------------------------

class PerformanceModel:
    """Learns a performance model mapping params -> throughput/latency.

    Uses a simple cost model:
    - For each OpType, tracks throughput as a function of batch_size and seq_len
    - Builds a piecewise-linear approximation for predictions

    This avoids needing actual ML models while still providing
    meaningful optimization guidance.
    """

    def __init__(self):
        self._profiles: dict[OpType, OpProfile] = {}
        self._lock = threading.Lock()

    def record_sample(self, sample: OpSample) -> None:
        with self._lock:
            if sample.op_type not in self._profiles:
                self._profiles[sample.op_type] = OpProfile(op_type=sample.op_type)

            profile = self._profiles[sample.op_type]
            profile.samples.append(sample)
            profile.last_updated = time.time()

            # Keep last 1000 samples per op
            if len(profile.samples) > 1000:
                profile.samples.pop(0)

            # Recompute aggregates
            durations = [s.duration_ms for s in profile.samples]
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            profile.avg_duration_ms = sum(durations) / n
            profile.p50_duration_ms = durations_sorted[n // 2]
            profile.p95_duration_ms = durations_sorted[int(n * 0.95)]
            profile.p99_duration_ms = durations_sorted[int(n * 0.99)]

            avg_input = sum(s.input_size for s in profile.samples) / n
            profile.throughput_tok_s = avg_input / (profile.avg_duration_ms / 1000.0) if profile.avg_duration_ms > 0 else 0.0

    def get_profile(self, op_type: OpType) -> OpProfile | None:
        with self._lock:
            return self._profiles.get(op_type)

    def predict_cost_ms(self, op_type: OpType, batch_size: int, seq_len: int) -> float:
        """Predict operation cost in ms for given params."""
        profile = self.get_profile(op_type)
        if profile is None or not profile.samples:
            return 10.0  # Default fallback

        # Find nearest samples
        with self._lock:
            close = [
                s for s in profile.samples
                if abs(s.batch_size - batch_size) <= max(1, batch_size // 2)
                and abs(s.seq_len - seq_len) <= max(64, seq_len // 2)
            ]

        if close:
            return sum(s.duration_ms for s in close) / len(close)

        # Scale from average
        avg_sample = profile.samples[len(profile.samples) // 2]
        scale = (batch_size / max(avg_sample.batch_size, 1)) * (seq_len / max(avg_sample.seq_len, 1))
        return profile.avg_duration_ms * scale

    def predict_throughput(self, batch_size: int, seq_len: int) -> float:
        """Predict end-to-end throughput in tokens/sec."""
        costs = []
        for op_type in (OpType.ATTENTION, OpType.MLP, OpType.NORM, OpType.EMBEDDING):
            costs.append(self.predict_cost_ms(op_type, batch_size, seq_len))

        # Communication cost scales with model parallelism
        comm_profile = self.get_profile(OpType.COMMUNICATION)
        if comm_profile and comm_profile.samples:
            costs.append(comm_profile.avg_duration_ms)

        total_ms = sum(costs) + 1.0
        return (batch_size * seq_len) / (total_ms / 1000.0)

    def all_profiles(self) -> dict[str, Any]:
        result = {}
        with self._lock:
            for op_type, profile in self._profiles.items():
                result[op_type.value] = {
                    "samples": len(profile.samples),
                    "avg_duration_ms": round(profile.avg_duration_ms, 3),
                    "p50_ms": round(profile.p50_duration_ms, 3),
                    "p95_ms": round(profile.p95_duration_ms, 3),
                    "p99_ms": round(profile.p99_duration_ms, 3),
                    "throughput_tok_s": round(profile.throughput_tok_s, 1),
                }
        return result


# ---------------------------------------------------------------------------
# Parameter Tuner
# ---------------------------------------------------------------------------

class ParameterTuner:
    """Searches for optimal parameters using hill-climbing + random restarts.

    Strategy:
    1. Start with current params
    2. Propose small random perturbations
    3. Measure throughput impact
    4. Accept if better
    5. Periodically do random restarts to escape local optima
    """

    def __init__(self, perf_model: PerformanceModel):
        self._perf_model = perf_model
        self._best_params = TunableParams()
        self._best_throughput = 0.0
        self._lock = threading.Lock()

    def propose(self, current_params: TunableParams) -> TunableParams:
        """Propose a parameter configuration to try."""
        candidate = TunableParams(**asdict(current_params))

        # Randomly perturb one parameter
        r = random.randint(0, 4)
        if r < 1:
            candidate.batch_size = max(1, current_params.batch_size + self._rand_delta(2))
        elif r % 5 < 2:
            candidate.kv_cache_quant_bits = self._cycle_quant_bits(current_params.kv_cache_quant_bits)
        elif r % 5 < 3:
            candidate.speculative_decoding_enabled = not current_params.speculative_decoding_enabled
        elif r % 5 < 4:
            candidate.speculative_decoding_k = max(1, min(10, current_params.speculative_decoding_k + self._rand_delta(1)))
        else:
            candidate.chunk_size = max(64, min(4096, current_params.chunk_size + self._rand_delta(128)))

        return candidate

    def _rand_delta(self, max_delta: int) -> int:
        return random.randint(-max_delta, max_delta)

    def _cycle_quant_bits(self, current: int) -> int:
        options = [16, 8, 4]
        idx = options.index(current) if current in options else 0
        return options[(idx + 1) % len(options)]

    def update(self, params: TunableParams, throughput: float) -> bool:
        """Update the best-known params if throughput improved.

        Returns True if this is a new best.
        """
        with self._lock:
            if throughput > self._best_throughput:
                self._best_params = TunableParams(**asdict(params))
                self._best_throughput = throughput
                return True
        return False

    @property
    def best_params(self) -> TunableParams:
        with self._lock:
            return TunableParams(**asdict(self._best_params))

    @property
    def best_throughput(self) -> float:
        with self._lock:
            return self._best_throughput


# ---------------------------------------------------------------------------
# Self-Optimizing Engine
# ---------------------------------------------------------------------------

class SelfOptimizingEngine:
    """Continuously profiles operations and auto-tunes parameters.

    Usage:
        engine = SelfOptimizingEngine()
        engine.start()

        # During inference:
        engine.record_operation(OpType.ATTENTION, duration_ms=2.3, batch_size=4, seq_len=512)
        engine.record_operation(OpType.MLP, duration_ms=1.8, batch_size=4, seq_len=512)

        # Get recommended params:
        params = engine.get_optimal_params()
        print(params.batch_size, params.kv_cache_quant_bits)

        # Stop:
        engine.stop()
    """

    def __init__(
        self,
        model_name: str = "",
        profile_dir: str | None = None,
        tune_interval_seconds: float = 60.0,
        warmup_seconds: float = 30.0,
        exploration_noise: float = 0.1,
        apply_params: Callable[[TunableParams], None] | None = None,
    ):
        self._model_name = model_name
        self._profile_dir = Path(profile_dir or os.path.join(
            os.path.expanduser("~"), ".distllm", "profiles"
        ))
        self._profile_dir.mkdir(parents=True, exist_ok=True)

        self._tune_interval = tune_interval_seconds
        self._warmup_seconds = warmup_seconds
        self._exploration_noise = exploration_noise
        self._apply_params = apply_params  # Callback to apply params to live system

        self._perf_model = PerformanceModel()
        self._tuner = ParameterTuner(self._perf_model)
        self._current_params = TunableParams()
        self._current_throughput: float = 0.0
        self._total_requests: int = 0
        self._start_time = time.time()

        self._lock = threading.Lock()
        self._running = False
        self._tune_thread: threading.Thread | None = None
        self._sample_buffer: list[OpSample] = []
        self._buffer_lock = threading.Lock()

        # Load saved profile if available
        self._load_profile()

    # -----------------------------------------------------------------------
    # Recording Operations
    # -----------------------------------------------------------------------

    def record_operation(
        self,
        op_type: OpType,
        duration_ms: float,
        batch_size: int = 1,
        seq_len: int = 0,
        input_size: int = 0,
        hidden_dim: int = 0,
        precision: str = "fp16",
    ) -> None:
        """Record a single operation's timing.

        Args:
            op_type: Type of operation (attention, mlp, comm, etc.)
            duration_ms: Duration in milliseconds.
            batch_size: Batch size used.
            seq_len: Sequence length.
            input_size: Total input tokens (batch * seq).
            hidden_dim: Model hidden dimension.
            precision: Precision used (fp16, bf16, int8).
        """
        sample = OpSample(
            op_type=op_type,
            duration_ms=duration_ms,
            input_size=input_size or (batch_size * max(seq_len, 1)),
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            precision=precision,
            timestamp=time.time(),
        )
        with self._buffer_lock:
            self._sample_buffer.append(sample)

        # Periodically flush to performance model
        if len(self._sample_buffer) >= 50:
            self._flush_samples()

    def _flush_samples(self) -> None:
        with self._buffer_lock:
            samples = self._sample_buffer
            self._sample_buffer = []

        for sample in samples:
            self._perf_model.record_sample(sample)

    # -----------------------------------------------------------------------
    # Tuning Loop
    # -----------------------------------------------------------------------

    def _tune_loop(self) -> None:
        """Background tuning loop: profile -> propose -> measure -> adapt."""
        last_tune = 0.0
        iteration = 0

        while self._running:
            now = time.time()
            elapsed = now - self._start_time

            # Warmup: just collect data
            if elapsed < self._warmup_seconds:
                time.sleep(5.0)
                continue

            # Periodically flush remaining samples
            self._flush_samples()

            # Tune at intervals
            if now - last_tune >= self._tune_interval:
                iteration += 1
                last_tune = now

                # Propose new params
                candidate = self._tuner.propose(self._current_params)

                # Estimate throughput for candidate
                estimated = self._perf_model.predict_throughput(
                    candidate.batch_size,
                    candidate.max_seq_len or 512,
                )

                # Accept if better
                is_better = estimated > self._current_throughput
                if is_better or iteration % 5 == 0:
                    with self._lock:
                        self._current_params = candidate
                        self._current_throughput = estimated

                    self._tuner.update(candidate, estimated)

                    # Apply to live system if callback is registered
                    if is_better and self._apply_params is not None:
                        try:
                            self._apply_params(candidate)
                            logger.info(
                                f"SelfOptimizing: applied new params to live system "
                                f"(throughput {estimated:.0f} tok/s)"
                            )
                        except Exception as e:
                            logger.warning(f"SelfOptimizing: failed to apply params: {e}")

                    if is_better:
                        logger.info(
                            f"SelfOptimizing: iteration {iteration}, "
                            f"throughput {estimated:.0f} tok/s, "
                            f"batch={candidate.batch_size}, "
                            f"kv_cache_quant={candidate.kv_cache_quant_bits}bit, "
                            f"spec={candidate.speculative_decoding_enabled}(k={candidate.speculative_decoding_k})"
                        )

                # Occasionally do random restart
                if iteration >= 4 and iteration % 10 == 0:
                    self._tuner = ParameterTuner(self._perf_model)
                    logger.info("SelfOptimizing: random restart")

                self._save_profile()

            time.sleep(5.0)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the background tuning thread."""
        if self._running:
            return
        self._running = True
        self._tune_thread = threading.Thread(target=self._tune_loop, daemon=True)
        self._tune_thread.start()
        logger.info("SelfOptimizingEngine started")

    def set_apply_callback(self, callback: Callable[[TunableParams], None]) -> None:
        """Register a callback to apply tuned parameters to the live system.

        The callback receives TunableParams and should update system configuration
        (batch size, KV cache quantization, speculative decoding, etc.).
        """
        self._apply_params = callback

    def stop(self) -> None:
        """Stop the tuning thread and save profile."""
        self._running = False
        self._flush_samples()
        self._save_profile()
        logger.info("SelfOptimizingEngine stopped")

    def get_optimal_params(self) -> TunableParams:
        """Get the current optimal parameter configuration."""
        return self._tuner.best_params

    def get_current_params(self) -> TunableParams:
        with self._lock:
            return TunableParams(**asdict(self._current_params))

    def record_request(self, tokens_generated: int, total_time_ms: float) -> None:
        """Record an end-to-end request for throughput tracking."""
        with self._lock:
            self._total_requests += 1
            throughput = tokens_generated / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0
            self._current_throughput = throughput

    def get_suggested_batch_size(self, max_batch: int = 64) -> int:
        """Get the suggested batch size based on profiling."""
        params = self.get_optimal_params()
        return min(params.batch_size, max_batch)

    def get_suggested_kv_cache_quant(self) -> int:
        return self.get_optimal_params().kv_cache_quant_bits

    def should_enable_speculative_decoding(self) -> bool:
        return self.get_optimal_params().speculative_decoding_enabled

    # -----------------------------------------------------------------------
    # Profile Persistence
    # -----------------------------------------------------------------------

    def _profile_path(self) -> Path:
        name = self._model_name.replace("/", "_") if self._model_name else "default"
        return self._profile_dir / f"{name}.json"

    def _save_profile(self) -> None:
        path = self._profile_path()
        data = {
            "model_name": self._model_name,
            "best_params": self._tuner.best_params.to_dict(),
            "current_params": self._current_params.to_dict(),
            "best_throughput": self._tuner.best_throughput,
            "current_throughput": self._current_throughput,
            "total_requests": self._total_requests,
            "profiles": self._perf_model.all_profiles(),
            "saved_at": time.time(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Failed to save profile: {e}")

    def _load_profile(self) -> None:
        path = self._profile_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)

            if "best_params" in data:
                self._tuner._best_params = TunableParams.from_dict(data["best_params"])
            if "best_throughput" in data:
                self._tuner._best_throughput = data["best_throughput"]

            logger.info(f"Loaded saved profile from {path}")
        except Exception as e:
            logger.debug(f"Failed to load profile: {e}")

    def stats(self) -> dict[str, Any]:
        self._flush_samples()
        profiles = self._perf_model.all_profiles()
        best = self._tuner.best_params
        return {
            "total_requests": self._total_requests,
            "uptime_seconds": int(time.time() - self._start_time),
            "best_params": best.to_dict(),
            "best_throughput": self._tuner.best_throughput,
            "current_throughput": self._current_throughput,
            "per_operation": profiles,
        }

    def summary(self) -> str:
        s = self.stats()
        lines = [
            f"SelfOptimizingEngine: {s['total_requests']} reqs, {s['uptime_seconds']}s uptime",
            f"  Best throughput: {s['best_throughput']:.0f} tok/s",
            f"  Current throughput: {s['current_throughput']:.0f} tok/s",
            f"  Best params: batch={s['best_params']['batch_size']}, "
            f"kv_cache_quant={s['best_params']['kv_cache_quant_bits']}bit, "
            f"spec_decode={s['best_params']['speculative_decoding_enabled']}(k={s['best_params']['speculative_decoding_k']})",
        ]
        for op_name, op_data in s['per_operation'].items():
            lines.append(f"  {op_name}: {op_data['avg_duration_ms']}ms avg, {op_data['throughput_tok_s']} tok/s")
        return "\n".join(lines)
