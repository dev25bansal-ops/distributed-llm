"""Adaptive compression during idle periods.

When cluster utilization is below a configurable threshold for a minimum
duration, automatically run compression jobs (quantization, pruning) on
loaded models. During high load, swap to the compressed variant for higher
throughput.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class IdleDetectorConfig:
    """Configuration for idle detection."""
    utilization_threshold_pct: float = 30.0
    idle_duration_s: int = 60
    check_interval_s: int = 15


class IdleDetector:
    """Detects when cluster utilization is below a threshold.

    Accepts a callable that returns the current utilization fraction
    (0.0 = idle, 1.0 = fully loaded).
    """

    def __init__(
        self,
        utilization_fn: Callable[[], float],
        config: IdleDetectorConfig | None = None,
    ):
        self._utilization_fn = utilization_fn
        self._config = config or IdleDetectorConfig()
        self._idle_since: float | None = None

    @property
    def is_idle(self) -> bool:
        """True when utilization has been below threshold long enough."""
        now = time.monotonic()
        try:
            util = self._utilization_fn()
        except Exception:
            self._idle_since = None
            return False
        below = util < (self._config.utilization_threshold_pct / 100.0)

        if below:
            if self._idle_since is None:
                self._idle_since = now
            return (now - self._idle_since) >= self._config.idle_duration_s
        else:
            self._idle_since = None
            return False

    @property
    def idle_duration(self) -> float:
        """Seconds since utilization first dropped below threshold (0 if busy)."""
        if self._idle_since is None:
            return 0.0
        return time.monotonic() - self._idle_since

    def reset(self) -> None:
        """Force reset idle state (called after a compression job starts)."""
        self._idle_since = None


@dataclass
class CompressionJob:
    """Metadata about a completed or in-progress compression job."""
    model_name: str
    model_path: str
    compressed_path: str
    method: str
    started_at: float
    finished_at: float | None = None
    success: bool = False
    error: str | None = None


class SimpleCompressor:
    """Runs model compression using the existing quantization infrastructure.

    Loads the model with quantization applied (e.g. BitsAndBytes) and saves
    the compressed version to a designated output directory.
    """

    def __init__(
        self,
        output_base: str = "",
        method: str = "int4",
        calibration_samples: int = 128,
    ):
        self._output_base = output_base
        self._method = method
        self._calibration_samples = calibration_samples

    def compress(
        self,
        model_name: str,
        model_path: str,
        tag: str = "compressed",
    ) -> str:
        """Compress a model and return the path to the compressed version.

        Args:
            model_name: Human-readable model name (for logging).
            model_path: Path or HuggingFace ID of the model to compress.
            tag: Suffix for the output directory (e.g. "compressed").

        Returns:
            Path to the compressed model directory.
        """
        import gc
        import torch

        output_dir = os.path.join(self._output_base or "/tmp/distllm-compress",
                                  f"{os.path.basename(model_path)}-{tag}")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Compressing {model_name} ({model_path}) → {output_dir}")

        bits = 4
        if self._method in ("int8",):
            bits = 8
        use_awq = "awq" in self._method
        use_gptq = "gptq" in self._method or "gptq" in self._method

        dtype = torch.float16 if bits <= 16 else torch.float32

        if use_awq or use_gptq:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Compressed {model_name} to {output_dir}")
            except Exception as e:
                logger.error(f"Compression failed for {model_name}: {e}")
                raise
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(bits == 4),
                    load_in_8bit=(bits == 8),
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Compressed {model_name} to {output_dir} via BitsAndBytes")
            except ImportError:
                logger.warning("BitsAndBytes not available; saving model as-is")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dir


@dataclass
class AdaptiveCompressionConfig:
    """Configuration for the AdaptiveCompressionManager."""
    enabled: bool = True
    idle_threshold_pct: float = 30.0
    idle_duration_s: int = 60
    check_interval_s: int = 15
    compression_method: str = "int4"
    calibration_samples: int = 128
    output_dir: str = "/tmp/distllm-compress"


class AdaptiveCompressionManager:
    """Orchestrates compression during idle and swaps to compressed variants.

    Background thread monitors cluster utilization. When idle for a minimum
    duration, picks a loaded model (via hot-swap manager), compresses it,
    and registers the compressed variant. When load increases, the caller
    can swap to the compressed variant for higher throughput.
    """

    def __init__(
        self,
        config: AdaptiveCompressionConfig | None = None,
        utilization_fn: Callable[[], float] | None = None,
        hot_swap_mgr: Any | None = None,
        compressor: SimpleCompressor | None = None,
        on_compression_complete: Callable[[CompressionJob], None] | None = None,
    ):
        self._config = config or AdaptiveCompressionConfig()
        self._utilization_fn = utilization_fn or (lambda: 0.0)
        self._hot_swap_mgr = hot_swap_mgr
        self._compressor = compressor or SimpleCompressor(
            output_base=self._config.output_dir,
            method=self._config.compression_method,
            calibration_samples=self._config.calibration_samples,
        )
        self._on_compression_complete = on_compression_complete

        self._idle_detector = IdleDetector(
            utilization_fn=self._utilization_fn,
            config=IdleDetectorConfig(
                utilization_threshold_pct=self._config.idle_threshold_pct,
                idle_duration_s=self._config.idle_duration_s,
                check_interval_s=self._config.check_interval_s,
            ),
        )

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._jobs: list[CompressionJob] = []
        self._compressing_now = False
        self._compressed_model_variants: dict[str, str] = {}

    @property
    def jobs(self) -> list[CompressionJob]:
        with self._lock:
            return list(self._jobs)

    @property
    def is_compressing(self) -> bool:
        return self._compressing_now

    @property
    def compressed_variants(self) -> dict[str, str]:
        with self._lock:
            return dict(self._compressed_model_variants)

    def _get_utilization(self) -> float:
        try:
            return self._utilization_fn()
        except Exception:
            return 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._config.enabled:
            logger.info("Adaptive compression is disabled")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="adaptive-compression",
        )
        self._thread.start()
        logger.info(
            f"Adaptive compression started "
            f"(idle <{self._config.idle_threshold_pct}% for "
            f"{self._config.idle_duration_s}s, every {self._config.check_interval_s}s)"
        )

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def _run_loop(self) -> None:
        while self._running.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Adaptive compression tick failed")
            self._running.wait(self._config.check_interval_s)

    def _tick(self) -> None:
        if self._compressing_now:
            return

        if not self._idle_detector.is_idle:
            return

        candidate = self._pick_candidate()
        if candidate is None:
            return

        self._idle_detector.reset()
        self._start_compression(*candidate)

    def _pick_candidate(self) -> tuple[str, str] | None:
        if self._hot_swap_mgr is None:
            return None

        try:
            loaded = self._hot_swap_mgr.list_loaded_models()
        except Exception:
            loaded = []

        if not loaded:
            return None

        with self._lock:
            already_compressed = set(self._compressed_model_variants.keys())

        for entry in loaded:
            name = entry.get("name", "")
            path = entry.get("path", "")
            if name and path and name not in already_compressed:
                return name, path

        return None

    def _start_compression(self, name: str, path: str) -> None:
        self._compressing_now = True
        job = CompressionJob(
            model_name=name,
            model_path=path,
            compressed_path="",
            method=self._config.compression_method,
            started_at=time.time(),
        )
        with self._lock:
            self._jobs.append(job)

        thread = threading.Thread(
            target=self._run_compression,
            args=(name, path, job),
            daemon=True,
            name=f"compress-{name}",
        )
        thread.start()

    def _run_compression(self, name: str, path: str, job: CompressionJob) -> None:
        logger.info(f"Starting compression of {name} ({path})")
        try:
            compressed_path = self._compressor.compress(name, path)
            job.compressed_path = compressed_path
            job.success = True
            job.finished_at = time.time()

            with self._lock:
                self._compressed_model_variants[name] = compressed_path

            if self._hot_swap_mgr is not None:
                try:
                    model_info = self._hot_swap_mgr.list_loaded_models()
                    layers = 32
                    for entry in model_info:
                        if entry.get("name") == name:
                            layers = entry.get("total_layers", 32)
                            break
                    variant_name = f"{name}-compressed"
                    self._hot_swap_mgr.register_model(
                        variant_name, compressed_path, layers,
                    )
                    logger.info(f"Registered compressed variant: {variant_name}")
                except Exception:
                    logger.exception("Failed to register compressed variant")

            if self._on_compression_complete:
                try:
                    self._on_compression_complete(job)
                except Exception:
                    logger.exception("Compression completion callback failed")

            logger.info(f"Compression of {name} completed: {compressed_path}")
        except Exception as e:
            job.success = False
            job.error = str(e)
            job.finished_at = time.time()
            logger.error(f"Compression of {name} failed: {e}")
        finally:
            self._compressing_now = False

    def get_compressed_path(self, model_name: str) -> str | None:
        with self._lock:
            return self._compressed_model_variants.get(model_name)
