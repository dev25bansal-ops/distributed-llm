"""Tests for distllm.dist.partition.quant_bench — zero mocks, GPU-agnostic."""

from __future__ import annotations

import time
from dataclasses import asdict

import pytest

from distllm.dist.partition.quant_bench import (
    QuantBenchmarker,
    QuantBenchmarkResult,
    QuantBenchmarkSuite,
)


# ── QuantBenchmarkResult ──────────────────────────────────────────────────────


class TestQuantBenchmarkResult:
    """Dataclass holding one method's benchmark outcome."""

    def test_defaults(self) -> None:
        r = QuantBenchmarkResult(method="test", gpu_id=0, gpu_name="A100")
        assert r.method == "test"
        assert r.gpu_id == 0
        assert r.gpu_name == "A100"
        assert r.matmul_tflops == 0.0
        assert r.speedup_vs_fp16 == 1.0
        assert r.memory_reduction_actual == 1.0
        assert r.kernel_available is False
        assert r.dequant_overhead_ms == 0.0
        assert r.benchmark_time_s == 0.0
        assert r.error == ""

    def test_all_fields_explicit(self) -> None:
        r = QuantBenchmarkResult(
            method="bnb_8bit",
            gpu_id=1,
            gpu_name="H100",
            matmul_tflops=125.0,
            speedup_vs_fp16=1.8,
            memory_reduction_actual=0.5,
            kernel_available=True,
            dequant_overhead_ms=0.12,
            benchmark_time_s=2.3,
            error="",
        )
        assert r.matmul_tflops == 125.0
        assert r.speedup_vs_fp16 == 1.8
        assert r.memory_reduction_actual == 0.5
        assert r.kernel_available is True
        assert r.dequant_overhead_ms == 0.12
        assert r.benchmark_time_s == 2.3
        assert r.error == ""
        assert r.method == "bnb_8bit"
        assert r.gpu_id == 1
        assert r.gpu_name == "H100"

    def test_edge_empty_method_name(self) -> None:
        r = QuantBenchmarkResult(method="", gpu_id=-1, gpu_name="")
        assert r.method == ""
        assert r.gpu_id == -1
        assert r.gpu_name == ""

    def test_error_string_non_empty(self) -> None:
        r = QuantBenchmarkResult(
            method="test", gpu_id=0, gpu_name="GPU-0", error="CUDA OOM"
        )
        assert r.error == "CUDA OOM"

    def test_negative_tflops(self) -> None:
        """Negative tflops should be storable (edge data)."""
        r = QuantBenchmarkResult(
            method="test", gpu_id=0, gpu_name="GPU", matmul_tflops=-1.0
        )
        assert r.matmul_tflops == -1.0

    def test_is_dataclass(self) -> None:
        r = QuantBenchmarkResult(method="x", gpu_id=0, gpu_name="y")
        d = asdict(r)
        assert isinstance(d, dict)
        assert d["method"] == "x"


# ── QuantBenchmarkSuite ───────────────────────────────────────────────────────


class TestQuantBenchmarkSuite:
    """Suite-level aggregation, property, summary, and profile conversion."""

    def test_empty_suite(self) -> None:
        s = QuantBenchmarkSuite(gpu_id=0, gpu_name="A100")
        assert s.gpu_id == 0
        assert s.gpu_name == "A100"
        assert s.fp16_tflops == 0.0
        assert s.results == {}
        assert isinstance(s.timestamp, float)
        assert s.timestamp > 0.0

    def test_best_method_empty(self) -> None:
        s = QuantBenchmarkSuite(gpu_id=1, gpu_name="H100")
        assert s.best_method == "none"

    def test_best_method_single(self) -> None:
        r = QuantBenchmarkResult(
            method="bnb_8bit",
            gpu_id=0,
            gpu_name="A100",
            speedup_vs_fp16=1.5,
            matmul_tflops=40.0,
        )
        s = QuantBenchmarkSuite(gpu_id=0, gpu_name="A100", results={"bnb_8bit": r})
        assert s.best_method == "bnb_8bit"

    def test_best_method_picks_max_speedup(self) -> None:
        r1 = QuantBenchmarkResult(
            method="bnb_4bit",
            gpu_id=0,
            gpu_name="A100",
            speedup_vs_fp16=2.0,
            matmul_tflops=55.0,
        )
        r2 = QuantBenchmarkResult(
            method="bnb_8bit",
            gpu_id=0,
            gpu_name="A100",
            speedup_vs_fp16=1.5,
            matmul_tflops=40.0,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0,
            gpu_name="A100",
            results={"bnb_4bit": r1, "bnb_8bit": r2},
        )
        assert s.best_method == "bnb_4bit"

    def test_best_method_tie_first_inserted(self) -> None:
        """max() returns the first key encountered on tie (dict insertion order)."""
        r_a = QuantBenchmarkResult(
            method="alpha",
            gpu_id=0,
            gpu_name="GPU",
            speedup_vs_fp16=1.0,
        )
        r_z = QuantBenchmarkResult(
            method="zeta",
            gpu_id=0,
            gpu_name="GPU",
            speedup_vs_fp16=1.0,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0,
            gpu_name="GPU",
            results={"alpha": r_a, "zeta": r_z},
        )
        assert s.best_method == "alpha"

    def test_summary_empty(self) -> None:
        s = QuantBenchmarkSuite(gpu_id=0, gpu_name="A100", fp16_tflops=50.0)
        text = s.summary()
        assert "GPU 0 (A100)" in text
        assert "FP16 TFLOPS: 50.0" in text

    def test_summary_with_results(self) -> None:
        r = QuantBenchmarkResult(
            method="bnb_8bit",
            gpu_id=0,
            gpu_name="A100",
            matmul_tflops=40.0,
            speedup_vs_fp16=1.5,
            memory_reduction_actual=0.5,
            kernel_available=True,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0,
            gpu_name="A100",
            fp16_tflops=50.0,
            results={"bnb_8bit": r},
        )
        text = s.summary()
        assert "bnb_8bit" in text
        assert "40.0 TFLOPS" in text
        assert "1.50x vs fp16" in text
        assert "available" in text

    def test_summary_unavailable(self) -> None:
        r = QuantBenchmarkResult(
            method="fp8_e4m3",
            gpu_id=0,
            gpu_name="GPU",
            kernel_available=False,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0, gpu_name="GPU", results={"fp8_e4m3": r}
        )
        assert "unavailable" in s.summary()

    def test_summary_sorts_methods(self) -> None:
        r2 = QuantBenchmarkResult(
            method="z_method",
            gpu_id=0,
            gpu_name="GPU",
            matmul_tflops=10.0,
            speedup_vs_fp16=1.0,
            kernel_available=True,
        )
        r1 = QuantBenchmarkResult(
            method="a_method",
            gpu_id=0,
            gpu_name="GPU",
            matmul_tflops=20.0,
            speedup_vs_fp16=2.0,
            kernel_available=True,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0,
            gpu_name="GPU",
            results={"z_method": r2, "a_method": r1},
        )
        text = s.summary()
        # a_method should appear before z_method (sorted)
        a_pos = text.index("a_method")
        z_pos = text.index("z_method")
        assert a_pos < z_pos

    def test_to_quant_profiles_empty(self) -> None:
        s = QuantBenchmarkSuite(gpu_id=0, gpu_name="GPU")
        assert s.to_quant_profiles() == {}

    def test_to_quant_profiles_filters_unavailable(self) -> None:
        r_avail = QuantBenchmarkResult(
            method="bnb_8bit",
            gpu_id=0,
            gpu_name="GPU",
            speedup_vs_fp16=2.0,
            memory_reduction_actual=0.5,
            kernel_available=True,
        )
        r_unavail = QuantBenchmarkResult(
            method="fp8_e4m3",
            gpu_id=0,
            gpu_name="GPU",
            kernel_available=False,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0,
            gpu_name="GPU",
            results={"bnb_8bit": r_avail, "fp8_e4m3": r_unavail},
        )
        profiles = s.to_quant_profiles()
        assert "bnb_8bit" in profiles
        assert "fp8_e4m3" not in profiles
        assert profiles["bnb_8bit"]["speed_penalty"] == pytest.approx(0.5)
        assert profiles["bnb_8bit"]["memory_reduction"] == 0.5

    def test_to_quant_profiles_penalty_clamp_low_speedup(self) -> None:
        """speed_penalty is clamped: 1.0 / max(speedup, 0.01)."""
        r = QuantBenchmarkResult(
            method="slow",
            gpu_id=0,
            gpu_name="GPU",
            speedup_vs_fp16=0.0,
            memory_reduction_actual=1.0,
            kernel_available=True,
        )
        s = QuantBenchmarkSuite(
            gpu_id=0, gpu_name="GPU", results={"slow": r}
        )
        profiles = s.to_quant_profiles()
        # max(0.0, 0.01) = 0.01  ->  1.0 / 0.01 = 100.0
        assert profiles["slow"]["speed_penalty"] == 100.0

    def test_timestamp_approximates_now(self) -> None:
        before = time.time()
        s = QuantBenchmarkSuite(gpu_id=0, gpu_name="GPU")
        after = time.time()
        assert before <= s.timestamp <= after


# ── QuantBenchmarker ──────────────────────────────────────────────────────────


class TestQuantBenchmarker:
    """Benchmarker — tests are GPU-agnostic (pass with or without CUDA).

    Assertions focus on structural properties (types, shapes, invariants)
    rather than specific numeric values which vary by hardware.
    """

    # -- helper ----------------------------------------------------------------

    @staticmethod
    def maker() -> QuantBenchmarker:
        return QuantBenchmarker()

    # -- benchmark_gpu ---------------------------------------------------------

    def test_benchmark_gpu_returns_suite_type(self) -> None:
        """benchmark_gpu always returns a QuantBenchmarkSuite."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=0)
        assert isinstance(suite, QuantBenchmarkSuite)
        assert suite.gpu_id == 0

    def test_benchmark_gpu_contains_all_methods(self) -> None:
        """All three methods are always present in the results dict."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=0)
        for method in ("bnb_8bit", "bnb_4bit", "fp8_e4m3"):
            assert method in suite.results
            r = suite.results[method]
            assert isinstance(r, QuantBenchmarkResult)
            assert r.method == method
            assert r.gpu_id == 0

    def test_benchmark_gpu_non_negative_benchmark_time(self) -> None:
        """benchmark_time_s should be >= 0 for every result."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=0)
        for r in suite.results.values():
            assert r.benchmark_time_s >= 0.0

    def test_benchmark_gpu_fp16_tflops_is_float(self) -> None:
        """fp16_tflops is always a float (may be 0 without GPU, positive with)."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=0)
        assert isinstance(suite.fp16_tflops, float)
        assert suite.fp16_tflops >= 0.0

    def test_benchmark_gpu_gpu_name_is_string(self) -> None:
        """gpu_name is always a non-empty string."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=0)
        assert isinstance(suite.gpu_name, str)
        assert len(suite.gpu_name) > 0

    def test_benchmark_gpu_negative_id(self) -> None:
        """A negative GPU id should still produce a valid suite."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=-1)
        assert isinstance(suite, QuantBenchmarkSuite)
        assert suite.gpu_id == -1
        for r in suite.results.values():
            assert r.gpu_id == -1

    def test_benchmark_gpu_large_gpu_id(self) -> None:
        """A large GPU id should produce a suite (may or may not have real GPU)."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=999)
        assert isinstance(suite, QuantBenchmarkSuite)
        assert suite.gpu_id == 999

    def test_benchmark_gpu_called_twice_independent(self) -> None:
        """Calling benchmark_gpu twice yields separate suite objects."""
        bench = self.maker()
        s1 = bench.benchmark_gpu(0)
        s2 = bench.benchmark_gpu(0)
        assert s1 is not s2

    def test_benchmark_gpu_result_methods_match_suite_gpu_id(self) -> None:
        """Result objects inside the suite reference the same GPU id."""
        bench = self.maker()
        suite = bench.benchmark_gpu(gpu_id=2)
        assert suite.gpu_id == 2
        for r in suite.results.values():
            assert r.gpu_id == 2

    # -- benchmark_all_gpus ----------------------------------------------------

    def test_benchmark_all_gpus_returns_list(self) -> None:
        """benchmark_all_gpus always returns a list."""
        bench = self.maker()
        suites = bench.benchmark_all_gpus()
        assert isinstance(suites, list)
        # Every element is a QuantBenchmarkSuite
        for s in suites:
            assert isinstance(s, QuantBenchmarkSuite)

    def test_benchmark_all_gpus_results_have_unique_gpu_ids(self) -> None:
        """Each suite in the list has a distinct gpu_id (if any)."""
        bench = self.maker()
        suites = bench.benchmark_all_gpus()
        gpu_ids = [s.gpu_id for s in suites]
        assert len(gpu_ids) == len(set(gpu_ids))

    # -- _get_gpu_name ---------------------------------------------------------

    def test_get_gpu_name_returns_string(self) -> None:
        bench = self.maker()
        name = bench._get_gpu_name(0)  # noqa: SLF001
        assert isinstance(name, str)
        assert len(name) > 0

    # -- _device_count ---------------------------------------------------------

    def test_device_count_is_int(self) -> None:
        bench = self.maker()
        count = bench._device_count()  # noqa: SLF001
        assert isinstance(count, int)
        assert count >= 0

    # -- _get_device -----------------------------------------------------------

    def test_get_device_or_none(self) -> None:
        bench = self.maker()
        dev = bench._get_device(0)  # noqa: SLF001
        # Returns None when no GPU, or a torch.device when CUDA is available
        import torch  # noqa: F811

        assert dev is None or isinstance(dev, torch.device)

    # -- _estimate_memory_reduction --------------------------------------------

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("bnb_8bit", 0.5),
            ("bnb_4bit", 0.25),
            ("gptq", 0.25),
            ("awq", 0.25),
            ("fp8_e4m3", 0.5),
            ("fp8_e5m2", 0.5),
            ("int8", 0.5),
            ("nf4", 0.25),
            ("unknown_method", 1.0),
            ("", 1.0),
        ],
    )
    def test_estimate_memory_reduction(self, method: str, expected: float) -> None:
        bench = self.maker()
        assert bench._estimate_memory_reduction(method) == expected  # noqa: SLF001

    # -- _make_timer / _elapsed ------------------------------------------------

    def test_make_timer_with_none_device(self) -> None:
        """Passing None returns time-based timer."""
        bench = self.maker()
        timer = bench._make_timer(None)  # noqa: SLF001
        assert isinstance(timer, tuple)
        assert timer[0] == "time"
        assert isinstance(timer[1], float)

    def test_elapsed_with_none_device_non_negative(self) -> None:
        bench = self.maker()
        start = bench._make_timer(None)  # noqa: SLF001
        elapsed = bench._elapsed(start, None)  # noqa: SLF001
        assert elapsed >= 0.0

    def test_elapsed_with_none_device_advances(self) -> None:
        """Verify elapsed increases after a tiny delay."""
        bench = self.maker()
        start = time.time()
        _ = time.sleep(0.01)
        elapsed = bench._elapsed(("time", start), None)  # noqa: SLF001
        assert elapsed >= 0.005

    def test_make_timer_with_unknown_device_type(self) -> None:
        """A device with type != 'cuda' returns ('time', float)."""
        bench = self.maker()
        # Simulate a device-like object with type 'cpu'
        class FakeDevice:
            type = "cpu"

        timer = bench._make_timer(FakeDevice())  # noqa: SLF001
        assert timer[0] == "time"
        assert isinstance(timer[1], float)

    # -- _synchronize ----------------------------------------------------------

    def test_synchronize_none_device(self) -> None:
        bench = self.maker()
        bench._synchronize(None)  # noqa: SLF001

    def test_synchronize_string_device(self) -> None:
        """Passing a non-device object should not raise."""
        bench = self.maker()
        bench._synchronize("cpu")  # noqa: SLF001

    # -- internal bench methods (structural checks) ----------------------------

    def test_bench_fp16_matmul_returns_float(self) -> None:
        bench = self.maker()
        result = bench._bench_fp16_matmul(0)  # noqa: SLF001
        assert isinstance(result, float)

    def test_bench_int8_matmul_returns_tuple(self) -> None:
        bench = self.maker()
        result = bench._bench_int8_matmul(0)  # noqa: SLF001
        assert isinstance(result, tuple)
        assert len(result) == 2
        tflops, available = result
        assert isinstance(tflops, float)
        assert isinstance(available, bool)

    def test_bench_int4_matmul_returns_tuple(self) -> None:
        bench = self.maker()
        result = bench._bench_int4_matmul(0)  # noqa: SLF001
        assert isinstance(result, tuple)
        assert len(result) == 2
        tflops, available = result
        assert isinstance(tflops, float)
        assert isinstance(available, bool)

    def test_bench_fp8_matmul_returns_tuple(self) -> None:
        bench = self.maker()
        result = bench._bench_fp8_matmul(0)  # noqa: SLF001
        assert isinstance(result, tuple)
        assert len(result) == 2
        tflops, available = result
        assert isinstance(tflops, float)
        assert isinstance(available, bool)

    # -- edge / error cases ----------------------------------------------------

    def test_benchmark_gpu_suite_has_timestamp(self) -> None:
        bench = self.maker()
        suite = bench.benchmark_gpu(0)
        assert isinstance(suite.timestamp, float)
        assert suite.timestamp > 0.0

    def test_benchmark_gpu_result_error_field(self) -> None:
        """error field is always a string."""
        bench = self.maker()
        suite = bench.benchmark_gpu(0)
        for r in suite.results.values():
            assert isinstance(r.error, str)

    def test_benchmark_gpu_result_kernel_available_is_bool(self) -> None:
        """kernel_available is always a boolean."""
        bench = self.maker()
        suite = bench.benchmark_gpu(0)
        for r in suite.results.values():
            assert isinstance(r.kernel_available, bool)
