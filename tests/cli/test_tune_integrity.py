"""Output-integrity tests for ``distllm tune batch``.

Defect: when every probe failed (no server on the target port, bad API key),
``run_tune_batch`` still exited 0, wrote a results file containing literal
``Infinity`` (invalid JSON per RFC 8259), and printed a recommendation table
built from a fabricated measurement. These tests pin the fixed contract:

* all probes fail -> SystemExit(1), no output file, clear diagnostic;
* mixed results   -> output file is strictly valid JSON (no Infinity/NaN),
  failed probes are null, and the recommendation comes only from real samples.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _strict_loads(text: str):
    """json.loads that rejects Infinity/-Infinity/NaN (RFC 8259 strict)."""

    def _reject(value):  # parse_constant hook
        raise ValueError(f"non-finite constant in JSON: {value}")

    return json.loads(text, parse_constant=_reject)


def _fake_client(*, fail: bool, latency_s: float = 0.02):
    """Build a patched httpx.Client whose posts either succeed or fail."""

    def post(url, **kwargs):
        if fail:
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(status_code=401),
            )
            return resp
        return MagicMock()  # raise_for_status() is a no-op

    inst = MagicMock()
    inst.post.side_effect = post
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=inst)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _run(console, output, max_batch=2):
    from distllm.cli.tune import run_tune_batch

    run_tune_batch(
        model="gpt2",
        gpu_count=1,
        target_latency_ms=500.0,
        host="localhost",
        port=59999,
        max_batch=max_batch,
        max_tokens=8192,
        output=output,
        console=console,
    )


class TestTuneBatchAllProbesFail:
    def test_all_probes_fail_exits_nonzero_without_output_file(self, tmp_path):
        out = tmp_path / "tune.json"
        console = MagicMock()

        with patch("httpx.Client", return_value=_fake_client(fail=True)):
            with pytest.raises(SystemExit) as exc:
                _run(console, str(out))

        assert exc.value.code == 1
        assert not out.exists(), "a failed tuning run must not write a results file"

        printed = " ".join(
            str(c.args[0]) for c in console.print.call_args_list if c.args
        )
        assert "all probes failed" in printed.lower()
        assert "localhost:59999" in printed
        # No recommendation table on the failure path. The table is printed as
        # a rich Table object (whose str() is just a repr), so assert on the
        # object type, not on rendered text.
        from rich.table import Table

        assert not any(
            isinstance(c.args[0], Table)
            for c in console.print.call_args_list
            if c.args
        ), "failure path must not print a recommendation table"
        assert "Results saved" not in printed

    def test_all_probes_connection_refused_exits_nonzero(self, tmp_path):
        out = tmp_path / "tune-conn.json"
        console = MagicMock()

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(side_effect=httpx.ConnectError("refused"))
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client", return_value=ctx):
            with pytest.raises(SystemExit) as exc:
                _run(console, str(out))

        assert exc.value.code == 1
        assert not out.exists()


class TestTuneBatchMixedResults:
    def _mixed_client(self):
        """Batch size 1 succeeds, batch size 2 fails (401)."""
        calls = {"n": 0}

        def post(url, **kwargs):
            # 3 samples per batch size; max_batch=2 probes bs=1 then bs=2.
            calls["n"] += 1
            if calls["n"] <= 3:
                return MagicMock()  # raise_for_status() no-op -> success
            resp = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(status_code=401),
            )
            return resp

        inst = MagicMock()
        inst.post.side_effect = post
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=inst)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_mixed_results_valid_strict_json_and_null_failed_probes(self, tmp_path):
        out = tmp_path / "tune-mixed.json"
        console = MagicMock()

        with patch("httpx.Client", return_value=self._mixed_client()):
            _run(console, str(out))

        assert out.exists()
        data = _strict_loads(out.read_text(encoding="utf-8"))

        probes = {p["batch_size"]: p for p in data["probe_results"]}
        assert probes[1]["avg_latency_ms"] is not None
        assert probes[2]["avg_latency_ms"] is None
        assert probes[2]["throughput_tok_s"] is None
        assert probes[2]["errors"] == 3

        # Recommendation must come from the only real sample.
        assert data["optimal_batch_size"] == 1
        assert data["expected_latency_ms"] is not None
        assert data["expected_throughput_tok_s"] is not None
        assert data["expected_latency_ms"] == probes[1]["avg_latency_ms"]

    def test_mixed_results_success_path_shape_unchanged(self, tmp_path):
        out = tmp_path / "tune-ok.json"
        console = MagicMock()

        with patch("httpx.Client", return_value=_fake_client(fail=False)):
            _run(console, str(out), max_batch=4)

        data = _strict_loads(out.read_text(encoding="utf-8"))
        assert set(data) == {
            "model", "gpu_count", "target_latency_ms", "optimal_batch_size",
            "recommended_max_tokens_per_batch", "expected_throughput_tok_s",
            "expected_latency_ms", "probe_results",
        }
        assert [p["batch_size"] for p in data["probe_results"]] == [1, 2, 4]
        assert all(p["avg_latency_ms"] is not None for p in data["probe_results"])
        assert all(p["errors"] == 0 for p in data["probe_results"])
        # Default target latency (500 ms) is met by the fake ~20 ms probes:
        # the highest-throughput viable probe (largest batch size) wins.
        assert data["optimal_batch_size"] == 4
