"""Tests for distllm.dist.partition.cli module.

Zero mocks — uses only real objects from the module.
No GPU, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import pytest


# ============================================================================
# build_parser
# ============================================================================


class TestBuildParser:
    """build_parser() returns an ArgumentParser with correct defaults."""

    def test_returns_argument_parser(self) -> None:
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default_values(self) -> None:
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.hidden_size == 4096
        assert args.intermediate_size == 11008
        assert args.num_layers == 32
        assert args.num_heads == 32
        assert args.head_dim == 128
        assert args.vocab_size == 32000
        assert args.nodes is None
        assert args.gpu_counts is None
        assert args.hostnames is None
        assert args.batch_size == 1
        assert args.seq_len == 4096
        assert args.compare is False
        assert args.save is None
        assert args.load is None
        assert args.verbose is False

    def test_explicit_values(self) -> None:
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--hidden-size",
                "8192",
                "--intermediate-size",
                "28672",
                "--num-layers",
                "80",
                "--num-heads",
                "64",
                "--head-dim",
                "64",
                "--vocab-size",
                "50000",
                "--nodes",
                "node-a,node-b,node-c",
                "--gpu-counts",
                "node-a:2,node-b:4,node-c:1",
                "--hostnames",
                "node-a:10.0.0.1,node-b:10.0.0.2",
                "--batch-size",
                "4",
                "--seq-len",
                "8192",
                "--compare",
                "--save",
                "test_out.json",
                "--verbose",
            ]
        )
        assert args.hidden_size == 8192
        assert args.intermediate_size == 28672
        assert args.num_layers == 80
        assert args.num_heads == 64
        assert args.head_dim == 64
        assert args.vocab_size == 50000
        assert args.nodes == "node-a,node-b,node-c"
        assert args.gpu_counts == "node-a:2,node-b:4,node-c:1"
        assert args.hostnames == "node-a:10.0.0.1,node-b:10.0.0.2"
        assert args.batch_size == 4
        assert args.seq_len == 8192
        assert args.compare is True
        assert args.save == "test_out.json"
        assert args.verbose is True

    def test_boundary_values(self) -> None:
        """Smallest sensible values for integer arguments."""
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--hidden-size",
                "64",
                "--num-layers",
                "1",
                "--num-heads",
                "1",
                "--batch-size",
                "1",
                "--seq-len",
                "1",
            ]
        )
        assert args.hidden_size == 64
        assert args.num_layers == 1
        assert args.num_heads == 1
        assert args.batch_size == 1
        assert args.seq_len == 1

    def test_zero_values_accepted(self) -> None:
        """ArgumentParser does not reject zero values for these int args."""
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--hidden-size",
                "0",
                "--num-layers",
                "0",
                "--batch-size",
                "0",
                "--seq-len",
                "0",
            ]
        )
        assert args.hidden_size == 0
        assert args.num_layers == 0
        assert args.batch_size == 0
        assert args.seq_len == 0

    def test_negative_values_accepted(self) -> None:
        """ArgumentParser accepts negative ints (validation is in run_partitioner)."""
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--hidden-size",
                "-1",
                "--num-layers",
                "-5",
                "--batch-size",
                "-2",
            ]
        )
        assert args.hidden_size == -1
        assert args.num_layers == -5
        assert args.batch_size == -2

    def test_single_node_string(self) -> None:
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--nodes", "node-x"])
        assert args.nodes == "node-x"

    def test_empty_nodes_string(self) -> None:
        """An empty --nodes string is treated as a single empty element (truthy)."""
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--nodes", ""])
        assert args.nodes == ""

    def test_load_only_flag(self) -> None:
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--load", "plan.json"])
        assert args.load == "plan.json"
        assert args.save is None

    def test_compare_flag_alone(self) -> None:
        from distllm.dist.partition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--compare"])
        assert args.compare is True


# ============================================================================
# display_plan
# ============================================================================


class TestDisplayPlan:
    """display_plan() reads and prints a JSON partition plan."""

    def test_valid_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import display_plan

        data = _make_plan_data(
            max_node_time_ms=123.45,
            throughput=678.9,
            assignments=[("node-a", 0, 5, 50.0)],
        )
        path = _write_temp_json(data)
        try:
            display_plan(path)
            out = capsys.readouterr().out
            assert "Max node time:" in out
            assert "123.45" in out
            assert "throughput" in out.lower()
            assert "node-a" in out
            assert "layers [0, 5)" in out
        finally:
            os.unlink(path)

    def test_empty_assignments(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import display_plan

        data = _make_plan_data(assignments=[])
        path = _write_temp_json(data)
        try:
            display_plan(path)
            out = capsys.readouterr().out
            assert "Saved Partition Plan:" in out
            assert "Assignments:" in out
        finally:
            os.unlink(path)

    def test_zero_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import display_plan

        data = _make_plan_data(max_node_time_ms=0.0, throughput=0.0)
        path = _write_temp_json(data)
        try:
            display_plan(path)
            out = capsys.readouterr().out
            assert "0.0" in out
        finally:
            os.unlink(path)

    def test_multiple_assignments(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import display_plan

        data = _make_plan_data(
            assignments=[
                ("node-a", 0, 16, 100.0),
                ("node-b", 16, 32, 120.0),
            ],
        )
        path = _write_temp_json(data)
        try:
            display_plan(path)
            out = capsys.readouterr().out
            assert "node-a" in out
            assert "node-b" in out
            assert "16, 32" in out
        finally:
            os.unlink(path)

    def test_missing_file_raises(self) -> None:
        from distllm.dist.partition.cli import display_plan

        with pytest.raises(FileNotFoundError):
            display_plan("/tmp/__nonexistent_plan_test_file_xyz.json")

    def test_invalid_json_raises(self) -> None:
        from distllm.dist.partition.cli import display_plan

        path = _write_temp_json_text("{invalid json}")
        try:
            with pytest.raises(json.JSONDecodeError):
                display_plan(path)
        finally:
            os.unlink(path)


# ============================================================================
# run_partitioner  (async, no GPU required)
# ============================================================================


class TestRunPartitioner:
    """run_partitioner() async entry point — uses real objects end to end."""

    async def test_default_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args([])
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert out.strip(), "Expected some output from the partitioner"

    async def test_with_compare(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(["--compare"])
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert "Strategy comparison" in out

    async def test_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(["--verbose"])
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert out.strip(), "Expected output even with verbose mode"

    async def test_save_plan(self) -> None:
        from distllm.dist.partition.cli import build_parser, run_partitioner

        save_path = tempfile.mktemp(suffix=".json")
        try:
            args = build_parser().parse_args(["--save", save_path])
            await run_partitioner(args)
            assert os.path.isfile(save_path)
            with open(save_path) as f:
                data = json.load(f)
            assert "solution" in data
            assert "config" in data
            assert "max_node_time_ms" in data["solution"]
            assert "assignments" in data["solution"]
        finally:
            if os.path.isfile(save_path):
                os.unlink(save_path)

    async def test_save_with_compare(self) -> None:
        from distllm.dist.partition.cli import build_parser, run_partitioner

        save_path = tempfile.mktemp(suffix=".json")
        try:
            args = build_parser().parse_args(
                ["--save", save_path, "--compare"]
            )
            await run_partitioner(args)
            assert os.path.isfile(save_path)
        finally:
            if os.path.isfile(save_path):
                os.unlink(save_path)

    async def test_custom_model_params(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Small model config — fewer layers, smaller hidden size."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--hidden-size",
                "2048",
                "--intermediate-size",
                "8192",
                "--num-layers",
                "4",
                "--num-heads",
                "8",
                "--batch-size",
                "2",
                "--seq-len",
                "1024",
            ]
        )
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert out.strip()

    async def test_explicit_single_node(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Explicit node name flows through to output."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--nodes",
                "worker-1",
                "--gpu-counts",
                "worker-1:1",
            ]
        )
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert "worker-1" in out

    async def test_explicit_two_nodes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Two nodes triggers the DP multi-node path."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--nodes",
                "alpha,beta",
                "--gpu-counts",
                "alpha:1,beta:1",
                "--hostnames",
                "alpha:127.0.0.1,beta:127.0.0.1",
                "--num-layers",
                "8",
                "--hidden-size",
                "1024",
            ]
        )
        await run_partitioner(args)
        out = capsys.readouterr().out
        # Should mention both nodes in summaries
        assert "alpha" in out and "beta" in out

    async def test_explicit_hostnames(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--hostnames parameter is parsed and forwarded correctly."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--nodes",
                "n1,n2",
                "--gpu-counts",
                "n1:1,n2:2",
                "--hostnames",
                "n1:10.0.0.1,n2:10.0.0.2",
                "--num-layers",
                "4",
                "--hidden-size",
                "512",
            ]
        )
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert "n1" in out
        assert "n2" in out

    async def test_invalid_gpu_counts_format_raises(self) -> None:
        """Missing colon in gpu-counts raises ValueError during parse."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--nodes",
                "worker-1",
                "--gpu-counts",
                "worker-1",
            ]
        )
        with pytest.raises(ValueError):
            await run_partitioner(args)

    async def test_invalid_hostnames_format_raises(self) -> None:
        """Missing colon in hostnames raises ValueError during parse."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--nodes",
                "n1",
                "--gpu-counts",
                "n1:1",
                "--hostnames",
                "n1",
            ]
        )
        with pytest.raises(ValueError):
            await run_partitioner(args)

    async def test_oom_zero_memory_gpu(self, capsys: pytest.CaptureFixture[str]) -> None:
        """GPU with zero memory is handled gracefully (falls to CPU path)."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--nodes",
                "gpu-zero",
                "--gpu-counts",
                "gpu-zero:0",
                "--num-layers",
                "2",
                "--hidden-size",
                "256",
            ]
        )
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert out.strip()

    async def test_low_layer_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Single layer is the minimum reasonable input."""
        from distllm.dist.partition.cli import build_parser, run_partitioner

        args = build_parser().parse_args(
            [
                "--num-layers",
                "1",
                "--hidden-size",
                "256",
                "--intermediate-size",
                "512",
            ]
        )
        await run_partitioner(args)
        out = capsys.readouterr().out
        assert out.strip()


# ============================================================================
# main
# ============================================================================


class TestMain:
    """main() entry point."""

    def test_is_callable(self) -> None:
        from distllm.dist.partition.cli import main

        assert callable(main)

    def test_with_load_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main() with --load should display the plan and return early."""
        from distllm.dist.partition.cli import main

        data = _make_plan_data(
            max_node_time_ms=42.0,
            throughput=100.0,
            assignments=[("n0", 0, 10, 42.0)],
        )
        path = _write_temp_json(data)
        old_argv = sys.argv
        try:
            sys.argv = ["test_prog", "--load", path]
            main()
            out = capsys.readouterr().out
            assert "Saved Partition Plan:" in out
            assert "42.0" in out
        finally:
            sys.argv = old_argv
            os.unlink(path)

    def test_with_load_missing_file_raises(self) -> None:
        from distllm.dist.partition.cli import main

        old_argv = sys.argv
        try:
            sys.argv = [
                "test_prog",
                "--load",
                "/tmp/__missing_plan_test_file.json",
            ]
            with pytest.raises(FileNotFoundError):
                main()
        finally:
            sys.argv = old_argv


# ============================================================================
# helpers
# ============================================================================


def _make_plan_data(
    max_node_time_ms: float = 100.0,
    throughput: float = 500.0,
    assignments: list[tuple[str, int, int, float]] | None = None,
) -> dict:
    """Build a minimal partition-plan dictionary matching the CLI schema."""
    if assignments is None:
        assignments = [("node-0", 0, 10, 50.0)]
    return {
        "solution": {
            "max_node_time_ms": max_node_time_ms,
            "estimated_throughput_tok_s": throughput,
            "assignments": [
                {
                    "node_id": n,
                    "start_layer": s,
                    "end_layer": e,
                    "estimated_time_ms": t,
                }
                for n, s, e, t in assignments
            ],
        },
        "config": {
            "hidden_size": 4096,
            "num_layers": 32,
            "batch_size": 1,
            "seq_len": 4096,
        },
    }


def _write_temp_json(data: dict) -> str:
    """Write *data* to a temp JSON file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _write_temp_json_text(text: str) -> str:
    """Write raw *text* to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        f.write(text)
    return path
