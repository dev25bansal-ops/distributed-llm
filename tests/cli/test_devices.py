"""Tests for the ``distllm devices`` hardware discovery command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from distllm.cli.devices import (
    _cpu_info,
    _format_bytes,
    _memory_total_available,
    _network_interfaces,
    _safe_run,
    _topology_info,
    print_devices,
    run_devices,
)
from distllm.cli.main import app


runner = CliRunner()


class TestRunDevices:
    """``run_devices`` returns a fully-populated topology dict."""

    def test_returns_required_keys(self):
        result = run_devices()
        assert isinstance(result, dict)
        for key in ("hostname", "platform", "gpus", "cpu", "memory", "network", "topology"):
            assert key in result, f"missing key: {key}"

    def test_platform_metadata(self):
        result = run_devices()
        plat = result["platform"]
        assert "system" in plat
        assert "release" in plat
        assert "python" in plat

    def test_cpu_block_shape(self):
        result = run_devices()
        cpu = result["cpu"]
        assert "count" in cpu
        assert "architecture" in cpu
        assert "numa_nodes" in cpu
        assert isinstance(cpu["count"], int)
        assert cpu["count"] >= 1
        assert isinstance(cpu["numa_nodes"], int)
        assert cpu["numa_nodes"] >= 1

    def test_memory_block_shape(self):
        result = run_devices()
        mem = result["memory"]
        assert "total_bytes" in mem
        assert "available_bytes" in mem
        # At least one of the two backends (Linux/Windows) should populate
        # total_bytes.  If the host is neither, both are 0 and we just verify
        # the keys exist.
        assert isinstance(mem["total_bytes"], int)
        assert isinstance(mem["available_bytes"], int)

    def test_gpus_is_list(self):
        result = run_devices()
        assert isinstance(result["gpus"], list)

    def test_network_is_list(self):
        result = run_devices()
        assert isinstance(result["network"], list)

    def test_topology_block(self):
        result = run_devices()
        topo = result["topology"]
        assert "nvlink" in topo
        assert "pcie_links" in topo
        assert isinstance(topo["nvlink"], bool)


class TestSafeRun:
    def test_returns_empty_on_missing_binary(self):
        assert _safe_run(["definitely-not-a-real-binary-xyz123"]) == ""

    def test_returns_stdout_on_success(self):
        # python -c prints to stdout; should return the value.
        out = _safe_run(["python", "-c", "print('hello')"])
        assert "hello" in out

    def test_handles_timeout_gracefully(self):
        # Sleep longer than the default timeout.
        out = _safe_run(["python", "-c", "import time; time.sleep(10)"], timeout=0.1)
        assert out == ""


class TestFormatBytes:
    def test_known_values(self):
        assert _format_bytes(0) == "n/a"
        # 1 GiB = 1073741824 bytes
        s = _format_bytes(1024 ** 3)
        assert "GiB" in s
        assert "1.0" in s
        # 512 MiB
        s = _format_bytes(512 * 1024 * 1024)
        assert "MiB" in s
        assert "512.0" in s


class TestCpuInfo:
    def test_returns_basic_shape(self):
        cpu = _cpu_info()
        assert cpu["count"] >= 1
        assert cpu["architecture"] != "unknown"
        assert cpu["numa_nodes"] >= 1


class TestNetworkInterfaces:
    def test_returns_list(self):
        net = _network_interfaces()
        assert isinstance(net, list)
        assert len(net) >= 1
        for iface in net:
            assert "name" in iface
            assert "ipv4" in iface
            assert "up" in iface


class TestTopologyInfo:
    def test_returns_expected_keys(self):
        topo = _topology_info()
        assert "nvlink" in topo
        assert "pcie_links" in topo


class TestPrintDevices:
    """``print_devices`` should not raise on any reasonable environment."""

    def test_prints_no_cuda_env(self):
        from rich.console import Console

        result = run_devices()
        # Just verify it doesn't raise; rich Console renders to a stream we
        # don't capture here.
        print_devices(result, Console(record=True, width=120))

    def test_handles_empty_gpus(self):
        """An empty GPU list shouldn't crash the printer."""
        from rich.console import Console

        fake = {
            "hostname": "test-host",
            "platform": {"system": "Linux", "release": "test", "python": "3.11"},
            "gpus": [],
            "cpu": {"count": 4, "architecture": "x86_64", "numa_nodes": 1, "model": "test"},
            "memory": {"total_bytes": 0, "available_bytes": 0},
            "network": [],
            "topology": {"nvlink": False, "pcie_links": []},
        }
        console = Console(record=True, width=120)
        print_devices(fake, console)
        output = console.export_text()
        assert "test-host" in output
        assert "GPUs" in output

    def test_handles_populated_topology(self):
        from rich.console import Console

        fake = {
            "hostname": "gpu-host",
            "platform": {"system": "Linux", "release": "5.15", "python": "3.11"},
            "gpus": [{
                "index": 0,
                "name": "NVIDIA H100",
                "vram_bytes": 80 * 1024 ** 3,
                "cuda_version": "12.4",
                "driver_version": "550.54",
                "compute_capability": "9.0",
            }],
            "cpu": {"count": 32, "architecture": "x86_64", "numa_nodes": 2, "model": "AMD EPYC"},
            "memory": {"total_bytes": 256 * 1024 ** 3, "available_bytes": 128 * 1024 ** 3},
            "network": [{"name": "eth0", "ipv4": "10.0.0.1", "mac": "aa:bb:cc:dd:ee:ff", "up": True}],
            "topology": {"nvlink": True, "pcie_links": ["NVIDIA PCIe devices: 4"]},
        }
        console = Console(record=True, width=120)
        print_devices(fake, console)
        output = console.export_text()
        assert "gpu-host" in output
        assert "H100" in output
        assert "NUMA" in output
        assert "10.0.0.1" in output
        assert "NVLink" in output


class TestDevicesCliCommand:
    """End-to-end CLI invocation."""

    def test_devices_help(self):
        result = runner.invoke(app, ["devices", "--help"])
        assert result.exit_code == 0
        assert "hardware topology" in result.output

    def test_devices_runs(self):
        result = runner.invoke(app, ["devices"])
        # Exit code 0 even when no GPU is present — the command is read-only.
        assert result.exit_code == 0, result.output

    def test_devices_json_output_is_valid_json(self):
        result = runner.invoke(app, ["devices", "--json"])
        assert result.exit_code == 0, result.output
        # CliRunner renders to stdout via rich Console; the JSON portion should
        # be parseable.  rich may wrap with ANSI; find the first '{' and last '}'.
        out = result.output
        start = out.find("{")
        end = out.rfind("}")
        assert start != -1 and end != -1 and end > start
        data = json.loads(out[start:end + 1])
        assert "hostname" in data
        assert "gpus" in data
        assert "cpu" in data