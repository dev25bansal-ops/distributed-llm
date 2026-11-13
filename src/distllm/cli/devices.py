"""CLI command: distllm devices — discover local node hardware topology.

Discovers and pretty-prints:
- GPUs (index, name, VRAM, CUDA version, driver version)
- CPUs (count, architecture, NUMA nodes)
- Memory (total / available)
- Network interfaces + IPv4 addresses
- NVLink / PCIe topology (best-effort)

Designed to run inside the worker container/host so an operator can
copy/paste the output into a bug report or feed it into a sizing tool.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


def _safe_run(cmd: list[str], timeout: float = 5.0) -> str:
    """Run *cmd*; return stdout (stripped) or empty string on any error.

    Never raises — discovery is best-effort, we don't want a missing
    ``nvidia-smi`` or ``lspci`` to take down the whole report.
    """
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return (out.stdout or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _read_memory_psutil() -> tuple[int, int]:
    """Portable memory probe for platforms without /proc (macOS and others).

    psutil is a core dependency, so this works on every supported platform;
    /proc is still preferred on Linux only because it needs no import.
    """
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except (ImportError, OSError, AttributeError):
        return 0, 0


def _read_meminfo_linux() -> tuple[int, int]:
    """Parse /proc/meminfo. Returns (total_bytes, available_bytes)."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                val = val.strip().split()[0] if val.strip() else "0"
                try:
                    info[key] = int(val)
                except ValueError:
                    continue
        total_kb = info.get("MemTotal", 0)
        avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
        return total_kb * 1024, avail_kb * 1024
    except OSError:
        return 0, 0


def _read_meminfo_windows() -> tuple[int, int]:
    """Best-effort Windows memory probe via ctypes GlobalMemoryStatusEx."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except (OSError, AttributeError):
        pass
    return 0, 0


def _memory_total_available(system: str | None = None) -> tuple[int, int]:
    """Return (total, available) bytes for the current host.

    ``system`` is ``platform.system()`` unless the caller passes one in, which
    exists so the platform branches are testable without a Mac.
    """
    if system is None:
        system = platform.system()
    if system == "Windows":
        return _read_meminfo_windows()
    if system == "Linux":
        return _read_meminfo_linux()
    return _read_memory_psutil()


def _cpu_info() -> dict[str, Any]:
    """Return {count, architecture, numa_nodes, model}."""
    info: dict[str, Any] = {
        "count": os.cpu_count() or 0,
        "architecture": platform.machine() or platform.processor() or "unknown",
        "numa_nodes": 0,
        "model": platform.processor() or "unknown",
    }
    # Linux: parse /sys/devices/system/node/ for NUMA topology
    if sys.platform != "win32" and os.path.isdir("/sys/devices/system/node"):
        try:
            nodes = [
                n for n in os.listdir("/sys/devices/system/node")
                if n.startswith("node") and n[4:].isdigit()
            ]
            info["numa_nodes"] = len(nodes)
        except OSError:
            pass
    # Windows fallback: assume single NUMA node unless WMIC reports otherwise
    if info["numa_nodes"] == 0 and platform.system() == "Windows":
        out = _safe_run(["wmic", "cpu", "get", "NumberOfLogicalProcessors", "/value"])
        if out:
            # /value emits "NumberOfLogicalProcessors=N"; tolerate whitespace.
            for line in out.splitlines():
                if "=" in line:
                    _, _, v = line.partition("=")
                    v = v.strip()
                    if v.isdigit():
                        info["count"] = max(info["count"], int(v))
        info["numa_nodes"] = 1
    if info["numa_nodes"] == 0:
        info["numa_nodes"] = 1
    return info


def _gpu_info() -> list[dict[str, Any]]:
    """Return per-GPU dicts using torch first, nvidia-smi as fallback."""
    gpus: list[dict[str, Any]] = []

    # 1) Try torch (matches what the engine actually sees at runtime).
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "unknown"
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append({
                    "index": idx,
                    "name": props.name,
                    "vram_bytes": int(getattr(props, "total_memory", 0)),
                    "cuda_version": cuda_ver,
                    "driver_version": "n/a",
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass

    # 1b) Apple Silicon: no CUDA and no nvidia-smi, but the MPS backend is a
    # real accelerator. Without this a MacBook reports "(none detected)".
    if not gpus:
        try:
            import torch  # type: ignore

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpus.append({
                    "index": 0,
                    "name": f"Apple Silicon GPU (MPS, {platform.machine()})",
                    # Unified memory: no separate VRAM figure, so report host
                    # memory rather than a misleading zero.
                    "vram_bytes": _read_memory_psutil()[0],
                    "cuda_version": "n/a (Metal)",
                    "driver_version": platform.release(),
                    "compute_capability": "n/a",
                })
        except ImportError:
            pass

    # 2) Fallback to nvidia-smi so the command works even when torch is
    # CPU-only or missing; preserves order and fills in driver/CUDA versions.
    if not gpus:
        smi = _safe_run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        # There is no `cuda_version` query field: nvidia-smi answers
        # `Field "cuda_version" is not a valid field to query.` on stderr and
        # exits 0, and _safe_run reads stdout only, so a query for it silently
        # yields "". The CUDA version is read from the header line instead —
        # `nvidia-smi` with no arguments prints `CUDA Version: 12.4` in its
        # table header, which is the documented way to get it.
        cuda_short = _nvidia_smi_cuda_version()
        for line in smi.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                vram_mib = int(parts[2])
            except ValueError:
                vram_mib = 0
            try:
                gpu_index = int(parts[0])
            except ValueError:
                continue
            gpus.append({
                "index": gpu_index,
                "name": parts[1],
                "vram_bytes": vram_mib * 1024 * 1024,
                "cuda_version": cuda_short,
                "driver_version": parts[3],
                "compute_capability": parts[4],
            })
    return gpus


def _nvidia_smi_cuda_version() -> str:
    """Return the driver's CUDA version, or "unknown".

    Read from the header of plain `nvidia-smi` output. Two spellings exist
    across driver generations: older drivers print ``CUDA Version: 12.4`` and
    newer ones (621.x here) print ``CUDA UMD Version: 13.5``; both are tried.
    The previous `--query-gpu=cuda_version` is not a valid field, so the old
    code shipped whatever diagnostic nvidia-smi printed — the string "Field"
    on a machine whose driver rejects the query.
    """
    header = _safe_run(["nvidia-smi"])
    for line in header.splitlines():
        for marker in ("CUDA UMD Version:", "CUDA Version:"):
            if marker in line:
                value = line.split(marker, 1)[1].strip().strip("|").strip()
                return value or "unknown"
    return "unknown"


def _driver_version_linux() -> str | None:
    """Read NVIDIA kernel module version on Linux for an authoritative driver string."""
    for path in ("/proc/driver/nvidia/version",):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                line = fh.readline().strip()
                # Line is e.g. "NVRM version: NVIDIA UNIX Open Kernel Module ..."
                if line:
                    return line.split("NVIDIA", 1)[1].strip() if "NVIDIA" in line else line
        except OSError:
            continue
    return None


def _network_interfaces() -> list[dict[str, Any]]:
    """Return [{name, ipv4, mac}] using socket + /sys (Linux) or psutil fallback."""
    interfaces: list[dict[str, Any]] = []
    # IPv4 address table — cheap and works on every platform
    try:
        import psutil  # type: ignore

        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        for name, addr_list in addrs.items():
            ipv4 = next((a.address for a in addr_list if a.family == socket.AF_INET), "")
            mac = next((a.address for a in addr_list if a.family == psutil.AF_LINK), "")
            up = stats[name].isup if name in stats else False
            interfaces.append({"name": name, "ipv4": ipv4, "mac": mac, "up": up})
    except ImportError:
        # No psutil — best-effort hostname + loopback only.
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except OSError:
            ip, hostname = "", ""
        interfaces.append({
            "name": hostname or "host",
            "ipv4": ip,
            "mac": "",
            "up": True,
        })
    return interfaces


def _topology_info() -> dict[str, Any]:
    """Best-effort NVLink / PCIe link info (empty if not discoverable)."""
    info: dict[str, Any] = {"nvlink": False, "pcie_links": []}

    # nvidia-smi topo -m prints an ASCII matrix; just confirm the binary works.
    if shutil.which("nvidia-smi"):
        out = _safe_run(["nvidia-smi", "topo", "-m"])
        if out and "NVLink" in out:
            info["nvlink"] = True
        info["pcie_links"] = []  # Don't dump the full matrix — too noisy for CLI

    # Linux: enumerate PCIe devices matching NVIDIA vendor 0x10de.
    if sys.platform != "win32" and os.path.isdir("/sys/bus/pci/devices"):
        nvidia_count = 0
        try:
            for dev in os.listdir("/sys/bus/pci/devices"):
                vendor_path = f"/sys/bus/pci/devices/{dev}/vendor"
                try:
                    with open(vendor_path, encoding="utf-8") as fh:
                        if fh.read().strip() == "0x10de":
                            nvidia_count += 1
                except OSError:
                    continue
        except OSError:
            pass
        info["pcie_links"] = [f"NVIDIA PCIe devices: {nvidia_count}"]
    return info


def _format_bytes(n: int) -> str:
    if n <= 0:
        return "n/a"
    gib = n / (1024 ** 3)
    if gib >= 1:
        return f"{gib:.1f} GiB"
    mib = n / (1024 ** 2)
    return f"{mib:.1f} MiB"


def _format_pretty(obj: Any) -> Any:
    """Convert the discovery result into JSON-friendly primitives."""
    if isinstance(obj, dict):
        return {k: _format_pretty(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_format_pretty(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def run_devices(system: str | None = None) -> dict[str, Any]:
    """Discover node hardware topology.

    Returns a dict suitable for both pretty-printing and JSON serialization.
    Keys: ``hostname``, ``platform``, ``gpus``, ``cpu``, ``memory``,
    ``network``, ``topology``.

    ``system`` overrides ``platform.system()`` for the memory probe so the
    macOS/Linux/Windows branches can be exercised off their own platform.
    """
    if system is None:
        system = platform.system()
    mem_total, mem_avail = _memory_total_available(system)
    gpus = _gpu_info()
    cpu = _cpu_info()
    network = _network_interfaces()
    topology = _topology_info()
    driver = _driver_version_linux() if system == "Linux" else None
    if driver and gpus:
        # Annotate the first GPU with the kernel module line so it's not lost;
        # nvidia-smi's driver_version can be misleading on containerized hosts.
        gpus[0]["driver_version"] = gpus[0].get("driver_version", "n/a") or "n/a"
        gpus[0]["driver_module"] = driver[:80]

    return {
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "gpus": gpus,
        "cpu": cpu,
        "memory": {
            "total_bytes": mem_total,
            "available_bytes": mem_avail,
        },
        "network": network,
        "topology": topology,
    }


def print_devices(result: dict[str, Any], console) -> None:
    """Render *result* as a series of rich tables on *console*."""
    from rich.table import Table

    # Header
    plat = result.get("platform", {})
    console.print(
        f"[bold]{result.get('hostname', '?')}[/bold] — "
        f"{plat.get('system', '?')} {plat.get('release', '?')} · "
        f"Python {plat.get('python', '?')}"
    )

    # GPUs
    gpus = result.get("gpus", [])
    table = Table(title=f"GPUs ({len(gpus)})")
    table.add_column("Idx", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("VRAM", style="green", justify="right")
    table.add_column("CUDA", style="magenta")
    table.add_column("Driver", style="blue")
    table.add_column("Compute", style="dim")
    if not gpus:
        table.add_row("-", "(none detected)", "-", "-", "-", "-")
    else:
        for g in gpus:
            table.add_row(
                str(g.get("index", "?")),
                str(g.get("name", "?")),
                _format_bytes(g.get("vram_bytes", 0)),
                str(g.get("cuda_version", "?")),
                str(g.get("driver_version", "?")),
                str(g.get("compute_capability", "?")),
            )
    console.print(table)

    # CPU
    cpu = result.get("cpu", {})
    ct = Table(title="CPU")
    ct.add_column("Metric", style="cyan")
    ct.add_column("Value", style="green")
    ct.add_row("Logical cores", str(cpu.get("count", "?")))
    ct.add_row("Architecture", str(cpu.get("architecture", "?")))
    ct.add_row("NUMA nodes", str(cpu.get("numa_nodes", "?")))
    ct.add_row("Model", str(cpu.get("model", "?")))
    console.print(ct)

    # Memory
    mem = result.get("memory", {})
    mt = Table(title="Memory")
    mt.add_column("Metric", style="cyan")
    mt.add_column("Value", style="green")
    mt.add_row("Total", _format_bytes(mem.get("total_bytes", 0)))
    mt.add_row("Available", _format_bytes(mem.get("available_bytes", 0)))
    console.print(mt)

    # Network
    net = result.get("network", [])
    nt = Table(title=f"Network Interfaces ({len(net)})")
    nt.add_column("Name", style="cyan")
    nt.add_column("IPv4", style="white")
    nt.add_column("MAC", style="dim")
    nt.add_column("Up", style="green")
    if not net:
        nt.add_row("(none)", "-", "-", "-")
    else:
        from rich.text import Text

        for iface in net:
            mac = str(iface.get("mac", "-")) or "-"
            # Rich interprets :shortcode: sequences as emoji even inside a
            # table cell, so a MAC address like e6:cd:9f:23:1d:69 rendered as
            # "e6💿9f:23:1d:69" (and then crashed any cp1252 console). A Text
            # object is printed verbatim, with no markup pass.
            nt.add_row(
                str(iface.get("name", "?")),
                str(iface.get("ipv4", "-")) or "-",
                Text(mac),
                "yes" if iface.get("up") else "no",
            )
    console.print(nt)

    # Topology
    topo = result.get("topology", {})
    tt = Table(title="Topology")
    tt.add_column("Metric", style="cyan")
    tt.add_column("Value", style="green")
    tt.add_row("NVLink", "yes" if topo.get("nvlink") else "no")
    for entry in topo.get("pcie_links", []):
        tt.add_row("PCIe", str(entry))
    console.print(tt)
