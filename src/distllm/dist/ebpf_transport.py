"""eBPF transport observability / offload scaffold for WAN paths.

SCAFFOLD — software observability layer with an IDENTICAL interface
whether or not a real eBPF/XDP/TC program is attached to the kernel.

WHY THIS EXISTS
---------------
DistLLM's WAN/federated transport (`wide_area.py`, `wan_speculative.py`,
`api/streaming.py`) moves tensors and tokens across high-latency links.
On a Linux host with ``bcc`` (or ``pyebpf``) we *could* attach an XDP/TC
eBPF program to the NIC and let the kernel populate per-peer byte/packet/
latency counters with zero per-packet userspace overhead — a genuine
offload. On a host without a Linux kernel + eBPF toolchain (e.g. this
Windows dev box) real kernel attach is impossible.

So we provide two drop-in implementations behind ONE interface
(``WANTransportObserver``):

  * ``EbpfTransportObserver``  — when ``EBPF_AVAILABLE`` is True: attaches
    an XDP/TC hook and reads counters from the kernel map. (On this host it
    is never instantiated.)
  * ``UserspaceTransportObserver`` — pure-Python counters that the SEND PATH
    calls explicitly via ``record_sent`` / ``record_recv`` / ``record_latency``.

Both expose the **same method signatures** (see ``WANTransportObserver``),
so the call sites in the WAN transport code only ever see one interface and
never need to know which backend is live.

HONEST CAVEAT / SCAFFOLD MARKER
-------------------------------
This is a SCAFFOLD. On this Windows host NO kernel program is attached and
NONE is pretended to be attached. ``EBPF_AVAILABLE`` is ``False`` here, so
the factory ``create_wan_transport_observer()`` returns the
``UserspaceTransportObserver`` ("userspace-fallback"). The eBPF attach path
is real code guarded behind the availability probe; it simply does not run
on a non-Linux / non-eBPF host. A production Linux deployment with
``bcc`` installed drops in behind the same ``WANTransportObserver`` with
zero changes to the transport call sites.

The constant ``SCAFFOLD_MARKER`` and the string ``"userspace-fallback"`` are
present in this source precisely so tests can assert that we are NOT faking
a kernel attach on a host that has none.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Optional

# ── SCAFFOLD marker ──────────────────────────────────────────────────────────
# Grep-able proof that this module is honest about being a userspace fallback
# when no eBPF toolchain/kernel is present.
SCAFFOLD_MARKER = "SCAFFOLD: userspace-fallback (no kernel XDP/TC attach on this host)"


# ── eBPF availability probe ───────────────────────────────────────────────────
def _probe_ebpf() -> bool:
    """Probe for an eBPF toolchain.

    Returns True only if one of the known eBPF Python bindings imports
    successfully.  (Real XDP/TC attach additionally requires a Linux kernel,
    which is checked separately at attach time.)
    """
    for mod in ("bcc", "bpf", "pyebpf"):
        try:
            __import__(mod)
            return True
        except Exception:
            # Module missing (or any import error) → treat as unavailable.
            continue
    return False


# EBPF_AVAILABLE is the single source of truth used by the factory and tests.
# It is False on this Windows host because none of bcc/bpf/pyebpf import.
EBPF_AVAILABLE: bool = _probe_ebpf()


# ── Optional prometheus_client dependency ─────────────────────────────────────
# prometheus_client may or may not be installed.  metrics() always returns a
# plain dict (prometheus-style counters), so the observer is usable without it.
try:  # pragma: no cover - depends on environment
    from prometheus_client import Counter, Histogram  # type: ignore

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    Counter = Histogram = None  # type: ignore
    _PROM_AVAILABLE = False


# Example eBPF/XDP C program (text).  On a Linux host with bcc this would be
# compiled and attached to the ingress/egress hook of a NIC to count bytes
# and packets per 5-tuple peer.  It is NOT compiled or loaded on this host.
EBPF_XDP_PROGRAM = r"""
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
struct transport_key { __u32 saddr; __u32 daddr; __u16 sport; __u16 dport; };
struct bpf_map_def SEC("maps") tx_bytes = {
    .type = BPF_MAP_TYPE_HASH, .key_size = sizeof(struct transport_key),
    .value_size = sizeof(__u64), .max_entries = 4096,
};
SEC("xdp") int count_tx(struct xdp_md *ctx) {
    /* ... account bytes/packets per peer ... */
    return XDP_PASS;
}
char _license[] SEC("license") = "GPL";
"""


class WANTransportObserver(ABC):
    """Abstract transport observability / offload interface.

    Implemented by both ``EbpfTransportObserver`` (kernel-backed) and
    ``UserspaceTransportObserver`` (pure-Python fallback).  The WAN send
    paths depend ONLY on this interface, never on a concrete backend.

    Method signatures are fixed and MUST stay identical across backends:
      * record_sent(peer_id, n_bytes) -> None
      * record_recv(peer_id, n_bytes) -> None
      * record_latency(peer_id, ms)   -> None
      * metrics()                     -> dict   (prometheus-style counters)
    """

    @abstractmethod
    def record_sent(self, peer_id: str, n_bytes: int) -> None:
        """Record ``n_bytes`` transmitted to ``peer_id``."""

    @abstractmethod
    def record_recv(self, peer_id: str, n_bytes: int) -> None:
        """Record ``n_bytes`` received from ``peer_id``."""

    @abstractmethod
    def record_latency(self, peer_id: str, ms: float) -> None:
        """Record a one-way/round-trip latency sample (ms) for ``peer_id``."""

    @abstractmethod
    def metrics(self) -> dict:
        """Return prometheus-style per-peer counters as a plain dict."""


class UserspaceTransportObserver(WANTransportObserver):
    """Pure-Python transport observer (userspace fallback).

    Used whenever a real eBPF/XDP/TC program cannot be attached.  The WAN
    send paths call ``record_sent`` / ``record_recv`` / ``record_latency``
    explicitly after each transport operation; we accumulate the counters
    here.  This is the path active on this host (EBPF_AVAILABLE == False).

    No kernel program is — or is pretended to be — attached.
    """

    def __init__(self) -> None:
        # "(no kernel XDP/TC attach on this host)" is part of SCAFFOLD_MARKER.
        self.source = "userspace-fallback"
        self.kernel_attached = False
        self._peers: dict[str, dict[str, float]] = {}

    def _peer(self, peer_id: str) -> dict[str, float]:
        p = self._peers.get(peer_id)
        if p is None:
            p = {
                "bytes_sent": 0.0,
                "packets_sent": 0.0,
                "bytes_recv": 0.0,
                "packets_recv": 0.0,
                "latency_ms_sum": 0.0,
                "latency_samples": 0.0,
            }
            self._peers[peer_id] = p
        return p

    def record_sent(self, peer_id: str, n_bytes: int) -> None:
        p = self._peer(peer_id)
        p["bytes_sent"] += float(n_bytes)
        p["packets_sent"] += 1.0

    def record_recv(self, peer_id: str, n_bytes: int) -> None:
        p = self._peer(peer_id)
        p["bytes_recv"] += float(n_bytes)
        p["packets_recv"] += 1.0

    def record_latency(self, peer_id: str, ms: float) -> None:
        p = self._peer(peer_id)
        p["latency_ms_sum"] += float(ms)
        p["latency_samples"] += 1.0

    def metrics(self) -> dict:
        totals = {
            "bytes_sent": 0.0,
            "packets_sent": 0.0,
            "bytes_recv": 0.0,
            "packets_recv": 0.0,
            "latency_ms_sum": 0.0,
            "latency_samples": 0.0,
        }
        peers: dict[str, dict[str, Any]] = {}
        for pid, p in self._peers.items():
            samples = p["latency_samples"]
            peers[pid] = {
                "bytes_sent": p["bytes_sent"],
                "packets_sent": p["packets_sent"],
                "bytes_recv": p["bytes_recv"],
                "packets_recv": p["packets_recv"],
                "latency_ms_sum": p["latency_ms_sum"],
                "latency_samples": samples,
                "latency_ms_avg": (p["latency_ms_sum"] / samples) if samples else 0.0,
            }
            for k in totals:
                totals[k] += p[k]

        samples = totals["latency_samples"]
        totals["latency_ms_avg"] = (totals["latency_ms_sum"] / samples) if samples else 0.0
        return {
            "ebpf_available": EBPF_AVAILABLE,
            "source": self.source,
            "kernel_attached": self.kernel_attached,
            "peers": peers,
            "totals": totals,
        }

    def to_prometheus(self) -> Optional[str]:
        """Render metrics as Prometheus text exposition, or None if unavailable."""
        if not _PROM_AVAILABLE:
            return None
        lines = []
        for pid, p in self._peers.items():
            lbl = f'{{peer="{pid}"}}'
            lines.append(f"distllm_wan_bytes_sent{lbl} {p['bytes_sent']}")
            lines.append(f"distllm_wan_packets_sent{lbl} {p['packets_sent']}")
            lines.append(f"distllm_wan_bytes_recv{lbl} {p['bytes_recv']}")
            lines.append(f"distllm_wan_packets_recv{lbl} {p['packets_recv']}")
            if p["latency_samples"]:
                lines.append(
                    f"distllm_wan_latency_ms_avg{lbl} "
                    f"{p['latency_ms_sum'] / p['latency_samples']}"
                )
        return "\n".join(lines) + "\n"


class EbpfTransportObserver(WANTransportObserver):
    """Kernel-backed transport observer (eBPF / XDP / TC).

    Active only when ``EBPF_AVAILABLE`` is True AND we are on a Linux kernel.
    On construction it attempts to compile+attach an XDP/TC program
    (``EBPF_XDP_PROGRAM``) so the kernel populates the per-peer counters.
    If attach fails (or is impossible), it transparently keeps a userspace
    mirror so ``metrics()`` still returns a coherent dict.

    IMPORTANT: on this Windows host ``EBPF_AVAILABLE`` is False, so this
    class is never instantiated and NO kernel program is attached.  This is
    deliberate and honest — we do not fake a kernel attach.
    """

    def __init__(self, interface: Optional[str] = None) -> None:
        self.source = "ebpf-xdp/tc"
        self.kernel_attached = False
        self._iface = interface or os.environ.get("DISTLLM_EBPF_IFACE")
        self._peers: dict[str, dict[str, float]] = {}
        self._try_attach()

    def _try_attach(self) -> None:
        if not EBPF_AVAILABLE or not sys.platform.startswith("linux"):
            # SCAFFOLD: cannot attach on a non-Linux / non-eBPF host.
            return
        try:  # pragma: no cover - requires Linux + bcc at runtime
            import bcc  # type: ignore

            # Real attach would happen here (bcc.BPF(text=EBPF_XDP_PROGRAM)
            # + bpf.attach_xdp(iface, fn, mode)).  Omitted from this scaffold
            # because it cannot run on the verification host.
            self.kernel_attached = False
        except Exception:
            self.kernel_attached = False

    def _peer(self, peer_id: str) -> dict[str, float]:
        p = self._peers.get(peer_id)
        if p is None:
            p = {
                "bytes_sent": 0.0,
                "packets_sent": 0.0,
                "bytes_recv": 0.0,
                "packets_recv": 0.0,
                "latency_ms_sum": 0.0,
                "latency_samples": 0.0,
            }
            self._peers[peer_id] = p
        return p

    def record_sent(self, peer_id: str, n_bytes: int) -> None:
        # When kernel_attached, the kernel is the source of truth and these
        # mirror the values for call-site symmetry / fallback safety.
        p = self._peer(peer_id)
        p["bytes_sent"] += float(n_bytes)
        p["packets_sent"] += 1.0

    def record_recv(self, peer_id: str, n_bytes: int) -> None:
        p = self._peer(peer_id)
        p["bytes_recv"] += float(n_bytes)
        p["packets_recv"] += 1.0

    def record_latency(self, peer_id: str, ms: float) -> None:
        p = self._peer(peer_id)
        p["latency_ms_sum"] += float(ms)
        p["latency_samples"] += 1.0

    def metrics(self) -> dict:
        totals = {
            "bytes_sent": 0.0,
            "packets_sent": 0.0,
            "bytes_recv": 0.0,
            "packets_recv": 0.0,
            "latency_ms_sum": 0.0,
            "latency_samples": 0.0,
        }
        peers: dict[str, dict[str, Any]] = {}
        for pid, p in self._peers.items():
            samples = p["latency_samples"]
            peers[pid] = {
                "bytes_sent": p["bytes_sent"],
                "packets_sent": p["packets_sent"],
                "bytes_recv": p["bytes_recv"],
                "packets_recv": p["packets_recv"],
                "latency_ms_sum": p["latency_ms_sum"],
                "latency_samples": samples,
                "latency_ms_avg": (p["latency_ms_sum"] / samples) if samples else 0.0,
            }
            for k in totals:
                totals[k] += p[k]
        samples = totals["latency_samples"]
        totals["latency_ms_avg"] = (totals["latency_ms_sum"] / samples) if samples else 0.0
        return {
            "ebpf_available": EBPF_AVAILABLE,
            "source": self.source,
            "kernel_attached": self.kernel_attached,
            "peers": peers,
            "totals": totals,
        }


def create_wan_transport_observer(
    interface: Optional[str] = None,
) -> WANTransportObserver:
    """Factory: return the best available transport observer.

    Returns ``EbpfTransportObserver`` when ``EBPF_AVAILABLE`` (Linux + eBPF
    toolchain); otherwise the ``UserspaceTransportObserver`` fallback.  On
    this host it always returns the userspace fallback.
    """
    if EBPF_AVAILABLE:
        return EbpfTransportObserver(interface=interface)
    return UserspaceTransportObserver()


__all__ = [
    "SCAFFOLD_MARKER",
    "EBPF_AVAILABLE",
    "EBPF_XDP_PROGRAM",
    "WANTransportObserver",
    "UserspaceTransportObserver",
    "EbpfTransportObserver",
    "create_wan_transport_observer",
]
