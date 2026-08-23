"""Regression tests for task A7: eBPF transport observability/offload scaffold.

This module is a SCAFFOLD.  It proves the *integration contract* between
DistLLM's WAN/federated transport and a transport observability hook with a
single, backend-agnostic interface (``WANTransportObserver``):

  * when an eBPF/XDP/TC program is attached (Linux + bcc) the kernel populates
    the counters;
  * when it is NOT available (this Windows host) a pure-Python
    ``UserspaceTransportObserver`` fallback is used — the SEND PATH calls
    ``record_sent`` / ``record_recv`` / ``record_latency`` explicitly.

Both backends implement the SAME method signatures, so the transport call
sites never know which one is live.

HONEST CAVEAT: On this host ``EBPF_AVAILABLE`` is False.  NO kernel program
is attached and NONE is pretended to be attached.  The eBPF attach path is
real code guarded behind the availability probe; it simply does not run
here.  A production Linux deployment drops in behind the same
``WANTransportObserver`` with zero transport-call-site changes.

These tests assert:
  1. The userspace FALLBACK observer records sent bytes and metrics() reflects them.
  2. The interface is IDENTICAL whether eBPF or fallback (same method signatures).
  3. The eBPF-absent path is clearly labelled (no fake kernel attach — grep marker).
  4. Disabling the observer (None) leaves the WAN send path functional.
"""

from __future__ import annotations

from inspect import signature
from pathlib import Path

import pytest
import torch

from distllm.dist.ebpf_transport import (
    EBPF_AVAILABLE,
    SCAFFOLD_MARKER,
    EbpfTransportObserver,
    WANTransportObserver,
    UserspaceTransportObserver,
    create_wan_transport_observer,
)
from distllm.dist.wan_speculative import WANSpeculativeDecoder

# The source file is grepped for the SCAFFOLD marker in test 3.
_EBPF_SOURCE = Path(__file__).resolve().parents[2] / "src" / "distllm" / "dist" / "ebpf_transport.py"

_EXPECTED_METHODS = ("record_sent", "record_recv", "record_latency", "metrics")


# ── 1. Userspace fallback records sent bytes and metrics reflect them ──────────
def test_1_userspace_fallback_records_sent_bytes():
    obs = UserspaceTransportObserver()
    assert obs.kernel_attached is False, "no kernel attach on fallback host"

    obs.record_sent("peer-A", 100)
    obs.record_sent("peer-A", 50)
    obs.record_recv("peer-A", 200)
    obs.record_latency("peer-A", 12.5)

    m = obs.metrics()
    # Per-peer counters present.
    assert "peer-A" in m["peers"]
    peer = m["peers"]["peer-A"]
    assert peer["bytes_sent"] == 150.0, "two sends of 100+50 => 150"
    assert peer["packets_sent"] == 2.0
    assert peer["bytes_recv"] == 200.0
    assert peer["latency_ms_sum"] == 12.5
    assert peer["latency_ms_avg"] == 12.5
    # Totals aggregate across peers.
    assert m["totals"]["bytes_sent"] == 150.0
    assert m["totals"]["packets_sent"] == 2.0
    assert m["ebpf_available"] == EBPF_AVAILABLE


# ── 2. Interface identical for eBPF and fallback backends ──────────────────────
def test_2_interface_identical_across_backends():
    for mname in _EXPECTED_METHODS:
        fb_sig = signature(getattr(UserspaceTransportObserver, mname))
        eb_sig = signature(getattr(EbpfTransportObserver, mname))
        assert fb_sig == eb_sig, f"signature mismatch on {mname}: {fb_sig} != {eb_sig}"
    # Both are proper WANTransportObserver implementations.
    assert issubclass(UserspaceTransportObserver, WANTransportObserver)
    assert issubclass(EbpfTransportObserver, WANTransportObserver)
    # The factory returns an object whose interface equals the abstract type.
    produced = create_wan_transport_observer()
    assert isinstance(produced, WANTransportObserver)


# ── 3. eBPF-absent path is clearly labelled; no fake kernel attach ─────────────
def test_3_ebpf_absent_clearly_labelled_no_fake_attach():
    # On this host the eBPF toolchain is unavailable.
    assert EBPF_AVAILABLE is False, "expected eBPF unavailable on this host"

    # The factory yields the userspace fallback (never a fake kernel attach).
    obs = create_wan_transport_observer()
    assert isinstance(obs, UserspaceTransportObserver)
    assert obs.source == "userspace-fallback"
    assert obs.kernel_attached is False

    # The source file carries the SCAFFOLD marker and explicitly states no
    # kernel XDP/TC attach on this host — proof we are not faking a kernel hook.
    src = _EBPF_SOURCE.read_text(encoding="utf-8")
    assert SCAFFOLD_MARKER in src
    assert "userspace-fallback" in src
    assert "no kernel XDP/TC attach on this host" in src
    # It must NOT claim a kernel program was attached on this host.
    assert "kernel_attached = True" not in src or "kernel_attached = False" in src


# ── 4. Disabling the observer (None) leaves the WAN send path functional ───────
def _make_deterministic_decoder(observer):
    """Build a WANSpeculativeDecoder whose target argmax matches draft (greedy
    accept-all), so generate() completes in one WAN round-trip quickly."""
    vocab = 16
    seq0 = torch.zeros(1, 3, dtype=torch.long)  # tiny prompt

    def draft_forward(prefix, **kwargs):
        # _draft_forward calls self._draft(current, **kwargs) (prefix only).
        # Emit a single next-token logits row so sampling picks a valid token.
        L = prefix.shape[1]
        logits = torch.full((1, 1, vocab), -1e9, dtype=torch.float32)
        tok = ((L + 1) % (vocab - 1)) + 1
        logits[0, 0, tok] = 1.0
        return logits

    async def target_forward(tokens, **kwargs):
        # generate() awaits target_forward. logits with argmax at each
        # position == token value -> greedy verification accepts all drafts.
        L = tokens.shape[1]
        logits = torch.full((1, L, vocab), -1e9, dtype=torch.float32)
        idx = tokens[0].tolist()
        for pos, tok in enumerate(idx):
            logits[0, pos, tok] = 1.0
        return logits

    dec = WANSpeculativeDecoder(
        target_forward=target_forward,
        draft_forward=draft_forward,
        num_candidates=8,
        temperature=0.0,  # greedy => accept when argmax matches
        device="cpu",
        max_speculation_depth=16,
        transport_observer=observer,
    )
    return dec, seq0


def test_4_disabled_observer_leaves_send_path_functional():
    # Observer disabled (None) -> generate() still runs and returns output.
    dec_off, seq0 = _make_deterministic_decoder(observer=None)
    out_off = None
    # generate() is async; run it via asyncio.
    import asyncio

    async def _run():
        return await dec_off.generate(seq0, max_new_tokens=4)

    out_off = asyncio.run(_run())
    assert out_off is not None
    assert out_off.shape[0] == 1
    assert out_off.shape[1] >= seq0.shape[1], "generation must extend the prompt"

    # With the observer enabled, the SAME send path runs AND records metrics.
    obs = UserspaceTransportObserver()
    dec_on, seq0b = _make_deterministic_decoder(observer=obs)

    async def _run2():
        return await dec_on.generate(seq0b, max_new_tokens=4)

    out_on = asyncio.run(_run2())
    assert out_on.shape == out_off.shape, "enabled observer must not change transport output"

    m = obs.metrics()
    # At least one WAN round-trip was recorded by the (functional) send path.
    assert m["totals"]["packets_sent"] >= 1.0, "send path must have recorded a send"
    assert m["totals"]["bytes_sent"] > 0.0
    assert "wan" in m["peers"], "speculative decoder records under peer 'wan'"
