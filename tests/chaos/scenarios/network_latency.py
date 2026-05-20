"""Chaos scenario: network latency injection.

Simulates increased network latency between coordinator and worker nodes
using tc (traffic control). Verifies that the system degrades gracefully
under high-latency network conditions.

Run manually:
    python tests/chaos/scenarios/network_latency.py --target host:port --latency 200ms

Or via pytest:
    pytest tests/chaos/scenarios/test_network_latency.py -v
"""

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class LatencyConfig:
    target_host: str
    target_port: int
    latency_ms: int = 100
    jitter_ms: int = 10
    duration_s: float = 30.0
    loss_percent: float = 0.0


class NetworkLatencyInjector:
    """Injects network latency using tc (traffic control) on Linux."""

    def __init__(self, config: LatencyConfig):
        self.config = config
        self._active = False

    def apply(self) -> bool:
        """Apply network latency using tc."""
        if os.name != "posix":
            logger.warning("Network latency injection requires Linux")
            return False

        dev = self._find_interface()
        if not dev:
            logger.error("Could not find network interface")
            return False

        try:
            subprocess.run(
                ["tc", "qdisc", "add", "dev", dev, "root", "netem",
                 "delay", f"{self.config.latency_ms}ms", f"{self.config.jitter_ms}ms",
                 "distribution", "normal"],
                check=True, capture_output=True,
            )
            if self.config.loss_percent > 0:
                subprocess.run(
                    ["tc", "qdisc", "change", "dev", dev, "root", "netem",
                     "loss", f"{self.config.loss_percent}%"],
                    check=True, capture_output=True,
                )
            self._active = True
            logger.info(f"Applied {self.config.latency_ms}ms ±{self.config.jitter_ms}ms latency on {dev}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply latency: {e.stderr.decode()}")
            return False

    def remove(self) -> bool:
        """Remove network latency."""
        if os.name != "posix":
            return False
        dev = self._find_interface()
        if not dev:
            return False
        try:
            subprocess.run(["tc", "qdisc", "del", "dev", dev, "root"],
                           check=True, capture_output=True)
            self._active = False
            logger.info(f"Removed latency rules from {dev}")
            return True
        except subprocess.CalledProcessError:
            return False

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *args):
        self.remove()

    @staticmethod
    def _find_interface() -> Optional[str]:
        """Find the default network interface."""
        try:
            result = subprocess.run(
                ["ip", "route", "get", "8.8.8.8"],
                capture_output=True, text=True, check=True,
            )
            parts = result.stdout.split()
            if "dev" in parts:
                idx = parts.index("dev")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        except (subprocess.CalledProcessError, IndexError):
            pass
        return None


def test_latency_injection_and_removal():
    """Verify latency is applied and removed correctly."""
    if os.name != "posix":
        return  # skip on non-Linux

    injector = NetworkLatencyInjector(
        LatencyConfig(target_host="localhost", target_port=50050, latency_ms=50)
    )
    assert injector.apply()
    time.sleep(3)  # Let it take effect
    assert injector._active
    assert injector.remove()
    assert not injector._active
