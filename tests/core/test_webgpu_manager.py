"""Tests for WebGPUManager using real objects via load_module pattern."""

from __future__ import annotations

import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_wgpu_mod = load_module("distllm/core/webgpu_manager.py")
WebGPUManager = _wgpu_mod.WebGPUManager
BrowserGPU = _wgpu_mod.BrowserGPU
WebGPUNode = _wgpu_mod.WebGPUNode


class TestBrowserGPU:
    """BrowserGPU dataclass construction."""

    def test_minimal(self) -> None:
        gpu = BrowserGPU(session_id="sess-1")
        assert gpu.session_id == "sess-1"
        assert gpu.is_available is True
        assert gpu.gpu_vendor == ""
        assert gpu.requests_served == 0

    def test_custom_fields(self) -> None:
        gpu = BrowserGPU(
            session_id="sess-2",
            gpu_vendor="NVIDIA",
            gpu_model="RTX 4090",
            vram_mb=24576,
            webgpu_features=["shader-f16"],
        )
        assert gpu.gpu_vendor == "NVIDIA"
        assert gpu.vram_mb == 24576


class TestWebGPUNode:
    """WebGPUNode dataclass construction."""

    def test_minimal(self) -> None:
        gpu = BrowserGPU(session_id="s")
        node = WebGPUNode(node_id="wgpu-1", session_id="s", gpu_info=gpu)
        assert node.status == "active"
        assert node.current_request is None


class TestWebGPUManagerConstruction:
    """WebGPUManager construction with defaults."""

    def test_default_params(self) -> None:
        mgr = WebGPUManager()
        assert mgr._max_nodes == 100
        assert mgr._heartbeat_timeout == 30.0
        assert mgr._enable_contribution is True

    def test_custom_params(self) -> None:
        mgr = WebGPUManager(max_nodes=10, heartbeat_timeout_s=60.0, enable_contribution=False)
        assert mgr._max_nodes == 10
        assert mgr._heartbeat_timeout == 60.0
        assert mgr._enable_contribution is False

    def test_initial_stats(self) -> None:
        mgr = WebGPUManager()
        stats = mgr.stats()
        assert stats["total_registrations"] == 0
        assert stats["total_requests"] == 0
        assert stats["total_tokens_served"] == 0
        assert stats["active_nodes"] == 0


class TestWebGPUManagerRegistration:
    """Register browser GPU contributors."""

    def test_register_browser(self) -> None:
        mgr = WebGPUManager()
        session_id = mgr.register_browser({
            "gpu_vendor": "NVIDIA",
            "gpu_model": "RTX 4090",
            "vram_mb": 24576,
            "webgpu_features": ["shader-f16"],
        })
        assert session_id != ""
        assert len(session_id) == 16

    def test_register_increments_stats(self) -> None:
        mgr = WebGPUManager()
        mgr.register_browser({"gpu_vendor": "AMD"})
        stats = mgr.stats()
        assert stats["total_registrations"] == 1
        assert stats["active_nodes"] == 1

    def test_register_multiple(self) -> None:
        mgr = WebGPUManager()
        s1 = mgr.register_browser({"gpu_vendor": "NVIDIA"})
        s2 = mgr.register_browser({"gpu_vendor": "AMD"})
        assert s1 != s2
        stats = mgr.stats()
        assert stats["active_nodes"] == 2

    def test_max_nodes_enforced(self) -> None:
        mgr = WebGPUManager(max_nodes=2)
        mgr.register_browser({"gpu_vendor": "A"})
        mgr.register_browser({"gpu_vendor": "B"})
        empty = mgr.register_browser({"gpu_vendor": "C"})
        assert empty == ""  # Rejected

    def test_register_minimal_info(self) -> None:
        mgr = WebGPUManager()
        session_id = mgr.register_browser({})
        assert session_id != ""

    def test_get_available_node_after_register(self) -> None:
        mgr = WebGPUManager()
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        node = mgr.get_available_node()
        assert node is not None
        assert node.status == "active"

    def test_get_available_node_when_empty(self) -> None:
        mgr = WebGPUManager()
        node = mgr.get_available_node()
        assert node is None


class TestWebGPUManagerHeartbeat:
    """Heartbeat management."""

    def test_heartbeat_updates_time(self) -> None:
        mgr = WebGPUManager()
        session_id = mgr.register_browser({"gpu_vendor": "NVIDIA"})
        old_gpu = mgr._sessions[session_id]
        old_time = old_gpu.last_heartbeat

        time.sleep(0.01)
        result = mgr.heartbeat(session_id)
        assert result is True
        assert mgr._sessions[session_id].last_heartbeat > old_time

    def test_heartbeat_unknown_session(self) -> None:
        mgr = WebGPUManager()
        result = mgr.heartbeat("nonexistent")
        assert result is False

    def test_unregister(self) -> None:
        mgr = WebGPUManager()
        session_id = mgr.register_browser({"gpu_vendor": "NVIDIA"})
        mgr.unregister(session_id)
        stats = mgr.stats()
        assert stats["active_nodes"] == 0

    def test_unregister_unknown(self) -> None:
        mgr = WebGPUManager()
        # Should not raise
        mgr.unregister("unknown")


class TestWebGPUManagerNodeLifecycle:
    """Node busy/free lifecycle."""

    def test_mark_busy(self) -> None:
        mgr = WebGPUManager()
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        node = mgr.get_available_node()
        assert node is not None

        mgr.mark_busy(node.node_id, "req-1")
        assert mgr._nodes[node.node_id].status == "busy"

    def test_mark_free(self) -> None:
        mgr = WebGPUManager()
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        node = mgr.get_available_node()
        assert node is not None

        mgr.mark_busy(node.node_id, "req-1")
        mgr.mark_free(node.node_id, tokens_served=100)

        assert mgr._nodes[node.node_id].status == "active"
        assert mgr._nodes[node.node_id].gpu_info.requests_served == 1
        assert mgr._nodes[node.node_id].gpu_info.total_tokens == 100

    def test_stats_update_on_free(self) -> None:
        mgr = WebGPUManager()
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        node = mgr.get_available_node()
        mgr.mark_busy(node.node_id, "req-1")
        mgr.mark_free(node.node_id, tokens_served=50)

        stats = mgr.stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens_served"] == 50

    def test_get_available_node_skips_busy(self) -> None:
        mgr = WebGPUManager()
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        node = mgr.get_available_node()
        mgr.mark_busy(node.node_id, "req-1")

        second = mgr.get_available_node()
        assert second is None


class TestWebGPUManagerStaleCleanup:
    """Stale node cleanup."""

    def test_cleanup_stale_timeout(self) -> None:
        mgr = WebGPUManager(heartbeat_timeout_s=0.01)
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        time.sleep(0.02)
        mgr._cleanup_stale()
        stats = mgr.stats()
        assert stats["active_nodes"] == 0

    def test_cleanup_stale_not_removed_within_timeout(self) -> None:
        mgr = WebGPUManager(heartbeat_timeout_s=60.0)
        mgr.register_browser({"gpu_vendor": "NVIDIA"})
        mgr._cleanup_stale()
        stats = mgr.stats()
        assert stats["active_nodes"] == 1


class TestWebGPUManagerClientHTML:
    """Client HTML page generation."""

    def test_get_client_html(self) -> None:
        mgr = WebGPUManager()
        html = mgr.get_client_html()
        assert isinstance(html, str)
        assert "DistLLM WebGPU Client" in html
        assert "<!DOCTYPE html>" in html

    def test_client_html_has_connect_button(self) -> None:
        mgr = WebGPUManager()
        html = mgr.get_client_html()
        assert "connect-btn" in html

    def test_client_html_has_javascript(self) -> None:
        mgr = WebGPUManager()
        html = mgr.get_client_html()
        assert "<script>" in html
        assert "detectGPU" in html
