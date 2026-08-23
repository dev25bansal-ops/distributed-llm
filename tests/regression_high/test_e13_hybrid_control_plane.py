"""E13 regression: hybrid managed control-plane (doc + worker-agent scaffold).

Strategy E13 delivers an ARCHITECTURE brief and a THIN worker-agent scaffold —
NOT a real managed SaaS. These tests assert exactly that contract:

  (1) ``docs/HYBRID_CONTROL_PLANE.md`` exists and references the EXISTING
      modules it maps onto (coordinator / registry / metering) by name.
  (2) ``distllm.cloud.worker_agent`` imports, and ``register_worker()`` is
      callable and returns a clear scaffold/registration-shaped result WITHOUT
      touching the network (the HTTP layer is mocked via an injected poster).

Nothing here claims a real DistLLM Cloud SaaS exists — the scaffold results are
explicitly labelled as such.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ── Locate the repo root / doc regardless of cwd ───────────────────────────

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("could not locate repo root from test file")


DOC_PATH = _repo_root() / "docs" / "HYBRID_CONTROL_PLANE.md"


# ── (1) Documentation assertions ───────────────────────────────────────────

class TestHybridControlPlaneDoc:
    def test_doc_exists(self):
        assert DOC_PATH.is_file(), f"missing architecture doc: {DOC_PATH}"

    def test_doc_references_existing_modules_by_name(self):
        text = DOC_PATH.read_text(encoding="utf-8")
        # Must map the hosted control plane onto modules that ALREADY exist.
        for needle in (
            "distllm.core.coordinator",
            "distllm.backends.registry",
            "distllm.core.metering",
            "distllm.core.placement",
            "distllm.core.plugin_marketplace",
            "distllm.dist.p2p",
        ):
            assert needle in text, f"doc must reference existing module '{needle}'"

    def test_doc_states_data_plane_stays_on_customer_infra(self):
        text = DOC_PATH.read_text(encoding="utf-8").lower()
        # Sovereignty positioning: compute/data stay with the customer.
        assert "data plane" in text
        assert "control plane" in text
        assert "sovereign" in text  # matches sovereign / sovereignty

    def test_doc_is_honest_about_scaffold_status(self):
        text = DOC_PATH.read_text(encoding="utf-8").lower()
        # Must NOT claim a shipping SaaS; must flag scaffold / not built.
        assert "scaffold" in text
        assert "not built" in text


# ── (2) Worker-agent scaffold assertions ───────────────────────────────────

class TestWorkerAgentScaffold:
    def test_module_imports(self):
        import distllm.cloud.worker_agent as wa

        assert hasattr(wa, "register_worker")
        assert hasattr(wa, "collect_capabilities")
        assert hasattr(wa, "WorkerCapabilities")
        assert hasattr(wa, "RegistrationResult")

    def test_collect_capabilities_returns_control_metadata_only(self):
        from distllm.cloud.worker_agent import collect_capabilities

        caps = collect_capabilities(region="eu-sovereign", gpu_count=4)
        payload = caps.to_payload()
        assert payload["region"] == "eu-sovereign"
        assert payload["gpu_count"] == 4
        assert isinstance(payload["backends"], list)
        # Sovereignty contract: NO content/data-plane fields cross the boundary.
        for forbidden in ("prompt", "weights", "kv_cache", "completion", "gradients"):
            assert forbidden not in payload

    def test_capabilities_map_onto_existing_placement_type(self):
        from distllm.cloud.worker_agent import collect_capabilities
        from distllm.core.placement import LinkInfo

        caps = collect_capabilities(region="us", latency_ms=12.0, bandwidth_gbps=40.0)
        link = caps.to_link_info()
        assert isinstance(link, LinkInfo)
        assert link.region == "us"
        assert link.latency_ms == 12.0

    def test_register_worker_is_callable_and_mocks_http(self):
        from distllm.cloud.worker_agent import register_worker

        sent: dict = {}

        def fake_poster(url, payload, headers):
            # Assert we never leak content and that auth/mTLS-ready headers exist.
            sent["url"] = url
            sent["payload"] = payload
            sent["headers"] = headers
            return {"accepted": True}

        result = register_worker(
            "https://cloud.distllm.ai",
            "tok_secret",
            poster=fake_poster,
        )

        # Registration-shaped result, no network touched.
        assert result.ok is True
        assert result.status == "registered"
        assert result.worker_id
        assert result.coordinator_url == "https://cloud.distllm.ai"
        assert result.response == {"accepted": True}
        # The poster saw the control-metadata payload + bearer auth header.
        assert sent["url"].endswith("/api/v1/cloud/workers/register")
        assert sent["headers"]["Authorization"] == "Bearer tok_secret"
        assert sent["payload"]["schema"] == "distllm.cloud.worker_registration/v1"
        # Honest: even a successful mock is labelled a scaffold.
        assert "scaffold" in result.detail.lower()

    def test_register_worker_offline_returns_scaffold_not_raise(self):
        from distllm.cloud.worker_agent import register_worker

        def boom(url, payload, headers):
            raise ConnectionError("no live coordinator")

        result = register_worker("https://nope.invalid", "tok", poster=boom)
        assert result.ok is False
        assert result.status == "scaffold"
        assert result.payload  # payload was still prepared
        assert "scaffold" in result.detail.lower()

    def test_register_worker_validates_inputs(self):
        from distllm.cloud.worker_agent import register_worker

        with pytest.raises(ValueError):
            register_worker("", "tok")
        with pytest.raises(ValueError):
            register_worker("https://cloud.distllm.ai", "")

    def test_result_is_json_serializable_shape(self):
        import json

        from distllm.cloud.worker_agent import register_worker

        result = register_worker(
            "https://cloud.distllm.ai",
            "tok",
            poster=lambda u, p, h: {"ok": 1},
        )
        blob = json.loads(json.dumps(result.to_dict()))
        assert blob["worker_id"] == result.worker_id
        assert blob["status"] == "registered"
