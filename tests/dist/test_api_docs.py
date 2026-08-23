"""Tests for the api_docs module — OpenAPI spec generation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from distllm.dist.api_docs import build_distributed_spec, generate_spec


class TestBuildDistributedSpec:
    """Tests for build_distributed_spec()."""

    def test_default_base_url(self) -> None:
        """Default base_url is http://localhost:8000."""
        spec = build_distributed_spec()
        assert spec["servers"] == [
            {"url": "http://localhost:8000", "description": "Coordinator API"}
        ]

    def test_custom_base_url(self) -> None:
        """A custom base_url is reflected in the servers list."""
        spec = build_distributed_spec(base_url="https://coordinator.example.com:443")
        assert spec["servers"] == [
            {
                "url": "https://coordinator.example.com:443",
                "description": "Coordinator API",
            }
        ]

    def test_openapi_version(self) -> None:
        """Spec declares OpenAPI 3.1.0."""
        spec = build_distributed_spec()
        assert spec["openapi"] == "3.1.0"

    def test_info_section(self) -> None:
        """Info section has title, version, description, and contact."""
        spec = build_distributed_spec()
        info = spec["info"]
        assert info["title"] == "DistLLM Distributed Layer API"
        assert info["version"] == "2.0.0"
        assert "description" in info
        assert "contact" in info

    def test_paths_contains_expected_endpoints(self) -> None:
        """All expected endpoint groups are present in paths."""
        spec = build_distributed_spec()
        paths = spec["paths"]

        # Cluster management
        assert "/admin/v1/nodes" in paths
        assert "/admin/v1/nodes/{node_id}/ready" in paths
        assert "/admin/v1/cluster/status" in paths
        assert "/admin/v1/cluster/rebalance" in paths

        # Federation
        assert "/v1/federation/heartbeat" in paths
        assert "/v1/federation/health" in paths
        assert "/v1/federation/peers" in paths
        assert "/v1/federation/gossip" in paths

        # Recovery
        assert "/admin/v1/recovery/status" in paths
        assert "/admin/v1/recovery/history" in paths
        assert "/admin/v1/recovery/drill" in paths

        # Marketplace
        assert "/api/v1/marketplace/listings" in paths
        assert "/api/v1/marketplace/jobs" in paths

        # Provisioning
        assert "/api/v1/provisioning/deployments" in paths
        assert "/api/v1/provisioning/deployments/{deployment_id}" in paths

        # Cache
        assert "/api/v1/cache/warm" in paths
        assert "/api/v1/cache/migrate" in paths

        # Power management
        assert "/admin/v1/power/status" in paths
        assert "/admin/v1/power/auto-tune" in paths

        # Multi-tenant
        assert "/admin/v1/tenants" in paths
        assert "/admin/v1/tenants/{tenant_id}" in paths

        # Cloud regions
        assert "/api/v1/regions" in paths

    def test_components_contains_security_schemes(self) -> None:
        """Components include BearerAuth and ApiKeyAuth security schemes."""
        spec = build_distributed_spec()
        schemes = spec["components"]["securitySchemes"]
        assert "BearerAuth" in schemes
        assert schemes["BearerAuth"]["type"] == "http"
        assert schemes["BearerAuth"]["scheme"] == "bearer"
        assert "ApiKeyAuth" in schemes
        assert schemes["ApiKeyAuth"]["type"] == "apiKey"

    def test_components_contains_schemas(self) -> None:
        """Components include all expected schema definitions."""
        spec = build_distributed_spec()
        schemas = spec["components"]["schemas"]
        expected = {
            "NodeInfo",
            "NodeRegistration",
            "ClusterStatus",
            "ClusterLoad",
            "FederationHealth",
            "PeerInfo",
            "RecoveryStatus",
            "RecoveryEvent",
            "DrillResult",
            "DeploymentRequest",
        }
        assert set(schemas) == expected

    def test_schema_structure(self) -> None:
        """Key schemas have expected required / default fields."""
        spec = build_distributed_spec()
        schemas = spec["components"]["schemas"]

        reg = schemas["NodeRegistration"]
        assert reg["required"] == ["node_id", "host", "port"]

        dr = schemas["DeploymentRequest"]
        assert dr["required"] == ["tenant_id", "model_name"]
        assert dr["properties"]["gpu_count"]["default"] == 1

        cs = schemas["ClusterStatus"]
        assert cs["properties"]["status"]["enum"] == ["ok", "degraded", "starting"]

    def test_result_is_valid_openapi_dict(self) -> None:
        """The returned dict is JSON-serializable (no non-serializable types)."""
        spec = build_distributed_spec()
        # Should not raise
        json.dumps(spec)

    def test_empty_prefix_no_effect_on_structure(self) -> None:
        """Passing an empty string as base_url still produces a valid spec."""
        spec = build_distributed_spec(base_url="")
        assert spec["servers"] == [{"url": "", "description": "Coordinator API"}]
        # All other structure must be intact
        assert "/admin/v1/nodes" in spec["paths"]

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8080",
            "https://coord.example.com",
            "http://localhost",
            "https://10.0.0.1:443",
        ],
    )
    def test_various_base_url_formats(self, url: str) -> None:
        """Various URL formats should all work."""
        spec = build_distributed_spec(base_url=url)
        assert spec["servers"][0]["url"] == url

    def test_path_object_has_methods(self) -> None:
        """Each path entry contains HTTP method keys (get, post, delete, etc.)."""
        spec = build_distributed_spec()
        for path, methods in spec["paths"].items():
            assert isinstance(methods, dict)
            assert len(methods) >= 1

    def test_tags_are_consistent(self) -> None:
        """Every operation has a tags list with at least one tag."""
        spec = build_distributed_spec()
        for path, methods in spec["paths"].items():
            for method, operation in methods.items():
                assert "tags" in operation
                assert len(operation["tags"]) >= 1

    def test_security_not_required_on_public_endpoints(self) -> None:
        """Some endpoints do not require auth (health, status, etc.)."""
        spec = build_distributed_spec()
        public_paths = [
            "/admin/v1/cluster/status",
            "/v1/federation/health",
            "/v1/federation/peers",
            "/admin/v1/recovery/status",
            "/admin/v1/recovery/history",
        ]
        for path in public_paths:
            for method in spec["paths"][path].values():
                assert "security" not in method, f"{path} should be public"

    def test_path_parameters_required_flag(self) -> None:
        """Path parameters have required: true."""
        spec = build_distributed_spec()
        path_spec = spec["paths"]["/admin/v1/nodes/{node_id}/ready"]
        params = path_spec["post"]["parameters"]
        assert params[0]["name"] == "node_id"
        assert params[0]["required"] is True

    def test_query_parameters_default(self) -> None:
        """Query parameter min_gpu_memory_gb has a default of 80."""
        spec = build_distributed_spec()
        path_spec = spec["paths"]["/api/v1/regions"]
        params = path_spec["get"]["parameters"]
        param = [p for p in params if p["name"] == "min_gpu_memory_gb"][0]
        assert param["schema"]["default"] == 80

    def test_for_response_401_indicating_auth(self) -> None:
        """Endpoints with security have a 401 response documented."""
        spec = build_distributed_spec()
        methods = spec["paths"]["/admin/v1/nodes"]["get"]
        assert "401" in methods["responses"]


class TestGenerateSpec:
    """Tests for generate_spec()."""

    def test_returns_spec_dict_when_no_path(self) -> None:
        """Calling generate_spec() without output_path returns the spec dict."""
        spec = generate_spec()
        assert isinstance(spec, dict)
        assert spec["openapi"] == "3.1.0"

    def test_writes_to_file_when_output_path_given(self) -> None:
        """With an output_path, the spec is written as JSON to that file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            spec = generate_spec(output_path=tmp_path)
            # Must still return the spec dict
            assert isinstance(spec, dict)

            # File must exist and contain valid JSON matching the dict
            written = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
            assert written == spec
        finally:
            os.unlink(tmp_path)

    def test_output_is_pretty_printed(self) -> None:
        """The written JSON uses 2-space indentation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            generate_spec(output_path=tmp_path)
            raw = Path(tmp_path).read_text(encoding="utf-8")
            # Pretty-printed JSON has newlines between top-level keys
            assert "\n  " in raw
        finally:
            os.unlink(tmp_path)

    def test_custom_base_url_via_env_var(self) -> None:
        """The COORDINATOR_URL env var overrides the default base_url."""
        original = os.environ.get("COORDINATOR_URL")
        try:
            os.environ["COORDINATOR_URL"] = "https://custom-env.example.com"
            spec = generate_spec()
            assert spec["servers"][0]["url"] == "https://custom-env.example.com"
        finally:
            if original is not None:
                os.environ["COORDINATOR_URL"] = original
            else:
                os.environ.pop("COORDINATOR_URL", None)

    def test_none_output_path_returns_spec(self) -> None:
        """Passing None as output_path behaves like the default (no file write)."""
        spec = generate_spec(output_path=None)
        assert isinstance(spec, dict)
        assert spec["openapi"] == "3.1.0"

    def test_output_path_with_env_override(self) -> None:
        """Writing to file with a custom COORDINATOR_URL env var."""
        original = os.environ.get("COORDINATOR_URL")
        try:
            os.environ["COORDINATOR_URL"] = "https://with-file.example.com"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                tmp_path = f.name

            try:
                spec = generate_spec(output_path=tmp_path)
                written = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
                assert written["servers"][0]["url"] == "https://with-file.example.com"
            finally:
                os.unlink(tmp_path)
        finally:
            if original is not None:
                os.environ["COORDINATOR_URL"] = original
            else:
                os.environ.pop("COORDINATOR_URL", None)
