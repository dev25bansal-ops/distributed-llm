"""Tests: Docker builds, CI/CD workflows, secrets/security, proto compilation."""

from pathlib import Path

import pytest
import yaml

REPO_DIR = Path(__file__).parent.parent.parent


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check(path, *kw):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for k in kw:
        assert k in text, f"Missing '{k}' in {path}"
    return True


DOCKERFILE = REPO_DIR / "Dockerfile"
DOCKER_12_1 = REPO_DIR / "Dockerfile.cuda12.1"
DOCKER_12_6 = REPO_DIR / "Dockerfile.cuda12.6"
ENTRYPOINT = REPO_DIR / "docker-entrypoint.sh"
COMPOSE = REPO_DIR / "docker-compose.yml"
DOCKERIGNORE = REPO_DIR / ".dockerignore"
CI_YML = REPO_DIR / ".github" / "workflows" / "ci.yml"
RELEASE_YML = REPO_DIR / ".github" / "workflows" / "release.yml"
SECURITY_YML = REPO_DIR / ".github" / "workflows" / "security.yml"
CONTAINER_SCAN = REPO_DIR / ".github" / "workflows" / "container-scan.yml"
SECRETS_BASELINE = REPO_DIR / ".secrets.baseline"
INSTALL_SH = REPO_DIR / "install.sh"
PROTO = REPO_DIR / "proto" / "node.proto"


# ===========================================================================
# 12.1 Docker Builds
# ===========================================================================


class TestDockerBuilds:
    def test_dockerfile_exists(self):
        assert DOCKERFILE.exists()

    def test_multi_stage(self):
        _check(DOCKERFILE, "AS builder", "AS runtime")

    def test_healthcheck(self):
        _check(DOCKERFILE, "HEALTHCHECK", "curl -f http://localhost:8000/health")

    def test_non_root_user(self):
        _check(DOCKERFILE, "USER distllm", "groupadd -r distllm")

    def test_proto_copied(self):
        _check(DOCKERFILE, "COPY proto/")

    def test_exposes_ports(self):
        _check(DOCKERFILE, "EXPOSE 8000 50051")

    def test_entrypoint(self):
        _check(DOCKERFILE, "docker-entrypoint.sh")

    def test_cuda12_1_exists(self):
        assert DOCKER_12_1.exists()

    def test_cuda12_6_exists(self):
        assert DOCKER_12_6.exists()

    def test_cuda12_1_healthcheck(self):
        _check(DOCKER_12_1, "HEALTHCHECK")

    def test_cuda12_6_healthcheck(self):
        _check(DOCKER_12_6, "HEALTHCHECK")

    def test_entrypoint_exists(self):
        assert ENTRYPOINT.exists()

    def test_entrypoint_env_vars(self):
        _check(ENTRYPOINT, "DISTLLM_NODE_ID", "DISTLLM_MODEL", "DISTLLM_START_LAYER",
               "DISTLLM_END_LAYER", "DISTLLM_TOTAL_LAYERS")

    def test_compose_exists(self):
        assert COMPOSE.exists()

    def test_compose_three_services(self):
        dc = _load(COMPOSE)
        assert all(s in dc["services"] for s in ("coordinator", "node_0", "node_1"))

    def test_compose_gpu_reservations(self):
        dc = _load(COMPOSE)
        for name, svc in dc["services"].items():
            devs = svc.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
            assert any("nvidia" in str(d) for d in devs), f"{name} missing GPU"

    def test_dockerignore_proto_not_excluded(self):
        text = open(DOCKERIGNORE, encoding="utf-8").read()
        assert "proto/" not in text

    def test_dockerignore_pycache(self):
        _check(DOCKERIGNORE, "__pycache__/")


# ===========================================================================
# 12.2 CI/CD Pipelines
# ===========================================================================


class TestCICDWorkflows:
    def test_ci_yml_exists(self):
        assert CI_YML.exists()

    def test_ci_has_lint_job(self):
        _check(CI_YML, "lint:")

    def test_ci_has_test_job(self):
        _check(CI_YML, "\n  test:")

    def test_ci_has_security_sast(self):
        _check(CI_YML, "security-sast")

    def test_ci_has_security_deps(self):
        _check(CI_YML, "security-deps")

    def test_ci_has_security_secrets(self):
        _check(CI_YML, "security-secrets")

    def test_ci_has_docker_build(self):
        _check(CI_YML, "build-docker")

    def test_ci_has_fuzz_test(self):
        _check(CI_YML, "fuzz-test")

    def test_ci_has_load_test(self):
        _check(CI_YML, "load-test")

    def test_release_yml_exists(self):
        assert RELEASE_YML.exists()

    def test_release_trigger_on_version_tags(self):
        _check(RELEASE_YML, "tags:", "v*")

    def test_release_has_docker_push(self):
        _check(RELEASE_YML, "build-docker", "ghcr.io")

    def test_release_has_pypi(self):
        _check(RELEASE_YML, "publish-pypi")

    def test_release_has_helm(self):
        _check(RELEASE_YML, "publish-helm")

    def test_release_has_sbom(self):
        _check(RELEASE_YML, "generate-sbom")

    def test_security_yml_exists(self):
        assert SECURITY_YML.exists()

    def test_security_workflow_weekly_schedule(self):
        _check(SECURITY_YML, "schedule", "cron")

    def test_container_scan_yml_exists(self):
        assert CONTAINER_SCAN.exists()

    def test_container_scan_has_trivy(self):
        _check(CONTAINER_SCAN, "trivy")

    def test_container_scan_has_grype(self):
        _check(CONTAINER_SCAN, "grype")


# ===========================================================================
# 12.3 Secrets & Security
# ===========================================================================


class TestSecretsAndSecurity:
    def test_secrets_baseline_exists(self):
        assert SECRETS_BASELINE.exists()

    def test_install_sh_exists(self):
        assert INSTALL_SH.exists()

    def test_install_sh_checks_docker(self):
        _check(INSTALL_SH, "docker")

    def test_install_sh_cuda_selection(self):
        _check(INSTALL_SH, "Dockerfile.cuda")

    def test_install_sh_checks_prerequisites(self):
        _check(INSTALL_SH, "curl", "git")

    def test_install_sh_health_check(self):
        _check(INSTALL_SH, "health")

    def test_bandit_in_ci(self):
        _check(CI_YML, "bandit")

    def test_pip_audit_in_ci(self):
        _check(CI_YML, "pip-audit")

    def test_detect_secrets_in_ci(self):
        _check(CI_YML, "detect-secrets")


# ===========================================================================
# 12.4 Proto
# ===========================================================================


class TestProto:
    def test_proto_file_exists(self):
        assert PROTO.exists()

    def test_proto_package(self):
        _check(PROTO, "package distributed_llm;")

    def test_proto_forward_pass_request(self):
        _check(PROTO, "ForwardPassRequest")

    def test_proto_forward_pass_response(self):
        _check(PROTO, "ForwardPassResponse")

    def test_proto_health_check(self):
        _check(PROTO, "HealthCheck")

    def test_proto_error_codes(self):
        for code in ("UNKNOWN", "MODEL_ERROR", "OOM", "TIMEOUT", "INVALID_INPUT", "NODE_UNREACHABLE", "CIRCUIT_BREAKER_OPEN"):
            _check(PROTO, code)

    def test_proto_node_service_rpcs(self):
        for rpc in ("ForwardPass", "HealthCheck", "GetNodeInfo", "MoEForward", "Ping"):
            _check(PROTO, f"rpc {rpc}")

    def test_proto_coordinator_service_rpcs(self):
        for rpc in ("RegisterNode", "Infer", "StreamInfer", "ListModels"):
            _check(PROTO, f"rpc {rpc}")

    def test_proto_kv_cache(self):
        _check(PROTO, "KVCache", "KVLayerCache")

    def test_proto_gossip(self):
        _check(PROTO, "GossipAdvertisement", "GossipResponse")

    def test_proto_moe(self):
        _check(PROTO, "MoEForwardRequest", "MoEForwardResponse")
