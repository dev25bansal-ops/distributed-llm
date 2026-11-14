"""Packaging-artifact regressions (2026-11-13 packaging audit).

Every failure here is one an installed user hits *after* pip install — the
repo checkout can be entirely healthy while the wheel/sdist drop the files or
declare floors that the shipped code cannot run under:

* pyproject floors for grpcio/protobuf were below the gencode the wheel ships
  (node_pb2_grpc.py raises RuntimeError below grpcio 1.80.0; node_pb2.py
  raises VersionError below protobuf 6.31.1), so a resolver whose pick
  satisfied the declaration still produced an unimportable install.
* Dashboard static assets and the correctness registry were not declared as
  package data, so setuptools dropped them from wheel and sdist.
* The `distllm.dist` package-data glob promised "*.proto" but no .proto
  existed there.
* integrations/docker/docker-compose.gpu.yml had two YAML merge keys in one
  mapping (a construct error for docker compose's Go parser).
* The kustomize overlays pointed a single `patches` entry with `target:` at a
  two-document file, which kustomize v5 rejects.
* The Helm chart's poddisruptionbudget.yaml template called an undefined
  helper and was ungated; the chart could not render at all.

These tests re-derive each requirement from the artifacts (the gencode, the
compose file, the kustomize sources) rather than restating constants, so a
future regeneration that bumps the gencode fails here until the floors move
with it.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MANIFEST = REPO_ROOT / "MANIFEST.in"
GENCODE_GRPC = REPO_ROOT / "src" / "distllm" / "dist" / "node_pb2_grpc.py"
GENCODE_PB2 = REPO_ROOT / "src" / "distllm" / "dist" / "node_pb2.py"
PROTO_ROOT = REPO_ROOT / "proto" / "node.proto"
PROTO_PKG = REPO_ROOT / "src" / "distllm" / "dist" / "node.proto"
COMPOSE_GPU = REPO_ROOT / "integrations" / "docker" / "docker-compose.gpu.yml"
KUSTOMIZE = REPO_ROOT / "integrations" / "kubernetes" / "kustomize"
HELM_TEMPLATES = REPO_ROOT / "integrations" / "kubernetes" / "helm" / "distllm" / "templates"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _deps() -> list[str]:
    return _pyproject()["project"]["dependencies"]


def _dep_floor(name: str) -> tuple[int, ...]:
    """Parse the declared floor of `name` out of [project].dependencies."""
    for dep in _deps():
        m = re.match(rf"{re.escape(name)}>=([0-9][0-9.]*)", dep)
        if m:
            return tuple(int(part) for part in m.group(1).split("."))
    raise AssertionError(f"no floor declared for {name!r} in dependencies: {_deps()}")


def _geq(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Version-tuple >= that treats (7, 35) == (7, 35, 0)."""
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) >= b + (0,) * (width - len(b))


# ===========================================================================
# grpcio / protobuf floors must cover the shipped gencode
# ===========================================================================


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_grpcio_floor_covers_shipped_gencode():
    # The shipped node_pb2_grpc.py hard-requires its generator's grpcio.
    text = GENCODE_GRPC.read_text(encoding="utf-8")
    m = re.search(r"GRPC_GENERATED_VERSION = '([0-9.]+)'", text)
    assert m, "GRPC_GENERATED_VERSION not found in node_pb2_grpc.py"
    required = tuple(int(p) for p in m.group(1).split("."))
    floor = _dep_floor("grpcio")
    assert _geq(floor, required), (
        f"pyproject declares grpcio>={'.'.join(map(str, floor))} but the shipped "
        f"gencode raises RuntimeError below grpcio>={m.group(1)}"
    )


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_protobuf_floor_covers_shipped_gencode():
    # node_pb2.py validates the runtime against the generator's 3-tuple.
    text = GENCODE_PB2.read_text(encoding="utf-8")
    m = re.search(
        r"ValidateProtobufRuntimeVersion\(\s*_runtime_version\.Domain\.PUBLIC,\s*"
        r"(\d+),\s*(\d+),\s*(\d+),",
        text,
    )
    assert m, "ValidateProtobufRuntimeVersion tuple not found in node_pb2.py"
    required = tuple(int(g) for g in m.groups())
    floor = _dep_floor("protobuf")
    assert _geq(floor, required), (
        f"pyproject declares protobuf>={'.'.join(map(str, floor))} but the shipped "
        f"gencode raises VersionError below {'.'.join(map(str, required))}"
    )


# ===========================================================================
# package data: dashboard assets, proto, correctness registry
# ===========================================================================


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_dashboard_assets_declared_as_package_data():
    data = _pyproject()["tool"]["setuptools"]["package-data"]
    assert "distllm.dashboard" in data, (
        "dashboard static assets are not declared as package data; installed "
        "builds serve 'Dashboard not found'"
    )
    globs = " ".join(data["distllm.dashboard"])
    assert "static/" in globs and "static_v2/" in globs


def test_manifest_covers_dashboard_assets():
    text = MANIFEST.read_text(encoding="utf-8")
    assert "distllm/dashboard/static" in text
    assert "distllm/dashboard/static_v2" in text


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_proto_ships_as_package_data_with_content():
    # The declared glob must match a real file, byte-identical to the root
    # proto (the old promise pointed at a nonexistent src/distllm/dist/*.proto).
    assert PROTO_PKG.exists(), (
        "src/distllm/dist/node.proto is missing; the distllm.dist '*.proto' "
        "package-data declaration matches nothing"
    )
    assert (
        PROTO_PKG.read_bytes() == PROTO_ROOT.read_bytes()
    ), "packaged node.proto drifted from proto/node.proto"
    data = _pyproject()["tool"]["setuptools"]["package-data"]
    assert "*.proto" in data["distllm.dist"]


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_correctness_registry_shipped_as_package_data():
    data = _pyproject()["tool"]["setuptools"]["package-data"]
    assert "correctness_registry.jsonl" in data.get("distllm.core", []), (
        "the committed determinism evidence is not shipped; an installed copy "
        "reports replay_from_seed(1234, 8, 128) as matches=False"
    )
    # And the file it points at must exist with at least one record.
    registry = REPO_ROOT / "src" / "distllm" / "core" / "correctness_registry.jsonl"
    assert registry.exists()
    assert len(registry.read_text(encoding="utf-8").strip().splitlines()) >= 1


def test_correctness_registry_default_write_is_not_site_packages():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from distllm.core import correctness_harness as ch

    reg = ch.DeterminismRegistry()
    # Recording must never target the package directory (read-only in
    # site-packages); reads still consult the shipped baseline.
    assert reg.path != ch.DEFAULT_REGISTRY_PATH
    assert ch.DEFAULT_REGISTRY_PATH in reg.read_paths


# ===========================================================================
# docker compose: no duplicate merge keys
# ===========================================================================


def _duplicate_merge_key_paths(path: Path) -> list[str]:
    """Paths of mappings containing more than one `<<` key."""
    node = yaml.compose(path.read_text(encoding="utf-8"))
    bad: list[str] = []

    def walk(n, chain: str) -> None:
        if isinstance(n, yaml.MappingNode):
            merges = [k for k, _ in n.value if getattr(k, "value", None) == "<<"]
            if len(merges) > 1:
                bad.append(chain)
            for k, v in n.value:
                walk(v, f"{chain}/{getattr(k, 'value', '?')}")
        elif isinstance(n, yaml.SequenceNode):
            for i, v in enumerate(n.value):
                walk(v, f"{chain}[{i}]")

    walk(node, "")
    return bad


def test_compose_gpu_has_no_duplicate_merge_keys():
    # Two `<<:` keys in one mapping is a construct error for docker compose's
    # Go parser ("mapping key \"<<\" already defined"), rejecting the file.
    assert _duplicate_merge_key_paths(COMPOSE_GPU) == []


def test_compose_gpu_merges_resolve_to_image_and_gpu():
    doc = yaml.safe_load(COMPOSE_GPU.read_text(encoding="utf-8"))
    for name in ("coordinator", "worker-1", "worker-2"):
        svc = doc["services"][name]
        assert svc.get("image", "").startswith("distllm:"), name
        devices = svc["deploy"]["resources"]["reservations"]["devices"]
        assert any(d.get("driver") == "nvidia" for d in devices), name


# ===========================================================================
# kustomize: patches entries point at single-document files
# ===========================================================================


@pytest.mark.parametrize("overlay", ["dev", "staging", "prod"])
def test_overlay_patches_are_single_document_and_match_target(overlay: str):
    kdir = KUSTOMIZE / "overlays" / overlay
    kustomization = yaml.safe_load((kdir / "kustomization.yaml").read_text(encoding="utf-8"))
    checked = 0
    for entry in kustomization.get("patches", []):
        if "path" not in entry or "target" not in entry:
            continue  # inline patches with target are fine
        docs = [
            d for d in yaml.safe_load_all((kdir / entry["path"]).read_text(encoding="utf-8")) if d
        ]
        assert len(docs) == 1, (
            f"{overlay}: patches entry targets {entry['target']} but "
            f"{entry['path']} has {len(docs)} documents; kustomize v5 rejects "
            "a targeted multi-document patch"
        )
        assert docs[0]["kind"] == entry["target"]["kind"], (
            f"{overlay}: {entry['path']} is a {docs[0]['kind']}, target says "
            f"{entry['target']['kind']}"
        )
        assert docs[0]["metadata"]["name"] == entry["target"]["name"], (
            f"{overlay}: {entry['path']} patches "
            f"{docs[0]['metadata']['name']!r}, target says {entry['target']['name']!r}"
        )
        checked += 1
    assert checked >= 2, f"{overlay}: expected coordinator+worker patch entries"


@pytest.mark.parametrize("overlay", ["dev", "staging", "prod"])
def test_overlay_declares_worker_patch_file(overlay: str):
    kdir = KUSTOMIZE / "overlays" / overlay
    assert (kdir / "worker-patch.yaml").exists()
    assert (kdir / "coordinator-patch.yaml").exists()
    # Both files must exist AND be referenced by name, not by accident.
    text = (kdir / "kustomization.yaml").read_text(encoding="utf-8")
    assert "path: worker-patch.yaml" in text
    assert "path: coordinator-patch.yaml" in text


# ===========================================================================
# Helm chart: every include resolves to a defined template
# ===========================================================================


def test_helm_templates_have_no_undefined_includes():
    defined: set[str] = set()
    includes: dict[str, set[str]] = {}
    for tpl in HELM_TEMPLATES.glob("*"):
        text = tpl.read_text(encoding="utf-8")
        for m in re.finditer(r'\{\{-?\s*define\s+"([^"]+)"', text):
            defined.add(m.group(1))
        found = set(re.findall(r'\{\{-?\s*include\s+"([^"]+)"', text))
        if found:
            includes[tpl.name] = found
    missing = {name: sorted(inc - defined) for name, inc in includes.items() if inc - defined}
    assert not missing, f"includes without define: {missing}"


def test_helm_chart_has_single_gated_pdb_template():
    pdbs = sorted(p.name for p in HELM_TEMPLATES.glob("*pdb*"))
    assert pdbs == ["pdb.yaml"], (
        f"expected exactly one PDB template, found {pdbs} — duplicate PDB "
        "templates render two PodDisruptionBudgets for one workload"
    )
    text = (HELM_TEMPLATES / "pdb.yaml").read_text(encoding="utf-8")
    # Gated: values can disable it, and the sibling gate value exists.
    assert "{{- if .Values.podDisruptionBudget.enabled }}" in text
    values = yaml.safe_load(
        (REPO_ROOT / "integrations" / "kubernetes" / "helm" / "distllm" / "values.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "enabled" in values["podDisruptionBudget"]
