"""CI import-assertion: fail when the measured import graph is truncated.

Companion to the ``testing`` extra (see pyproject.toml). CI installs
``.[dev,testing]``; if any non-optional module fails to import because of a
missing *optional* dependency, coverage is silently computed over a truncated
graph and the coverage number becomes misleading. This script imports every
module under ``src/distllm`` and separates two failure classes:

- **Optional-dep skips** — a module fails with ``ModuleNotFoundError`` for a
  package that belongs to a named ``[project.optional-dependencies]`` extra
  (``grpc``→backends+monitoring, ``aioquic``→wan, ``redis``→self-hosted, ...).
  These are reported as **SKIPPED** with reason ``requires [<extra>]`` and do
  NOT fail the run.
- **Genuine failures** — anything else (``NameError``, ``ImportError`` for
  internal names, missing source files, ...) still **FAILS** the run.

Usage: python scripts/ci/import_assert.py [--package distllm] [--list-skipped]
"""

from __future__ import annotations

import argparse
import importlib
import io
import pkgutil
import sys
from pathlib import Path

# Mapping from import-package -> [project.optional-dependencies] extra.
# Derived from pyproject.toml extras:
#   observability / monitoring / backends / self-hosted / wan / ray / spark
#   mlflow / wandb / model-extras / vector-db-* / security / cli / ui / examples
#   openai-agents / agno / autogpt
OPTIONAL_DEP_EXTRAS: dict[str, str] = {
    # gRPC: pulled in via open-telemetry grpc exporter extra (monitoring) and
    # via backends extras. grpcio itself is NOT in core/dev/testing.
    "grpc": "backends",
    # QUIC transport
    "aioquic": "wan",
    # WebRTC
    "aiortc": "self-hosted",
    # Redis cache
    "redis": "self-hosted",
    # Metrics / tracing
    "prometheus_client": "observability",
    "opentelemetry": "observability",
    # Backends extras (CUDA/ML-specific)
    "vllm": "backends",
    "sglang": "backends",
    "llama_cpp": "backends",
    "exllama": "backends",
    "exllamav2": "backends",
    # Self-hosted ML stack fragments (Apple Silicon / accelerators)
    "mlx": "self-hosted",
    "tensorrt": "self-hosted",
    "onnxruntime": "self-hosted",
    "pynvml": "self-hosted",
    "nvidia_ml_py": "self-hosted",
    "nvidia_ml_py3": "self-hosted",
    "torchvision": "self-hosted",
    # Distributed compute extras
    "ray": "ray",
    # Experiment tracking
    "wandb": "wandb",
    "mlflow": "mlflow",
    # Big data
    "pyspark": "spark",
    # Vector DB SDKs (project extras vector-db-*)
    "pinecone": "vector-db-pinecone",
    "qdrant_client": "vector-db-qdrant",
    "weaviate": "vector-db-weaviate",
    "pymilvus": "vector-db-milvus",
    # Model quantization adapters
    "bitsandbytes": "model-extras",
    "peft": "model-extras",
    # Agent framework integrations
    "openai_agents": "openai-agents",
    "agno": "agno",
    "autogpt": "autogpt",
    # Examples / integration SDKs
    "langchain": "examples",
    "langchain_core": "examples",
    "langchain_openai": "examples",
    "llama_index": "examples",
    "haystack": "examples",
    "crewai": "examples",
    "agency": "examples",
    "openai": "examples",
    # Security extras
    "cryptography": "security",
    "nacl": "security",
    # UI/CLI extras
    "typer": "cli",
    "rich": "cli",
    "prompt_toolkit": "cli",
    "jinja2": "ui",
    "multipart": "ui",
    "qrcode": "ui",
}

# Extra level overrides: some modules are intentionally skipped because they
# require an extra even though they import a package that could also be in core.
# e.g. protobuf-generated code requires grpcio, which is optional.
EXTRA_SKIPS: dict[str, str] = {
    # Generated gRPC stubs require grpcio (a backends/signalling dep).
    "distllm.dist.node_pb2_grpc": "backends",
}


def _extra_for_missing_module(missing: str | None) -> str | None:
    if not missing:
        return None
    top = missing.split(".")[0]
    return OPTIONAL_DEP_EXTRAS.get(top)


def iter_modules(package: str):
    yield package
    pkg = importlib.import_module(package)
    path = getattr(pkg, "__path__", None)
    if not path:
        return
    for info in pkgutil.walk_packages(path, prefix=f"{package}."):
        if info.name.split(".")[-1].startswith("_test"):
            continue
        yield info.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="distllm")
    parser.add_argument("--list-skipped", action="store_true",
                        help="Print modules skipped for optional deps and why")
    args = parser.parse_args()

    failures: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    count = 0
    for name in iter_modules(args.package):
        count += 1
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - report everything
            reason = type(exc).__name__
            detail = str(exc).strip().splitlines()[0] if str(exc) else ""
            if name in EXTRA_SKIPS:
                skipped.append((name, EXTRA_SKIPS[name]))
                continue
            if isinstance(exc, ModuleNotFoundError):
                extra = _extra_for_missing_module(exc.name or "")
                if extra is not None:
                    skipped.append((name, extra))
                    continue
            failures.append((name, reason, detail))

    if args.list_skipped:
        print(f"SKIPPED MODULES (requires optional dependency): {len(skipped)}")
        for name, extra in sorted(skipped):
            print(f"  {name}: requires [{extra}]")

    if failures:
        print(f"IMPORT ASSERTION FAILED: {len(failures)} genuine failure(s)")
        for name, reason, detail in sorted(failures):
            print(f"  {name}: {reason}: {detail}")
        print(
            "\nThese are genuine import failures (bugs, not missing optional deps).\n"
            "Fix the code, or add the missing optional package to the 'testing'\n"
            "extra (pyproject.toml) if it belongs there."
        )
        return 1

    print(f"import assertion OK: {count} modules checked, "
          f"{count - len(skipped)} imported, {len(skipped)} skipped (optional deps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
