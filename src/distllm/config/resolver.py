"""Unified configuration resolver.

Loads ``DistLLMSettings`` with full precedence:

    CLI overrides > env vars (``DISTLLM__*``) > YAML > defaults

All three entry points (coordinator, API server, worker) use this resolver
instead of duplicating config logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.config.settings import DistLLMSettings


def _find_config(candidates: list[str]) -> str | None:
    """Return the first existing path from *candidates*, or ``None``."""
    for path in candidates:
        expanded = os.path.expanduser(os.path.expandvars(path))
        if os.path.exists(expanded):
            return expanded
    return None


class ConfigResolver:
    """Resolves configuration from YAML, env vars, and CLI overrides.

    Typical usage::

        resolver = ConfigResolver.from_cli("coordinator", args)
        settings = resolver.resolve()
    """

    COMMON_CONFIG_CANDIDATES = [
        "config.yaml",
        "~/.config/distllm/config.yaml",
        "/etc/distllm/config.yaml",
    ]

    # ── Entry-point-specific argument groups ──────────────────────────

    COORDINATOR_ARGS: list[dict] = [
        {"name": "--model", "type": str, "required": True, "help": "Model name"},
        {"name": "--port", "type": int, "default": 50050, "help": "gRPC port"},
        {"name": "--dtype", "type": str, "default": "float16", "choices": ["float16", "float32", "bfloat16"]},
        {"name": "--nodes", "type": str, "nargs": "+", "help": "host:port:start:end per node"},
        {"name": "--total-layers", "type": int, "help": "Total layers in model"},
        {"name": "--local", "action": "store_true", "help": "Run full model locally"},
        {"name": "--chat", "action": "store_true", "help": "Start interactive chat mode"},
        {"name": "--trust-remote-code", "action": "store_true"},
        {"name": "--cluster-key", "type": str, "default": None, "help": "Shared cluster auth key"},
        {"name": "--model-cache-dir", "type": str, "default": None, "help": "Model cache directory"},
        {"name": "--redundancy", "type": int, "default": 1, "help": "Redundant peers per stage"},
        {"name": "--min-reputation", "type": float, "default": 0.0, "help": "Minimum reputation (0.0-1.0)"},
        {"name": "--federate", "action": "store_true", "help": "Enable federation"},
        {"name": "--federation-cluster-id", "type": str, "default": "default"},
        {"name": "--federation-port", "type": int, "default": 50060, "help": "Federation listen port"},
        {"name": "--federation-seed", "type": str, "default": None, "action": "append"},
        {"name": "--distribute-weights", "action": "store_true", "default": True},
    ]

    API_ARGS: list[dict] = [
        {"name": "--model", "type": str, "default": None, "help": "Model name (overrides config)"},
        {"name": "--host", "type": str, "default": None, "help": "Server host (overrides config)"},
        {"name": "--port", "type": int, "default": None, "help": "Server port (overrides config)"},
        {"name": "--dtype", "type": str, "default": None, "help": "Model dtype (overrides config)"},
        {"name": "--local", "action": "store_true", "help": "Load model locally"},
        {"name": "--config", "type": str, "default": None, "help": "Path to config.yaml"},
        {"name": "--quantization", "type": str, "default": "none",
         "choices": ["none", "bitsandbytes_4bit", "bitsandbytes_8bit", "gptq"]},
        {"name": "--nodes", "type": str, "nargs": "+", "help": "host:port:start:end per node"},
        {"name": "--total-layers", "type": int, "help": "Total layers in model"},
    ]

    WORKER_ARGS: list[dict] = [
        {"name": "--node-id", "type": str, "required": True, "help": "Unique node identifier"},
        {"name": "--model", "type": str, "required": True, "help": "HuggingFace model name"},
        {"name": "--start-layer", "type": int, "required": True},
        {"name": "--end-layer", "type": int, "required": True},
        {"name": "--total-layers", "type": int, "required": True},
        {"name": "--port", "type": int, "default": 50051, "help": "gRPC port"},
        {"name": "--coordinator-host", "type": str, "default": "localhost"},
        {"name": "--coordinator-port", "type": int, "default": 50050},
        {"name": "--device", "type": str, "default": "auto", "choices": ["auto", "cuda", "rocm", "mps", "xpu", "cpu"]},
        {"name": "--dtype", "type": str, "default": "float16", "choices": ["float16", "float32", "bfloat16"]},
        {"name": "--quantization-method", "type": str, "default": "none", "choices": ["none", "bnb_4bit", "bnb_8bit"]},
        {"name": "--expert-ids", "type": int, "nargs": "*", "default": []},
        {"name": "--insecure", "action": "store_true", "help": "Disable TLS for gRPC (dev only)"},
        {"name": "--tls-cert", "type": str, "default": None},
        {"name": "--tls-key", "type": str, "default": None},
        {"name": "--tls-ca", "type": str, "default": None},
        {"name": "--cluster-key", "type": str, "default": None},
        {"name": "--model-cache-dir", "type": str, "default": None},
        {"name": "--max-workers", "type": int, "default": 4},
        {"name": "--weight-source", "type": str, "default": None, "help": "host:port for P2P weight pull"},
        {"name": "--privacy-split", "action": "store_true"},
        {"name": "--privacy-prefix-layers", "type": int, "default": 0},
        {"name": "--privacy-suffix-layers", "type": int, "default": 0},
        {"name": "--compression-method", "type": str, "default": "none",
         "choices": ["none", "ptq_int8", "ptq_int4", "pruning_structured", "distillation", "auto"]},
        {"name": "--pruning-ratio", "type": float, "default": 0.0},
        {"name": "--distillation-teacher", "type": str, "default": None},
    ]

    COMMON_ARGS: list[dict] = [
        {"name": "--debug", "action": "store_true", "help": "Enable debug mode"},
        {"name": "--validate-config", "action": "store_true", "help": "Validate config and exit"},
    ]

    ARG_NAME_KEY = "name"

    @staticmethod
    def _register_args(parser: argparse.ArgumentParser, definitions: list[dict]) -> None:
        """Register argument definitions on *parser* without mutating the originals."""
        for kwargs in definitions:
            kwargs = dict(kwargs)  # shallow copy to avoid mutating class-level dicts
            name = kwargs.pop(ConfigResolver.ARG_NAME_KEY)
            parser.add_argument(name, **kwargs)

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        config_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ):
        self._config_path = config_path
        self._cli_overrides = cli_overrides or {}

    @classmethod
    def from_cli(cls, entry_point: str, argv: list[str] | None = None) -> "ConfigResolver":
        """Parse CLI args for *entry_point* and return a resolver.

        Args:
            entry_point: One of ``"coordinator"``, ``"api"``, ``"worker"``.
            argv: CLI arguments (defaults to ``sys.argv[1:]``).
        """
        parser = argparse.ArgumentParser(
            description=f"DistLLM {entry_point.title()}",
            allow_abbrev=False,
        )

        if entry_point == "coordinator":
            group_defs = cls.COMMON_ARGS + cls.COORDINATOR_ARGS
        elif entry_point == "api":
            group_defs = cls.COMMON_ARGS + cls.API_ARGS
        elif entry_point == "worker":
            group_defs = cls.COMMON_ARGS + cls.WORKER_ARGS
        else:
            raise ValueError(f"Unknown entry_point: {entry_point}")

        cls._register_args(parser, group_defs)

        args = parser.parse_args(argv)

        # Early exit for validate-config
        if getattr(args, "validate_config", False):
            cls._validate_only()
            raise SystemExit(0)

        # Determine config path
        config_path = cls._resolve_config_path(entry_point, args)

        # Build CLI overrides
        cli_overrides = cls._build_overrides(entry_point, args)

        return cls(config_path=config_path, cli_overrides=cli_overrides or None)

    # ── Resolution ───────────────────────────────────────────────────

    def resolve(self) -> DistLLMSettings:
        """Load and return a fully resolved ``DistLLMSettings`` instance."""
        return DistLLMSettings.from_yaml(
            config_path=self._config_path,
            cli_overrides=self._cli_overrides,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_only() -> None:
        DistLLMSettings.validate_startup()
        print("Config validation passed")

    @staticmethod
    def _resolve_config_path(entry_point: str, args: argparse.Namespace) -> str | None:
        # API server accepts an explicit --config path
        if entry_point == "api" and getattr(args, "config", None):
            return args.config

        # Auto-discover: cwd, then common locations
        candidates = list(ConfigResolver.COMMON_CONFIG_CANDIDATES)
        # Also check relative to the entry point's source directory
        if entry_point == "api":
            script_dir = Path(__file__).resolve().parent.parent
            candidates.insert(0, str(script_dir / ".." / "config.yaml"))
        elif entry_point == "coordinator":
            script_dir = Path(__file__).resolve().parent.parent
            candidates.insert(0, str(script_dir / ".." / "config.yaml"))

        found = _find_config(candidates)
        if entry_point != "worker" and found:
            logger.info(f"Using config: {found}")
        return found

    @staticmethod
    def _build_overrides(entry_point: str, args: argparse.Namespace) -> dict[str, Any]:
        overrides: dict[str, Any] = {}

        if entry_point == "api":
            if getattr(args, "model", None):
                overrides.setdefault("model", {})["name"] = args.model
            if getattr(args, "dtype", None):
                overrides.setdefault("model", {})["dtype"] = args.dtype
            if getattr(args, "host", None):
                overrides.setdefault("coordinator", {})["host"] = args.host
            if getattr(args, "port", None):
                overrides.setdefault("coordinator", {})["api_port"] = args.port
            q = getattr(args, "quantization", "none")
            if q and q != "none":
                overrides.setdefault("quantization", {})["method"] = q

        return overrides or None
