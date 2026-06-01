"""CLI entry point for the DistLLM Coordinator.

Extracted from coordinator.py to keep the Coordinator class focused on
orchestration logic.  Registered as the ``distllm-coordinator`` console
script in pyproject.toml.
"""

from __future__ import annotations

import argparse
import os

from loguru import logger

from distllm.config.settings import DistLLMSettings
from distllm.core.coordinator import Coordinator
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.debug import set_debug_mode
from distllm.dist.federation import FederationConfig


def _resolve_cluster_key() -> str | None:
    """Resolve cluster key from environment variable or file.

    Resolution order:
    1. DISTLLM_CLUSTER_KEY environment variable
    2. ~/.distllm/cluster_key file
    3. None (no key)
    """
    key = os.environ.get("DISTLLM_CLUSTER_KEY", "")
    if key:
        return key
    key_path = os.path.expanduser("~/.distllm/cluster_key")
    if os.path.isfile(key_path):
        try:
            with open(key_path) as f:
                return f.read().strip()
        except OSError:
            pass
    return None


def main():
    from distllm.config.resolver import ConfigResolver

    parser = argparse.ArgumentParser(description="DistLLM Coordinator")
    ConfigResolver._register_args(parser, ConfigResolver.COMMON_ARGS + ConfigResolver.COORDINATOR_ARGS)
    args = parser.parse_args()

    if args.validate_config:
        DistLLMSettings.validate_startup()
        logger.info("Config validation passed")
        return

    if args.debug:
        set_debug_mode(True)

    # Discover config path and load settings if available
    config_path = ConfigResolver._resolve_config_path("coordinator", args)
    settings = DistLLMSettings.from_yaml(config_path=config_path) if config_path else None

    federation_cfg = None
    if args.federate:
        federation_cfg = FederationConfig(
            enabled=True,
            cluster_id=args.federation_cluster_id,
            listen_port=args.federation_port,
            seed_nodes=args.federation_seed or [],
        )

    # Build CoordinatorConfig: YAML/env as base, CLI args override
    if settings is not None:
        config = CoordinatorConfig.from_settings(settings)
        config.model_name = args.model or config.model_name
        config.dtype = args.dtype or config.dtype
        config.port = args.port
        config.trust_remote_code = args.trust_remote_code or None
        config.cluster_key = args.cluster_key or config.cluster_key or _resolve_cluster_key()
        config.model_cache_dir = args.model_cache_dir or config.model_cache_dir
        config.redundancy = args.redundancy
        config.federation_config = federation_cfg or config.federation_config
    else:
        config = CoordinatorConfig(
            model_name=args.model,
            port=args.port,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code or None,
            cluster_key=args.cluster_key or _resolve_cluster_key(),
            model_cache_dir=args.model_cache_dir,
            redundancy=args.redundancy,
            federation_config=federation_cfg,
        )
    coordinator = Coordinator(config=config)
    coordinator._distribute_weights = args.distribute_weights

    # Initialize model router if chat_router config is available
    if settings is not None and getattr(settings, 'chat_router', None):
        cr = settings.chat_router
        if cr.enabled:
            coordinator.init_model_router(cr)

    if args.local:
        coordinator.load_local_model()
        if args.chat:
            logger.info(f"Model loaded: {args.model}")
            while True:
                prompt = input("\nPrompt (or 'quit' to exit): ")
                if prompt.lower() in ('quit', 'exit'):
                    break
                result = coordinator.generate(prompt, max_new_tokens=128)
                logger.info(f"Result: {result}")
        else:
            coordinator.start()
    else:
        if args.nodes:
            for i, node_str in enumerate(args.nodes):
                parts = node_str.split(":")
                coordinator.manual_register(
                    node_id=f"node_{i}",
                    host=parts[0],
                    port=int(parts[1]),
                    start_layer=int(parts[2]),
                    end_layer=int(parts[3]),
                    total_layers=args.total_layers,
                )
        coordinator.start()


if __name__ == "__main__":
    main()
