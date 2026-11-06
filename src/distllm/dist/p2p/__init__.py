"""P2P networking subpackage for distributed inference.

Provides peer discovery, gossip protocol, load balancing,
routing, and transport abstractions for cross-node communication.

Uses lazy ``__getattr__`` imports so that importing a light submodule
(e.g. ``distllm.dist.p2p.quic_transport``) does not force heavyweight
optional dependencies (torch, aioquic) to be installed.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}


def _register(module: str, *symbols: str) -> None:
    for sym in symbols:
        _LAZY_IMPORTS[sym] = (module, sym)


_register(
    "distllm.dist.p2p.discovery",
    "FederationPeerDiscovery", "PeerInfo",
)
_register("distllm.dist.p2p.gossip", "GossipProtocol")
_register(
    "distllm.dist.p2p.load_balancer",
    "FederationLoadBalancer", "RemoteClusterLoad",
)
_register("distllm.dist.p2p.router", "FederationRouter")
_register("distllm.dist.p2p.transport", "KVCacheTransfer")

__all__ = [
    "FederationPeerDiscovery",
    "PeerInfo",
    "GossipProtocol",
    "FederationLoadBalancer",
    "RemoteClusterLoad",
    "FederationRouter",
    "P2PTransport",
]

# Backward-compatibility alias: ``P2PTransport`` is the old public name
# for ``KVCacheTransfer`` exported by ``p2p.transport``.
_LAZY_ALIASES: dict[str, tuple[str, str]] = {
    "P2PTransport": ("distllm.dist.p2p.transport", "KVCacheTransfer"),
}


def __getattr__(name: str):
    if name in _LAZY_ALIASES:
        import importlib
        module_path, symbol = _LAZY_ALIASES[name]
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, symbol = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'distllm.dist.p2p' has no attribute {name!r}")


def __dir__():
    return list(__all__)
