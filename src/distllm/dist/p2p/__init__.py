"""P2P networking subpackage for distributed inference.

Provides peer discovery, gossip protocol, load balancing,
routing, and transport abstractions for cross-node communication.
"""

from __future__ import annotations
from distllm.dist.p2p.discovery import FederationPeerDiscovery, PeerInfo
from distllm.dist.p2p.gossip import GossipProtocol
from distllm.dist.p2p.load_balancer import FederationLoadBalancer, RemoteClusterLoad
from distllm.dist.p2p.router import FederationRouter
from distllm.dist.p2p.transport import KVCacheTransfer as P2PTransport

__all__ = [
    "FederationPeerDiscovery",
    "PeerInfo",
    "GossipProtocol",
    "FederationLoadBalancer",
    "RemoteClusterLoad",
    "FederationRouter",
    "P2PTransport",
]
