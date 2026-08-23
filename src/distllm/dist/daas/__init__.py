"""DaaS (Distributed as a Service) subpackage.

Provides server-side coordination for DaaS deployments including
tenant-aware dispatch, usage metering, multi-tenant resource isolation,
and tenant-aware load balancing.
"""

from __future__ import annotations

from distllm.dist.daas.tenant_dispatcher import (
    Priority,
    Request,
    TenantDispatcher,
    TenantQueueState,
)

from distllm.dist.daas.usage_meter import TenantUsage, UsageMeter

from distllm.dist.daas.resource_isolation import (
    ResourceIsolator,
    TenantResourceState,
)

from distllm.dist.daas.load_balancer import (
    NodeInfo,
    NodeScore,
    TenantAwareLoadBalancer,
)

__all__ = [
    # tenant_dispatcher
    "TenantDispatcher",
    "TenantQueueState",
    "Request",
    "Priority",
    # usage_meter
    "UsageMeter",
    "TenantUsage",
    # resource_isolation
    "ResourceIsolator",
    "TenantResourceState",
    # load_balancer
    "TenantAwareLoadBalancer",
    "NodeInfo",
    "NodeScore",
]
