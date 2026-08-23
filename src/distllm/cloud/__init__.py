"""Cloud Provider SDK Wrappers.

Provides unified access to AWS, GCP, and Azure pricing and availability APIs,
as well as GPU spot market orchestration (RunPod, Vast.ai, Salad).
"""

from distllm.cloud.common import (
    AvailabilityChecker,
    AvailabilityInfo,
    GPUSpec,
    InstanceSpec,
    PriceQuote,
    PricingFetcher,
    ProviderSession,
    multi_provider_session,
)
from distllm.cloud.spot_orchestrator import (
    BidResult,
    BidStatus,
    CostReport,
    GPUInstance,
    GPU_ON_DEMAND_REFERENCE,
    GPUSpotMarket,
    InstanceStatus,
    normalize_gpu_name,
    Provider,
    ProviderProtocol,
    SpotOrchestrator,
)

__all__ = [
    # Common
    "AvailabilityChecker",
    "AvailabilityInfo",
    "GPUSpec",
    "InstanceSpec",
    "PriceQuote",
    "PricingFetcher",
    "ProviderSession",
    "multi_provider_session",
    # Spot orchestrator
    "BidResult",
    "BidStatus",
    "CostReport",
    "GPUInstance",
    "GPU_ON_DEMAND_REFERENCE",
    "GPUSpotMarket",
    "InstanceStatus",
    "normalize_gpu_name",
    "Provider",
    "ProviderProtocol",
    "SpotOrchestrator",
]
