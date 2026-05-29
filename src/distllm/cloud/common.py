"""Cloud Provider SDK Wrappers — common abstractions.

Base classes and interfaces for cloud provider integrations.
All providers implement PricingFetcher and AvailabilityChecker.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from loguru import logger


@dataclass
class GPUSpec:
    """Specification for a GPU accelerator."""
    name: str  # e.g., "A100", "V100", "T4"
    memory_gb: float
    generation: str = ""  # e.g., "Ampere", "Volta"
    cuda_cores: int = 0
    tensor_cores: int = 0
    tdp_watts: int = 0


@dataclass
class InstanceSpec:
    """Full specification for a GPU instance type."""
    provider: str
    instance_type: str
    region: str = ""
    gpu: GPUSpec | None = None
    gpu_count: int = 1
    vcpus: int = 0
    ram_gb: float = 0.0
    storage_gb: float = 0.0
    network_gbps: float = 0.0

    @property
    def total_gpu_memory_gb(self) -> float:
        if self.gpu:
            return self.gpu.memory_gb * self.gpu_count
        return 0.0


@dataclass
class PriceQuote:
    """A price quote for an instance type."""
    provider: str
    instance_type: str
    region: str
    on_demand_hourly: float = 0.0
    spot_hourly: float = 0.0
    reserved_1yr_hourly: float = 0.0
    reserved_3yr_hourly: float = 0.0
    currency: str = "USD"
    timestamp: float = field(default_factory=time.time)

    @property
    def spot_discount_pct(self) -> float:
        if self.on_demand_hourly <= 0:
            return 0.0
        return (1 - self.spot_hourly / self.on_demand_hourly) * 100


@dataclass
class AvailabilityInfo:
    """Availability information for an instance type."""
    provider: str
    instance_type: str
    region: str
    available: bool = True
    spot_available: bool = True
    capacity_status: str = "available"  # "available", "limited", "unavailable"
    spot_interrupt_rate: float = 0.0  # 0.0-1.0


class PricingFetcher(ABC):
    """Abstract base class for cloud pricing fetchers."""

    @abstractmethod
    def fetch_gpu_pricing(self, regions: list[str] | None = None) -> list[PriceQuote]:
        """Fetch GPU instance pricing.

        Args:
            regions: Optional list of regions. None = all known regions.

        Returns:
            List of PriceQuote for GPU instances.
        """

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (aws, gcp, azure)."""


class AvailabilityChecker(ABC):
    """Abstract base class for cloud availability checkers."""

    @abstractmethod
    def check_availability(
        self, instance_type: str, region: str
    ) -> AvailabilityInfo:
        """Check availability for an instance type in a region.

        Args:
            instance_type: The instance type to check.
            region: The region to check.

        Returns:
            AvailabilityInfo with current availability status.
        """

    @abstractmethod
    def list_gpu_instances(self, region: str | None = None) -> list[InstanceSpec]:
        """List available GPU instance types.

        Args:
            region: Optional region filter.

        Returns:
            List of InstanceSpec for available GPU instances.
        """


class ProviderSession:
    """Context manager for cloud provider sessions.

    Manages authentication, connection pooling, and cleanup.

    Usage::

        with ProviderSession("aws", region="us-east-1") as session:
            prices = session.fetch_pricing()
            avail = session.check_availability("p4d.24xlarge", "us-east-1")
    """

    def __init__(self, provider: str, region: str = "", **config: Any):
        self.provider = provider
        self.region = region
        self.config = config
        self._fetcher: PricingFetcher | None = None
        self._checker: AvailabilityChecker | None = None
        self._closed = False

    def __enter__(self) -> ProviderSession:
        self._init_provider()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            logger.debug(f"Provider session closed: {self.provider}")

    def _init_provider(self) -> None:
        if self.provider == "aws":
            from distllm.cloud.aws import AWSAvailabilityChecker, AWSPricingFetcher
            self._fetcher = AWSPricingFetcher()
            self._checker = AWSAvailabilityChecker()
        elif self.provider == "gcp":
            from distllm.cloud.gcp import GCPAvailabilityChecker, GCPPricingFetcher
            self._fetcher = GCPPricingFetcher()
            self._checker = GCPAvailabilityChecker()
        elif self.provider == "azure":
            from distllm.cloud.azure import AzureAvailabilityChecker, AzurePricingFetcher
            self._fetcher = AzurePricingFetcher()
            self._checker = AzureAvailabilityChecker()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def fetch_pricing(self, regions: list[str] | None = None) -> list[PriceQuote]:
        if not self._fetcher:
            self._init_provider()
        return self._fetcher.fetch_pricing(regions)  # type: ignore

    def check_availability(self, instance_type: str, region: str = "") -> AvailabilityInfo:
        if not self._checker:
            self._init_provider()
        return self._checker.check_availability(instance_type, region or self.region)  # type: ignore

    def list_gpu_instances(self, region: str | None = None) -> list[InstanceSpec]:
        if not self._checker:
            self._init_provider()
        return self._checker.list_gpu_instances(region or self.region)  # type: ignore


@contextmanager
def multi_provider_session(
    providers: list[str], region: str = ""
) -> Generator[dict[str, ProviderSession], None, None]:
    """Context manager for multiple provider sessions.

    Usage::

        with multi_provider_session(["aws", "gcp", "azure"]) as sessions:
            for name, session in sessions.items():
                prices = session.fetch_pricing()
    """
    sessions: dict[str, ProviderSession] = {}
    try:
        for provider in providers:
            sessions[provider] = ProviderSession(provider, region)
            sessions[provider].__enter__()
        yield sessions
    finally:
        for session in sessions.values():
            try:
                session.close()
            except Exception:
                pass
