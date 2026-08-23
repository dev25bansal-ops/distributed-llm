"""Bring Your Own GPU Marketplace — peer-to-peer GPU sharing.

Allows GPU owners to list their hardware for inference serving and
lets users request compute from the marketplace. Integrates with the
reputation system for trust-based matching.

Features:
- GPU listing with pricing, availability, and capability specs
- Job posting with model requirements, SLA, and budget
- Automatic matching of jobs to providers based on capability and price
- Trust/reputation-based filtering
- Usage metering and billing integration
- Provider earnings tracking
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.event_bus import EventBus
    from distllm.core.persistence import StorageBackend


class ListingStatus(enum.Enum):
    """GPU listing status."""
    ACTIVE = "active"
    BUSY = "busy"
    OFFLINE = "offline"
    PAUSED = "paused"


class JobStatus(enum.Enum):
    """Marketplace job status."""
    OPEN = "open"
    MATCHED = "matched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GPUListing:
    """A GPU listing in the marketplace."""
    listing_id: str
    provider_id: str
    provider_name: str = ""

    # Hardware specs
    gpu_name: str = ""
    gpu_memory_bytes: int = 0
    gpu_count: int = 1
    cpu_cores: int = 0
    ram_bytes: int = 0

    # Pricing (per GPU-hour)
    price_per_hour: float = 0.0
    price_per_million_tokens: float = 0.0
    currency: str = "USD"

    # Availability
    status: ListingStatus = ListingStatus.ACTIVE
    available_from: float = 0.0
    available_until: float = 0.0
    max_concurrent_jobs: int = 1
    current_jobs: int = 0

    # Capabilities
    supported_models: list[str] = field(default_factory=list)
    supported_dtypes: list[str] = field(default_factory=lambda: ["float16"])
    max_batch_size: int = 8
    supports_streaming: bool = True
    supports_quantization: bool = False
    supports_lora: bool = False

    # Network
    region: str = ""
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0

    # Carbon
    carbon_intensity: float = 0.0  # gCO2/kWh
    renewable_pct: float = 0.0

    # Trust
    reputation_score: float = 0.5
    total_jobs_completed: int = 0
    uptime_pct: float = 100.0

    # Metadata
    source: str = "peer"  # "peer", "cloud", "federated"
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        return (
            self.status == ListingStatus.ACTIVE
            and self.current_jobs < self.max_concurrent_jobs
        )

    @property
    def effective_score(self) -> float:
        """Score for matching: balances price, reputation, and capability."""
        price_score = 1.0 / max(self.price_per_hour, 0.01)
        rep_score = self.reputation_score
        perf_score = self.gpu_memory_bytes / (100 * 1024**3)  # Normalized to 100GB
        return (price_score * 0.3 + rep_score * 0.5 + perf_score * 0.2)


@dataclass
class MarketplaceJob:
    """A compute job posted to the marketplace."""
    job_id: str
    requester_id: str

    # Requirements
    model_name: str = ""
    min_gpu_memory_bytes: int = 0
    min_gpu_count: int = 1
    min_cpu_cores: int = 0
    min_ram_bytes: int = 0
    required_dtype: str = "float16"
    requires_streaming: bool = True
    requires_quantization: bool = False
    requires_lora: bool = False

    # Budget
    max_price_per_hour: float = 0.0
    max_price_per_million_tokens: float = 0.0
    max_budget_total: float = 0.0

    # SLA
    max_latency_ms: float = 5000.0
    min_uptime_pct: float = 99.0
    preferred_regions: list[str] = field(default_factory=list)
    min_reputation: float = 0.3

    # Status
    status: JobStatus = JobStatus.OPEN
    matched_listing_id: str = ""
    matched_provider_id: str = ""

    # Usage tracking
    tokens_generated: int = 0
    cost_accumulated: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    # Metadata
    created_at: float = field(default_factory=time.time)
    priority: int = 2  # 0=critical, 1=high, 2=normal, 3=low
    tags: list[str] = field(default_factory=list)

    @property
    def duration_hours(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) / 3600.0
        if self.started_at:
            return (time.time() - self.started_at) / 3600.0
        return 0.0


@dataclass
class ProviderEarnings:
    """Earnings summary for a GPU provider."""
    provider_id: str
    total_earnings: float = 0.0
    total_gpu_hours: float = 0.0
    total_tokens_served: int = 0
    total_jobs: int = 0
    current_month_earnings: float = 0.0
    pending_payout: float = 0.0
    last_payout_at: float = 0.0


class Marketplace:
    """GPU marketplace for peer-to-peer compute sharing.

    Manages listings, job matching, and usage tracking.

    Args:
        event_bus: Optional event bus for emitting marketplace events.
        backend: Optional :class:`StorageBackend` for durable storage.
            When provided all mutations are persisted and the marketplace
            is pre-loaded from the backend on construction.  When
            ``None`` the marketplace operates entirely in-memory.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        backend: StorageBackend | None = None,
    ):
        self._listings: dict[str, GPUListing] = {}
        self._jobs: dict[str, MarketplaceJob] = {}
        self._earnings: dict[str, ProviderEarnings] = {}
        self._event_bus = event_bus
        self._backend = backend
        self._lock = __import__("threading").Lock()

        if self._backend is not None:
            self._backend.initialize()
            self._load_from_backend()

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event through the event bus if configured."""
        if self._event_bus is not None:
            try:
                self._event_bus.publish(event_type, payload)
            except Exception:
                logger.exception(f"Failed to emit {event_type}")

    # ── Backend hydration & persistence ─────────────────────────────────

    def _load_from_backend(self) -> None:
        """Load persisted state into the in-memory caches."""
        assert self._backend is not None
        # Listings
        for d in self._backend.load_all_listings():
            listing = GPUListing(
                listing_id=d["listing_id"],
                provider_id=d["provider_id"],
                provider_name=d.get("provider_name", ""),
                gpu_name=d.get("gpu_name", ""),
                gpu_memory_bytes=d.get("gpu_memory_bytes", 0),
                gpu_count=d.get("gpu_count", 1),
                cpu_cores=d.get("cpu_cores", 0),
                ram_bytes=d.get("ram_bytes", 0),
                price_per_hour=d.get("price_per_hour", 0.0),
                price_per_million_tokens=d.get("price_per_million_tokens", 0.0),
                currency=d.get("currency", "USD"),
                status=ListingStatus(d.get("status", "active")),
                available_from=d.get("available_from", 0.0),
                available_until=d.get("available_until", 0.0),
                max_concurrent_jobs=d.get("max_concurrent_jobs", 1),
                current_jobs=d.get("current_jobs", 0),
                supported_models=d.get("supported_models", []),
                supported_dtypes=d.get("supported_dtypes", ["float16"]),
                max_batch_size=d.get("max_batch_size", 8),
                supports_streaming=d.get("supports_streaming", True),
                supports_quantization=d.get("supports_quantization", False),
                supports_lora=d.get("supports_lora", False),
                region=d.get("region", ""),
                bandwidth_mbps=d.get("bandwidth_mbps", 0.0),
                latency_ms=d.get("latency_ms", 0.0),
                carbon_intensity=d.get("carbon_intensity", 0.0),
                renewable_pct=d.get("renewable_pct", 0.0),
                reputation_score=d.get("reputation_score", 0.5),
                total_jobs_completed=d.get("total_jobs_completed", 0),
                uptime_pct=d.get("uptime_pct", 100.0),
                source=d.get("source", "peer"),
                created_at=d["created_at"],
                last_updated=d["last_updated"],
                tags=d.get("tags", []),
            )
            self._listings[listing.listing_id] = listing

        # Jobs
        for d in self._backend.load_all_jobs():
            job = MarketplaceJob(
                job_id=d["job_id"],
                requester_id=d["requester_id"],
                model_name=d.get("model_name", ""),
                min_gpu_memory_bytes=d.get("min_gpu_memory_bytes", 0),
                min_gpu_count=d.get("min_gpu_count", 1),
                min_cpu_cores=d.get("min_cpu_cores", 0),
                min_ram_bytes=d.get("min_ram_bytes", 0),
                required_dtype=d.get("required_dtype", "float16"),
                requires_streaming=d.get("requires_streaming", True),
                requires_quantization=d.get("requires_quantization", False),
                requires_lora=d.get("requires_lora", False),
                max_price_per_hour=d.get("max_price_per_hour", 0.0),
                max_price_per_million_tokens=d.get("max_price_per_million_tokens", 0.0),
                max_budget_total=d.get("max_budget_total", 0.0),
                max_latency_ms=d.get("max_latency_ms", 5000.0),
                min_uptime_pct=d.get("min_uptime_pct", 99.0),
                preferred_regions=d.get("preferred_regions", []),
                min_reputation=d.get("min_reputation", 0.3),
                status=JobStatus(d.get("status", "open")),
                matched_listing_id=d.get("matched_listing_id", ""),
                matched_provider_id=d.get("matched_provider_id", ""),
                tokens_generated=d.get("tokens_generated", 0),
                cost_accumulated=d.get("cost_accumulated", 0.0),
                started_at=d.get("started_at", 0.0),
                completed_at=d.get("completed_at", 0.0),
                created_at=d["created_at"],
                priority=d.get("priority", 2),
                tags=d.get("tags", []),
            )
            self._jobs[job.job_id] = job

        # Provider earnings
        for listing in self._listings.values():
            pid = listing.provider_id
            if pid not in self._earnings:
                # Check backend for persisted earnings
                ed = self._backend.load_provider_earnings(pid)
                if ed:
                    self._earnings[pid] = ProviderEarnings(
                        provider_id=ed["provider_id"],
                        total_earnings=ed.get("total_earnings", 0.0),
                        total_gpu_hours=ed.get("total_gpu_hours", 0.0),
                        total_tokens_served=ed.get("total_tokens_served", 0),
                        total_jobs=ed.get("total_jobs", 0),
                        current_month_earnings=ed.get("current_month_earnings", 0.0),
                        pending_payout=ed.get("pending_payout", 0.0),
                        last_payout_at=ed.get("last_payout_at", 0.0),
                    )
                else:
                    self._earnings[pid] = ProviderEarnings(provider_id=pid)

    def _persist_listing(self, listing: GPUListing) -> None:
        """Persist a listing to the backend (no-op if no backend)."""
        if self._backend is None:
            return
        self._backend.save_listing({
            "listing_id": listing.listing_id,
            "provider_id": listing.provider_id,
            "provider_name": listing.provider_name,
            "gpu_name": listing.gpu_name,
            "gpu_memory_bytes": listing.gpu_memory_bytes,
            "gpu_count": listing.gpu_count,
            "cpu_cores": listing.cpu_cores,
            "ram_bytes": listing.ram_bytes,
            "price_per_hour": listing.price_per_hour,
            "price_per_million_tokens": listing.price_per_million_tokens,
            "currency": listing.currency,
            "status": listing.status.value,
            "available_from": listing.available_from,
            "available_until": listing.available_until,
            "max_concurrent_jobs": listing.max_concurrent_jobs,
            "current_jobs": listing.current_jobs,
            "supported_models": listing.supported_models,
            "supported_dtypes": listing.supported_dtypes,
            "max_batch_size": listing.max_batch_size,
            "supports_streaming": listing.supports_streaming,
            "supports_quantization": listing.supports_quantization,
            "supports_lora": listing.supports_lora,
            "region": listing.region,
            "bandwidth_mbps": listing.bandwidth_mbps,
            "latency_ms": listing.latency_ms,
            "carbon_intensity": listing.carbon_intensity,
            "renewable_pct": listing.renewable_pct,
            "reputation_score": listing.reputation_score,
            "total_jobs_completed": listing.total_jobs_completed,
            "uptime_pct": listing.uptime_pct,
            "source": listing.source,
            "created_at": listing.created_at,
            "last_updated": listing.last_updated,
            "tags": listing.tags,
        })

    def _persist_job(self, job: MarketplaceJob) -> None:
        """Persist a job to the backend (no-op if no backend)."""
        if self._backend is None:
            return
        self._backend.save_job({
            "job_id": job.job_id,
            "requester_id": job.requester_id,
            "model_name": job.model_name,
            "min_gpu_memory_bytes": job.min_gpu_memory_bytes,
            "min_gpu_count": job.min_gpu_count,
            "min_cpu_cores": job.min_cpu_cores,
            "min_ram_bytes": job.min_ram_bytes,
            "required_dtype": job.required_dtype,
            "requires_streaming": job.requires_streaming,
            "requires_quantization": job.requires_quantization,
            "requires_lora": job.requires_lora,
            "max_price_per_hour": job.max_price_per_hour,
            "max_price_per_million_tokens": job.max_price_per_million_tokens,
            "max_budget_total": job.max_budget_total,
            "max_latency_ms": job.max_latency_ms,
            "min_uptime_pct": job.min_uptime_pct,
            "preferred_regions": job.preferred_regions,
            "min_reputation": job.min_reputation,
            "status": job.status.value,
            "matched_listing_id": job.matched_listing_id,
            "matched_provider_id": job.matched_provider_id,
            "tokens_generated": job.tokens_generated,
            "cost_accumulated": job.cost_accumulated,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "created_at": job.created_at,
            "priority": job.priority,
            "tags": job.tags,
        })

    def _persist_earnings(self, earnings: ProviderEarnings) -> None:
        """Persist provider earnings to the backend (no-op if no backend)."""
        if self._backend is None:
            return
        self._backend.save_provider_earnings({
            "provider_id": earnings.provider_id,
            "total_earnings": earnings.total_earnings,
            "total_gpu_hours": earnings.total_gpu_hours,
            "total_tokens_served": earnings.total_tokens_served,
            "total_jobs": earnings.total_jobs,
            "current_month_earnings": earnings.current_month_earnings,
            "pending_payout": earnings.pending_payout,
            "last_payout_at": earnings.last_payout_at,
        })

    # ── Listing Management ──────────────────────────────────────────────

    def create_listing(
        self,
        provider_id: str,
        gpu_name: str,
        gpu_memory_bytes: int,
        price_per_hour: float,
        gpu_count: int = 1,
        **kwargs: Any,
    ) -> GPUListing:
        """Create a new GPU listing."""
        listing_id = f"gpu-{uuid.uuid4().hex[:12]}"
        listing = GPUListing(
            listing_id=listing_id,
            provider_id=provider_id,
            gpu_name=gpu_name,
            gpu_memory_bytes=gpu_memory_bytes,
            gpu_count=gpu_count,
            price_per_hour=price_per_hour,
            **kwargs,
        )
        with self._lock:
            self._listings[listing_id] = listing
            if provider_id not in self._earnings:
                self._earnings[provider_id] = ProviderEarnings(provider_id=provider_id)
        self._persist_listing(listing)
        logger.info(f"Created listing {listing_id}: {gpu_name} at ${price_per_hour}/hr")
        return listing

    def update_listing(self, listing_id: str, **updates: Any) -> GPUListing | None:
        """Update a GPU listing."""
        with self._lock:
            listing = self._listings.get(listing_id)
            if not listing:
                return None
            old_status = listing.status
            for key, value in updates.items():
                if hasattr(listing, key):
                    setattr(listing, key, value)
            listing.last_updated = time.time()

        self._persist_listing(listing)
        # Emit status change event outside the lock
        if "status" in updates and listing.status != old_status:
            self._emit("listing.status_changed", {
                "listing_id": listing_id,
                "provider_id": listing.provider_id,
                "old_status": old_status.value,
                "new_status": listing.status.value,
            })
        return listing

    def remove_listing(self, listing_id: str) -> bool:
        """Remove a GPU listing."""
        with self._lock:
            removed = self._listings.pop(listing_id, None) is not None
        if removed and self._backend is not None:
            self._backend.delete_listing(listing_id)
        return removed

    def get_listing(self, listing_id: str) -> GPUListing | None:
        """Get a specific listing."""
        with self._lock:
            return self._listings.get(listing_id)

    def list_listings(
        self,
        status: ListingStatus | None = None,
        min_gpu_memory: int = 0,
        max_price: float = 0.0,
        region: str = "",
        max_carbon_gco2_kwh: float = 0.0,
        source: str = "",
    ) -> list[GPUListing]:
        """List available GPU listings with optional filters.

        Args:
            status: Filter by listing status.
            min_gpu_memory: Minimum GPU memory in bytes.
            max_price: Maximum price per hour.
            region: Filter by region.
            max_carbon_gco2_kwh: Maximum carbon intensity (gCO2/kWh). 0 = no filter.
            source: Filter by source ("peer", "cloud", "federated").
        """
        with self._lock:
            results = list(self._listings.values())

        if status:
            results = [l for l in results if l.status == status]
        if min_gpu_memory:
            results = [l for l in results if l.gpu_memory_bytes >= min_gpu_memory]
        if max_price > 0:
            results = [l for l in results if l.price_per_hour <= max_price]
        if region:
            results = [l for l in results if l.region == region]
        if max_carbon_gco2_kwh > 0:
            results = [l for l in results if l.carbon_intensity > 0 and l.carbon_intensity <= max_carbon_gco2_kwh]
        if source:
            results = [l for l in results if l.source == source]

        return sorted(results, key=lambda l: l.effective_score, reverse=True)

    # ── Job Management ──────────────────────────────────────────────────

    def post_job(
        self,
        requester_id: str,
        model_name: str,
        min_gpu_memory_bytes: int = 0,
        max_price_per_hour: float = 0.0,
        **kwargs: Any,
    ) -> MarketplaceJob:
        """Post a compute job to the marketplace."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = MarketplaceJob(
            job_id=job_id,
            requester_id=requester_id,
            model_name=model_name,
            min_gpu_memory_bytes=min_gpu_memory_bytes,
            max_price_per_hour=max_price_per_hour,
            **kwargs,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._persist_job(job)
        logger.info(f"Posted job {job_id}: {model_name}")
        return job

    def match_job(self, job_id: str, max_carbon_gco2_kwh: float = 0.0) -> GPUListing | None:
        """Find the best matching GPU listing for a job.

        Matching criteria:
        1. Hardware meets minimum requirements
        2. Price within budget
        3. Reputation meets minimum
        4. SLA requirements met
        5. Carbon intensity within limit (if specified)
        6. Sorted by effective score (price + reputation + performance)

        Args:
            job_id: The job to match.
            max_carbon_gco2_kwh: Maximum carbon intensity. 0 = no filter.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.OPEN:
                return None

            candidates = []
            for listing in self._listings.values():
                if not listing.is_available:
                    continue
                if listing.gpu_memory_bytes < job.min_gpu_memory_bytes:
                    continue
                if job.max_price_per_hour > 0 and listing.price_per_hour > job.max_price_per_hour:
                    continue
                if listing.reputation_score < job.min_reputation:
                    continue
                if job.min_uptime_pct > 0 and listing.uptime_pct < job.min_uptime_pct:
                    continue
                if job.preferred_regions and listing.region not in job.preferred_regions:
                    continue
                if max_carbon_gco2_kwh > 0 and listing.carbon_intensity > 0:
                    if listing.carbon_intensity > max_carbon_gco2_kwh:
                        continue
                candidates.append(listing)

            if not candidates:
                return None

            # Sort by effective score
            candidates.sort(key=lambda l: l.effective_score, reverse=True)
            best = candidates[0]

            # Match
            job.status = JobStatus.MATCHED
            job.matched_listing_id = best.listing_id
            job.matched_provider_id = best.provider_id
            best.current_jobs += 1
            if best.current_jobs >= best.max_concurrent_jobs:
                best.status = ListingStatus.BUSY

            logger.info(f"Matched job {job_id} to listing {best.listing_id}")

        self._persist_job(job)
        self._persist_listing(best)
        # Emit outside the lock to avoid deadlocks
        self._emit("job.matched", {
            "job_id": job_id,
            "listing_id": best.listing_id,
            "provider_id": best.provider_id,
            "gpu_name": best.gpu_name,
            "price_per_hour": best.price_per_hour,
        })
        return best

    def start_job(self, job_id: str) -> bool:
        """Mark a job as running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.MATCHED:
                return False
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

        self._persist_job(job)
        self._emit("job.started", {
            "job_id": job_id,
            "started_at": job.started_at,
            "matched_provider_id": job.matched_provider_id,
        })
        return True

    def complete_job(self, job_id: str, tokens_generated: int = 0) -> bool:
        """Mark a job as completed and update earnings."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.RUNNING:
                return False

            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.tokens_generated = tokens_generated

            # Calculate cost
            listing = self._listings.get(job.matched_listing_id)
            earnings: ProviderEarnings | None = None
            if listing:
                hours = job.duration_hours
                cost = hours * listing.price_per_hour
                job.cost_accumulated = cost

                # Update provider earnings
                earnings = self._earnings.get(job.matched_provider_id)
                if earnings:
                    earnings.total_earnings += cost
                    earnings.total_gpu_hours += hours
                    earnings.total_tokens_served += tokens_generated
                    earnings.total_jobs += 1
                    earnings.pending_payout += cost
                    earnings.current_month_earnings += cost

                # Update listing
                listing.current_jobs = max(0, listing.current_jobs - 1)
                listing.total_jobs_completed += 1
                if listing.status == ListingStatus.BUSY:
                    listing.status = ListingStatus.ACTIVE

            logger.info(
                f"Completed job {job_id}: {tokens_generated} tokens, "
                f"${job.cost_accumulated:.4f}"
            )

        self._persist_job(job)
        if listing:
            self._persist_listing(listing)
        if earnings:
            self._persist_earnings(earnings)
        # Snapshot for event payload
        self._emit("job.completed", {
            "job_id": job_id,
            "tokens_generated": tokens_generated,
            "cost_accumulated": job.cost_accumulated,
            "duration_hours": job.duration_hours,
            "matched_provider_id": job.matched_provider_id,
        })
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
                return False

            prev_status = job.status.value
            listing = self._listings.get(job.matched_listing_id)
            if listing:
                listing.current_jobs = max(0, listing.current_jobs - 1)
                if listing.status == ListingStatus.BUSY:
                    listing.status = ListingStatus.ACTIVE

            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()

        self._persist_job(job)
        if listing:
            self._persist_listing(listing)
        self._emit("job.cancelled", {
            "job_id": job_id,
            "previous_status": prev_status,
            "matched_provider_id": job.matched_provider_id,
        })
        return True

    def fail_job(self, job_id: str, error: str = "") -> bool:
        """Mark a job as failed.

        Args:
            job_id: The job to fail.
            error: Optional error description.

        Returns:
            True if the job was transitioned to FAILED.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False

            listing = self._listings.get(job.matched_listing_id)
            if listing:
                listing.current_jobs = max(0, listing.current_jobs - 1)
                if listing.status == ListingStatus.BUSY:
                    listing.status = ListingStatus.ACTIVE

            job.status = JobStatus.FAILED
            job.completed_at = time.time()

        self._persist_job(job)
        if listing:
            self._persist_listing(listing)
        self._emit("job.failed", {
            "job_id": job_id,
            "error": error,
            "matched_provider_id": job.matched_provider_id,
        })
        return True

    def get_job(self, job_id: str) -> MarketplaceJob | None:
        """Get a specific job."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(
        self,
        requester_id: str = "",
        status: JobStatus | None = None,
    ) -> list[MarketplaceJob]:
        """List jobs with optional filters."""
        with self._lock:
            results = list(self._jobs.values())
        if requester_id:
            results = [j for j in results if j.requester_id == requester_id]
        if status:
            results = [j for j in results if j.status == status]
        return sorted(results, key=lambda j: j.created_at, reverse=True)

    # ── Earnings ────────────────────────────────────────────────────────

    def get_provider_earnings(self, provider_id: str) -> ProviderEarnings | None:
        """Get earnings summary for a provider."""
        with self._lock:
            return self._earnings.get(provider_id)

    def get_marketplace_stats(self) -> dict[str, Any]:
        """Get overall marketplace statistics."""
        with self._lock:
            listings = list(self._listings.values())
            jobs = list(self._jobs.values())
            return {
                "total_listings": len(listings),
                "active_listings": sum(
                    1 for l in listings if l.status == ListingStatus.ACTIVE
                ),
                "total_jobs": len(jobs),
                "open_jobs": sum(1 for j in jobs if j.status == JobStatus.OPEN),
                "running_jobs": sum(1 for j in jobs if j.status == JobStatus.RUNNING),
                "completed_jobs": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
                "total_volume_usd": sum(j.cost_accumulated for j in jobs),
                "total_tokens_served": sum(j.tokens_generated for j in jobs),
            "avg_price_per_hour": (
                sum(l.price_per_hour for l in listings) / len(listings)
                if listings else 0.0
            ),
        }

    # ── Cloud Integration ──────────────────────────────────────────────

    def add_cloud_listings(self, cloud_prices: list[dict[str, Any]]) -> int:
        """Import cloud GPU instances as marketplace listings.

        Converts cloud pricing data (from PricingManager) into GPUListing
        entries so they appear alongside peer listings in unified matching.

        Args:
            cloud_prices: List of dicts with keys: provider, instance_type,
                region, gpu_type, gpu_count, gpu_memory_gb, on_demand_price,
                spot_price, carbon_intensity.

        Returns:
            Number of cloud listings created.
        """
        created = 0
        for p in cloud_prices:
            listing_id = f"cloud-{p.get('provider', '')}-{p.get('instance_type', '')}-{p.get('region', '')}"
            gpu_bytes = int(p.get("gpu_memory_gb", 0) * 1024**3)
            on_demand = p.get("on_demand_price", 0.0)
            spot = p.get("spot_price", 0.0)
            price = spot if spot > 0 else on_demand

            listing = GPUListing(
                listing_id=listing_id,
                provider_id=p.get("provider", ""),
                provider_name=p.get("provider", ""),
                gpu_name=p.get("gpu_type", p.get("instance_type", "")),
                gpu_memory_bytes=gpu_bytes,
                gpu_count=p.get("gpu_count", 1),
                price_per_hour=price,
                region=p.get("region", ""),
                carbon_intensity=p.get("carbon_intensity", 0.0),
                source="cloud",
                status=ListingStatus.ACTIVE,
                reputation_score=1.0,
                uptime_pct=99.9,
            )
            with self._lock:
                self._listings[listing_id] = listing
            self._persist_listing(listing)
            created += 1
        logger.info(f"Imported {created} cloud listings into marketplace")
        return created

    def get_unified_listings(
        self,
        min_gpu_memory: int = 0,
        max_price: float = 0.0,
        max_carbon_gco2_kwh: float = 0.0,
        max_latency_ms: float = 0.0,
    ) -> list[GPUListing]:
        """Get all listings (peer + cloud) sorted by effective score.

        This is the main entry point for cross-provider matching.
        """
        return self.list_listings(
            min_gpu_memory=min_gpu_memory,
            max_price=max_price,
            max_carbon_gco2_kwh=max_carbon_gco2_kwh,
        )


@dataclass
class CloudCostModel:
    """Converts cloud instance prices into marketplace-compatible format.

    Maps cloud provider instance types to GPUListing-compatible dicts
    that can be imported via Marketplace.add_cloud_listings().
    """

    @staticmethod
    def from_pricing_data(prices: list[Any]) -> list[dict[str, Any]]:
        """Convert InstancePricing objects to marketplace-compatible dicts.

        Args:
            prices: List of InstancePricing or objects with compatible attributes.

        Returns:
            List of dicts ready for Marketplace.add_cloud_listings().
        """
        result = []
        for p in prices:
            if hasattr(p, "__dict__"):
                d = p.__dict__
            elif isinstance(p, dict):
                d = p
            else:
                continue
            result.append({
                "provider": d.get("provider", ""),
                "instance_type": d.get("instance_type", ""),
                "region": d.get("region", ""),
                "gpu_type": d.get("gpu_type", ""),
                "gpu_count": d.get("gpu_count", 1),
                "gpu_memory_gb": d.get("gpu_memory_gb", 0.0),
                "on_demand_price": d.get("on_demand_price", d.get("price_per_hour", 0.0)),
                "spot_price": d.get("spot_price", 0.0),
                "carbon_intensity": d.get("carbon_intensity", 0.0),
            })
        return result
