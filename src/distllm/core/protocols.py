"""Protocol interfaces for distributed LLM components.

Defines abstract interfaces (using typing.Protocol) so that the
coordinator and other components depend on abstractions rather than
concrete implementations, enabling testability and flexibility.
"""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class INodeClient(Protocol):
    """Interface for a gRPC client connecting to a worker node."""
    host: str
    port: int

    def health_check(self) -> object:
        """Check node health. Returns a health response proto."""
        ...

    def forward(self, request: object) -> object:
        """Send a forward pass request to the node. Returns response proto."""
        ...

    def close(self) -> None:
        """Close the gRPC channel."""
        ...


@runtime_checkable
class ITokenizer(Protocol):
    """Interface for a tokenizer compatible with HuggingFace tokenizers."""
    eos_token_id: int | None
    bos_token_id: int | None
    pad_token_id: int | None
    vocab_size: int

    def encode(self, text: str, **kwargs: Any) -> list[int] | torch.Tensor:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int] | torch.Tensor, **kwargs: Any) -> str:
        """Decode token IDs back to text."""
        ...


@runtime_checkable
class IModelPartitioner(Protocol):
    """Interface for a model partitioner that loads and runs model layers."""
    full_model: object
    tokenizer: ITokenizer | None
    embed_input: Any | None
    is_last_node: bool

    def load_full_model(self) -> None:
        """Load the full model."""
        ...

    def load_layer_subset(self, start: int, end: int, total: int, device: str) -> None:
        """Load a subset of model layers."""
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through loaded layers."""
        ...

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states (last node only)."""
        ...


@runtime_checkable
class ICacheBackend(Protocol):
    """Interface for a prefix cache backend."""

    def lookup(self, tokens: list[int]) -> tuple[int, object]:
        """Lookup tokens in the cache. Returns (prefix_len, entry)."""
        ...

    def store(self, tokens: list[int], entry: object) -> None:
        """Store tokens and associated entry in the cache."""
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...


@runtime_checkable
class IMetricsExporter(Protocol):
    """Interface for a Prometheus-style metrics exporter."""

    def record(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        ...


@runtime_checkable
class INodeFactory(Protocol):
    """Factory interface for creating node clients."""

    def create_node(
        self,
        node_id: str,
        host: str,
        port: int,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        use_tls: bool = True,
        ca_cert: str | None = None,
    ) -> INodeClient:
        """Create a new node client."""
        ...


@runtime_checkable
class IResourceManager(Protocol):
    """Interface for node lifecycle and health management."""

    def check_circuit_breaker(self, node_id: str) -> bool:
        """Check if a node's circuit breaker is open."""
        ...

    def record_success(self, node_id: str) -> None:
        """Record a successful node operation."""
        ...

    def record_failure(self, node_id: str) -> None:
        """Record a node failure."""
        ...

    def health_check_all(self, nodes: dict[str, Any]) -> dict[str, dict]:
        """Check health of all registered nodes."""
        ...

    def close_all(self, nodes: dict[str, Any]) -> None:
        """Close all node connections."""
        ...


@runtime_checkable
class ICacheManager(Protocol):
    """Interface for cache management (prefix cache, KV cache)."""

    def lookup_prefix(self, tokens: list[int]) -> int:
        """Lookup prefix match length for tokens."""
        ...

    def maybe_chunk(self, tokens: list[int]) -> list[list[int]]:
        """Apply chunked prefill if enabled."""
        ...


@runtime_checkable
class ITokenGenerator(Protocol):
    """Interface for token sampling and generation."""

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Sample next token from logits."""
        ...

    def sample_batch(
        self,
        logits: torch.Tensor,
        batch: list[Any],
    ) -> tuple[torch.Tensor, list[dict[str, Any] | None]]:
        """Sample next tokens for a batch with constraints."""
        ...


@runtime_checkable
class IPipelineOrchestrator(Protocol):
    """Interface for distributed pipeline orchestration."""

    @property
    def nodes(self) -> dict[str, Any]:
        """Registered nodes."""
        ...

    @property
    def node_order(self) -> list[str]:
        """Ordered list of node IDs."""
        ...

    def register_node(self, node_id: str, host: str, port: int, start_layer: int, end_layer: int, **kwargs: Any) -> None:
        """Register a new node."""
        ...

    def validate_layer_assignment(self, node_id: str, start_layer: int, end_layer: int) -> None:
        """Validate layer assignment."""
        ...

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
    ) -> torch.Tensor:
        """Run input through all nodes via gRPC."""
        ...
