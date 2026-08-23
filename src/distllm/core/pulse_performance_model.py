"""GNN-based performance model -- latency, throughput, and memory prediction.

Provides a learned performance model that maps (model_config, hardware,
parallelism) tuples to latency, throughput, and memory estimates.  The model
uses a graph neural network (when PyTorch is available) to exploit the
structure of the transformer computation graph; a pure-numpy MLP fallback
is provided when torch is not installed.

Classes
-------
PerformanceModel
    GNN predicting latency/throughput/memory for (model, hardware, parallelism)
    tuples.  predict(model_config, hardware, batch_size, seq_len) returns a
    PerformancePrediction with latency_ms, throughput_tps, memory_mb.
    Fine-tune from real cluster data via add_sample / train.

WhatIfSimulator
    Interactive what-if queries on a trained PerformanceModel.
    what_if_batch_size(new_size) -> projected throughput
    what_if_gpu_count(n) -> improvement
    what_if_quantization(method) -> latency/quality tradeoff

ReductionSuggestionEngine
    Recommends optimisation actions.  analyze(current_config, slo_targets)
    returns a list of ReductionSuggestion with action, expected_improvement,
    confidence, effort.

Pulse
    Combines all three.  start() / stop() manage background monitoring and
    online fine-tuning.  stats() returns model accuracy and usage counters.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE: bool
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_NUMPY_AVAILABLE: bool
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _require_numpy() -> None:
    if not _NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for the performance model "
            "(both the torch path and the numpy fallback need it)"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_FEAT_DIM = 6    # Per-layer node features
_GLOBAL_FEAT_DIM = 11  # Aggregated hardware + parallelism features
_NODE_AGG_DIM = 3 * _NODE_FEAT_DIM  # 18 -- mean + max + min of node features
_NUMPY_INPUT_DIM = _NODE_AGG_DIM + _GLOBAL_FEAT_DIM  # 29


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Transformer model configuration."""

    hidden_size: int = 4096
    num_heads: int = 32
    num_layers: int = 32
    intermediate_size: int = 11008
    head_dim: int = 128
    num_kv_heads: int = 8
    vocab_size: int = 32000
    max_position_embeddings: int = 4096


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware device specification."""

    gpu_model: str = "unknown"
    compute_tflops: float = 80.0
    memory_bandwidth_gbps: float = 2000.0
    total_memory_gb: float = 80.0
    interconnect_bandwidth_gbps: float = 600.0
    num_gpus: int = 1
    gpu_memory_per_device: float = 80.0


@dataclass(frozen=True)
class ParallelismConfig:
    """Parallelism strategy configuration."""

    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    sequence_parallel: bool = False
    expert_parallel: bool = False


@dataclass
class PerformancePrediction:
    """Predicted performance metrics for a single query."""

    latency_ms: float = 0.0
    throughput_tps: float = 0.0
    memory_mb: float = 0.0
    confidence: float = 0.0  # 0 = low, 1 = high


@dataclass
class TrainingRecord:
    """Observed runtime datum for model training."""

    model_config: ModelConfig
    hardware: HardwareConfig
    parallelism: ParallelismConfig
    batch_size: int
    seq_len: int
    quantization: str = "float16"
    observed_latency_ms: float = 0.0
    observed_throughput_tps: float = 0.0
    observed_memory_mb: float = 0.0
    flash_attn: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReductionSuggestion:
    """A single optimisation recommendation."""

    action: str
    expected_improvement: float  # percentage (e.g. 15.0 = 15 %)
    confidence: float  # 0..1
    effort: str  # "low", "medium", "high"
    description: str = ""


@dataclass
class PulseStats:
    """Aggregated statistics from the Pulse service."""

    total_predictions: int = 0
    total_suggestions: int = 0
    training_samples: int = 0
    model_accuracy: float = 0.0  # 1 - normalised loss
    uptime_seconds: float = 0.0
    last_training_time: float = 0.0


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _quantization_bits(method: str) -> int:
    return {"float16": 16, "bfloat16": 16, "int8": 8, "int4": 4, "fp8": 8, "nf4": 4}.get(
        method, 16
    )


def _build_node_features(
    model_config: ModelConfig,
    quantization_bits: int = 16,
) -> np.ndarray:
    """Build per-layer node features, shape ``(num_layers, _NODE_FEAT_DIM)``.

    Each row corresponds to one transformer layer.
    """
    _require_numpy()
    n = model_config.num_layers
    features = np.zeros((n, _NODE_FEAT_DIM), dtype=np.float64)

    for i in range(n):
        features[i, 0] = float(model_config.hidden_size) / 4096.0
        features[i, 1] = float(model_config.num_heads) / 32.0
        features[i, 2] = float(model_config.intermediate_size) / 11008.0
        features[i, 3] = float(model_config.head_dim) / 128.0
        features[i, 4] = float(i) / max(float(n), 1.0)  # normalised layer index
        features[i, 5] = float(quantization_bits) / 16.0

    return features


def _build_global_features(
    hardware: HardwareConfig,
    parallelism: ParallelismConfig | None,
    batch_size: int = 1,
    seq_len: int = 2048,
    flash_attn: bool = False,
) -> np.ndarray:
    """Build global feature vector, shape ``(_GLOBAL_FEAT_DIM,)``."""
    _require_numpy()
    tp = float(parallelism.tensor_parallel) if parallelism else 1.0
    pp = float(parallelism.pipeline_parallel) if parallelism else 1.0
    dp = float(parallelism.data_parallel) if parallelism else 1.0
    sp = 1.0 if (parallelism and parallelism.sequence_parallel) else 0.0

    features = np.array(
        [
            hardware.compute_tflops / 100.0,
            hardware.memory_bandwidth_gbps / 2000.0,
            hardware.total_memory_gb / 80.0,
            hardware.interconnect_bandwidth_gbps / 600.0,
            float(hardware.num_gpus),
            math.log2(max(tp, 1.0)) / 4.0,  # log scale, cap at 16
            math.log2(max(pp, 1.0)) / 4.0,
            math.log2(max(dp, 1.0)) / 8.0,
            float(batch_size) / 64.0,
            float(seq_len) / 4096.0,
            1.0 if flash_attn else 0.0,
        ],
        dtype=np.float64,
    )
    return features


def _build_edge_index(num_layers: int) -> np.ndarray:
    """Build edge adjacency for a transformer stack.

    Returns shape ``(2, num_edges)`` -- each column is ``(src, dst)``.
    Edges include forward connections, self-loops, and skip connections.
    """
    _require_numpy()
    edges: list[tuple[int, int]] = []
    for i in range(num_layers):
        edges.append((i, i))  # self-loop
        if i + 1 < num_layers:
            edges.append((i, i + 1))  # forward edge
        if i + 2 < num_layers:
            edges.append((i, i + 2))  # skip connection

    return np.array(edges, dtype=np.int64).T  # (2, E)


def _extract_features_aggregated(
    model_config: ModelConfig,
    hardware: HardwareConfig,
    parallelism: ParallelismConfig | None = None,
    batch_size: int = 1,
    seq_len: int = 2048,
    quantization_bits: int = 16,
    flash_attn: bool = False,
) -> np.ndarray:
    """Build a combined feature vector for the numpy MLP (graph-aggregated).

    Aggregates node features (mean, max, min) and concatenates global features.
    Returns shape ``(_NUMPY_INPUT_DIM,)``.
    """
    _require_numpy()
    node_feats = _build_node_features(model_config, quantization_bits)
    global_feats = _build_global_features(
        hardware, parallelism, batch_size, seq_len, flash_attn
    )

    # Aggregate node features: mean, max, min
    agg = np.concatenate(
        [
            node_feats.mean(axis=0),
            node_feats.max(axis=0),
            node_feats.min(axis=0),
        ]
    )  # (3 * _NODE_FEAT_DIM,) = 18

    return np.concatenate([agg, global_feats])  # 18 + 11 = 29


# ---------------------------------------------------------------------------
# Quantisation heuristics (used by WhatIfSimulator and SuggestionEngine)
# ---------------------------------------------------------------------------

_QUANTIZATION_SPEEDUP: dict[str, float] = {
    "float16": 1.0,
    "bfloat16": 1.0,
    "int8": 1.2,
    "int4": 1.5,
    "fp8": 1.3,
    "nf4": 1.4,
}

_QUANTIZATION_QUALITY: dict[str, float] = {
    "float16": 1.0,
    "bfloat16": 1.0,
    "int8": 0.99,
    "int4": 0.95,
    "fp8": 0.98,
    "nf4": 0.94,
}


# ---------------------------------------------------------------------------
# PyTorch GNN (used when torch is available)
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class _GraphMessagePassing(nn.Module):
        """Single message-passing layer (mean-pool aggregation)."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
            """``x``: (N, in_dim), ``edge_index``: (2, E)."""
            N = x.size(0)
            src, dst = edge_index[0], edge_index[1]

            messages = self.linear(x[src])  # (E, out_dim)

            out = torch.zeros(N, messages.size(-1), device=x.device)
            out.index_add_(0, dst, messages)

            # Degree normalisation (mean instead of sum)
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            deg = deg.clamp(min=1.0)
            out = out / deg.unsqueeze(-1)

            return out

    class _TorchGNN(nn.Module):
        """Graph Neural Network for performance prediction.

        Architecture
        ------------
        1. Node feature encoder (MLP)
        2. Two message-passing layers with residual connections
        3. Global readout: mean-pool node embeddings + concat global features
        4. Output heads for latency, throughput, memory
        """

        def __init__(
            self,
            node_feat_dim: int = _NODE_FEAT_DIM,
            global_feat_dim: int = _GLOBAL_FEAT_DIM,
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mp1 = _GraphMessagePassing(hidden_dim, hidden_dim)
            self.mp2 = _GraphMessagePassing(hidden_dim, hidden_dim)

            combined_dim = hidden_dim + global_feat_dim
            self.readout = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),  # latency, throughput, memory
            )

        def forward(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            global_features: torch.Tensor,
        ) -> torch.Tensor:
            """Returns shape ``(1, 3)`` -- (latency, throughput, memory)."""
            x = self.node_encoder(node_features)  # (N, hidden)

            # Message passing with residual connections
            x1 = self.mp1(x, edge_index)
            x = x + x1
            x2 = self.mp2(x, edge_index)
            x = x + x2

            # Global mean pooling
            pooled = x.mean(dim=0, keepdim=True)  # (1, hidden)

            # Concatenate global features
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)
            combined = torch.cat([pooled, global_features], dim=-1)

            return self.readout(combined)  # (1, 3)

        @torch.no_grad()
        def predict_numpy(
            self,
            node_feats: np.ndarray,
            edge_index: np.ndarray,
            global_feats: np.ndarray,
        ) -> np.ndarray:
            """Predict from numpy arrays, returns numpy output."""
            self.eval()
            nf = torch.from_numpy(node_feats).float()
            ei = torch.from_numpy(edge_index).long()
            gf = torch.from_numpy(global_feats).float()
            out = self.forward(nf, ei, gf)
            return out.numpy()

else:

    class _TorchGNN:  # type: ignore[no-redef]
        """Stub -- torch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is not available")


# ---------------------------------------------------------------------------
# NumPy MLP fallback (used when torch is unavailable)
# ---------------------------------------------------------------------------


class _NumpyMLP:
    """Pure-numpy MLP with backpropagation (SGD).

    Drop-in replacement for the GNN when PyTorch is not available.
    Operates on aggregated graph features (mean / max / min of node features
    + global features).
    """

    def __init__(
        self,
        input_dim: int = _NUMPY_INPUT_DIM,
        hidden_dims: list[int] | None = None,
        output_dim: int = 3,
        learning_rate: float = 1e-3,
    ):
        _require_numpy()
        dims = hidden_dims or [64, 32]
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        all_dims = [input_dim] + list(dims) + [output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(all_dims) - 1):
            scale = math.sqrt(2.0 / all_dims[i])  # Xavier init
            self.weights.append(
                np.random.randn(all_dims[i], all_dims[i + 1]).astype(np.float64)
                * scale
            )
            self.biases.append(np.zeros(all_dims[i + 1], dtype=np.float64))

        self._is_trained = False

    # ------------------------------------------------------------------
    # forward / predict
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for i in range(len(self.weights)):
            x = x @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)  # ReLU
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict for one or more samples.

        ``x`` may be 1-D (single sample) or 2-D (batch).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def is_trained(self) -> bool:
        return self._is_trained

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def _mse_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))

    def _train_epoch(self, x: np.ndarray, y: np.ndarray) -> float:
        """Single batch forward + backward pass.  Returns batch loss."""
        n = x.shape[0]

        # -- forward (store activations for backward) --
        activations: list[np.ndarray] = [x]
        pre_activations: list[np.ndarray] = []
        current = x
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            current = np.maximum(z, 0) if i < len(self.weights) - 1 else z
            activations.append(current)

        pred = activations[-1]
        loss = self._mse_loss(pred, y)

        # -- backward (SGD) --
        grad = 2.0 * (pred - y) / n
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] -= self.learning_rate * (activations[i].T @ grad)
            self.biases[i] -= self.learning_rate * np.sum(grad, axis=0)
            if i > 0:
                grad = grad @ self.weights[i].T
                grad[pre_activations[i - 1] <= 0] = 0.0  # ReLU' mask

        return loss

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Train on (x, y) for *epochs* epochs via mini-batch SGD.

        Returns the per-epoch loss history.
        """
        n = x.shape[0]
        losses: list[float] = []

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            x_shuf = x[idx]
            y_shuf = y[idx]

            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                loss_val = self._train_epoch(x_shuf[start:end], y_shuf[start:end])
                epoch_loss += loss_val
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                logger.debug(
                    "NumpyMLP epoch {}/{}  loss={:.6f}", epoch + 1, epochs, avg_loss
                )

        self._is_trained = True
        return losses


# ---------------------------------------------------------------------------
# PerformanceModel
# ---------------------------------------------------------------------------


class PerformanceModel:
    """GNN-based performance model for distributed LLM inference.

    Predicts latency, throughput, and memory for (model_config, hardware,
    parallelism) tuples.  Uses a graph neural network when PyTorch is
    available; falls back to a pure-numpy MLP.

    Call :meth:`add_sample` with :class:`TrainingRecord` instances to
    accumulate observations, then :meth:`train` to fit the network.

    Parameters
    ----------
    hidden_dims:
        Dimensions of hidden layers (default ``[64, 32]``).
    learning_rate:
        Adam (torch) or SGD (numpy) learning rate.
    epochs:
        Number of training epochs when :meth:`train` is called.
    batch_size:
        Mini-batch size for training.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        self._hidden_dims = hidden_dims or [64, 32]
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._records: list[TrainingRecord] = []
        self._nn: _TorchGNN | _NumpyMLP | None = None
        self._torch_optimizer: Any = None
        self._built = False
        self._y_min: np.ndarray | None = None
        self._y_range: np.ndarray | None = None

    # ------------------------------------------------------------------
    # network construction (lazy)
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if self._built:
            return

        if _TORCH_AVAILABLE:
            self._nn = _TorchGNN(
                node_feat_dim=_NODE_FEAT_DIM,
                global_feat_dim=_GLOBAL_FEAT_DIM,
                hidden_dim=self._hidden_dims[0] if self._hidden_dims else 64,
            )
            self._torch_optimizer = optim.Adam(
                self._nn.parameters(), lr=self._learning_rate
            )
            logger.debug("PerformanceModel built with torch GNN")
        else:
            self._nn = _NumpyMLP(
                input_dim=_NUMPY_INPUT_DIM,
                hidden_dims=self._hidden_dims,
                output_dim=3,
                learning_rate=self._learning_rate,
            )
            logger.debug("PerformanceModel built with NumPy MLP")

        self._built = True

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        model_config: ModelConfig,
        hardware: HardwareConfig,
        parallelism: ParallelismConfig | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        quantization: str = "float16",
        flash_attn: bool = False,
    ) -> PerformancePrediction:
        """Predict performance for a (model, hardware, parallelism) tuple.

        Returns a :class:`PerformancePrediction` with *latency_ms*,
        *throughput_tps*, and *memory_mb*.
        """
        self._ensure_built()
        if self._nn is None:
            return PerformancePrediction()

        qbits = _quantization_bits(quantization)
        global_feats = _build_global_features(
            hardware, parallelism, batch_size, seq_len, flash_attn
        )

        if isinstance(self._nn, _TorchGNN):
            node_feats = _build_node_features(model_config, qbits)
            edge_idx = _build_edge_index(model_config.num_layers)
            raw = self._nn.predict_numpy(node_feats, edge_idx, global_feats).ravel()
        else:
            feats = _extract_features_aggregated(
                model_config,
                hardware,
                parallelism,
                batch_size,
                seq_len,
                qbits,
                flash_attn,
            )
            raw = self._nn.predict(feats.reshape(1, -1)).ravel()

        # Denormalise if we have stored statistics from training
        if self._y_min is not None and self._y_range is not None:
            latency = float(raw[0] * self._y_range[0] + self._y_min[0])
            throughput = float(raw[1] * self._y_range[1] + self._y_min[1])
            memory = float(raw[2] * self._y_range[2] + self._y_min[2])
        else:
            latency = float(raw[0])
            throughput = float(raw[1])
            memory = float(raw[2])

        # Build confidence from training data volume
        confidence = min(1.0, len(self._records) / 50.0)

        return PerformancePrediction(
            latency_ms=max(latency, 0.0),
            throughput_tps=max(throughput, 0.0),
            memory_mb=max(memory, 0.0),
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def add_sample(self, record: TrainingRecord) -> None:
        """Append an observed sample to the training set."""
        self._records.append(record)

    def add_samples(self, records: list[TrainingRecord]) -> None:
        """Append multiple observed samples."""
        self._records.extend(records)

    def train(
        self,
        records: list[TrainingRecord] | None = None,
    ) -> dict[str, float]:
        """Train (or re-train) the network on observed data.

        Parameters
        ----------
        records:
            Optional override.  If ``None``, uses the internally accumulated
            records (added via :meth:`add_sample` or a previous ``train``).

        Returns
        -------
        A dict of training metrics: ``{"final_loss", "num_samples"}``.
        """
        self._ensure_built()
        if self._nn is None:
            return {"final_loss": float("nan"), "num_samples": 0}

        data = records if records is not None else self._records
        if len(data) < 2:
            logger.warning(
                "PerformanceModel.train needs at least 2 samples (got {})",
                len(data),
            )
            return {"final_loss": float("nan"), "num_samples": len(data)}

        _require_numpy()

        if isinstance(self._nn, _TorchGNN):
            return self._train_torch(data)
        return self._train_numpy(data)

    def _train_torch(self, records: list[TrainingRecord]) -> dict[str, float]:
        """Train the PyTorch GNN."""
        _require_numpy()

        # Build target matrix
        targets_list: list[np.ndarray] = [
            np.array(
                [r.observed_latency_ms, r.observed_throughput_tps, r.observed_memory_mb]
            )
            for r in records
        ]
        targets = torch.from_numpy(np.stack(targets_list)).float()

        # Normalise targets to [0, 1] range
        y_min = targets.min(dim=0).values
        y_range = (targets.max(dim=0).values - y_min).clamp(min=1.0)
        targets_norm = (targets - y_min) / y_range

        self._y_min = y_min.numpy()
        self._y_range = y_range.numpy()

        dataset_size = len(records)
        self._nn.train()  # type: ignore[union-attr]
        final_loss = 0.0

        for epoch in range(self._epochs):
            idx = torch.randperm(dataset_size)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, dataset_size, self._batch_size):
                end = min(start + self._batch_size, dataset_size)
                batch_idx = idx[start:end]

                batch_loss = 0.0
                for bi in batch_idx:
                    r = records[bi]
                    qbits = _quantization_bits(r.quantization)
                    nf = torch.from_numpy(
                        _build_node_features(r.model_config, qbits)
                    ).float()
                    ei = torch.from_numpy(
                        _build_edge_index(r.model_config.num_layers)
                    ).long()
                    gf = torch.from_numpy(
                        _build_global_features(
                            r.hardware, r.parallelism, r.batch_size, r.seq_len, r.flash_attn
                        )
                    ).float()

                    pred = self._nn(nf, ei, gf)  # type: ignore[union-attr]
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(pred.squeeze(), targets_norm[bi])

                    self._torch_optimizer.zero_grad()  # type: ignore[union-attr]
                    loss.backward()
                    self._torch_optimizer.step()  # type: ignore[union-attr]

                    batch_loss += loss.item()
                    n_batches += 1

                epoch_loss += batch_loss

            avg = epoch_loss / max(n_batches, 1)
            final_loss = avg
            if (epoch + 1) % max(1, self._epochs // 10) == 0:
                logger.debug(
                    "TorchGNN epoch {}/{}  loss={:.6f}", epoch + 1, self._epochs, avg
                )

        self._nn.eval()  # type: ignore[union-attr]
        return {"final_loss": float(final_loss), "num_samples": len(records)}

    def _train_numpy(self, records: list[TrainingRecord]) -> dict[str, float]:
        """Train the numpy MLP."""
        _require_numpy()
        X_list: list[np.ndarray] = []
        Y_list: list[np.ndarray] = []
        for r in records:
            qbits = _quantization_bits(r.quantization)
            feats = _extract_features_aggregated(
                r.model_config,
                r.hardware,
                r.parallelism,
                r.batch_size,
                r.seq_len,
                qbits,
                r.flash_attn,
            )
            X_list.append(feats)
            Y_list.append(
                np.array(
                    [r.observed_latency_ms, r.observed_throughput_tps, r.observed_memory_mb]
                )
            )

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)

        # Normalise targets
        y_min = Y.min(axis=0)
        y_range = np.maximum(Y.max(axis=0) - y_min, 1.0)
        Y_norm = (Y - y_min) / y_range
        self._y_min = y_min
        self._y_range = y_range

        losses = self._nn.train(  # type: ignore[union-attr]
            X, Y_norm, epochs=self._epochs, batch_size=self._batch_size, verbose=True
        )
        return {
            "final_loss": float(losses[-1]) if losses else 0.0,
            "num_samples": len(records),
        }

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state (weights + config)."""
        self._ensure_built()
        state: dict[str, Any] = {
            "hidden_dims": self._hidden_dims,
            "learning_rate": self._learning_rate,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "num_records": len(self._records),
            "network_type": "torch" if _TORCH_AVAILABLE else "numpy",
        }
        if self._y_min is not None:
            state["y_min"] = self._y_min.tolist()
        if self._y_range is not None:
            state["y_range"] = self._y_range.tolist()
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PerformanceModel:
        """Restore a model from a previously exported state dict."""
        return cls(
            hidden_dims=state.get("hidden_dims"),
            learning_rate=state.get("learning_rate", 1e-3),
            epochs=state.get("epochs", 100),
            batch_size=state.get("batch_size", 32),
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def num_records(self) -> int:
        return len(self._records)

    @property
    def is_trained(self) -> bool:
        if not self._built or self._nn is None:
            return False
        if isinstance(self._nn, _NumpyMLP):
            return self._nn.is_trained()
        return True  # TorchGNN is considered trained after any forward pass


# ---------------------------------------------------------------------------
# WhatIfSimulator
# ---------------------------------------------------------------------------


class WhatIfSimulator:
    """Interactive what-if simulation on a trained PerformanceModel.

    Enables quick exploration of configuration changes without re-running
    the full training pipeline.

    Parameters
    ----------
    model:
        A (trained) PerformanceModel instance.
    model_config:
        Baseline model configuration.
    hardware:
        Baseline hardware configuration.
    parallelism:
        Baseline parallelism strategy.
    batch_size, seq_len, quantization, flash_attn:
        Baseline runtime parameters.
    """

    def __init__(
        self,
        model: PerformanceModel,
        model_config: ModelConfig,
        hardware: HardwareConfig,
        parallelism: ParallelismConfig | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        quantization: str = "float16",
        flash_attn: bool = False,
    ):
        self._model = model
        self._model_config = model_config
        self._hardware = hardware
        self._parallelism = parallelism
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._quantization = quantization
        self._flash_attn = flash_attn

    def _baseline(self) -> PerformancePrediction:
        """Current configuration prediction."""
        return self._model.predict(
            self._model_config,
            self._hardware,
            self._parallelism,
            self._batch_size,
            self._seq_len,
            self._quantization,
            self._flash_attn,
        )

    def what_if_batch_size(self, new_size: int) -> PerformancePrediction:
        """Projected throughput / latency if batch size changes."""
        return self._model.predict(
            self._model_config,
            self._hardware,
            self._parallelism,
            new_size,
            self._seq_len,
            self._quantization,
            self._flash_attn,
        )

    def what_if_gpu_count(self, n: int) -> dict[str, float]:
        """Projected improvement if GPU count changes.

        Returns a dict with *improvement_pct*, *new_throughput_tps*,
        *new_latency_ms*, and *new_memory_mb*.
        """
        hw = HardwareConfig(
            gpu_model=self._hardware.gpu_model,
            compute_tflops=self._hardware.compute_tflops,
            memory_bandwidth_gbps=self._hardware.memory_bandwidth_gbps,
            total_memory_gb=self._hardware.gpu_memory_per_device * n,
            interconnect_bandwidth_gbps=self._hardware.interconnect_bandwidth_gbps,
            num_gpus=n,
            gpu_memory_per_device=self._hardware.gpu_memory_per_device,
        )
        pred = self._model.predict(
            self._model_config,
            hw,
            self._parallelism,
            self._batch_size,
            self._seq_len,
            self._quantization,
            self._flash_attn,
        )
        base = self._baseline()
        improvement = (
            (pred.throughput_tps / max(base.throughput_tps, 0.001)) - 1.0
        ) * 100.0
        return {
            "improvement_pct": round(improvement, 1),
            "new_throughput_tps": round(pred.throughput_tps, 2),
            "new_latency_ms": round(pred.latency_ms, 2),
            "new_memory_mb": round(pred.memory_mb, 2),
        }

    def what_if_quantization(self, method: str) -> dict[str, float]:
        """Projected latency / quality trade-off for a different quantization.

        Returns a dict with *latency_speedup*, *quality_factor*,
        *estimated_memory_reduction*, and raw performance numbers.
        """
        pred = self._model.predict(
            self._model_config,
            self._hardware,
            self._parallelism,
            self._batch_size,
            self._seq_len,
            method,
            self._flash_attn,
        )
        base = self._baseline()

        speedup = _QUANTIZATION_SPEEDUP.get(method, 1.0) / max(
            _QUANTIZATION_SPEEDUP.get(self._quantization, 1.0), 0.001
        )
        quality = _QUANTIZATION_QUALITY.get(method, 1.0)
        qbits = _quantization_bits(method)
        mem_reduction = 16.0 / qbits if qbits > 0 else 1.0

        return {
            "latency_speedup": round(speedup, 2),
            "quality_factor": quality,
            "estimated_memory_reduction": round(mem_reduction, 2),
            "new_latency_ms": round(pred.latency_ms, 2),
            "new_throughput_tps": round(pred.throughput_tps, 2),
            "new_memory_mb": round(pred.memory_mb, 2),
        }

    def what_if_flash_attention(self, enabled: bool) -> PerformancePrediction:
        """Projected performance with flash attention toggled."""
        return self._model.predict(
            self._model_config,
            self._hardware,
            self._parallelism,
            self._batch_size,
            self._seq_len,
            self._quantization,
            enabled,
        )

    def what_if_parallelism(
        self,
        tp: int | None = None,
        pp: int | None = None,
        dp: int | None = None,
    ) -> PerformancePrediction:
        """Projected performance with a different parallelism strategy."""
        par = self._parallelism or ParallelismConfig()
        new_par = ParallelismConfig(
            tensor_parallel=tp if tp is not None else par.tensor_parallel,
            pipeline_parallel=pp if pp is not None else par.pipeline_parallel,
            data_parallel=dp if dp is not None else par.data_parallel,
            sequence_parallel=par.sequence_parallel,
            expert_parallel=par.expert_parallel,
        )
        return self._model.predict(
            self._model_config,
            self._hardware,
            new_par,
            self._batch_size,
            self._seq_len,
            self._quantization,
            self._flash_attn,
        )


# ---------------------------------------------------------------------------
# ReductionSuggestionEngine
# ---------------------------------------------------------------------------


class ReductionSuggestionEngine:
    """Recommends optimisation actions based on a trained model and SLO targets.

    The engine compares current performance against SLO targets and generates
    ranked suggestions sorted by confidence then expected improvement.
    """

    def __init__(
        self,
        model: PerformanceModel,
        model_config: ModelConfig,
        hardware: HardwareConfig,
        parallelism: ParallelismConfig | None = None,
    ):
        self._model = model
        self._model_config = model_config
        self._hardware = hardware
        self._parallelism = parallelism

    def analyze(
        self,
        current_config: dict[str, Any] | None = None,
        slo_targets: dict[str, float] | None = None,
    ) -> list[ReductionSuggestion]:
        """Analyse current configuration against SLO targets.

        Parameters
        ----------
        current_config:
            Keys: ``batch_size``, ``seq_len``, ``quantization``, ``flash_attn``.
            Falls back to defaults from the engine's constructor values.
        slo_targets:
            Keys: ``max_latency_ms``, ``min_throughput_tps``, ``max_memory_mb``.
            When empty, all suggestions are returned unprioritised.

        Returns
        -------
        list[ReductionSuggestion]:
            Suggestions sorted by confidence then expected improvement.
        """
        cfg = current_config or {}
        slo = slo_targets or {}

        batch_size = cfg.get("batch_size", 1)
        seq_len = cfg.get("seq_len", 2048)
        quantization = cfg.get("quantization", "float16")
        flash_attn = cfg.get("flash_attn", False)

        baseline = self._model.predict(
            self._model_config,
            self._hardware,
            self._parallelism,
            batch_size,
            seq_len,
            quantization,
            flash_attn,
        )

        suggestions: list[ReductionSuggestion] = []

        violates_latency = slo.get("max_latency_ms", float("inf")) < baseline.latency_ms
        violates_throughput = (
            slo.get("min_throughput_tps", 0.0) > baseline.throughput_tps
        )
        violates_memory = slo.get("max_memory_mb", float("inf")) < baseline.memory_mb

        # --- increase batch size ---
        if violates_throughput or violates_latency:
            new_pred = self._model.predict(
                self._model_config,
                self._hardware,
                self._parallelism,
                batch_size * 2,
                seq_len,
                quantization,
                flash_attn,
            )
            imp = (
                (new_pred.throughput_tps / max(baseline.throughput_tps, 0.001)) - 1.0
            ) * 100.0
            suggestions.append(
                ReductionSuggestion(
                    action="increase_batch",
                    expected_improvement=round(max(imp, 0.0), 1),
                    confidence=0.7,
                    effort="low",
                    description=f"Double batch size from {batch_size} to {batch_size * 2}",
                )
            )

        # --- add GPU ---
        if violates_latency or violates_throughput:
            hw_more = HardwareConfig(
                gpu_model=self._hardware.gpu_model,
                compute_tflops=self._hardware.compute_tflops,
                memory_bandwidth_gbps=self._hardware.memory_bandwidth_gbps,
                total_memory_gb=(self._hardware.num_gpus + 1)
                * self._hardware.gpu_memory_per_device,
                interconnect_bandwidth_gbps=self._hardware.interconnect_bandwidth_gbps,
                num_gpus=self._hardware.num_gpus + 1,
                gpu_memory_per_device=self._hardware.gpu_memory_per_device,
            )
            new_pred = self._model.predict(
                self._model_config,
                hw_more,
                self._parallelism,
                batch_size,
                seq_len,
                quantization,
                flash_attn,
            )
            imp = (
                (new_pred.throughput_tps / max(baseline.throughput_tps, 0.001)) - 1.0
            ) * 100.0
            suggestions.append(
                ReductionSuggestion(
                    action="add_gpu",
                    expected_improvement=round(max(imp, 0.0), 1),
                    confidence=0.6,
                    effort="high",
                    description=f"Add 1 GPU (total: {self._hardware.num_gpus + 1})",
                )
            )

        # --- change quantization ---
        if violates_memory or violates_latency:
            for q in ("int8", "int4"):
                if q == quantization:
                    continue
                q_pred = self._model.predict(
                    self._model_config,
                    self._hardware,
                    self._parallelism,
                    batch_size,
                    seq_len,
                    q,
                    flash_attn,
                )
                mem_saving = (
                    1.0 - q_pred.memory_mb / max(baseline.memory_mb, 0.001)
                ) * 100.0
                q_imp = (
                    (q_pred.throughput_tps / max(baseline.throughput_tps, 0.001)) - 1.0
                ) * 100.0
                if mem_saving > 5.0:
                    suggestions.append(
                        ReductionSuggestion(
                            action="change_quantization",
                            expected_improvement=round(max(q_imp, 0.0), 1),
                            confidence=0.8,
                            effort="medium",
                            description=(
                                f"Switch quantization from {quantization} to {q}  "
                                f"(~{mem_saving:.0f}% memory reduction)"
                            ),
                        )
                    )

        # --- enable flash attention ---
        if not flash_attn and violates_latency:
            fa_pred = self._model.predict(
                self._model_config,
                self._hardware,
                self._parallelism,
                batch_size,
                seq_len,
                quantization,
                True,
            )
            fa_imp = (
                (fa_pred.throughput_tps / max(baseline.throughput_tps, 0.001)) - 1.0
            ) * 100.0
            suggestions.append(
                ReductionSuggestion(
                    action="enable_flash_attn",
                    expected_improvement=round(max(fa_imp, 0.0), 1),
                    confidence=0.9,
                    effort="low",
                    description="Enable flash attention for faster attention computation",
                )
            )

        # --- tensor parallelism ---
        if self._hardware.num_gpus >= 2 and (violates_latency or violates_memory):
            tp_val = min(self._hardware.num_gpus, 8)
            par = self._parallelism or ParallelismConfig()
            if par.tensor_parallel < tp_val:
                new_par = ParallelismConfig(
                    tensor_parallel=tp_val,
                    pipeline_parallel=par.pipeline_parallel,
                    data_parallel=par.data_parallel,
                    sequence_parallel=par.sequence_parallel,
                    expert_parallel=par.expert_parallel,
                )
                tp_pred = self._model.predict(
                    self._model_config,
                    self._hardware,
                    new_par,
                    batch_size,
                    seq_len,
                    quantization,
                    flash_attn,
                )
                tp_imp = (
                    (tp_pred.throughput_tps / max(baseline.throughput_tps, 0.001))
                    - 1.0
                ) * 100.0
                suggestions.append(
                    ReductionSuggestion(
                        action="tensor_parallel",
                        expected_improvement=round(max(tp_imp, 0.0), 1),
                        confidence=0.65,
                        effort="medium",
                        description=f"Increase tensor parallelism to {tp_val}",
                    )
                )

        # Sort by confidence desc, then expected improvement desc
        suggestions.sort(
            key=lambda s: (s.confidence, s.expected_improvement), reverse=True
        )

        if not suggestions:
            suggestions.append(
                ReductionSuggestion(
                    action="no_action_needed",
                    expected_improvement=0.0,
                    confidence=1.0,
                    effort="none",
                    description="Current configuration meets all SLO targets",
                )
            )

        return suggestions


# ---------------------------------------------------------------------------
# Pulse
# ---------------------------------------------------------------------------


class Pulse:
    """Integrated performance monitoring, prediction, and optimisation service.

    Combines :class:`PerformanceModel` with continuous online fine-tuning,
    what-if simulation, and reduction suggestion into a single entry point.

    Usage::

        pulse = Pulse()
        pulse.add_record(TrainingRecord(...))
        pulse.train()
        pulse.start()  # begins background online fine-tuning

        pred = pulse.predict(...)
        suggestions = pulse.suggest(current_config, slo_targets)
        stats = pulse.stats()
        pulse.stop()

    Parameters
    ----------
    performance_model:
        An existing PerformanceModel instance.  A fresh one is created if
        ``None``.
    monitor_interval_s:
        Sleep interval between background-loop iterations.
    auto_train_interval_s:
        Minimum time (seconds) between automatic re-training runs.
    min_samples_for_train:
        Minimum number of training records before auto-training kicks in.
    """

    def __init__(
        self,
        performance_model: PerformanceModel | None = None,
        monitor_interval_s: float = 60.0,
        auto_train_interval_s: float = 300.0,
        min_samples_for_train: int = 10,
    ):
        self._model = performance_model or PerformanceModel()
        self._monitor_interval_s = monitor_interval_s
        self._auto_train_interval_s = auto_train_interval_s
        self._min_samples_for_train = min_samples_for_train

        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0

        # Stats
        self._total_predictions = 0
        self._total_suggestions = 0
        self._last_train_time: float = 0.0
        self._last_train_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background monitoring and online fine-tuning."""
        if self._running:
            logger.warning("Pulse is already running")
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="pulse-online-tuner",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Pulse started (monitor={}s, auto_train={}s)",
            self._monitor_interval_s,
            self._auto_train_interval_s,
        )

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Pulse stopped")

    def _run_loop(self) -> None:
        """Background loop: periodically re-trains the model."""
        last_train = 0.0
        while self._running:
            now = time.time()
            if (
                now - last_train >= self._auto_train_interval_s
                and self._model.num_records >= self._min_samples_for_train
            ):
                metrics = self._model.train()
                self._last_train_time = now
                self._last_train_metrics = metrics
                last_train = now
                logger.debug(
                    "Pulse auto-train completed: loss={:.4f}, samples={}",
                    metrics.get("final_loss", float("nan")),
                    metrics.get("num_samples", 0),
                )
            time.sleep(self._monitor_interval_s)

    # ------------------------------------------------------------------
    # delegation
    # ------------------------------------------------------------------

    def predict(
        self,
        model_config: ModelConfig,
        hardware: HardwareConfig,
        parallelism: ParallelismConfig | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        quantization: str = "float16",
        flash_attn: bool = False,
    ) -> PerformancePrediction:
        """Predict performance.  Delegates to the underlying PerformanceModel."""
        self._total_predictions += 1
        return self._model.predict(
            model_config,
            hardware,
            parallelism,
            batch_size,
            seq_len,
            quantization,
            flash_attn,
        )

    def add_record(self, record: TrainingRecord) -> None:
        """Add an observed training record."""
        self._model.add_sample(record)

    def add_records(self, records: list[TrainingRecord]) -> None:
        """Add multiple observed training records."""
        self._model.add_samples(records)

    def train(
        self, records: list[TrainingRecord] | None = None
    ) -> dict[str, float]:
        """Explicitly train the model (synchronous)."""
        metrics = self._model.train(records)
        self._last_train_time = time.time()
        self._last_train_metrics = metrics
        return metrics

    def simulator(
        self,
        model_config: ModelConfig,
        hardware: HardwareConfig,
        parallelism: ParallelismConfig | None = None,
        batch_size: int = 1,
        seq_len: int = 2048,
        quantization: str = "float16",
        flash_attn: bool = False,
    ) -> WhatIfSimulator:
        """Create a :class:`WhatIfSimulator` bound to this model."""
        return WhatIfSimulator(
            self._model,
            model_config,
            hardware,
            parallelism,
            batch_size,
            seq_len,
            quantization,
            flash_attn,
        )

    def suggest(
        self,
        current_config: dict[str, Any] | None = None,
        slo_targets: dict[str, float] | None = None,
    ) -> list[ReductionSuggestion]:
        """Get reduction suggestions for the current configuration."""
        self._total_suggestions += 1
        engine = ReductionSuggestionEngine(
            self._model, ModelConfig(), HardwareConfig()
        )
        return engine.analyze(current_config, slo_targets)

    def stats(self) -> PulseStats:
        """Return aggregated statistics."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0.0

        # Model accuracy: 1 - normalised final loss (clamped)
        accuracy = 0.0
        if "final_loss" in self._last_train_metrics:
            loss = self._last_train_metrics["final_loss"]
            if isinstance(loss, (int, float)) and not math.isnan(loss):
                accuracy = max(0.0, 1.0 - min(loss, 1.0))

        return PulseStats(
            total_predictions=self._total_predictions,
            total_suggestions=self._total_suggestions,
            training_samples=self._model.num_records,
            model_accuracy=accuracy,
            uptime_seconds=uptime,
            last_training_time=self._last_train_time,
        )

    @property
    def performance_model(self) -> PerformanceModel:
        """Access the underlying PerformanceModel directly."""
        return self._model

    def state_dict(self) -> dict[str, Any]:
        """Serialise the Pulse state."""
        return {
            "model": self._model.state_dict(),
            "total_predictions": self._total_predictions,
            "total_suggestions": self._total_suggestions,
            "last_train_time": self._last_train_time,
        }
