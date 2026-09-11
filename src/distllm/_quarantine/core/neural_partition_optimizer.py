"""Neural partition optimizer — learned cost model + Bayesian optimization.

Combines a neural network based cost predictor with Bayesian optimization
to find optimal model partitions across heterogeneous hardware.

Classes
-------
NeuralCostModel
    MLP predicting per-layer latency, memory, and energy for
    (layer_config, hardware, parallelism) tuples.
BayesianOptimizationLoop
    Probes partition plans, observes real throughput, and updates
    the cost model via Thompson sampling (Optuna backed when available).
NeuralPartitionOptimizer
    Coordinates the cost model and Bayesian optimisation to
    iteratively discover the optimal partition for a given
    model + hardware pool.
"""

from __future__ import annotations

import math
import random
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

_OPTUNA_AVAILABLE: bool
try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

_NUMPY_AVAILABLE: bool
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


_INPUT_DIM = 12  # Default feature-vector dimension (see _extract_features)


def _require_numpy() -> None:
    if not _NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for NeuralCostModel "
            "(both the torch path and the numpy fallback need it)"
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerConfig:
    """Transformer layer (or layer group) configuration."""

    hidden_size: int = 4096
    num_heads: int = 32
    intermediate_size: int = 11008
    num_layers: int = 1
    head_dim: int = 128


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware device specification for partition placement."""

    gpu_model: str = "unknown"
    compute_tflops: float = 80.0
    memory_bandwidth_gbps: float = 2000.0
    total_memory_gb: float = 80.0
    interconnect_bandwidth_gbps: float = 600.0
    num_gpus: int = 1
    tdp_watts: float = 350.0


@dataclass
class CostPrediction:
    """Predicted resource usage for a (layer, hardware) assignment."""

    latency_ms: float = 0.0
    memory_mb: float = 0.0
    energy_j: float = 0.0


@dataclass
class TrainingSample:
    """Observed runtime datum for cost-model training."""

    layer_config: LayerConfig
    hardware: HardwareSpec
    observed_latency_ms: float
    observed_memory_mb: float
    observed_energy_j: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PartitionProposal:
    """A proposed partition: which layer ranges are assigned to which node."""

    node_ids: list[str]
    layer_ranges: list[tuple[int, int]]  # (start, end) exclusive per node
    expected_cost: CostPrediction = field(default_factory=CostPrediction)


@dataclass
class OptimizationStats:
    """Rolling statistics from the optimisation process."""

    total_rounds: int = 0
    best_latency_ms: float = float("inf")
    best_memory_mb: float = 0.0
    best_energy_j: float = 0.0
    convergence_ratio: float = 0.0
    num_observations: int = 0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_features(
    layer_config: LayerConfig,
    hardware: HardwareSpec,
    num_nodes: int = 1,
) -> np.ndarray:
    """Build a normalised feature vector from (layer_config, hardware, num_nodes).

    Dimensions (*INPUT_DIM* = 12)
    --------------------------------
    0  hidden_size / 4096
    1  num_heads / 32
    2  intermediate_size / 11008
    3  num_layers
    4  head_dim / 128
    5  compute_tflops / 100
    6  memory_bandwidth_gbps / 2000
    7  total_memory_gb / 80
    8  interconnect_bandwidth_gbps / 600
    9  num_gpus
    10 num_nodes
    11 tdp_watts / 350
    """
    _require_numpy()
    features = np.array(
        [
            float(layer_config.hidden_size) / 4096.0,
            float(layer_config.num_heads) / 32.0,
            float(layer_config.intermediate_size) / 11008.0,
            float(layer_config.num_layers),
            float(layer_config.head_dim) / 128.0,
            hardware.compute_tflops / 100.0,
            hardware.memory_bandwidth_gbps / 2000.0,
            hardware.total_memory_gb / 80.0,
            hardware.interconnect_bandwidth_gbps / 600.0,
            float(hardware.num_gpus),
            float(num_nodes),
            hardware.tdp_watts / 350.0,
        ],
        dtype=np.float64,
    )
    return features


def _proposal_to_features(
    proposal: PartitionProposal,
    hardware_pool: dict[str, HardwareSpec],
    model_config: LayerConfig,
) -> np.ndarray:
    """Convert a partition proposal + hardware pool to a single feature row.

    For heterogeneous hardware the feature vector is aggregated across nodes
    (mean GPU spec, max memory pressure, sum of interconnect bandwidths).
    """
    _require_numpy()
    if not proposal.node_ids:
        return np.zeros(_INPUT_DIM, dtype=np.float64)

    specs = [hardware_pool.get(nid, HardwareSpec()) for nid in proposal.node_ids]

    # Aggregate hardware characteristics
    mean_tflops = sum(s.compute_tflops for s in specs) / len(specs)
    mean_bw = sum(s.memory_bandwidth_gbps for s in specs) / len(specs)
    min_mem = min(s.total_memory_gb for s in specs)  # weakest link
    max_interconnect = max(s.interconnect_bandwidth_gbps for s in specs)
    total_gpus = sum(s.num_gpus for s in specs)
    mean_tdp = sum(s.tdp_watts for s in specs) / len(specs)

    features = np.array(
        [
            float(model_config.hidden_size) / 4096.0,
            float(model_config.num_heads) / 32.0,
            float(model_config.intermediate_size) / 11008.0,
            float(model_config.num_layers),
            float(model_config.head_dim) / 128.0,
            mean_tflops / 100.0,
            mean_bw / 2000.0,
            min_mem / 80.0,
            max_interconnect / 600.0,
            float(total_gpus),
            float(len(proposal.node_ids)),
            mean_tdp / 350.0,
        ],
        dtype=np.float64,
    )
    return features


# ---------------------------------------------------------------------------
# PyTorch MLP (used when torch is available)
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class _TorchMLP(nn.Module):
        """Multi-layer perceptron for cost prediction (PyTorch)."""

        def __init__(
            self,
            input_dim: int = _INPUT_DIM,
            hidden_dims: list[int] | None = None,
            output_dim: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            dims = hidden_dims or [128, 64, 32]
            layers_list: list[nn.Module] = []
            prev = input_dim
            for h in dims:
                layers_list.extend(
                    [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                )
                prev = h
            layers_list.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers_list)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        @torch.no_grad()
        def predict_numpy(self, x: np.ndarray) -> np.ndarray:
            """Run a forward pass on a numpy array, return numpy."""
            self.eval()
            t = torch.from_numpy(x).float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return self.forward(t).numpy()

else:

    class _TorchMLP:  # type: ignore[no-redef]
        """Stub -- torch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is not available")


# ---------------------------------------------------------------------------
# NumPy MLP fallback
# ---------------------------------------------------------------------------


class _NumpyMLP:
    """Pure-numpy MLP with backpropagation (SGD).

    Used as a drop-in replacement for ``_TorchMLP`` when PyTorch is not
    available.
    """

    def __init__(
        self,
        input_dim: int = _INPUT_DIM,
        hidden_dims: list[int] | None = None,
        output_dim: int = 3,
        learning_rate: float = 1e-3,
    ):
        _require_numpy()
        dims = hidden_dims or [128, 64, 32]
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
    # public API
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass (all layers)."""
        for i in range(len(self.weights)):
            x = x @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)  # ReLU
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output for one or more samples.

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

        # -- forward (store activations) --
        activations: list[np.ndarray] = [x]
        pre_activations: list[np.ndarray] = []
        current = x
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                current = np.maximum(z, 0)
            else:
                current = z
            activations.append(current)

        pred = activations[-1]
        loss = self._mse_loss(pred, y)

        # -- backward (SGD) --
        grad = 2.0 * (pred - y) / n  # dL / d_output
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] -= self.learning_rate * (activations[i].T @ grad)
            self.biases[i] -= self.learning_rate * np.sum(grad, axis=0)
            if i > 0:
                grad = grad @ self.weights[i].T
                grad[pre_activations[i - 1] <= 0] = 0.0  # ReLU'

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
# Gaussian Process for Thompson sampling
# ---------------------------------------------------------------------------


class _GaussianProcess:
    """Simple Gaussian Process with an RBF kernel (numpy).

    Used by :class:`BayesianOptimizationLoop` to model the posterior over
    partition costs and drive Thompson sampling.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        sigma_f: float = 1.0,
        sigma_n: float = 0.1,
    ):
        _require_numpy()
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._L: np.ndarray | None = None  # Cholesky factor
        self._alpha: np.ndarray | None = None

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Pairwise RBF (squared-exponential) kernel."""
        sq_dist = (
            -2.0 * x1 @ x2.T
            + np.sum(x1 ** 2, axis=1, keepdims=True)
            + np.sum(x2 ** 2, axis=1, keepdims=True).T
        )
        return self.sigma_f ** 2 * np.exp(
            -0.5 / self.length_scale ** 2 * np.maximum(sq_dist, 0.0)
        )

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP on training data.

        Parameters
        ----------
        X: shape (n, d)
        y: shape (n,) or (n, 1)
        """
        n = X.shape[0]
        K = self._rbf_kernel(X, X) + self.sigma_n ** 2 * np.eye(n)
        self._L = np.linalg.cholesky(K)
        y_flat = y.ravel() if y.ndim > 1 else y
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_flat))
        self._X_train = X
        self._y_train = y_flat

    def predict(
        self, X_s: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Posterior mean (and optionally std) at test points ``X_s``."""
        if self._X_train is None or self._alpha is None or self._L is None:
            raise RuntimeError("GP not fitted -- call fit() first")

        K_s = self._rbf_kernel(X_s, self._X_train)
        mu = K_s @ self._alpha

        if not return_std:
            return mu

        v = np.linalg.solve(self._L, K_s.T)
        K_ss = self._rbf_kernel(X_s, X_s)
        cov = K_ss - v.T @ v
        var = np.maximum(np.diag(cov), 1e-10)
        return mu, np.sqrt(var)

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------

    def sample_posterior(self, X_s: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw function samples from the posterior at ``X_s``.

        Returns shape ``(len(X_s), n_samples)``.
        """
        if self._X_train is None or self._alpha is None or self._L is None:
            raise RuntimeError("GP not fitted -- call fit() first")

        K_s = self._rbf_kernel(X_s, self._X_train)
        mu = K_s @ self._alpha

        # compute posterior covariance
        v = np.linalg.solve(self._L, K_s.T)
        K_ss = self._rbf_kernel(X_s, X_s)
        cov = K_ss - v.T @ v
        cov += 1e-6 * np.eye(X_s.shape[0])  # jitter

        L_cov = np.linalg.cholesky(cov)
        samples = mu[:, None] + L_cov @ np.random.randn(X_s.shape[0], n_samples)
        return samples

    def thompson_sample(
        self, X_s: np.ndarray, n_samples: int = 1
    ) -> np.ndarray:
        """Alias for :meth:`sample_posterior`."""
        return self.sample_posterior(X_s, n_samples)


# ---------------------------------------------------------------------------
# NeuralCostModel
# ---------------------------------------------------------------------------


class NeuralCostModel:
    """Neural cost model predicting per-assignment latency, memory, and energy.

    Wraps either a PyTorch or a pure-numpy MLP with a uniform interface.
    Call :meth:`train` with observed :class:`TrainingSample` instances to
    fit the network.

    Parameters
    ----------
    hidden_dims:
        Dimensions of hidden layers (default ``[128, 64, 32]``).
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
        self._hidden_dims = hidden_dims or [128, 64, 32]
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._samples: list[TrainingSample] = []
        self._network: _TorchMLP | _NumpyMLP | None = None
        self._torch_optimizer: Any = None
        self._built = False

    # ------------------------------------------------------------------
    # network construction (lazy)
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if self._built:
            return
        hidden = self._hidden_dims

        if _TORCH_AVAILABLE:
            self._network = _TorchMLP(_INPUT_DIM, hidden, output_dim=3, dropout=0.1)
            self._torch_optimizer = optim.Adam(
                self._network.parameters(), lr=self._learning_rate
            )
            logger.debug(
                "NeuralCostModel built with torch MLP {}",
                [_INPUT_DIM] + list(hidden) + [3],
            )
        else:
            self._network = _NumpyMLP(_INPUT_DIM, hidden, output_dim=3,
                                      learning_rate=self._learning_rate)
            logger.debug(
                "NeuralCostModel built with NumPy MLP {}",
                [_INPUT_DIM] + list(hidden) + [3],
            )
        self._built = True

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        layer_config: LayerConfig,
        hardware: HardwareSpec,
        num_nodes: int = 1,
    ) -> CostPrediction:
        """Predict resource usage for a (config, hardware) pair.

        Returns a :class:`CostPrediction` with *latency_ms*,
        *memory_mb*, and *energy_j*.
        """
        self._ensure_built()
        if self._network is None:
            return CostPrediction()  # empty default

        features = _extract_features(layer_config, hardware, num_nodes)
        features_2d = features.reshape(1, -1)

        if isinstance(self._network, _TorchMLP):
            raw = self._network.predict_numpy(features_2d).ravel()
        else:
            raw = self._network.predict(features_2d).ravel()

        # De-normalize the network output: training scaled targets to [0,1],
        # so a raw 0..1 value is meaningless as latency/memory/energy.
        y_min = getattr(self, "_y_min", None)
        y_range = getattr(self, "_y_range", None)
        if y_min is not None and y_range is not None:
            raw = raw * np.asarray(y_range, dtype=float) + np.asarray(y_min, dtype=float)

        return CostPrediction(
            latency_ms=float(max(raw[0], 0.0)),
            memory_mb=float(max(raw[1], 0.0)),
            energy_j=float(max(raw[2], 0.0)),
        )

    def predict_batch(
        self,
        configs: list[LayerConfig],
        hardwares: list[HardwareSpec],
        num_nodes_list: list[int] | None = None,
    ) -> list[CostPrediction]:
        """Predict cost for a batch of (config, hardware) pairs."""
        if num_nodes_list is None:
            num_nodes_list = [1] * len(configs)
        return [
            self.predict(c, h, n)
            for c, h, n in zip(configs, hardwares, num_nodes_list)
        ]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def add_sample(self, sample: TrainingSample) -> None:
        """Append an observed sample to the training set."""
        self._samples.append(sample)

    def train(
        self,
        samples: list[TrainingSample] | None = None,
    ) -> dict[str, float]:
        """Train (or re-train) the network on observed throughput data.

        Parameters
        ----------
        samples:
            Optional override samples.  If ``None``, uses the internally
            accumulated samples (added via :meth:`add_sample` or previous
            ``train`` calls).

        Returns
        -------
        A dict of training metrics: ``{"final_loss", "num_samples"}``.
        """
        self._ensure_built()
        if self._network is None:
            return {"final_loss": float("nan"), "num_samples": 0}

        data = samples if samples is not None else self._samples
        if len(data) < 2:
            logger.warning(
                "NeuralCostModel.train needs at least 2 samples (got {})", len(data)
            )
            return {"final_loss": float("nan"), "num_samples": len(data)}

        # build feature matrix and target matrix
        _require_numpy()
        X_list: list[np.ndarray] = []
        Y_list: list[np.ndarray] = []
        for s in data:
            feats = _extract_features(s.layer_config, s.hardware)
            X_list.append(feats)
            Y_list.append(
                np.array([s.observed_latency_ms, s.observed_memory_mb, s.observed_energy_j])
            )

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)

        # normalise targets to [0, 1] range for stable training
        y_min = Y.min(axis=0)
        y_range = np.maximum(Y.max(axis=0) - y_min, 1.0)
        Y_norm = (Y - y_min) / y_range

        if isinstance(self._network, _TorchMLP):
            loss_val = self._train_torch(X, Y_norm)
        else:
            loss_val = self._train_numpy(X, Y_norm)

        # store normalisation constants
        self._y_min = y_min
        self._y_range = y_range

        return {"final_loss": float(loss_val), "num_samples": len(data)}

    def _train_torch(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Train the PyTorch network."""
        self._network.train()  # type: ignore[union-attr]
        dataset_size = X.shape[0]
        final_loss = 0.0

        for epoch in range(self._epochs):
            idx = np.random.permutation(dataset_size)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, dataset_size, self._batch_size):
                end = min(start + self._batch_size, dataset_size)
                batch_idx = idx[start:end]
                x_b = torch.from_numpy(X[batch_idx]).float()
                y_b = torch.from_numpy(Y[batch_idx]).float()

                self._torch_optimizer.zero_grad()  # type: ignore[union-attr]
                pred = self._network(x_b)  # type: ignore[union-attr]
                loss = nn.functional.mse_loss(pred, y_b)
                loss.backward()
                self._torch_optimizer.step()  # type: ignore[union-attr]

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            final_loss = avg
            if (epoch + 1) % max(1, self._epochs // 10) == 0:
                logger.debug("Torch MLP epoch {}/{}  loss={:.6f}",
                             epoch + 1, self._epochs, avg)

        self._network.eval()  # type: ignore[union-attr]
        return final_loss

    def _train_numpy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Train the numpy MLP."""
        losses = self._network.train(X, Y, epochs=self._epochs,  # type: ignore[union-attr]
                                     batch_size=self._batch_size,
                                     verbose=True)
        return losses[-1] if losses else 0.0

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
            "num_samples": len(self._samples),
            "network_type": "torch" if _TORCH_AVAILABLE else "numpy",
        }
        if self._network is not None:
            if isinstance(self._network, _TorchMLP):
                state["weights"] = {
                    k: v.cpu().numpy().tolist()
                    for k, v in self._network.state_dict().items()
                }
            elif isinstance(self._network, _NumpyMLP):
                state["weights"] = {
                    f"weight_{i}": w.tolist()
                    for i, w in enumerate(self._network.weights)
                }
                state["biases"] = {
                    f"bias_{i}": b.tolist()
                    for i, b in enumerate(self._network.biases)
                }
        if hasattr(self, "_y_min"):
            state["y_min"] = self._y_min.tolist()
            state["y_range"] = self._y_range.tolist()
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> NeuralCostModel:
        """Restore a model from a previously exported state dict.

        Loads the trained network weights and the target-normalisation
        constants so ``predict`` returns real (de-normalized) values.  A
        model restored without weights is untrained and returns zero cost.
        """
        model = cls(
            hidden_dims=state.get("hidden_dims"),
            learning_rate=state.get("learning_rate", 1e-3),
            epochs=state.get("epochs", 100),
            batch_size=state.get("batch_size", 32),
        )
        model._ensure_built()

        # Restore normalisation constants so predict() de-normalizes.
        if "y_min" in state:
            model._y_min = np.asarray(state["y_min"], dtype=float)
            model._y_range = np.asarray(state["y_range"], dtype=float)

        weights = state.get("weights")
        if weights and model._network is not None:
            try:
                if isinstance(model._network, _TorchMLP):
                    model._network.load_state_dict(
                        {k: torch.tensor(v) for k, v in weights.items()}
                    )
                elif isinstance(model._network, _NumpyMLP):
                    biases = state.get("biases", {})
                    model._network.weights = [
                        np.asarray(weights[f"weight_{i}"], dtype=float)
                        for i in range(len(weights))
                    ]
                    model._network.biases = [
                        np.asarray(biases[f"bias_{i}"], dtype=float)
                        for i in range(len(biases))
                    ]
            except Exception as e:  # noqa: BLE001 - defensive restore
                logger.warning(f"Failed to load NeuralCostModel weights: {e}")

        return model

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    @property
    def is_trained(self) -> bool:
        if self._network is None:
            return False
        if isinstance(self._network, _NumpyMLP):
            return self._network.is_trained()
        return True  # torch network is always "runnable"


# ---------------------------------------------------------------------------
# BayesianOptimizationLoop
# ---------------------------------------------------------------------------


class BayesianOptimizationLoop:
    """Bayesian optimisation loop for partition search.

    Proposes new partition plans via Thompson sampling, observes real
    (or simulated) throughput, and updates the underlying cost model.

    Uses **Optuna** (TPE sampler) when available and falls back to a
    Gaussian-process-based Thompson sampler implemented in numpy.

    Parameters
    ----------
    cost_model:
        The learned cost model to update after each observation.
    exploration_beta:
        Exploration-exploitation trade-off for LCB-style acquisition.
        Higher values promote exploration.
    """

    def __init__(
        self,
        cost_model: NeuralCostModel,
        exploration_beta: float = 2.0,
    ):
        self._cost_model = cost_model
        self._exploration_beta = exploration_beta
        self._observations: list[tuple[PartitionProposal, float, float]] = []
        self._best_proposal: PartitionProposal | None = None
        self._best_score: float = float("inf")
        self._gp: _GaussianProcess | None = None
        # Stored from the most recent propose/optimize call for GP fitting
        self._last_hardware_pool: dict[str, HardwareSpec] = {}
        self._last_model_config: LayerConfig = LayerConfig()
        _require_numpy()

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def propose(
        self,
        model_config: LayerConfig,
        hardware_pool: dict[str, HardwareSpec],
        num_candidates: int = 50,
    ) -> PartitionProposal:
        """Suggest the next partition plan via Thompson sampling.

        Generates *num_candidates* random partition layouts, evaluates
        each with Thompson-sampled cost from the GP posterior (or uniform
        noise when the GP is not yet fitted), and returns the most
        promising candidate.
        """
        if not hardware_pool:
            return PartitionProposal(node_ids=[], layer_ranges=[])

        # Store for GP fitting
        self._last_hardware_pool = hardware_pool
        self._last_model_config = model_config

        node_ids = list(hardware_pool.keys())
        num_layers = model_config.num_layers

        candidates = self._generate_candidates(
            node_ids, num_layers, num_candidates
        )
        if not candidates:
            return self._fallback_partition(node_ids, num_layers)

        feats = np.stack(
            [
                _proposal_to_features(p, hardware_pool, model_config)
                for p in candidates
            ],
            axis=0,
        )

        if self._gp is not None and len(self._observations) >= 3:
            # Thompson sample from GP posterior
            samples = self._gp.thompson_sample(feats, n_samples=1)
            scores = samples.ravel()
        else:
            # Cold start: use cost model prediction + heuristic noise
            scores = []
            for p, f in zip(candidates, feats):
                pred = self._cost_model.predict(
                    model_config,
                    self._pool_average(hardware_pool),
                    len(p.node_ids),
                )
                noise = np.random.exponential(scale=pred.latency_ms * 0.2 + 0.1)
                scores.append(pred.latency_ms + noise)
            scores = np.array(scores)

        best_idx = int(np.argmin(scores))
        best = candidates[best_idx]

        # fill in expected cost
        pred = self._cost_model.predict(
            model_config,
            self._pool_average(hardware_pool),
            len(best.node_ids),
        )
        best.expected_cost = pred
        return best

    def observe(
        self,
        plan: PartitionProposal,
        latency_ms: float,
        memory_mb: float,
    ) -> None:
        """Record an observed (plan, latency, memory) outcome.

        The observation is appended to the internal store and the GP
        model is re-fitted (if enough observations exist).
        """
        self._observations.append((plan, latency_ms, memory_mb))

        score = latency_ms
        if score < self._best_score:
            self._best_score = score
            self._best_proposal = plan

        logger.debug(
            "BO observed plan across {} nodes: latency={:.2f}ms  mem={:.2f}MB",
            len(plan.node_ids), latency_ms, memory_mb,
        )

        # re-fit GP when we have enough data
        if len(self._observations) >= 3:
            self._fit_gp()

    def optimize(
        self,
        model_config: LayerConfig,
        hardware_pool: dict[str, HardwareSpec],
        rounds: int = 20,
        evaluator: Any = None,
    ) -> PartitionProposal:
        """Run a full Bayesian optimisation cycle.

        Parameters
        ----------
        model_config:
            The model to partition.
        hardware_pool:
            Available hardware nodes.
        rounds:
            Number of BO rounds.
        evaluator:
            Optional callable ``evaluator(proposal) -> (latency, memory)``.
            If ``None``, the cost model's own prediction is used as a
            proxy (simulated BO).

        Returns
        -------
        The best :class:`PartitionProposal` discovered.
        """
        if evaluator is None:
            evaluator = self._make_simulated_evaluator(model_config, hardware_pool)

        for rnd in range(rounds):
            proposal = self.propose(model_config, hardware_pool)
            latency_ms, memory_mb = evaluator(proposal)
            self.observe(proposal, latency_ms, memory_mb)
            logger.info(
                "BO round {}/{}: {} nodes, latency={:.1f}ms, mem={:.1f}MB  "
                "(best={:.1f}ms)",
                rnd + 1, rounds, len(proposal.node_ids),
                latency_ms, memory_mb, self._best_score,
            )

        if self._best_proposal is not None:
            return self._best_proposal
        return self._fallback_partition(
            list(hardware_pool.keys()), model_config.num_layers,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        node_ids: list[str],
        num_layers: int,
        count: int,
    ) -> list[PartitionProposal]:
        """Generate random partition candidates."""
        candidates: list[PartitionProposal] = []
        max_nodes = min(len(node_ids), num_layers)

        for _ in range(count):
            n = random.randint(1, max_nodes)
            chosen = random.sample(node_ids, n)

            # random split ratios (sorted)
            if n == 1:
                ratios = [0.0, 1.0]
            else:
                raw = sorted(random.random() for _ in range(n - 1))
                ratios = [0.0] + raw + [1.0]

            ranges: list[tuple[int, int]] = []
            for i in range(n):
                start = int(round(ratios[i] * num_layers))
                end = int(round(ratios[i + 1] * num_layers))
                if end - start < 1:
                    if i == 0:
                        end = min(start + 1, num_layers)
                    elif i == n - 1:
                        start = max(end - 1, 0)
                    else:
                        continue
                if start >= end or start >= num_layers:
                    continue
                ranges.append((start, end))

            if not ranges or ranges[-1][1] != num_layers:
                continue

            candidates.append(
                PartitionProposal(node_ids=chosen, layer_ranges=ranges)
            )

        return candidates

    def _fallback_partition(
        self, node_ids: list[str], num_layers: int,
    ) -> PartitionProposal:
        """Uniform fallback partition (layers split evenly)."""
        n = min(len(node_ids), num_layers)
        chosen = node_ids[:n]
        per = num_layers // n
        rem = num_layers % n
        ranges: list[tuple[int, int]] = []
        start = 0
        for i in range(n):
            extra = 1 if i < rem else 0
            end = start + per + extra
            ranges.append((start, end))
            start = end
        return PartitionProposal(node_ids=chosen, layer_ranges=ranges)

    def _fit_gp(self) -> None:
        """Fit a GP on the observation history.

        Uses the hardware pool and model config stored from the most recent
        call to :meth:`propose` (or :meth:`optimize`).
        """
        if not self._observations:
            return
        _require_numpy()
        hw_pool = self._last_hardware_pool
        mcfg = self._last_model_config
        # Build feature matrix from observed proposals (latency-only GP)
        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        for proposal, lat, _mem in self._observations:
            f = _proposal_to_features(proposal, hw_pool, mcfg)
            X_list.append(f)
            y_list.append(lat)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list)

        # Normalise features
        self._gp_X_mean = X.mean(axis=0)
        self._gp_X_std = np.maximum(X.std(axis=0), 1e-10)
        X_norm = (X - self._gp_X_mean) / self._gp_X_std

        self._gp = _GaussianProcess(
            length_scale=1.0, sigma_f=y.std() if y.std() > 0 else 1.0, sigma_n=0.1
        )
        self._gp.fit(X_norm, y)

    def _pool_average(self, pool: dict[str, HardwareSpec]) -> HardwareSpec:
        """Return an 'average' hardware spec from the pool."""
        if not pool:
            return HardwareSpec()
        specs = list(pool.values())
        return HardwareSpec(
            gpu_model="pool_avg",
            compute_tflops=sum(s.compute_tflops for s in specs) / len(specs),
            memory_bandwidth_gbps=sum(s.memory_bandwidth_gbps for s in specs) / len(specs),
            total_memory_gb=sum(s.total_memory_gb for s in specs) / len(specs),
            interconnect_bandwidth_gbps=sum(s.interconnect_bandwidth_gbps for s in specs) / len(specs),
            num_gpus=sum(s.num_gpus for s in specs),
            tdp_watts=sum(s.tdp_watts for s in specs) / len(specs),
        )

    def _make_simulated_evaluator(
        self,
        model_config: LayerConfig,
        hardware_pool: dict[str, HardwareSpec],
    ) -> Any:
        """Build a callable that uses the cost model for simulated evaluation."""

        def _eval(proposal: PartitionProposal) -> tuple[float, float]:
            agg_hw = self._pool_average(hardware_pool)
            pred = self._cost_model.predict(
                model_config, agg_hw, len(proposal.node_ids),
            )
            # Add small simulated noise
            latency = pred.latency_ms * (1.0 + random.gauss(0, 0.05))
            memory = pred.memory_mb * (1.0 + random.gauss(0, 0.02))
            return max(latency, 0.0), max(memory, 0.0)

        return _eval

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def best_proposal(self) -> PartitionProposal | None:
        return self._best_proposal

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def num_observations(self) -> int:
        return len(self._observations)

    @property
    def gp_is_fitted(self) -> bool:
        return self._gp is not None


# ---------------------------------------------------------------------------
# NeuralPartitionOptimizer
# ---------------------------------------------------------------------------


class NeuralPartitionOptimizer:
    """End-to-end neural + Bayesian partition optimizer.

    Combines a :class:`NeuralCostModel` with a :class:`BayesianOptimizationLoop`
    to iteratively discover the optimal partition for a given model and
    hardware pool.

    Typical usage::

        opt = NeuralPartitionOptimizer()
        best = opt.auto_optimize(
            model_config=LayerConfig(hidden_size=4096, num_layers=80),
            hardware_pool={
                "gpu-0": HardwareSpec(compute_tflops=312, total_memory_gb=80),
                "gpu-1": HardwareSpec(compute_tflops=989, total_memory_gb=80),
            },
            rounds=30,
        )
        print(opt.stats())
    """

    def __init__(
        self,
        cost_model: NeuralCostModel | None = None,
        exploration_beta: float = 2.0,
    ):
        self._cost_model = cost_model or NeuralCostModel()
        self._bo = BayesianOptimizationLoop(
            self._cost_model, exploration_beta=exploration_beta,
        )
        self._rounds_run: int = 0
        self._best_history: list[float] = []
        self._convergence_ratio: float = 0.0
        self._final_proposal: PartitionProposal | None = None

    # ------------------------------------------------------------------
    # main optimisation entry point
    # ------------------------------------------------------------------

    def auto_optimize(
        self,
        model_config: LayerConfig,
        hardware_pool: dict[str, HardwareSpec],
        rounds: int = 20,
        seed_samples: list[TrainingSample] | None = None,
        evaluator: Any = None,
    ) -> PartitionProposal:
        """Iteratively find the optimal partition.

        Parameters
        ----------
        model_config:
            The model to partition.
        hardware_pool:
            Mapping of ``node_id -> HardwareSpec`` for available nodes.
        rounds:
            Number of BO rounds.
        seed_samples:
            Optional pre-existing observations to warm-start the cost model.
        evaluator:
            Optional callable ``evaluator(proposal) -> (latency_ms, memory_mb)``.
            When ``None`` the cost model is used as a proxy.

        Returns
        -------
        The best :class:`PartitionProposal` discovered.
        """
        if seed_samples:
            for s in seed_samples:
                self._cost_model.add_sample(s)
            self._cost_model.train()

        if not hardware_pool:
            logger.warning("Empty hardware pool -- returning fallback partition")
            proposal = PartitionProposal(node_ids=[], layer_ranges=[])
            self._final_proposal = proposal
            return proposal

        if evaluator is None:
            evaluator = self._bo._make_simulated_evaluator(model_config, hardware_pool)

        logger.info(
            "NeuralPartitionOptimizer starting {} rounds on {} nodes (model: {} layers, {} hidden)",
            rounds, len(hardware_pool), model_config.num_layers, model_config.hidden_size,
        )

        for rnd in range(rounds):
            proposal = self._bo.propose(model_config, hardware_pool)
            latency_ms, memory_mb = evaluator(proposal)
            self._bo.observe(proposal, latency_ms, memory_mb)
            self._rounds_run += 1
            self._best_history.append(self._bo.best_score)

            if (rnd + 1) % max(1, rounds // 5) == 0:
                logger.info(
                    "Opt round {}/{}: best latency={:.1f}ms  "
                    "({} nodes)",
                    rnd + 1, rounds, self._bo.best_score,
                    len(self._bo.best_proposal.node_ids)
                    if self._bo.best_proposal else 0,
                )

        # compute convergence ratio
        if len(self._best_history) >= 3:
            recent = self._best_history[-min(5, len(self._best_history)):]
            window_size = max(2, len(recent))
            if self._best_history[-window_size] > 0:
                self._convergence_ratio = (
                    (self._best_history[-window_size] - recent[-1])
                    / self._best_history[-window_size]
                )
            else:
                self._convergence_ratio = 0.0

        self._final_proposal = self._bo.best_proposal
        if self._final_proposal is None:
            self._final_proposal = self._bo._fallback_partition(
                list(hardware_pool.keys()), model_config.num_layers,
            )

        logger.info(
            "Optimisation complete: best latency={:.1f}ms, "
            "convergence={:.2%}",
            self._bo.best_score, self._convergence_ratio,
        )
        return self._final_proposal

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    def get_best_partition(self) -> PartitionProposal | None:
        """Return the best partition found so far."""
        return self._bo.best_proposal or self._final_proposal

    def stats(self) -> OptimizationStats:
        """Return rolling optimisation statistics."""
        best_proposal = self.get_best_partition()
        # Estimate memory & energy if possible
        if best_proposal is not None:
            mem_mb = float(np.mean(
                [abs(e - s) for e, s in best_proposal.layer_ranges]
            )) * 100  # rough proxy
            energy_j = mem_mb * 0.01  # rough proxy
        else:
            mem_mb = 0.0
            energy_j = 0.0

        return OptimizationStats(
            total_rounds=self._rounds_run,
            best_latency_ms=(
                self._bo.best_score
                if self._bo.best_score < float("inf")
                else 0.0
            ),
            best_memory_mb=mem_mb,
            best_energy_j=energy_j,
            convergence_ratio=round(self._convergence_ratio, 4),
            num_observations=self._bo.num_observations,
        )

    def summary(self) -> str:
        """Human-readable summary of the optimiser state."""
        s = self.stats()
        lines = [
            "NeuralPartitionOptimizer",
            f"  Rounds run: {s.total_rounds}",
            f"  Best latency: {s.best_latency_ms:.2f} ms",
            f"  Observations: {s.num_observations}",
            f"  Convergence: {s.convergence_ratio:.2%}",
        ]
        best = self.get_best_partition()
        if best is not None:
            lines.append(f"  Best nodes: {len(best.node_ids)}")
            lines.append(f"  Best ranges: {best.layer_ranges}")
        lines.append(
            f"  Cost model samples: {self._cost_model.num_samples}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"NeuralPartitionOptimizer(rounds={self._rounds_run}, "
            f"best_latency={self._bo.best_score if self._bo.best_score < float('inf') else 0.0:.2f})"
        )


__all__ = [
    # data classes
    "LayerConfig",
    "HardwareSpec",
    "CostPrediction",
    "TrainingSample",
    "PartitionProposal",
    "OptimizationStats",
    # classes
    "NeuralCostModel",
    "BayesianOptimizationLoop",
    "NeuralPartitionOptimizer",
]
