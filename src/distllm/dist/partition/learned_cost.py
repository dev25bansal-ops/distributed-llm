"""ML-based learned cost model for partition optimization.

Replaces the analytical FLOPS formula with a lightweight regression
model (gradient-boosted trees) trained on real runtime traces.  Falls
back to the analytical model when insufficient training data is available.

Typical usage::

    learned = LearnedCostModel(base_cost_model=analytical_model)
    learned.record(observation)          # from real inference
    learned.train()                      # retrain on collected data
    cost = learned.evaluate(node_id, start, end, batch_size, seq_len)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import TopologyGraph

# ---------------------------------------------------------------------------
# Shared parametric layer layout (F-051 fix).
#
# Training (_observation_to_features) and serving (FeatureExtractor.extract)
# must build their feature vectors from the SAME formulas, otherwise the
# model trains on one distribution and serves on another.  These constants
# and helpers mirror the parametric layer generator in
# profiles.GPUProfiler.estimate_layer_weights / _single_layer_weight_bytes /
# _single_layer_flops so both paths agree for any model whose LayerWeights
# were produced by that generator (the only generator in this codebase).
# ---------------------------------------------------------------------------

_DTYPE_BYTES = 2
_NUM_HEADS = 32
_HEAD_DIM = 128
_KV_CACHE_BYTES_PER_TOKEN = 2 * _NUM_HEADS * _HEAD_DIM * _DTYPE_BYTES


def _single_layer_weight_bytes(hidden_size: int, intermediate_size: int) -> int:
    """Weight bytes per transformer layer (attention + MLP + norms)."""
    qkv = 3 * hidden_size * hidden_size
    o_proj = hidden_size * hidden_size
    gate_up_down = 3 * hidden_size * intermediate_size
    norms = 4 * hidden_size
    return (qkv + o_proj + gate_up_down + norms) * _DTYPE_BYTES


def _single_layer_flops(hidden_size: int, intermediate_size: int) -> int:
    """Per-token FLOPS per transformer layer (default head config)."""
    qkv = 3 * 2 * hidden_size * hidden_size
    attn = 2 * 2 * _NUM_HEADS * _HEAD_DIM
    o_proj = 2 * hidden_size * hidden_size
    mlp = 3 * 2 * hidden_size * intermediate_size
    return qkv + attn + o_proj + mlp


def _estimate_hidden_intermediate(weight_memory_bytes: int) -> tuple[int, int]:
    """Recover (hidden_size, intermediate_size) estimates from one transformer
    layer's weight bytes.

    Used identically on the training and serving feature paths so both emit
    the same hidden/intermediate features for the same model.  Solves the
    parametric layout ``w/d = 4h^2 + 3*h*i + 4h`` self-consistently under
    the standard ``i ~= 11/3 * h``-scale assumption ``i = c*h``::

        w/d = (4 + 3c) * h^2 + 4h   ->   h = (-4 + sqrt(16 + 4*(4+3c)*w/d)) / (2*(4+3c))

    which always yields a non-negative intermediate (unlike solving for ``i``
    as a residual after a ``sqrt(w/7)`` hidden estimate, which goes negative
    for real LLM layouts).
    """
    w = weight_memory_bytes / _DTYPE_BYTES
    if w <= 0:
        return 0, 0
    # c = i/h ratio typical of modern LLM MLP blocks (e.g. 11008/4096 ≈ 2.69).
    c = 11008 / 4096
    a = 4 + 3 * c
    hidden = int((-4 + math.sqrt(16 + 4 * a * w)) / (2 * a))
    if hidden <= 0:
        return 0, 0
    intermediate = max(int(c * hidden), 0)
    return hidden, intermediate


@dataclass
class RuntimeObservation:
    """Recorded from a real inference run."""
    node_id: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    batch_size: int
    seq_len: int
    gpu_tflops: float
    gpu_mem_bw_gbps: float
    gpu_mem_bytes: int
    is_nvlink: bool
    is_infiniband: bool
    comm_bandwidth_gbps: float
    quant_bits: int = 16
    measured_latency_ms: float = 0.0
    measured_memory_bytes: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingMetrics:
    """Quality metrics for the learned model."""
    num_samples: int = 0
    mae_ms: float = 0.0
    mape_pct: float = 0.0
    r_squared: float = 0.0
    max_error_ms: float = 0.0
    trained_at: float = 0.0


class FeatureExtractor:
    """Converts a (node, layer_range, config) tuple into a feature vector."""

    @staticmethod
    def extract(
        node_id: str,
        start_layer: int,
        end_layer: int,
        batch_size: int,
        seq_len: int,
        gpu_profiles: dict[str, GPUProfile],
        layer_weights: list[LayerWeights],
        topology: TopologyGraph,
    ) -> list[float]:
        layers = layer_weights[start_layer:end_layer]
        gpu = gpu_profiles.get(node_id)

        num_layers = len(layers)
        total_flops = sum(l.flops_per_seq for l in layers) * batch_size * seq_len
        weight_mem = sum(l.weight_memory_bytes for l in layers)
        kv_mem = sum(l.kv_cache_bytes_per_token for l in layers) * batch_size * seq_len
        act_mem = max(
            (l.activation_memory_bytes for l in layers if l.activation_memory_bytes > 0),
            default=seq_len * 2,
        ) * batch_size * seq_len

        hidden_size = 0
        intermediate_size = 0
        for l in layers:
            if l.layer_type == "transformer" and l.weight_memory_bytes > 0:
                if l.hidden_size > 0 or l.intermediate_size > 0:
                    # Ground-truth dims annotated by the layer generator.
                    hidden_size = l.hidden_size
                    intermediate_size = l.intermediate_size
                else:
                    # Fallback: recover dims from weight bytes with the same
                    # estimator the training path uses (F-051 alignment).
                    hidden_size, intermediate_size = _estimate_hidden_intermediate(
                        l.weight_memory_bytes,
                    )
                break

        tflops = gpu.compute_tflops if gpu else 0.0
        mem_bw = gpu.memory_bandwidth_gbps if gpu else 0.0
        gpu_mem = gpu.total_memory_bytes if gpu else 0

        prev_node = None
        node_ids = topology.node_ids
        if node_id in node_ids:
            idx = node_ids.index(node_id)
            if idx > 0:
                prev_node = node_ids[idx - 1]

        comm_bw = 0.0
        is_nvlink = False
        is_infiniband = False
        if prev_node:
            comm_bw = topology.get_bandwidth(prev_node, node_id)
            for link in topology.links:
                if (link.source == prev_node and link.target == node_id) or \
                   (link.source == node_id and link.target == prev_node):
                    is_nvlink = link.is_nvlink
                    is_infiniband = link.is_infiniband
                    break

        return [
            float(num_layers),
            float(batch_size),
            float(seq_len),
            float(total_flops),
            float(weight_mem),
            float(kv_mem),
            float(act_mem),
            tflops,
            mem_bw,
            float(gpu_mem),
            comm_bw,
            1.0 if is_nvlink else 0.0,
            1.0 if is_infiniband else 0.0,
            float(hidden_size),
            float(intermediate_size),
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "num_layers", "batch_size", "seq_len",
            "total_flops", "weight_mem", "kv_mem", "act_mem",
            "gpu_tflops", "gpu_mem_bw_gbps", "gpu_mem_bytes",
            "comm_bandwidth_gbps", "is_nvlink", "is_infiniband",
            "hidden_size", "intermediate_size",
        ]


class GradientBoostedTrees:
    """Lightweight gradient-boosted tree ensemble (no external deps).

    Implements a simple GBR model with stumps (depth-1 trees) for
    fast training and inference on the small datasets typical of
    runtime profiling (hundreds to low thousands of samples).
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._trees: list[DecisionStump] = []
        self._base_prediction: float = 0.0
        self._trained = False

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        if not X or not y:
            return

        n = len(y)
        self._base_prediction = sum(y) / n

        residuals = [y[i] - self._base_prediction for i in range(n)]

        for _ in range(self.n_estimators):
            stump = DecisionStump(max_depth=self.max_depth)
            stump.fit(X, residuals)
            self._trees.append(stump)

            predictions = stump.predict_all(X)
            for i in range(n):
                residuals[i] -= self.learning_rate * predictions[i]

        self._trained = True

    def predict(self, x: list[float]) -> float:
        if not self._trained:
            raise RuntimeError("Model not trained")
        result = self._base_prediction
        for tree in self._trees:
            result += self.learning_rate * tree.predict_one(x)
        return max(result, 0.0)

    def is_trained(self) -> bool:
        return self._trained

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "base_prediction": self._base_prediction,
            "trees": [t.to_dict() for t in self._trees],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradientBoostedTrees:
        model = cls(
            n_estimators=data["n_estimators"],
            learning_rate=data["learning_rate"],
            max_depth=data["max_depth"],
        )
        model._base_prediction = data["base_prediction"]
        model._trees = [DecisionStump.from_dict(t) for t in data["trees"]]
        model._trained = True
        return model


class DecisionStump:
    """A simple decision tree with configurable depth."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self._tree: dict[str, Any] | None = None

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        self._tree = self._build(X, y, depth=0)

    def predict_one(self, x: list[float]) -> float:
        return self._traverse(x, self._tree)

    def predict_all(self, X: list[list[float]]) -> list[float]:
        return [self.predict_one(x) for x in X]

    def _build(self, X: list[list[float]], y: list[float], depth: int) -> dict[str, Any]:
        n = len(y)
        if n == 0:
            return {"value": 0.0}
        if depth >= self.max_depth or n <= 2:
            return {"value": sum(y) / n}

        best_feature = 0
        best_threshold = 0.0
        best_gain = -float("inf")
        total_var = self._variance(y) * n

        num_features = len(X[0]) if X else 0
        for feat in range(num_features):
            values = sorted(set(row[feat] for row in X))
            if len(values) < 2:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

            for thresh in thresholds[:10]:
                left_y = [y[i] for i in range(n) if X[i][feat] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat] > thresh]
                if not left_y or not right_y:
                    continue
                gain = total_var - self._variance(left_y) * len(left_y) - self._variance(right_y) * len(right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build(left_X, left_y, depth + 1),
            "right": self._build(right_X, right_y, depth + 1),
        }

    def _traverse(self, x: list[float], node: dict[str, Any] | None) -> float:
        if node is None:
            return 0.0
        if "value" in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])

    @staticmethod
    def _variance(y: list[float]) -> float:
        if not y:
            return 0.0
        mean = sum(y) / len(y)
        return sum((v - mean) ** 2 for v in y) / len(y)

    def to_dict(self) -> dict[str, Any]:
        return {"max_depth": self.max_depth, "tree": self._tree}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionStump:
        stump = cls(max_depth=data["max_depth"])
        stump._tree = data["tree"]
        return stump


class LearnedCostModel:
    """ML-based cost model that wraps an analytical base model.

    Collects runtime observations, trains a gradient-boosted model,
    and uses it for cost prediction.  Falls back to the analytical
    model when the learned model has insufficient data.

    Args:
        base_cost_model: Analytical fallback model.
        min_samples_to_train: Minimum observations before training.
        retrain_interval: Seconds between automatic retraining.
        save_path: Path to persist the learned model.
    """

    def __init__(
        self,
        base_cost_model: PartitionCostModel,
        gpu_profiles: dict[str, GPUProfile] | None = None,
        min_samples_to_train: int = 50,
        retrain_interval: float = 3600.0,
        save_path: str | Path | None = None,
    ):
        self._base = base_cost_model
        self._gpu_profiles = gpu_profiles or {}
        self._min_samples = min_samples_to_train
        self._retrain_interval = retrain_interval
        self._save_path = Path(save_path) if save_path else None

        self._observations: list[RuntimeObservation] = []
        self._model = GradientBoostedTrees(n_estimators=80, learning_rate=0.1, max_depth=3)
        self._metrics = TrainingMetrics()
        self._last_train_time: float = 0.0
        self._feature_cache: dict[str, list[float]] = {}

        if self._save_path and self._save_path.exists():
            self._load_model()

    def record(self, observation: RuntimeObservation) -> None:
        """Record a real runtime observation for training."""
        self._observations.append(observation)
        self._feature_cache.clear()

        if (len(self._observations) >= self._min_samples and
                time.time() - self._last_train_time > self._retrain_interval):
            self.train()

    def train(self) -> TrainingMetrics:
        """Train the learned model on collected observations."""
        if len(self._observations) < self._min_samples:
            logger.info(
                f"Not enough samples ({len(self._observations)}/{self._min_samples}) "
                f"to train learned cost model"
            )
            return self._metrics

        X: list[list[float]] = []
        y: list[float] = []

        for obs in self._observations:
            features = self._observation_to_features(obs)
            X.append(features)
            y.append(obs.measured_latency_ms)

        self._model.fit(X, y)

        predictions = [self._model.predict(x) for x in X]
        self._metrics = self._compute_metrics(y, predictions)
        self._metrics.trained_at = time.time()
        self._last_train_time = time.time()

        logger.info(
            f"Learned cost model trained on {len(X)} samples: "
            f"MAE={self._metrics.mae_ms:.2f}ms, "
            f"MAPE={self._metrics.mape_pct:.1f}%, "
            f"R²={self._metrics.r_squared:.3f}"
        )

        if self._save_path:
            self._save_model()

        return self._metrics

    def evaluate(
        self,
        node_id: str,
        start_layer_id: int,
        end_layer_id: int,
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> NodeCost:
        """Predict cost using the learned model (with analytical fallback)."""
        base_cost = self._base.evaluate(
            node_id, start_layer_id, end_layer_id, batch_size, seq_len,
        )

        if not self._model.is_trained():
            return base_cost

        try:
            features = self._extract_features(
                node_id, start_layer_id, end_layer_id, batch_size, seq_len,
            )
            predicted_ms = self._model.predict(features)
            predicted_ms = max(predicted_ms, 0.0)

            return NodeCost(
                node_id=base_cost.node_id,
                start_layer=base_cost.start_layer,
                end_layer=base_cost.end_layer,
                compute_time_ms=round(predicted_ms * (base_cost.compute_time_ms / max(base_cost.total_time_ms, 0.001)), 2),
                communication_time_ms=round(predicted_ms * (base_cost.communication_time_ms / max(base_cost.total_time_ms, 0.001)), 2),
                total_time_ms=round(predicted_ms, 2),
                memory_bytes=base_cost.memory_bytes,
                memory_available_bytes=base_cost.memory_available_bytes,
                fits_in_memory=base_cost.fits_in_memory,
            )
        except Exception as e:
            logger.debug(f"Learned model prediction failed, using analytical: {e}")
            return base_cost

    def evaluate_partition(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> list[NodeCost]:
        return [
            self.evaluate(nid, s, e, batch_size, seq_len)
            for nid, s, e in partition
        ]

    def max_latency(self, partition: list[tuple[str, int, int]], batch_size: int = 1, seq_len: int = 4096) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        return max(c.total_time_ms for c in costs) if costs else 0.0

    def combined_throughput(self, partition: list[tuple[str, int, int]], batch_size: int = 1, seq_len: int = 4096) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        if not costs:
            return 0.0
        bottleneck = max(c.total_time_ms for c in costs)
        if bottleneck <= 0:
            return 0.0
        return (batch_size * seq_len) / (bottleneck / 1000.0)

    @property
    def metrics(self) -> TrainingMetrics:
        return self._metrics

    @property
    def num_observations(self) -> int:
        return len(self._observations)

    @property
    def is_learned(self) -> bool:
        return self._model.is_trained()

    def _observation_to_features(self, obs: RuntimeObservation) -> list[float]:
        """Build the training feature vector for a runtime observation.

        F-051: this MUST stay field-for-field identical to
        FeatureExtractor.extract (the serving path) — same indices, same
        formulas.  The memory/flops features are reconstructed from the
        observation's (num_layers, hidden_size, intermediate_size) using the
        same parametric layer layout that FeatureExtractor recovers from
        LayerWeights via _estimate_hidden_intermediate / _single_layer_*.
        """
        h = obs.hidden_size
        i = obs.intermediate_size
        L = obs.num_layers
        B = obs.batch_size
        S = obs.seq_len

        # Same per-layer layout as profiles.GPUProfiler.estimate_layer_weights.
        weight_mem = _single_layer_weight_bytes(h, i) * L
        total_flops = _single_layer_flops(h, i) * L * B * S
        kv_mem = _KV_CACHE_BYTES_PER_TOKEN * L * B * S
        # Serve path: max(activation_memory_bytes) or fallback seq_len*2 when
        # every layer reports 0 (degenerate h=0 case).  Mirror exactly.
        act_per_token = 2 * h if h > 0 else 2 * S
        act_mem = act_per_token * B * S

        return [
            float(L),
            float(B),
            float(S),
            float(total_flops),
            float(weight_mem),
            float(kv_mem),
            float(act_mem),
            obs.gpu_tflops,
            obs.gpu_mem_bw_gbps,
            float(obs.gpu_mem_bytes),
            obs.comm_bandwidth_gbps,
            1.0 if obs.is_nvlink else 0.0,
            1.0 if obs.is_infiniband else 0.0,
            float(h),
            float(i),
        ]

    def _extract_features(
        self, node_id: str, start: int, end: int, batch_size: int, seq_len: int,
    ) -> list[float]:
        cache_key = f"{node_id}:{start}:{end}:{batch_size}:{seq_len}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        features = FeatureExtractor.extract(
            node_id, start, end, batch_size, seq_len,
            self._gpu_profiles, self._base._layer_weights, self._base._topology,
        )
        self._feature_cache[cache_key] = features
        return features

    def _compute_metrics(self, actual: list[float], predicted: list[float]) -> TrainingMetrics:
        n = len(actual)
        if n == 0:
            return TrainingMetrics()

        errors = [abs(a - p) for a, p in zip(actual, predicted)]
        mae = sum(errors) / n

        pct_errors = [abs(a - p) / max(abs(a), 0.001) * 100 for a, p in zip(actual, predicted)]
        mape = sum(pct_errors) / n

        mean_actual = sum(actual) / n
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        ss_tot = sum((a - mean_actual) ** 2 for a in actual)
        r_sq = 1 - ss_res / max(ss_tot, 1e-10)

        return TrainingMetrics(
            num_samples=n,
            mae_ms=round(mae, 3),
            mape_pct=round(mape, 2),
            r_squared=round(r_sq, 4),
            max_error_ms=round(max(errors), 3),
        )

    def _save_model(self) -> None:
        if not self._save_path:
            return
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "model": self._model.to_dict(),
                "metrics": {
                    "num_samples": self._metrics.num_samples,
                    "mae_ms": self._metrics.mae_ms,
                    "mape_pct": self._metrics.mape_pct,
                    "r_squared": self._metrics.r_squared,
                    "max_error_ms": self._metrics.max_error_ms,
                    "trained_at": self._metrics.trained_at,
                },
                "num_observations": len(self._observations),
            }
            with open(self._save_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Learned cost model saved to {self._save_path}")
        except Exception as e:
            logger.debug(f"Failed to save learned model: {e}")

    def _load_model(self) -> None:
        try:
            with open(self._save_path) as f:  # type: ignore[arg-type]
                data = json.load(f)
            self._model = GradientBoostedTrees.from_dict(data["model"])
            m = data.get("metrics", {})
            self._metrics = TrainingMetrics(
                num_samples=m.get("num_samples", 0),
                mae_ms=m.get("mae_ms", 0.0),
                mape_pct=m.get("mape_pct", 0.0),
                r_squared=m.get("r_squared", 0.0),
                max_error_ms=m.get("max_error_ms", 0.0),
                trained_at=m.get("trained_at", 0.0),
            )
            logger.info(f"Loaded learned cost model ({self._metrics.num_samples} samples, R²={self._metrics.r_squared:.3f})")
        except Exception as e:
            logger.debug(f"Could not load learned model: {e}")
