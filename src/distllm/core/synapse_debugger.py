"""Synapse: Python Debugger Integration for Distributed Model Debugging."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import Any

from loguru import logger


class DistributedBreakpoint:
    """A breakpoint that spans multiple distributed nodes."""

    def __init__(self, node_filter: str | None = None, layer_filter: str | None = None):
        self.node_filter = node_filter
        self.layer_filter = layer_filter
        self._hit_count = 0
        self._continue = threading.Event()
        self._continue.set()

    def should_break(self, node_id: str, layer_name: str, tensor_info: dict) -> bool:
        if self.node_filter and self.node_filter != node_id:
            return False
        if self.layer_filter and self.layer_filter not in layer_name:
            return False
        return True

    def hit(self, node_id: str, layer_name: str, tensor_info: dict) -> None:
        self._hit_count += 1
        self._continue.clear()
        logger.info(f"[SYNAPSE] Breakpoint hit: {node_id}/{layer_name}")
        logger.info(f"  Tensor: {tensor_info}")
        self._continue.wait()

    def resume(self) -> None:
        self._continue.set()

    @property
    def hit_count(self) -> int:
        return self._hit_count


class TensorInspector:
    """Inspect tensors during distributed execution."""

    @staticmethod
    def snapshot(tensor: Any, name: str = "") -> dict:
        if hasattr(tensor, 'shape'):
            shape = list(tensor.shape)
        elif hasattr(tensor, 'size'):
            shape = list(tensor.size())
        else:
            shape = []

        info = {"name": name, "shape": shape}
        if hasattr(tensor, 'dtype'):
            info["dtype"] = str(tensor.dtype)
        if hasattr(tensor, 'device'):
            info["device"] = str(tensor.device)
        if hasattr(tensor, 'min') and callable(getattr(tensor, 'min')):
            try:
                info["min"] = float(tensor.min())
                info["max"] = float(tensor.max())
                info["mean"] = float(tensor.mean())
                info["std"] = float(tensor.std())
                info["norm"] = float(tensor.norm())
            except Exception:
                pass
        if hasattr(tensor, 'numel'):
            info["num_elements"] = tensor.numel()
        if hasattr(tensor, 'grad') and tensor.grad is not None:
            info["grad_norm"] = float(tensor.grad.norm())
        return info

    @staticmethod
    def compare(tensor_a: Any, tensor_b: Any) -> dict:
        a = TensorInspector.snapshot(tensor_a, "A")
        b = TensorInspector.snapshot(tensor_b, "B")
        diff = {}
        if "min" in a and "min" in b:
            diff["max_diff"] = max(abs(a["min"] - b["min"]), abs(a["max"] - b["max"]))
        if "mean" in a and "mean" in b:
            diff["mean_diff"] = abs(a["mean"] - b["mean"])
        if hasattr(tensor_a, 'sub') and hasattr(tensor_b, 'sub'):
            try:
                diff_tensor = tensor_a.sub(tensor_b).abs()
                diff["l1"] = float(diff_tensor.mean())
                diff["l2"] = float(diff_tensor.norm())
                diff["cosine_sim"] = float(
                    (tensor_a.flatten() @ tensor_b.flatten())
                    / (tensor_a.norm() * tensor_b.norm() + 1e-8)
                )
            except Exception:
                pass
        return {"A": a, "B": b, "diff": diff}


class DistributedTracer:
    """Trace execution across distributed nodes."""

    def __init__(self):
        self._spans: list[dict] = []
        self._enabled = False
        self._lock = threading.RLock()

    def start(self) -> None:
        self._enabled = True
        logger.info("[SYNAPSE] Distributed tracing started")

    def stop(self) -> None:
        self._enabled = False
        logger.info(f"[SYNAPSE] Distributed tracing stopped ({len(self._spans)} spans)")

    def record(self, node_id: str, layer_name: str, event: str, duration_ms: float, metadata: dict | None = None) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._spans.append({
                "node_id": node_id,
                "layer": layer_name,
                "event": event,
                "duration_ms": round(duration_ms, 2),
                "timestamp": time.time(),
                "metadata": metadata or {},
            })

    def get_trace(self, node_filter: str | None = None) -> list[dict]:
        if node_filter:
            return [s for s in self._spans if s["node_id"] == node_filter]
        return list(self._spans)

    def export_chrome_trace(self, path: str = "") -> str:
        events = []
        for i, span in enumerate(self._spans):
            events.append({
                "name": f"{span['node_id']}/{span['layer']}",
                "cat": span["event"],
                "ph": "X",
                "ts": int(span["timestamp"] * 1_000_000),
                "dur": int(span["duration_ms"] * 1000),
                "tid": i,
                "pid": span["node_id"],
                "args": span["metadata"],
            })
        import json
        output = json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output

    @property
    def span_count(self) -> int:
        return len(self._spans)


class SynapseDebugger:
    """Integrated debugger for distributed model execution."""

    def __init__(self):
        self._tracer = DistributedTracer()
        self._breakpoints: list[DistributedBreakpoint] = []
        self._inspector = TensorInspector()
        self._debug_mode = os.environ.get("DISTLLM_DEBUG", "0") == "1"

    @property
    def tracer(self) -> DistributedTracer:
        return self._tracer

    @property
    def inspector(self) -> TensorInspector:
        return self._inspector

    def add_breakpoint(self, node_filter: str | None = None, layer_filter: str | None = None) -> DistributedBreakpoint:
        bp = DistributedBreakpoint(node_filter, layer_filter)
        self._breakpoints.append(bp)
        logger.info(f"[SYNAPSE] Breakpoint added: node={node_filter}, layer={layer_filter}")
        return bp

    def clear_breakpoints(self) -> None:
        self._breakpoints.clear()

    def check_breakpoints(self, node_id: str, layer_name: str, tensor_info: dict) -> bool:
        hit = False
        for bp in self._breakpoints:
            if bp.should_break(node_id, layer_name, tensor_info):
                bp.hit(node_id, layer_name, tensor_info)
                hit = True
        return hit

    def debug_hook(self, node_id: str, layer_name: str, input_tensor: Any, output_tensor: Any) -> None:
        if not self._debug_mode:
            return
        tensor_info = self._inspector.snapshot(output_tensor, layer_name)
        self.check_breakpoints(node_id, layer_name, tensor_info)

    def interactive_shell(self, namespace: dict | None = None) -> None:
        """Start an interactive Python shell within the debugger context."""
        try:
            import code
            vars = namespace or {}
            vars.update({"inspector": self._inspector, "tracer": self._tracer})
            code.interact(local=vars, banner="[SYNAPSE] Distributed debugger shell")
        except ImportError:
            logger.error("code module not available for interactive shell")

    def start_tracing(self) -> None:
        self._tracer.start()

    def stop_tracing(self, export_path: str = "") -> str:
        self._tracer.stop()
        if export_path:
            return self._tracer.export_chrome_trace(export_path)
        return ""

    @property
    def stats(self) -> dict:
        return {
            "debug_mode": self._debug_mode,
            "breakpoints": len(self._breakpoints),
            "spans": self._tracer.span_count,
        }
