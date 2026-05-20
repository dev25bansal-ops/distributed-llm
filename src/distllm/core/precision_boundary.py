"""Precision boundary handling for heterogeneous precision serving.

Handles precision conversion at tensor parallelism boundaries,
ensuring correct results when nodes use different precisions.
"""

from __future__ import annotations

import torch


# Conversion targets: always convert to highest precision in the path
CONVERSION_MAP = {
    "int4": torch.float16,
    "int8": torch.float16,
    "float8_e4m3fn": torch.float16,
    "float8_e5m2": torch.float16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class PrecisionBoundary:
    """Handles precision conversion at tensor parallelism boundaries.

    When nodes in a distributed pipeline use different precisions,
    tensors must be converted to a common precision at communication
    boundaries to avoid cascading quantization errors.

    Strategy: Always convert to the highest precision in the path
    (FP16/BF16). INT8 and FP8 tensors are dequantized before all-reduce
    or point-to-point communication.
    """

    @staticmethod
    def convert_precision(
        tensor: torch.Tensor,
        src_dtype: torch.dtype,
        dst_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a tensor from one precision to another.

        Args:
            tensor: Input tensor.
            src_dtype: Current dtype of the tensor.
            dst_dtype: Target dtype.

        Returns:
            Tensor in the target dtype.
        """
        if src_dtype == dst_dtype:
            return tensor

        # Direct conversion for float types
        if tensor.is_floating_point():
            return tensor.to(dst_dtype)

        # Dequantize integer types
        if src_dtype in (torch.int8, torch.int4):
            # For int8: assume symmetric quantization with scale=1.0
            # In production, scale/zero_point would be passed alongside
            return tensor.to(torch.float32).to(dst_dtype)

        # Fallback
        return tensor.to(dst_dtype)

    @staticmethod
    def get_boundary_dtype(node_a_precision: str, node_b_precision: str) -> torch.dtype:
        """Determine the dtype to use at the boundary between two nodes.

        Always returns the higher precision to avoid information loss.
        """
        rank_a = PRECISION_RANK.get(node_a_precision.lower(), 0)
        rank_b = PRECISION_RANK.get(node_b_precision.lower(), 0)

        if rank_a >= rank_b:
            return CONVERSION_MAP.get(node_a_precision.lower(), torch.float16)
        return CONVERSION_MAP.get(node_b_precision.lower(), torch.float16)

    @staticmethod
    def prepare_for_transfer(
        tensor: torch.Tensor,
        src_precision: str,
        dst_precision: str,
    ) -> tuple[torch.Tensor, dict]:
        """Prepare a tensor for cross-node transfer.

        Returns the converted tensor and metadata needed for the receiver
        to reconstruct the original precision if needed.

        Args:
            tensor: Tensor to transfer.
            src_precision: Source node's precision.
            dst_precision: Destination node's precision.

        Returns:
            Tuple of (converted_tensor, metadata_dict).
        """
        boundary_dtype = PrecisionBoundary.get_boundary_dtype(src_precision, dst_precision)
        converted = tensor.to(boundary_dtype)

        metadata = {
            "src_precision": src_precision,
            "dst_precision": dst_precision,
            "boundary_dtype": str(boundary_dtype),
            "shape": list(tensor.shape),
        }

        return converted, metadata


# Import rank from quality_sla to avoid duplication
PRECISION_RANK = {
    "int4": 0,
    "int8": 1,
    "float8_e4m3fn": 2,
    "float8_e5m2": 2,
    "float16": 3,
    "bfloat16": 4,
    "float32": 5,
}
