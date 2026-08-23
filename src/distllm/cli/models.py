"""Pydantic models for CLI argument validation.

Provides structured validation for the argparse-based CLI commands.
Each model mirrors a CLI command's arguments with type coercion,
range validation, and default values.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CoordinatorArgs(BaseModel):
    """Validated arguments for distllm-coordinator."""
    model_name: str = Field(..., description="Model name or path")
    port: int = Field(default=50050, ge=1024, le=65535)
    dtype: str = Field(default="float16", pattern=r"^(float16|float32|bfloat16)$")
    local: bool = Field(default=False)
    chat_mode: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    debug: bool = Field(default=False)

    @field_validator("model_name")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be empty")
        return v.strip()


class ApiServerArgs(BaseModel):
    """Validated arguments for distllm-api."""
    model_name: str = Field(..., description="Model name or path")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1024, le=65535)
    dtype: str = Field(default="float16", pattern=r"^(float16|float32|bfloat16)$")
    local: bool = Field(default=False)
    debug: bool = Field(default=False)
    no_auth: bool = Field(default=False)


class WorkerArgs(BaseModel):
    """Validated arguments for distllm-node."""
    node_id: str = Field(..., description="Unique worker node ID")
    model_name: str = Field(..., description="Model name")
    start_layer: int = Field(..., ge=0, description="First layer index")
    end_layer: int = Field(..., ge=0, description="Last layer index")
    total_layers: int = Field(..., ge=1, description="Total model layers")
    coordinator_host: str = Field(default="localhost")
    coordinator_port: int = Field(default=50050, ge=1024, le=65535)
    port: int = Field(default=50051, ge=1024, le=65535)
    dtype: str = Field(default="float16", pattern=r"^(float16|float32|bfloat16)$")

    @field_validator("end_layer")
    @classmethod
    def end_gte_start(cls, v: int, info: Any) -> int:
        if "start_layer" in info.data and v < info.data["start_layer"]:
            raise ValueError(f"end_layer ({v}) must be >= start_layer ({info.data['start_layer']})")
        return v

    @model_validator(mode="after")
    def _validate_layer_bounds(self) -> "WorkerArgs":
        """Ensure the worker covers at least one layer (no zero-layer workers)."""
        if self.start_layer >= self.total_layers:
            raise ValueError(
                f"start_layer ({self.start_layer}) must be < total_layers ({self.total_layers})"
            )
        if self.end_layer >= self.total_layers:
            raise ValueError(
                f"end_layer ({self.end_layer}) must be < total_layers ({self.total_layers})"
            )
        return self
