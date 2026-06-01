"""Text generation configuration."""

from pydantic import BaseModel, field_validator

__all__ = [
    "GenerationSettings",
]


class GenerationSettings(BaseModel):
    """Text generation configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"temperature must be 0-2.0, got {v}")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"top_p must be 0-1.0, got {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"top_k must be >= 0, got {v}")
        return v
