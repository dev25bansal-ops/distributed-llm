"""RAG pipeline, agent loop, and plugin system configuration classes."""

from typing import Any
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "RAGSettings",
    "AgentSettings",
    "PluginSettings",
]


class RAGSettings(BaseModel):
    """RAG pipeline with FAISS."""
    enabled: bool = False
    dimension: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 50
    index_path: str | None = None


class AgentSettings(BaseModel):
    """ReAct agent loop."""
    enabled: bool = False
    max_iterations: int = 10
    reflection_enabled: bool = True


class PluginSettings(BaseModel):
    """Plugin system configuration."""
    enabled: bool = True
    plugins: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in v:
            if isinstance(item, dict) and "module" in item:
                if "." not in item["module"]:
                    raise ValueError(f"Plugin module must be fully qualified, got {item['module']}")
        return v
