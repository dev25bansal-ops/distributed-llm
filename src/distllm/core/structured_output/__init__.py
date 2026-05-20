"""Structured Output Engine.

Provides grammar-constrained decoding, JSON schema enforcement,
streaming structured output with partial JSON parsing, and
output validation/repair.

Integrates with existing SchemaConstrainedDecoder, GBNFFSM, and
JSON schema infrastructure to provide a cohesive API.
"""

from __future__ import annotations

# Re-export from constrained_decoder for backward compatibility
from distllm.core.constrained_decoder import JSONSchemaConstraint, SchemaConstrainedDecoder

from distllm.core.structured_output.config import (
    GrammarConfig,
    SchemaConfig,
    StreamingConfig,
    StructuredOutputConfig,
    ValidationConfig,
)
from distllm.core.structured_output.engine import (
    GenerationResult,
    StructuredOutputEngine,
)
from distllm.core.structured_output.schema import GBNFGrammar, SchemaConverter
from distllm.core.structured_output.streaming import (
    BufferedAccumulator,
    PartialJSONParser,
    PartialResult,
    StructuredStreamChunk,
    StructuredStreamHandler,
)
from distllm.core.structured_output.validator import (
    OutputRepairer,
    RepairResult,
    SchemaValidator,
    ValidationResult,
)

__all__ = [
    "StructuredOutputEngine",
    "GenerationResult",
    "StructuredOutputConfig",
    "SchemaConfig",
    "GrammarConfig",
    "StreamingConfig",
    "ValidationConfig",
    "SchemaConverter",
    "GBNFGrammar",
    "StructuredStreamHandler",
    "StructuredStreamChunk",
    "PartialJSONParser",
    "PartialResult",
    "BufferedAccumulator",
    "SchemaValidator",
    "ValidationResult",
    "OutputRepairer",
    "RepairResult",
    "JSONSchemaConstraint",
    "SchemaConstrainedDecoder",
]
