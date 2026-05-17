# --- Stage 1: Build ---
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir --prefix=/install -e .

# --- Stage 2: Runtime ---
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.10
ARG DISTLLM_VERSION=0.4.0

# OCI labels
LABEL org.opencontainers.image.source="https://github.com/distributed-llm/distributed-llm"
LABEL org.opencontainers.image.version="${DISTLLM_VERSION}"
LABEL org.opencontainers.image.title="Distributed LLM"
LABEL org.opencontainers.image.description="Distributed LLM Inference System"

WORKDIR /app

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Security: Create non-root user
RUN groupadd -r distllm && useradd -r -g distllm -d /app -s /bin/bash distllm \
    && chown -R distllm:distllm /app

# Security: Switch to non-root user
USER distllm

# Configurable defaults via environment variables
ENV DISTLLM_NODE_ID=worker-0
ENV DISTLLM_MODEL=roneneldan/TinyStories-1M
ENV DISTLLM_START_LAYER=0
ENV DISTLLM_END_LAYER=3
ENV DISTLLM_TOTAL_LAYERS=8

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 50051

ENTRYPOINT ["sh", "-c", "distllm-node --node-id ${DISTLLM_NODE_ID} --model ${DISTLLM_MODEL} --start-layer ${DISTLLM_START_LAYER} --end-layer ${DISTLLM_END_LAYER} --total-layers ${DISTLLM_TOTAL_LAYERS}"]
