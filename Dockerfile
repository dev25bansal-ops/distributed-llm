# BUILDER stage — compile + install. This is the "dev" stage; the release stage is below.
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.10
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV NCCL_ALGO=Ring
ENV NCCL_PROTO=Simple

# Install system deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    python3-pip python3-setuptools python3-wheel \
    gcc g++ ninja-build git curl \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# NCCL: install from system (already in cuda-devel image) and ensure NCCL headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl-dev libnccl2 \
    && rm -rf /var/lib/apt/lists/*

# Install locked dependencies first (cached layer, pinning all transitive deps)
COPY requirements.lock .
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock

# Copy project files and install the package itself (without re-resolving deps)
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY proto/ proto/
RUN pip install --no-cache-dir --prefix=/install . --no-deps

# Torch with CUDA support. requirements.lock does not include torch, and
# without --prefix the wheel would live only in this builder stage and never
# reach the runtime image, so install it into /install as well.
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ==========================================================================
# RUNTIME stage — minimal, no build deps, no dev packages
# ==========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.10
ARG DISTLLM_VERSION=0.4.1

LABEL org.opencontainers.image.source="https://github.com/distributed-llm/distributed-llm"
LABEL org.opencontainers.image.version="${DISTLLM_VERSION}"
LABEL org.opencontainers.image.title="Distributed LLM"
LABEL org.opencontainers.image.description="Distributed LLM Inference System — production image"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NCCL_ALGO=Ring
ENV NCCL_PROTO=Simple
ENV NCCL_DEBUG=WARN

WORKDIR /app

# Install ONLY runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    libnccl2 \
    curl ca-certificates \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r distllm && \
    useradd -r -g distllm -d /app -s /bin/bash distllm && \
    chown -R distllm:distllm /app

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

USER distllm

# Default env for deployment.
# Model name/dtype are read via pydantic-settings: env_prefix "DISTLLM_" plus
# env_nested_delimiter "__", so model.name maps to DISTLLM_MODEL__NAME and
# model.dtype to DISTLLM_MODEL__DTYPE (verified with the installed
# pydantic-settings: DISTLLM__MODEL__NAME is silently IGNORED). Plain
# DISTLLM_MODEL / DISTLLM_DTYPE are not read by the server. Pipeline layer
# ranges and node identity are CLI flags (--node-id/--start-layer/...).
ENV DISTLLM_MODEL__NAME=meta-llama/Llama-3.2-1B
ENV DISTLLM_MODEL__DTYPE=float16

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 50051

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# L-11: To use a different CUDA version, pass --build-arg CUDA_VERSION=12.6
#   docker build --build-arg CUDA_VERSION=12.6 -t distllm:custom .
# The Dockerfile.cuda12.1 and Dockerfile.cuda12.6 variants are aliases
# for backward compatibility.
