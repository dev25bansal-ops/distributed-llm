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

# Ubuntu 22.04 ships pip 22.0.2 / setuptools 59.6.0. Setuptools <61 cannot read
# PEP 621 [project] metadata from pyproject.toml, and this repo has no setup.py
# or setup.cfg, so the build silently produced an empty UNKNOWN-0.0.0 wheel
# (961 bytes, no console scripts) and the runtime image had no `distllm-api`
# binary. The floors below are the first versions that understand this file:
# setuptools 61 (PEP 621) and pip 23 (Metadata-Version 2.4 wheels).
RUN python3 -m pip install --no-cache-dir --upgrade \
    "pip>=23.0" "setuptools>=68.0" wheel

WORKDIR /build

# Interpreter for the whole image: a venv keeps one consistent layout (Debian
# Python searches .../dist-packages while `pip --prefix` writes site-packages)
# and we copy it wholesale to the runtime at this same absolute path.
RUN python3 -m venv /install
ENV PATH=/install/bin:${PATH}

# Install PyTorch with CUDA support (pre-built wheel for speed)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# NCCL: already provided by the nvidia/cuda:*-devel base image (headers + libs).
# Installing the distro libnccl-dev/libnccl2 changes held packages and makes
# apt fail under -y ("Held packages were changed ... --allow-change-held-packages"),
# so we rely on the base image's NCCL instead.

# Install locked dependencies first (cached layer, pinning all transitive deps)
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Copy project files and install the package itself (without re-resolving deps)
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY proto/ proto/
# The package metadata (pyproject.toml) lives at the repo root, not in src/,
# so install the current directory. PIP finds the distllm package under
# src/ because pyproject sets package-dir = {"" = "src"}.
#
# Install the self-hosted extra too: core.coordinator imports transformers at
# module scope, and requirements.lock does not carry it, so a bare install
# produced an image that built fine and then died on
# `ModuleNotFoundError: No module named 'transformers'` when the coordinator
# started. torch is already installed from the CUDA index above and satisfies
# the extra's requirement, so pip will not pull a second copy.
RUN pip install --no-cache-dir "transformers>=4.48.0" \
    "safetensors>=0.4.0" "huggingface-hub>=0.20.0" "sentencepiece>=0.1.99" \
    "accelerate>=1.0.0" "nvidia-ml-py>=12.0"
RUN pip install --no-cache-dir . --no-deps

# Fail the build (not the container at runtime) if the package installed without
# its console scripts. A metadata-parsing failure yields an empty wheel that
# pip reports as success, and the image then starts with "distllm-api: not found".
RUN test -x /install/bin/distllm-api \
    && test -x /install/bin/distllm-coordinator \
    && test -x /install/bin/distllm-node \
    || { echo "ERROR: distllm console scripts missing from /install/bin"; ls -la /install/bin || true; exit 1; }

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
    curl ca-certificates \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv at the SAME absolute path it was built at: pyvenv.cfg and the
# console-script shebangs embed /install, so relocating it would break both.
COPY --from=builder /install /install
ENV PATH=/install/bin:${PATH}

# Create non-root user
RUN groupadd -r distllm && \
    useradd -r -g distllm -d /app -s /bin/bash distllm && \
    chown -R distllm:distllm /app

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

USER distllm

# Default env for deployment
ENV DISTLLM_NODE_ID=worker-0
ENV DISTLLM_MODEL=meta-llama/Llama-3.2-1B
ENV DISTLLM_START_LAYER=0
ENV DISTLLM_END_LAYER=3
ENV DISTLLM_TOTAL_LAYERS=8

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 50051

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# L-11: To use a different CUDA version, pass --build-arg CUDA_VERSION=12.6
#   docker build --build-arg CUDA_VERSION=12.6 -t distllm:custom .
# The Dockerfile.cuda12.1 and Dockerfile.cuda12.6 variants are aliases
# for backward compatibility.
