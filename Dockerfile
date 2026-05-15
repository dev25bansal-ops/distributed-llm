FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and OpenSSL
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root user
RUN groupadd -r distllm && useradd -r -g distllm -d /app -s /bin/bash distllm \
    && chown -R distllm:distllm /app

# Install application and dependencies
COPY --chown=distllm:distllm . .
RUN pip3 install --no-cache-dir -e .

# Security: Switch to non-root user
USER distllm

# Configurable defaults via environment variables
ENV DISTLLM_NODE_ID=worker-0
ENV DISTLLM_MODEL=roneneldan/TinyStories-1M
ENV DISTLLM_START_LAYER=0
ENV DISTLLM_END_LAYER=3
ENV DISTLLM_TOTAL_LAYERS=8

# Security: Run on non-privileged port
EXPOSE 8000 50051

ENTRYPOINT ["sh", "-c", "distllm-node --node-id ${DISTLLM_NODE_ID} --model ${DISTLLM_MODEL} --start-layer ${DISTLLM_START_LAYER} --end-layer ${DISTLLM_END_LAYER} --total-layers ${DISTLLM_TOTAL_LAYERS}"]
