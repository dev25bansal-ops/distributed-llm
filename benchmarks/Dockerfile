# Benchmark Environment Container
#
# Pinned system dependencies, CUDA, and PyTorch versions for
# bit-for-bit reproducible benchmarks across different machines.
#
# Usage:
#   docker build -f benchmarks/Dockerfile -t distllm-benchmark .
#   docker run --gpus all distllm-benchmark

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (pinned for reproducibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10=3.10.12-* \
    python3-pip=22.0.2-* \
    curl=7.81.0-* \
    git=1:2.34.1-* \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /benchmark

# Python dependencies pinned for reproducibility
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Install DistLLM from wheel
COPY dist/ dist/
RUN pip install dist/*.whl --no-cache-dir

# Copy benchmark suite
COPY benchmarks/ ./benchmarks/

# Hardware check
RUN python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Torch: {torch.__version__}')"

ENTRYPOINT ["python3", "-m", "benchmarks.run"]
