# DistLLM Development Container
FROM mcr.microsoft.com/devcontainers/universal:2

# Install additional tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        nvidia-cuda-toolkit \
        && rm -rf /var/lib/apt/lists/*

# Pre-install project dependencies
COPY requirements.txt /tmp/requirements.txt
COPY sdk/pyproject.toml /tmp/sdk-pyproject.toml
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
