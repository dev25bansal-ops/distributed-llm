FROM python:3.11-slim AS builder

WORKDIR /app
COPY dist/ dist/
RUN pip install dist/*.whl --no-cache-dir

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN pip install \
    gunicorn==22.0.0 \
    uvicorn[standard]==0.27.1 \
    redis==5.0.8 \
    grpcio==1.64.1 \
    --no-cache-dir

RUN groupadd -r distllm && useradd -r -g distllm -m distllm && \
    chown -R distllm:distllm /app
USER distllm

EXPOSE 8000 50051 9090

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["distllm-api"]
