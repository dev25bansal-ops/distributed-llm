#!/bin/sh
set -e

exec distllm-node \
  --node-id "${DISTLLM_NODE_ID}" \
  --model "${DISTLLM_MODEL}" \
  --start-layer "${DISTLLM_START_LAYER}" \
  --end-layer "${DISTLLM_END_LAYER}" \
  --total-layers "${DISTLLM_TOTAL_LAYERS}" \
  "$@"
