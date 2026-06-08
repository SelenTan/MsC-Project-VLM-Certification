#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/uceeht3/hf_cache
mkdir -p /tmp/uceeht3/vllm_cache
mkdir -p /tmp/uceeht3/torch_cache
mkdir -p /tmp/uceeht3/tmp

export HF_HOME=/tmp/uceeht3/hf_cache
export HUGGINGFACE_HUB_CACHE=/tmp/uceeht3/hf_cache/hub
export VLLM_CACHE_ROOT=/tmp/uceeht3/vllm_cache
export TORCH_HOME=/tmp/uceeht3/torch_cache
export TMPDIR=/tmp/uceeht3/tmp

if [ -f .env.local ]; then
  # Keep secrets out of git. Store HF_TOKEN in .env.local on the remote machine.
  set -a
  . ./.env.local
  set +a
fi

echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "TMPDIR=$TMPDIR"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi
