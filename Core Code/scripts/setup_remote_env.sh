#!/usr/bin/env bash
set -euo pipefail

cache_root="${VLM_CACHE_ROOT:-/tmp/${USER}}"

mkdir -p \
  "$cache_root/hf_cache/hub" \
  "$cache_root/hf_cache/transformers" \
  "$cache_root/hf_cache/datasets" \
  "$cache_root/vllm_cache" \
  "$cache_root/torch_cache" \
  "$cache_root/tmp"

export HF_HOME="$cache_root/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$cache_root/vllm_cache"
export TORCH_HOME="$cache_root/torch_cache"
export TMPDIR="$cache_root/tmp"
export HF_HUB_DISABLE_XET=1

if [ -f .env.local ]; then
  set -a
  . ./.env.local
  set +a
fi

echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "TMPDIR=$TMPDIR"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi
