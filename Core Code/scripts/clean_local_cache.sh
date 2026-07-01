#!/usr/bin/env bash
set -euo pipefail

cache_root="${VLM_CACHE_ROOT:-/tmp/${USER}}"

echo "Cache root: $cache_root"
echo "Before:"
du -sh "$cache_root" "$cache_root"/* 2>/dev/null || true

rm -rf "$cache_root/vllm_cache"/*
rm -rf "$cache_root/torch_cache"/*
rm -rf "$cache_root/tmp"/*

if [ "${1:-}" = "--all" ]; then
  rm -rf "$cache_root/hf_cache"/*
else
  echo 'Keeping Hugging Face model cache. Use: bash "Core Code/scripts/clean_local_cache.sh" --all'
fi

echo "After:"
du -sh "$cache_root" "$cache_root"/* 2>/dev/null || true
