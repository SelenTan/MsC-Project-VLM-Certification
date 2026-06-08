#!/usr/bin/env bash
set -euo pipefail

echo "Before:"
du -sh /tmp/uceeht3/* 2>/dev/null

rm -rf /tmp/uceeht3/vllm_cache/*
rm -rf /tmp/uceeht3/torch_cache/*
rm -rf /tmp/uceeht3/tmp/*

echo "After:"
du -sh /tmp/uceeht3/* 2>/dev/null
