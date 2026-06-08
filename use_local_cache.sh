mkdir -p /tmp/uceeht3/hf_cache
mkdir -p /tmp/uceeht3/vllm_cache
mkdir -p /tmp/uceeht3/torch_cache
mkdir -p /tmp/uceeht3/tmp

export HF_HOME=/tmp/uceeht3/hf_cache
export HUGGINGFACE_HUB_CACHE=/tmp/uceeht3/hf_cache/hub
export VLLM_CACHE_ROOT=/tmp/uceeht3/vllm_cache
export TORCH_HOME=/tmp/uceeht3/torch_cache
export TMPDIR=/tmp/uceeht3/tmp
