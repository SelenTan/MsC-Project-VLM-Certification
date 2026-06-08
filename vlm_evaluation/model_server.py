#!/usr/bin/env python3
"""Start and stop local vLLM servers for Qwen models."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib import error, request
from urllib.parse import urlparse


class ModelServerError(RuntimeError):
    pass


class ManagedServer:
    def __init__(self, process: Optional[subprocess.Popen[bytes]], log_path: Optional[Path]) -> None:
        self.process = process
        self.log_path = log_path

    @property
    def started_by_script(self) -> bool:
        return self.process is not None

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)


def endpoint_base(endpoint: str) -> str:
    if endpoint.endswith("/chat/completions"):
        return endpoint[: -len("/chat/completions")]
    return endpoint.rstrip("/")


def endpoint_models_url(endpoint: str) -> str:
    return f"{endpoint_base(endpoint)}/models"


def endpoint_available(endpoint: str, timeout: int = 5) -> bool:
    try:
        with request.urlopen(endpoint_models_url(endpoint), timeout=timeout) as response:
            return response.status == 200
    except (error.URLError, TimeoutError):
        return False


def parse_host_port(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"Endpoint must include host and port: {endpoint}")
    return parsed.hostname, parsed.port


def wait_for_endpoint(endpoint: str, process: subprocess.Popen[bytes], timeout_seconds: int, log_path: Path) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise ModelServerError(
                f"vLLM server exited before becoming ready. Check log: {log_path}"
            )
        if endpoint_available(endpoint, timeout=5):
            return
        time.sleep(5)
    raise ModelServerError(f"Timed out waiting for vLLM server. Check log: {log_path}")


def start_vllm_server(
    endpoint: str,
    model: str,
    served_model_name: str,
    log_path: Path,
    cuda_visible_devices: str,
    tensor_parallel_size: int,
    max_model_len: Optional[int],
    gpu_memory_utilization: Optional[float],
    limit_mm_per_prompt: Optional[str],
    mm_encoder_tp_mode: Optional[str],
    extra_args: tuple[str, ...],
    wait_timeout_seconds: int,
) -> ManagedServer:
    """Launch vLLM from scratch if the local internal API is not already ready."""
    if endpoint_available(endpoint):
        return ManagedServer(process=None, log_path=None)
    if shutil.which("vllm") is None:
        raise ModelServerError(
            "vLLM is not installed in this environment. Install project dependencies on the GPU machine first, "
            "for example: pip install -r requirement.txt"
        )

    _, port = parse_host_port(endpoint)
    command = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if limit_mm_per_prompt:
        json.loads(limit_mm_per_prompt)
        command.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])
    if mm_encoder_tp_mode:
        command.extend(["--mm-encoder-tp-mode", mm_encoder_tp_mode])
    command.extend(extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    wait_for_endpoint(endpoint, process, wait_timeout_seconds, log_path)
    return ManagedServer(process=process, log_path=log_path)
