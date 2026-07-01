#!/usr/bin/env python3
"""Start and stop local vLLM servers for Qwen models."""

from __future__ import annotations

import json
import os
import shlex
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


def auto_select_gpus(required_count: int, min_free_memory_mb: int) -> str:
    """Select the GPUs with the most free memory using nvidia-smi, or fail if not enough are available."""
    candidates = query_free_gpus(min_free_memory_mb)
    if len(candidates) < required_count:
        available = ", ".join(f"gpu{index}:{free_memory}MB" for index, free_memory in candidates) or "none"
        raise ModelServerError(
            f"Need {required_count} GPU(s) with at least {min_free_memory_mb}MB free each, "
            f"but only found: {available}"
        )
    selected = sorted(index for index, _ in candidates[:required_count])
    return ",".join(str(index) for index in selected)


def auto_select_gpus_by_balanced_memory(
    max_gpu_count: int,
    required_balanced_memory_mb: int,
    min_free_memory_mb: int,
) -> str:
    """Select the smallest GPU set whose TP-balanced usable memory meets the target requirement."""
    candidates = query_free_gpus(min_free_memory_mb)
    for gpu_count in range(1, max_gpu_count + 1):
        if len(candidates) < gpu_count:
            break
        selected_candidates = candidates[:gpu_count]
        min_selected_memory = min(free_memory for _, free_memory in selected_candidates)
        balanced_memory = gpu_count * min_selected_memory
        if balanced_memory >= required_balanced_memory_mb:
            selected = sorted(index for index, _ in selected_candidates)
            return ",".join(str(index) for index in selected)

    available = ", ".join(f"gpu{index}:{free_memory}MB" for index, free_memory in candidates) or "none"
    raise ModelServerError(
        f"Could not satisfy balanced GPU memory requirement. Need at least "
        f"{required_balanced_memory_mb}MB balanced usable memory across up to {max_gpu_count} GPU(s), "
        f"with each selected GPU above {min_free_memory_mb}MB free. Available: {available}"
    )


def query_free_gpus(min_free_memory_mb: int) -> list[tuple[int, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ModelServerError("Could not run nvidia-smi for GPU auto-selection.") from exc

    candidates: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, free_memory_text = [part.strip() for part in line.split(",", maxsplit=1)]
        index = int(index_text)
        free_memory_mb = int(free_memory_text)
        if free_memory_mb >= min_free_memory_mb:
            candidates.append((index, free_memory_mb))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def endpoint_base(endpoint: str) -> str:
    if endpoint.endswith("/chat/completions"):
        return endpoint[: -len("/chat/completions")]
    return endpoint.rstrip("/")


def endpoint_models_url(endpoint: str) -> str:
    return f"{endpoint_base(endpoint)}/models"


def endpoint_model_ids(endpoint: str, timeout: int = 5) -> Optional[set[str]]:
    try:
        with request.urlopen(endpoint_models_url(endpoint), timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    if response.status != 200 or not isinstance(response_data, dict):
        return None
    models = response_data.get("data")
    if not isinstance(models, list):
        return None
    return {
        model["id"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    }


def endpoint_available(endpoint: str, timeout: int = 5, expected_model: Optional[str] = None) -> bool:
    model_ids = endpoint_model_ids(endpoint, timeout)
    return model_ids is not None and (expected_model is None or expected_model in model_ids)


def parse_host_port(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"Endpoint must include host and port: {endpoint}")
    return parsed.hostname, parsed.port


def wait_for_endpoint(
    endpoint: str,
    expected_model: str,
    process: subprocess.Popen[bytes],
    timeout_seconds: int,
    log_path: Path,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise ModelServerError(
                f"vLLM server exited before becoming ready. Check log: {log_path}\n\n"
                f"Last log lines:\n{tail_log(log_path)}"
            )
        if endpoint_available(endpoint, timeout=5, expected_model=expected_model):
            return
        time.sleep(5)
    raise ModelServerError(
        f"Timed out waiting for vLLM server. Check log: {log_path}\n\n"
        f"Last log lines:\n{tail_log(log_path)}"
    )


def tail_log(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return "<log file does not exist>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) or "<log file is empty>"


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
    server_label: str = "vLLM",
) -> ManagedServer:
    """Launch vLLM from scratch if the local internal API is not already ready."""
    existing_model_ids = endpoint_model_ids(endpoint)
    if existing_model_ids is not None and served_model_name in existing_model_ids:
        return ManagedServer(process=None, log_path=None)
    if existing_model_ids is not None:
        raise ModelServerError(
            f"Endpoint {endpoint!r} is already serving {sorted(existing_model_ids)!r}, "
            f"not required model {served_model_name!r}."
        )
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
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    log_file = log_path.open("ab")
    process: Optional[subprocess.Popen[bytes]] = None
    try:
        command_text = shlex.join(command)
        print(f"Starting {server_label} server. Log: {log_path}", flush=True)
        log_file.write(f"\n\n=== vLLM command ===\n{command_text}\n".encode("utf-8"))
        log_file.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n".encode("utf-8"))
        log_file.write(f"HF_HUB_DISABLE_XET={env['HF_HUB_DISABLE_XET']}\n".encode("utf-8"))
        log_file.flush()
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        wait_for_endpoint(endpoint, served_model_name, process, wait_timeout_seconds, log_path)
    except Exception:
        if process is not None:
            ManagedServer(process=process, log_path=log_path).stop()
        raise
    finally:
        log_file.close()
    return ManagedServer(process=process, log_path=log_path)
