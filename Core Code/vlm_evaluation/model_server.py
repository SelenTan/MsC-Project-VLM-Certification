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


def auto_select_gpus(
    required_count: int,
    min_free_memory_mb: int,
    required_free_fraction: Optional[float] = None,
) -> str:
    """Select the GPUs with the most free memory using nvidia-smi, or fail if not enough are available."""
    candidates = query_usable_gpus(min_free_memory_mb, required_free_fraction)
    if len(candidates) < required_count:
        available = ", ".join(f"gpu{index}:{free_memory}MB" for index, free_memory in candidates) or "none"
        utilization_text = (
            f" and enough free memory for gpu_memory_utilization={required_free_fraction}"
            if required_free_fraction is not None
            else ""
        )
        raise ModelServerError(
            f"Need {required_count} GPU(s) with at least {min_free_memory_mb}MB free each, "
            f"a working PyTorch CUDA runtime{utilization_text}, but only found: {available}"
        )
    selected = sorted(index for index, _ in candidates[:required_count])
    return ",".join(str(index) for index in selected)


def auto_select_gpus_by_balanced_memory(
    max_gpu_count: int,
    required_balanced_memory_mb: int,
    min_free_memory_mb: int,
    required_free_fraction: Optional[float] = None,
) -> str:
    """Select the smallest GPU set whose TP-balanced usable memory meets the target requirement."""
    candidates = query_usable_gpus(min_free_memory_mb, required_free_fraction)
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
        f"with each selected GPU above {min_free_memory_mb}MB free and CUDA-runtime usable. Available: {available}"
    )


def query_usable_gpus(
    min_free_memory_mb: int,
    required_free_fraction: Optional[float] = None,
) -> list[tuple[int, int]]:
    """Return free-memory GPU candidates after removing devices PyTorch cannot initialize."""
    usable: list[tuple[int, int]] = []
    failed: list[str] = []
    for index, free_memory_mb, total_memory_mb in query_gpu_memory(min_free_memory_mb):
        if required_free_fraction is not None and free_memory_mb < total_memory_mb * required_free_fraction:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        error_text = cuda_runtime_error(env, str(index))
        if error_text is None:
            usable.append((index, free_memory_mb))
        else:
            failed.append(f"gpu{index}:{free_memory_mb}MB")
    if not usable and failed:
        raise ModelServerError(
            "nvidia-smi reports free GPUs, but PyTorch cannot initialize CUDA on them: "
            + ", ".join(failed)
        )
    return usable


def query_gpu_memory(min_free_memory_mb: int) -> list[tuple[int, int, int]]:
    """Return GPUs with enough free memory as (index, free MB, total MB), sorted by free memory."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ModelServerError("Could not run nvidia-smi for GPU auto-selection.") from exc

    candidates: list[tuple[int, int, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, free_memory_text, total_memory_text = [part.strip() for part in line.split(",", maxsplit=2)]
        index = int(index_text)
        free_memory_mb = int(free_memory_text)
        total_memory_mb = int(total_memory_text)
        if free_memory_mb >= min_free_memory_mb:
            candidates.append((index, free_memory_mb, total_memory_mb))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def query_free_gpus(min_free_memory_mb: int) -> list[tuple[int, int]]:
    return [(index, free_memory_mb) for index, free_memory_mb, _ in query_gpu_memory(min_free_memory_mb)]


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


def validate_cuda_runtime(env: dict[str, str], cuda_visible_devices: str) -> None:
    """Fail before vLLM startup when PyTorch cannot see a usable CUDA runtime."""
    requested_count = len([device for device in cuda_visible_devices.split(",") if device.strip()]) if cuda_visible_devices else None
    error_text = cuda_runtime_error(env, cuda_visible_devices)
    if error_text is not None:
        raise ModelServerError(
            "CUDA is not available to PyTorch in this session, so vLLM cannot start. "
            "Run on a GPU node/session and check nvidia-smi before starting the workflow."
            + (f"\nCUDA_VISIBLE_DEVICES={cuda_visible_devices or '<unset>'}\n{error_text}" if error_text else "")
        )
    if requested_count is not None:
        visible_count = cuda_device_count(env)
        if visible_count < requested_count:
            raise ModelServerError(
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} exposes {visible_count} CUDA device(s), "
                f"but {requested_count} were requested."
            )


def validate_gpu_memory_request(cuda_visible_devices: str, gpu_memory_utilization: Optional[float]) -> None:
    """Fail before vLLM startup when selected GPUs cannot satisfy the requested memory reservation."""
    if gpu_memory_utilization is None or not cuda_visible_devices:
        return
    requested_devices = [int(device.strip()) for device in cuda_visible_devices.split(",") if device.strip()]
    memory_by_index = {index: (free_mb, total_mb) for index, free_mb, total_mb in query_gpu_memory(0)}
    insufficient: list[str] = []
    missing: list[int] = []
    for index in requested_devices:
        memory = memory_by_index.get(index)
        if memory is None:
            missing.append(index)
            continue
        free_mb, total_mb = memory
        required_mb = total_mb * gpu_memory_utilization
        if free_mb < required_mb:
            insufficient.append(f"gpu{index}: {free_mb}MB free, needs {required_mb:.0f}MB")
    if missing:
        raise ModelServerError(f"Selected GPU(s) not found by nvidia-smi: {', '.join(map(str, missing))}")
    if insufficient:
        raise ModelServerError(
            "Selected GPU(s) do not have enough free memory for "
            f"gpu_memory_utilization={gpu_memory_utilization}: "
            + "; ".join(insufficient)
        )


def cuda_runtime_error(env: dict[str, str], cuda_visible_devices: str) -> Optional[str]:
    """Return a short CUDA initialization error, or None when PyTorch can use CUDA."""
    script = """
import sys
try:
    import torch
except Exception as exc:
    print(f"Could not import torch: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False.", file=sys.stderr)
    raise SystemExit(3)
device_count = torch.cuda.device_count()
print(f"torch CUDA device_count={device_count}")
if device_count:
    print(f"first CUDA device={torch.cuda.get_device_name(0)}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return None


def cuda_device_count(env: dict[str, str]) -> int:
    script = "import torch\nprint(torch.cuda.device_count())\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(result.stdout.strip() or "0") if result.returncode == 0 else 0


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
    validate_cuda_runtime(env, cuda_visible_devices)
    validate_gpu_memory_request(cuda_visible_devices, gpu_memory_utilization)

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
