#!/usr/bin/env python3
"""One-command workflow for AI checker validation and certification calculation."""

from __future__ import annotations

import sys
import traceback
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "vlm_evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "vlm_testing"))

from ai_checker import CheckerError, call_checker, load_annotation_guide, require_local_endpoint
from artifacts import (
    apply_checker_results,
    apply_human_label_sheet,
    create_run_dir,
    default_run_name,
    export_run_artifacts,
    load_records,
    record_key,
    sample_human_gold_pool,
    update_item_fields,
    validate_target_responses,
    write_human_label_sheet,
    write_json,
)
from benchmark_labels import (
    apply_benchmark_labels,
    default_benchmark_path,
    load_benchmark_rows,
    merge_existing_benchmark_labels,
    read_benchmark_csv,
    write_benchmark_csv,
)
from certification import alpha_grid, estimate_reliability, run_monte_carlo, summarize_certificates
from dataset_selection import select_dataset_dir
from error_experiments import (
    run_type_error_experiment,
    write_error_experiment_artifacts,
)
from model_server import ManagedServer, auto_select_gpus, auto_select_gpus_by_balanced_memory, start_vllm_server
from reset_dataset import reset_dataset_fields
from workflow_state import (
    append_judge_row,
    append_target_row,
    chunk_indexes,
    collect_chunk_results,
    delete_chunk_checkpoints,
    group_records_for_chunks,
    judge_results_path,
    load_human_gold_keys,
    load_local_chunks,
    load_manifest,
    load_run_status,
    missing_keys,
    parse_chunk_selection,
    progress_summary,
    records_for_chunks,
    save_local_chunks,
    successful_judge_rows,
    successful_target_rows,
    target_results_path,
    write_human_gold_keys,
    write_manifest,
    write_run_status,
)
from run_target_vlm import run_records as run_target_vlm_records


# Paths
DATASET_DIR = "Large Dataset"
DATASET_PATHS = {
    "Medium Dataset": "Medium Dataset",
    "Large Dataset": str(Path.home() / "datasets" / "Large_Dataset"),
}
RUNS_DIR = "runs"
ANNOTATION_GUIDE_PATH = "ANNOTATION_GUIDE.md"
BENCHMARK_LABELS_DIR = "benchmark_labels"
BENCHMARK_REQUIRED_DATASETS: tuple[str, ...] = ()
RUN_NAME = None

# Target VLM response generation
RUN_TARGET_VLM_FIRST = True
TARGET_CATEGORIES = None
TARGET_CUDA_VISIBLE_DEVICES = None
TARGET_GPU_MEMORY_UTILIZATION = 0.7
TARGET_LIMIT = None
TARGET_LIMIT_MM_PER_PROMPT = '{"image":2,"video":0}'
TARGET_MAX_GPUS = 2
TARGET_MAX_MODEL_LEN = 8192
TARGET_MAX_TOKENS = 160
TARGET_MIN_FREE_MEMORY_MB = 40000
TARGET_MM_ENCODER_TP_MODE = None
TARGET_MODEL_NAME = "mistralai/Pixtral-12B-2409"
TARGET_OVERWRITE_RESPONSES = False
TARGET_PORT = 8000
TARGET_PROGRESS_INTERVAL = 25
TARGET_REQUIRED_BALANCED_MEMORY_MB = 40000
TARGET_SERVER_WAIT_TIMEOUT_SECONDS = 3600
TARGET_TEMPERATURE = 0.0
TARGET_TENSOR_PARALLEL_SIZE = None
TARGET_TIMEOUT_SECONDS = 180
TARGET_VLLM_EXTRA_ARGS = (
    "--dtype", "float16",
    "--tokenizer-mode", "mistral",
)

# Automatic local checker deployment
ALLOW_NON_LOCAL_CHECKER = False
AUTO_DEPLOY_CHECKER = True
CHECKER_API_KEY_ENV = "LOCAL_CHECKER_API_KEY"
CHECKER_CUDA_VISIBLE_DEVICES = None
CHECKER_GPU_MEMORY_UTILIZATION = 0.35
CHECKER_LIMIT_MM_PER_PROMPT = None
CHECKER_MAX_MODEL_LEN = 8192
CHECKER_MIN_FREE_MEMORY_MB = 20000
CHECKER_MM_ENCODER_TP_MODE = None
CHECKER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CHECKER_NUM_GPUS = 1
CHECKER_PORT = 8001
CHECKER_INTERNAL_ENDPOINT = f"http://localhost:{CHECKER_PORT}/v1/chat/completions"
CHECKER_SERVER_WAIT_TIMEOUT_SECONDS = 1800
CHECKER_TENSOR_PARALLEL_SIZE = 1
CHECKER_VLLM_EXTRA_ARGS: tuple[str, ...] = ()

# Dataset and target selection
BALANCE_HUMAN_GOLD_BY_CATEGORY = True
TARGETS = ("visual_factuality", "robustness", "refusal_behavior")
DEFAULT_CHUNK_SIZE = 1000
# Human gold pool size per target; Monte Carlo samples N_M from this pool each repeat.
HUMAN_GOLD_PER_TARGET = 500
HUMAN_GOLD_PER_TARGET_BY_DATASET = {
    "Medium Dataset": 50,
    "Large Dataset": 500,
}
N_M = 50
N_M_GRID = (25, 50, 100)
N_J = 1000
N_J_BY_DATASET = {
    "Medium Dataset": 50,
    "Large Dataset": 1000,
}

# Random seeds
HUMAN_GOLD_SAMPLE_SEED = 42
MONTE_CARLO_SEED = 20260608

# Certification calculation
B = 1000
ZETA = 0.05
ALPHA_MIN = 0.01
ALPHA_MAX = 0.80
ALPHA_STEP = 0.01
ERROR_EXPERIMENT_ALPHA = 0.25
ERROR_EXPERIMENT_PROGRESS_INTERVAL = 10
ERROR_EXPERIMENT_REPEATS = 1000
PPI_RIDGE_PENALTY = 0.01

# Checker generation
CHECKER_MAX_TOKENS = 256
CHECKER_PROGRESS_INTERVAL = 25
CHECKER_TEMPERATURE = 0.0
CHECKER_TIMEOUT_SECONDS = 180
OVERWRITE_JUDGE_LABELS = False


def current_config(run_name: str) -> dict[str, Any]:
    """Collect every user-tunable workflow parameter into the saved run configuration."""
    return {
        "run_name": run_name,
        "dataset_dir": DATASET_DIR,
        "dataset_paths": DATASET_PATHS,
        "runs_dir": RUNS_DIR,
        "annotation_guide_path": ANNOTATION_GUIDE_PATH,
        "benchmark_labels_dir": BENCHMARK_LABELS_DIR,
        "benchmark_required_datasets": BENCHMARK_REQUIRED_DATASETS,
        "run_target_vlm_first": RUN_TARGET_VLM_FIRST,
        "target_model_name": TARGET_MODEL_NAME,
        "target_port": TARGET_PORT,
        "target_categories": TARGET_CATEGORIES,
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "target_limit": TARGET_LIMIT,
        "target_max_tokens": TARGET_MAX_TOKENS,
        "target_temperature": TARGET_TEMPERATURE,
        "target_timeout_seconds": TARGET_TIMEOUT_SECONDS,
        "target_overwrite_responses": TARGET_OVERWRITE_RESPONSES,
        "target_progress_interval": TARGET_PROGRESS_INTERVAL,
        "target_cuda_visible_devices": TARGET_CUDA_VISIBLE_DEVICES,
        "target_max_gpus": TARGET_MAX_GPUS,
        "target_required_balanced_memory_mb": TARGET_REQUIRED_BALANCED_MEMORY_MB,
        "target_min_free_memory_mb": TARGET_MIN_FREE_MEMORY_MB,
        "target_tensor_parallel_size": TARGET_TENSOR_PARALLEL_SIZE,
        "target_max_model_len": TARGET_MAX_MODEL_LEN,
        "target_gpu_memory_utilization": TARGET_GPU_MEMORY_UTILIZATION,
        "target_limit_mm_per_prompt": TARGET_LIMIT_MM_PER_PROMPT,
        "target_mm_encoder_tp_mode": TARGET_MM_ENCODER_TP_MODE,
        "target_vllm_extra_args": TARGET_VLLM_EXTRA_ARGS,
        "target_server_wait_timeout_seconds": TARGET_SERVER_WAIT_TIMEOUT_SECONDS,
        "checker_model_name": CHECKER_MODEL_NAME,
        "checker_port": CHECKER_PORT,
        "checker_internal_endpoint": CHECKER_INTERNAL_ENDPOINT,
        "checker_api_key_env": CHECKER_API_KEY_ENV,
        "allow_non_local_checker": ALLOW_NON_LOCAL_CHECKER,
        "auto_deploy_checker": AUTO_DEPLOY_CHECKER,
        "checker_cuda_visible_devices": CHECKER_CUDA_VISIBLE_DEVICES,
        "checker_num_gpus": CHECKER_NUM_GPUS,
        "checker_min_free_memory_mb": CHECKER_MIN_FREE_MEMORY_MB,
        "checker_tensor_parallel_size": CHECKER_TENSOR_PARALLEL_SIZE,
        "checker_max_model_len": CHECKER_MAX_MODEL_LEN,
        "checker_gpu_memory_utilization": CHECKER_GPU_MEMORY_UTILIZATION,
        "checker_limit_mm_per_prompt": CHECKER_LIMIT_MM_PER_PROMPT,
        "checker_mm_encoder_tp_mode": CHECKER_MM_ENCODER_TP_MODE,
        "checker_vllm_extra_args": CHECKER_VLLM_EXTRA_ARGS,
        "checker_server_wait_timeout_seconds": CHECKER_SERVER_WAIT_TIMEOUT_SECONDS,
        "targets": TARGETS,
        "human_gold_per_target": HUMAN_GOLD_PER_TARGET,
        "human_gold_per_target_by_dataset": HUMAN_GOLD_PER_TARGET_BY_DATASET,
        "n_m": N_M,
        "n_j": N_J,
        "n_m_grid": N_M_GRID,
        "n_j_by_dataset": N_J_BY_DATASET,
        "human_gold_sample_seed": HUMAN_GOLD_SAMPLE_SEED,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "repeats": B,
        "zeta": ZETA,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "alpha_step": ALPHA_STEP,
        "error_experiment_alpha": ERROR_EXPERIMENT_ALPHA,
        "error_experiment_progress_interval": ERROR_EXPERIMENT_PROGRESS_INTERVAL,
        "error_experiment_repeats": ERROR_EXPERIMENT_REPEATS,
        "ppi_ridge_penalty": PPI_RIDGE_PENALTY,
        "checker_max_tokens": CHECKER_MAX_TOKENS,
        "checker_progress_interval": CHECKER_PROGRESS_INTERVAL,
        "checker_temperature": CHECKER_TEMPERATURE,
        "checker_timeout_seconds": CHECKER_TIMEOUT_SECONDS,
        "overwrite_judge_labels": OVERWRITE_JUDGE_LABELS,
        "balance_human_gold_by_category": BALANCE_HUMAN_GOLD_BY_CATEGORY,
    }


def configured_n_j(dataset_dir: str) -> int:
    return N_J_BY_DATASET.get(dataset_dir, N_J)


def available_records_per_target(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["target"] for record in records)
    return {target: counts[target] for target in TARGETS}


def effective_n_j(dataset_dir: str, evaluation_records: list[dict[str, Any]]) -> int:
    configured = configured_n_j(dataset_dir)
    available = available_records_per_target(evaluation_records)
    smallest_pool = min(available.values()) if available else 0
    return min(configured, smallest_pool)


def configured_human_gold_per_target(dataset_dir: str) -> int:
    return HUMAN_GOLD_PER_TARGET_BY_DATASET.get(dataset_dir, HUMAN_GOLD_PER_TARGET)


def target_internal_endpoint() -> str:
    return f"http://localhost:{TARGET_PORT}/v1/chat/completions"


def maybe_start_target_server(run_dir: Path) -> ManagedServer:
    """Start the target VLM server only when selected records still need model responses."""
    if not RUN_TARGET_VLM_FIRST or not target_responses_needed():
        return ManagedServer(process=None, log_path=None)
    cuda_visible_devices = TARGET_CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = auto_select_gpus_by_balanced_memory(
            max_gpu_count=TARGET_MAX_GPUS,
            required_balanced_memory_mb=TARGET_REQUIRED_BALANCED_MEMORY_MB,
            min_free_memory_mb=TARGET_MIN_FREE_MEMORY_MB,
        )
    tensor_parallel_size = TARGET_TENSOR_PARALLEL_SIZE or len(cuda_visible_devices.split(","))
    print(f"Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}, tensor_parallel_size={tensor_parallel_size}")
    return start_vllm_server(
        endpoint=target_internal_endpoint(),
        model=TARGET_MODEL_NAME,
        served_model_name=TARGET_MODEL_NAME,
        log_path=run_dir / "logs" / "target_vlm.log",
        cuda_visible_devices=cuda_visible_devices,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=TARGET_MAX_MODEL_LEN,
        gpu_memory_utilization=TARGET_GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt=TARGET_LIMIT_MM_PER_PROMPT,
        mm_encoder_tp_mode=TARGET_MM_ENCODER_TP_MODE,
        extra_args=TARGET_VLLM_EXTRA_ARGS,
        wait_timeout_seconds=TARGET_SERVER_WAIT_TIMEOUT_SECONDS,
        server_label="target VLM",
    )


def target_responses_needed() -> bool:
    if not RUN_TARGET_VLM_FIRST:
        return False
    if TARGET_OVERWRITE_RESPONSES:
        return True
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
    return any(not record.get("target_model_response") for record in records)


def maybe_start_checker_server(run_dir: Path) -> ManagedServer:
    """Start the local checker server used to label target responses as success or failure."""
    if not AUTO_DEPLOY_CHECKER:
        return ManagedServer(process=None, log_path=None)
    cuda_visible_devices = CHECKER_CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = auto_select_gpus(CHECKER_NUM_GPUS, CHECKER_MIN_FREE_MEMORY_MB)
    print(f"Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}, tensor_parallel_size={CHECKER_TENSOR_PARALLEL_SIZE}")
    return start_vllm_server(
        endpoint=CHECKER_INTERNAL_ENDPOINT,
        model=CHECKER_MODEL_NAME,
        served_model_name=CHECKER_MODEL_NAME,
        log_path=run_dir / "logs" / "checker_vllm.log",
        cuda_visible_devices=cuda_visible_devices,
        tensor_parallel_size=CHECKER_TENSOR_PARALLEL_SIZE,
        max_model_len=CHECKER_MAX_MODEL_LEN,
        gpu_memory_utilization=CHECKER_GPU_MEMORY_UTILIZATION,
        limit_mm_per_prompt=CHECKER_LIMIT_MM_PER_PROMPT,
        mm_encoder_tp_mode=CHECKER_MM_ENCODER_TP_MODE,
        extra_args=CHECKER_VLLM_EXTRA_ARGS,
        wait_timeout_seconds=CHECKER_SERVER_WAIT_TIMEOUT_SECONDS,
        server_label="checker VLM",
    )


def judge_records(
    records: list[dict[str, Any]],
    annotation_guide: str,
    stage: str,
    checkpoint_path: Path | None = None,
    persist_to_dataset: bool = False,
    completed_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Call the checker, optionally checkpoint each result, and persist current-run evaluation labels."""
    rows: list[dict[str, Any]] = []
    completed_rows = completed_rows or {}
    progress_interval = max(1, CHECKER_PROGRESS_INTERVAL)
    for index, record in enumerate(records, start=1):
        key = record_key(record)
        if key in completed_rows:
            rows.append(completed_rows[key])
            if persist_to_dataset:
                apply_checker_results({key: record}, [completed_rows[key]])
            continue
        reuse_existing_label = (
            not persist_to_dataset
            and record.get("judge_label") is not None
            and not OVERWRITE_JUDGE_LABELS
        )
        if reuse_existing_label:
            result = {
                "record_key": record_key(record),
                "stage": stage,
                "item_id": record["item_id"],
                "image_id": record["image_id"],
                "image_type": record["image_type"],
                "target": record["target"],
                "qa_json_path": record["qa_json_path"],
                "judge_label": record["judge_label"],
                "judge_failure_reason": record.get("failure_reason"),
                "checker_raw_response": None,
                "checker_error": None,
            }
            rows.append(result)
            if checkpoint_path is not None:
                append_judge_row(checkpoint_path, result)
            continue
        if index == 1 or index == len(records) or index % progress_interval == 0:
            print(f"Checker {stage} {index}/{len(records)}: {record['item_id']} [{record['target']}]")
        try:
            result = call_checker(
                endpoint=CHECKER_INTERNAL_ENDPOINT,
                model=CHECKER_MODEL_NAME,
                annotation_guide=annotation_guide,
                record=record,
                max_tokens=CHECKER_MAX_TOKENS,
                temperature=CHECKER_TEMPERATURE,
                timeout=CHECKER_TIMEOUT_SECONDS,
                api_key_env=CHECKER_API_KEY_ENV,
            )
        except CheckerError as exc:
            failure_row = {
                "record_key": record_key(record),
                "stage": stage,
                "item_id": record["item_id"],
                "image_id": record["image_id"],
                "image_type": record["image_type"],
                "target": record["target"],
                "qa_json_path": record["qa_json_path"],
                "judge_label": None,
                "judge_failure_reason": None,
                "checker_raw_response": exc.raw_response,
                "checker_error": str(exc),
            }
            if checkpoint_path is not None:
                append_judge_row(checkpoint_path, failure_row)
            raise CheckerError(
                f"Checker failed for {record['item_id']} during {stage}: {exc}",
                raw_response=exc.raw_response,
            ) from exc
        result["record_key"] = record_key(record)
        result["stage"] = stage
        rows.append(result)
        completed_rows[key] = result
        if persist_to_dataset:
            apply_checker_results({record_key(record): record}, [result])
            record["judge_label"] = result["judge_label"]
            record["failure_reason"] = result["judge_failure_reason"]
        if checkpoint_path is not None:
            append_judge_row(checkpoint_path, result)
    return rows


def print_human_label_instructions(human_gold_records: list[dict[str, Any]]) -> None:
    target_counts = Counter(record["target"] for record in human_gold_records)
    counts_text = ", ".join(f"{target}: {target_counts[target]}" for target in TARGETS)
    print("\nHuman calibration labels are needed before certification.")
    print(f"Rows to label: {len(human_gold_records)} ({counts_text}).")
    print("This is the small human calibration set, not a full-dataset benchmark.")
    print("Fill human_label in the generated CSV label sheet.")
    for record in human_gold_records[:5]:
        print(
            f"- {record['item_id']} | {record['target']} | "
            f"{record['model_input_image_path']} | {record['qa_json_abs_path']}"
        )
    print("\nUse human_label = 0 for success and human_label = 1 for failure.")
    print("For failures, fill human_failure_reason if possible.")


def archive_run_artifacts(
    run_dir: Path,
    reliability_rows: list[dict[str, Any]],
    monte_carlo_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
    dataset_paths: dict[str, str] | None = None,
) -> None:
    """Write every reusable run artifact without mutating the working dataset."""
    export_run_artifacts(
        project_root=PROJECT_ROOT,
        dataset_dir=DATASET_DIR,
        run_dir=run_dir,
        reliability_rows=reliability_rows,
        monte_carlo_rows=monte_carlo_rows,
        certificate_rows=certificate_rows,
        dataset_paths=dataset_paths or DATASET_PATHS,
    )


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_dirs() -> list[Path]:
    runs_path = PROJECT_ROOT / RUNS_DIR
    if not runs_path.exists():
        return []
    return sorted(
        [
            path
            for path in runs_path.iterdir()
            if path.is_dir()
            and (path / "run_config.json").exists()
            and ((path / "chunks" / "manifest.jsonl").exists() or (path / "manifest.jsonl").exists())
        ],
        key=lambda path: path.name,
    )


def choose_run_dir() -> Path:
    runs = run_dirs()
    if not runs:
        raise SystemExit("No manifest-based runs found. Start a new run first.")
    print("\nExisting runs:")
    for index, run_dir in enumerate(runs, start=1):
        status = load_run_status(run_dir).get("status", "unknown")
        print(f"{index}. {run_dir.name} [{status}]")
    answer = input(f"Choose run [1-{len(runs)}]: ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(runs):
        raise SystemExit("Invalid run choice.")
    return runs[int(answer) - 1]


def start_new_run() -> Path:
    """Create a manifest-first run and save the fixed human-gold sample for final certification."""
    global DATASET_DIR

    DATASET_DIR = select_dataset_dir(PROJECT_ROOT, DATASET_DIR, DATASET_PATHS)
    reset_summary = reset_dataset_fields(PROJECT_ROOT, DATASET_DIR, DATASET_PATHS)
    print(
        f"Reset previous mutable labels/responses in {DATASET_DIR}: "
        f"{reset_summary['items_changed']} items across {reset_summary['files_changed']} QA files."
    )
    run_name = RUN_NAME or default_run_name(TARGET_MODEL_NAME)
    chunk_size = DEFAULT_CHUNK_SIZE

    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
    manifest_rows = group_records_for_chunks(records, chunk_size)
    human_gold_records = sample_human_gold_pool(
        records=records,
        targets=TARGETS,
        per_target=configured_human_gold_per_target(DATASET_DIR),
        seed=HUMAN_GOLD_SAMPLE_SEED,
        balance_by_image_type=BALANCE_HUMAN_GOLD_BY_CATEGORY,
    )

    run_dir = create_run_dir(PROJECT_ROOT, RUNS_DIR, run_name)
    config = current_config(run_dir.name)
    config["chunk_size"] = chunk_size
    write_json(run_dir / "run_config.json", config)
    write_manifest(run_dir, manifest_rows)
    write_human_gold_keys(run_dir, [record_key(record) for record in human_gold_records])
    write_run_status(
        run_dir,
        "created",
        dataset_dir=DATASET_DIR,
        total_records=len(manifest_rows),
        total_chunks=len(chunk_indexes(manifest_rows)),
        chunk_size=chunk_size,
    )
    print(f"\nCreated run: {run_dir}")
    print(f"Dataset: {DATASET_DIR}; chunk size: {chunk_size}")
    print(f"Manifest: {len(manifest_rows)} QA items in {len(chunk_indexes(manifest_rows))} chunks.")
    return run_dir


def target_args_for_config(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=config["dataset_dir"],
        port=config["target_port"],
        model=config["target_model_name"],
        api_key_env="LOCAL_VLM_API_KEY",
        categories=config["target_categories"],
        targets=tuple(config["targets"]),
        limit=None,
        max_tokens=config["target_max_tokens"],
        temperature=config["target_temperature"],
        timeout=config["target_timeout_seconds"],
        overwrite=config["target_overwrite_responses"],
        progress_interval=config.get("target_progress_interval", TARGET_PROGRESS_INTERVAL),
        dry_run=False,
    )


def materialize_target_rows(records: list[dict[str, Any]], rows_by_key: dict[str, dict[str, Any]]) -> None:
    for record in records:
        row = rows_by_key.get(record_key(record))
        if row is not None:
            update_item_fields(record, {"target_model_response": row["target_model_response"]})


def materialize_judge_rows(records: list[dict[str, Any]], rows_by_key: dict[str, dict[str, Any]]) -> None:
    for record in records:
        row = rows_by_key.get(record_key(record))
        if row is not None:
            update_item_fields(
                record,
                {
                    "judge_label": row["judge_label"],
                    "failure_reason": row["judge_failure_reason"],
                },
            )


def format_chunk_ranges(chunks: list[int]) -> str:
    """Format chunk indexes as compact ranges for terminal status messages."""
    if not chunks:
        return "none"
    ranges: list[str] = []
    start = previous = chunks[0]
    for chunk_index in chunks[1:]:
        if chunk_index == previous + 1:
            previous = chunk_index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = chunk_index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def chunk_status_by_index(run_dir: Path, manifest_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    summary = progress_summary(run_dir, manifest_rows)
    return {int(row["chunk_index"]): row for row in summary["chunks"]}


def next_resume_row(run_dir: Path, manifest_rows: list[dict[str, Any]], chunks: list[int]) -> dict[str, Any] | None:
    """Find the first selected chunk that still needs target or checker work."""
    by_index = chunk_status_by_index(run_dir, manifest_rows)
    for chunk_index in chunks:
        row = by_index[chunk_index]
        if not row["complete"]:
            return row
    return None


def selected_chunk_stage_ranges(
    run_dir: Path,
    manifest_rows: list[dict[str, Any]],
    chunks: list[int],
) -> dict[str, list[int]]:
    """Group selected chunks by completed, target-needed, and checker-needed status."""
    by_index = chunk_status_by_index(run_dir, manifest_rows)
    ranges = {"complete": [], "target VLM": [], "checker VLM": []}
    for chunk_index in chunks:
        row = by_index[chunk_index]
        ranges[row["stage"]].append(chunk_index)
    return ranges


def print_selected_chunk_status(run_dir: Path, manifest_rows: list[dict[str, Any]], chunks: list[int]) -> None:
    """Print the selected chunk range and the first checkpoint that will resume."""
    stage_ranges = selected_chunk_stage_ranges(run_dir, manifest_rows, chunks)
    print(f"Selected chunks: {format_chunk_ranges(chunks)}")
    print(f"Completed selected chunks: {len(stage_ranges['complete'])}/{len(chunks)} ({format_chunk_ranges(stage_ranges['complete'])})")
    print(f"Pending target VLM chunks: {format_chunk_ranges(stage_ranges['target VLM'])}")
    print(f"Pending checker VLM chunks: {format_chunk_ranges(stage_ranges['checker VLM'])}")
    resume = next_resume_row(run_dir, manifest_rows, chunks)
    if resume is None:
        print("Continue point: all selected chunks are complete.")
        return
    print(
        f"Continue point: {resume['chunk_name']} at {resume['stage']} "
        f"(target {resume['target_done']}/{resume['total']}, "
        f"checker {resume['judge_done']}/{resume['total']})."
    )


def print_next_chunk_hint(run_dir: Path, manifest_rows: list[dict[str, Any]], chunks: list[int] | None = None) -> None:
    """Print the next unfinished chunk and compact unfinished chunk ranges."""
    summary = progress_summary(run_dir, manifest_rows)
    selected_chunks = chunks or [int(row["chunk_index"]) for row in summary["chunks"]]
    unfinished = [int(row["chunk_index"]) for row in summary["chunks"] if not row["complete"]]
    resume = next_resume_row(run_dir, manifest_rows, selected_chunks)
    if resume is None:
        print("Next unfinished chunk: none.")
    else:
        print(
            f"Next unfinished chunk: {resume['chunk_name']} at {resume['stage']} "
            f"(target {resume['target_done']}/{resume['total']}, "
            f"checker {resume['judge_done']}/{resume['total']})."
        )
    print(f"Unfinished chunks: {format_chunk_ranges(unfinished)}")


def chunks_needing_target(records: list[dict[str, Any]], run_dir: Path, manifest_rows: list[dict[str, Any]], chunks: list[int]) -> list[int]:
    """Return selected chunks that still need target VLM responses."""
    needed: list[int] = []
    for chunk_index in chunks:
        completed = successful_target_rows(target_results_path(run_dir, chunk_index))
        chunk_records = records_for_chunks(records, manifest_rows, [chunk_index])
        if any(record_key(record) not in completed and not record.get("target_model_response") for record in chunk_records):
            needed.append(chunk_index)
    return needed


def chunks_needing_checker(run_dir: Path, manifest_rows: list[dict[str, Any]], chunks: list[int]) -> list[int]:
    """Return selected chunks that still need checker labels."""
    needed: list[int] = []
    for chunk_index in chunks:
        completed = successful_judge_rows(judge_results_path(run_dir, chunk_index))
        keys = {
            row["record_key"]
            for row in manifest_rows
            if int(row["chunk_index"]) == chunk_index
        }
        if not keys <= set(completed):
            needed.append(chunk_index)
    return needed


def run_chunks(run_dir: Path, chunks: list[int]) -> None:
    """Run target responses and checker labels for selected chunks with per-item checkpointing."""
    global DATASET_DIR

    config = load_run_config(run_dir)
    DATASET_DIR = config["dataset_dir"]
    if load_run_status(run_dir).get("status") == "completed":
        raise SystemExit("This run is already completed. Start a new run for a new experiment.")
    manifest_rows = load_manifest(run_dir)
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
    selected_records = records_for_chunks(records, manifest_rows, chunks)
    if not selected_records:
        raise SystemExit("Selected chunks contain no records.")

    print_selected_chunk_status(run_dir, manifest_rows, chunks)
    write_run_status(run_dir, "running", active_chunks=chunks, dataset_dir=DATASET_DIR)
    save_local_chunks(run_dir, chunks)
    require_local_endpoint(CHECKER_INTERNAL_ENDPOINT, ALLOW_NON_LOCAL_CHECKER)

    target_server = ManagedServer(process=None, log_path=None)
    checker_server = ManagedServer(process=None, log_path=None)
    try:
        target_chunks = chunks_needing_target(records, run_dir, manifest_rows, chunks)
        if RUN_TARGET_VLM_FIRST and target_chunks:
            target_server = maybe_start_target_server(run_dir)
        for position, chunk_index in enumerate(target_chunks, start=1):
            chunk_records = records_for_chunks(records, manifest_rows, [chunk_index])
            checkpoint_path = target_results_path(run_dir, chunk_index)
            completed = successful_target_rows(checkpoint_path)
            status = chunk_status_by_index(run_dir, manifest_rows)[chunk_index]
            print(
                f"Target VLM chunk {position}/{len(target_chunks)}: {status['chunk_name']} "
                f"({status['target_done']}/{status['total']} saved)."
            )
            run_target_vlm_records(
                target_args_for_config(config),
                chunk_records,
                completed,
                lambda row, path=checkpoint_path: append_target_row(path, row),
            )

        if target_server.started_by_script:
            target_server.stop()
            target_server = ManagedServer(process=None, log_path=None)

        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
        selected_records = records_for_chunks(records, manifest_rows, chunks)
        validate_target_responses(selected_records)

        checker_chunks = chunks_needing_checker(run_dir, manifest_rows, chunks)
        if checker_chunks:
            checker_server = maybe_start_checker_server(run_dir)
        annotation_guide = load_annotation_guide(PROJECT_ROOT / ANNOTATION_GUIDE_PATH)
        for position, chunk_index in enumerate(checker_chunks, start=1):
            chunk_records = records_for_chunks(records, manifest_rows, [chunk_index])
            checkpoint_path = judge_results_path(run_dir, chunk_index)
            completed = successful_judge_rows(checkpoint_path)
            status = chunk_status_by_index(run_dir, manifest_rows)[chunk_index]
            print(
                f"Checker VLM chunk {position}/{len(checker_chunks)}: {status['chunk_name']} "
                f"({status['judge_done']}/{status['total']} labelled)."
            )
            judge_records(
                chunk_records,
                annotation_guide,
                stage=f"chunk-{chunk_index:03d}",
                checkpoint_path=checkpoint_path,
                persist_to_dataset=True,
                completed_rows=completed,
            )
        summary = progress_summary(run_dir, manifest_rows)
        status = "chunks_completed" if summary["judge_done"] == summary["total_records"] else "running"
        write_run_status(run_dir, status, active_chunks=chunks, dataset_dir=DATASET_DIR)
        show_run_progress(run_dir)
    except Exception:
        error_path = run_dir / "logs" / "workflow_error.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Workflow failed. Full traceback saved to {error_path}", file=sys.stderr, flush=True)
        for log_name in ("target_vlm.log", "checker_vllm.log"):
            log_path = run_dir / "logs" / log_name
            if log_path.exists():
                print(f"Related vLLM log: {log_path}", file=sys.stderr, flush=True)
        raise
    finally:
        if checker_server.started_by_script:
            checker_server.stop()
        if target_server.started_by_script:
            target_server.stop()


def choose_chunks(run_dir: Path) -> list[int]:
    manifest_rows = load_manifest(run_dir)
    available = chunk_indexes(manifest_rows)
    summary = progress_summary(run_dir, manifest_rows)
    first_chunk = available[0]
    last_chunk = available[-1]
    print(
        f"\nRun has {summary['total_chunks']} chunks, "
        f"{summary['completed_chunks']} completed, {summary['judge_done']}/{summary['total_records']} judged."
    )
    print_next_chunk_hint(run_dir, manifest_rows)
    print(f"Chunk numbers: {first_chunk}-{last_chunk}")
    answer = input("Run which chunks? Example: 0 or 0-9: ").strip()
    return parse_chunk_selection(answer, available)


def continue_local_chunks() -> None:
    run_dir = choose_run_dir()
    chunks = load_local_chunks(run_dir)
    if not chunks:
        raise SystemExit("This machine has no saved local chunk assignment. Choose selected chunks first.")
    print("\nContinuing this machine's saved chunk assignment.")
    run_chunks(run_dir, chunks)


def run_selected_chunks() -> None:
    run_dir = choose_run_dir()
    chunks = choose_chunks(run_dir)
    run_chunks(run_dir, chunks)


def show_run_progress(run_dir: Path | None = None) -> None:
    run_dir = run_dir or choose_run_dir()
    manifest_rows = load_manifest(run_dir)
    summary = progress_summary(run_dir, manifest_rows)
    run_status = load_run_status(run_dir)
    status = run_status.get("status", "unknown")
    active_chunks = [int(index) for index in run_status.get("active_chunks", [])]
    print(f"\nRun: {run_dir.name} [{status}]")
    print(f"Chunks: {summary['completed_chunks']}/{summary['total_chunks']} completed")
    print(f"VLM responses saved: {summary['target_done']}/{summary['total_records']}")
    print(f"Checker labels completed: {summary['judge_done']}/{summary['total_records']} ({summary['completion_percent']}%)")
    if active_chunks:
        print(f"Active chunks on last run: {format_chunk_ranges(active_chunks)}")
    print_next_chunk_hint(run_dir, manifest_rows, active_chunks or None)
    print("Chunk checkpoints are temporary and are cleaned after final results are built.")
    if summary["judge_done"] == summary["total_records"]:
        print("Next step: choose 4. Build final results from completed chunks.")
    elif summary["target_done"] == summary["total_records"]:
        print("Next step: checker labels are not finished; continue this run.")
    else:
        print("Next step: VLM responses are not finished; continue this run.")


def snapshot_dataset_paths(run_dir: Path) -> dict[str, str] | None:
    snapshot_dir = run_dir / "dataset_snapshot"
    if not snapshot_dir.exists():
        return None
    return {**DATASET_PATHS, DATASET_DIR: str(snapshot_dir)}


def load_run_records(run_dir: Path, prefer_snapshot: bool = False) -> list[dict[str, Any]]:
    """Load records from the archived snapshot when available, otherwise from the working dataset."""
    source_paths = snapshot_dataset_paths(run_dir) if prefer_snapshot else None
    return load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, source_paths or DATASET_PATHS)


def ensure_benchmark_label_csv(
    dataset_dir: str,
    target_model_name: str,
    dataset_paths: dict[str, str] | None = None,
) -> Path:
    """Create or refresh the response-specific benchmark CSV while preserving valid manual labels."""
    benchmark_path = default_benchmark_path(PROJECT_ROOT, dataset_dir, target_model_name)
    fresh_rows = load_benchmark_rows(PROJECT_ROOT, dataset_dir, target_model_name, dataset_paths or DATASET_PATHS)
    if benchmark_path.exists():
        fresh_rows = merge_existing_benchmark_labels(fresh_rows, read_benchmark_csv(benchmark_path))
    write_benchmark_csv(benchmark_path, fresh_rows)
    return benchmark_path


def build_final_results() -> None:
    """Validate completed chunk files, materialize final labels, and calculate certificate artifacts."""
    global DATASET_DIR

    run_dir = choose_run_dir()
    config = load_run_config(run_dir)
    DATASET_DIR = config["dataset_dir"]
    target_model_name = config.get("target_model_name", TARGET_MODEL_NAME)
    manifest_rows = load_manifest(run_dir)
    run_status = load_run_status(run_dir)
    source_dataset_paths = snapshot_dataset_paths(run_dir) if run_status.get("status") == "completed" else None

    if source_dataset_paths is None:
        target_rows = collect_chunk_results(
            run_dir,
            manifest_rows,
            "target_responses.jsonl",
            value_field="target_model_response",
            error_field="target_error",
        )
        judge_rows = collect_chunk_results(
            run_dir,
            manifest_rows,
            "judge_labels.jsonl",
            value_field="judge_label",
            error_field="checker_error",
        )
        missing_target = missing_keys(manifest_rows, target_rows)
        missing_judge = missing_keys(manifest_rows, judge_rows)
        if missing_target or missing_judge:
            print(f"Missing target responses: {len(missing_target)}")
            print(f"Missing checker labels: {len(missing_judge)}")
            write_run_status(run_dir, "incomplete", dataset_dir=DATASET_DIR)
            return

        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
        materialize_target_rows(records, target_rows)
        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
        materialize_judge_rows(records, judge_rows)
        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, DATASET_PATHS)
        source_dataset_paths = DATASET_PATHS
    else:
        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, source_dataset_paths)

    benchmark_path = ensure_benchmark_label_csv(DATASET_DIR, target_model_name, source_dataset_paths)
    use_run_human_sheet = True
    if benchmark_path.exists():
        try:
            human_labelled_records = apply_benchmark_labels(records, read_benchmark_csv(benchmark_path))
        except ValueError as exc:
            if DATASET_DIR in BENCHMARK_REQUIRED_DATASETS:
                print(f"Benchmark labels are not ready: {exc}")
                print(f"Benchmark label file: {benchmark_path}")
                write_run_status(run_dir, "waiting_for_benchmark_labels", dataset_dir=DATASET_DIR)
                return
            print(f"Full benchmark labels are not complete yet: {benchmark_path}")
            print("Certification will use the small human calibration CSV for now.")
            print("For Type I/II paper experiments later, complete the benchmark CSV and choose 4 again.")
        else:
            human_gold_keys = load_human_gold_keys(run_dir)
            human_gold_records = [record for record in human_labelled_records if record_key(record) in human_gold_keys]
            evaluation_records = [record for record in human_labelled_records if record_key(record) not in human_gold_keys]
            use_run_human_sheet = False
            print(f"Using benchmark human labels: {benchmark_path}")

    if use_run_human_sheet:
        human_gold_keys = load_human_gold_keys(run_dir)
        human_gold_records = [record for record in records if record_key(record) in human_gold_keys]
        human_label_sheet_path = run_dir / "human_label_tasks.csv"
        if not human_label_sheet_path.exists():
            write_human_label_sheet(human_label_sheet_path, human_gold_records)
            print_human_label_instructions(human_gold_records)
            print(f"\nHuman label sheet created: {human_label_sheet_path}")
            print("Fill this CSV, then choose 4 again to build final results.")
            write_run_status(run_dir, "waiting_for_human_labels", dataset_dir=DATASET_DIR)
            return

        try:
            human_gold_records = apply_human_label_sheet(human_label_sheet_path, human_gold_records)
        except ValueError as exc:
            print(f"Human labels are not ready: {exc}")
            print(f"Human label sheet: {human_label_sheet_path}")
            write_run_status(run_dir, "waiting_for_human_labels", dataset_dir=DATASET_DIR)
            return
        gold_keys = {record_key(record) for record in human_gold_records}
        evaluation_records = [record for record in records if record_key(record) not in gold_keys]

    reliability_rows = [estimate_reliability(human_gold_records, target) for target in TARGETS]
    unreliable = [row for row in reliability_rows if not row["reliable"]]
    if unreliable:
        print("\nChecker unreliable. Certification will be summarized without Monte Carlo repeats.")
        for row in unreliable:
            print(f"- {row['target']}: {row['unreliable_reason']}")
        monte_carlo_rows: list[dict[str, Any]] = []
    else:
        n_j_for_run = effective_n_j(DATASET_DIR, evaluation_records)
        if n_j_for_run <= 0:
            raise ValueError("No evaluation records are available after selecting the human calibration set.")
        if n_j_for_run < configured_n_j(DATASET_DIR):
            print(f"Using N_J={n_j_for_run} because the evaluation pool is smaller than configured N_J={configured_n_j(DATASET_DIR)}.")
        print(f"Running {B} Monte Carlo repeats per target with N_J={n_j_for_run}...", flush=True)
        monte_carlo_rows = run_monte_carlo(
            human_gold_records=human_gold_records,
            evaluation_records=evaluation_records,
            targets=TARGETS,
            n_m=N_M,
            n_j=n_j_for_run,
            repeats=B,
            zeta=ZETA,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            alpha_step=ALPHA_STEP,
            seed=MONTE_CARLO_SEED,
        )
    certificate_rows = summarize_certificates(monte_carlo_rows, reliability_rows, TARGETS)
    archive_run_artifacts(
        run_dir=run_dir,
        reliability_rows=reliability_rows,
        monte_carlo_rows=monte_carlo_rows,
        certificate_rows=certificate_rows,
        dataset_paths=source_dataset_paths,
    )
    if use_run_human_sheet:
        print(f"Paper-style Type I/II experiments skipped: full benchmark labels are not complete.")
        print(f"To run them later, complete this CSV and choose 4 again: {benchmark_path}")
    else:
        try:
            run_paper_style_experiments(run_dir)
        except ValueError as exc:
            print(f"Paper-style experiments skipped: {exc}")
    deleted_checkpoints = delete_chunk_checkpoints(run_dir)
    if deleted_checkpoints:
        print(f"Cleaned {deleted_checkpoints} chunk checkpoint file(s).")
    write_run_status(run_dir, "completed", dataset_dir=DATASET_DIR)
    print(f"\nRun complete. Artifacts saved under {run_dir}")


def run_paper_style_experiments(run_dir: Path | None = None) -> None:
    """Run paper-style Direct/Noisy/Oracle/PPI Type I/II experiments from benchmark labels."""
    global DATASET_DIR

    if run_dir is None:
        run_dir = choose_run_dir()
    config = load_run_config(run_dir)
    DATASET_DIR = config["dataset_dir"]
    target_model_name = config.get("target_model_name", TARGET_MODEL_NAME)
    benchmark_path = default_benchmark_path(PROJECT_ROOT, DATASET_DIR, target_model_name)
    if not benchmark_path.exists():
        raise ValueError(f"benchmark labels not found: {benchmark_path}")

    source_dataset_paths = snapshot_dataset_paths(run_dir) or DATASET_PATHS
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS, source_dataset_paths)
    missing_judge_labels = [record_key(record) for record in records if record.get("judge_label") is None]
    if missing_judge_labels:
        examples = ", ".join(missing_judge_labels[:10])
        raise ValueError(f"{len(missing_judge_labels)} judge labels are missing from the saved dataset. Examples: {examples}")
    records = apply_benchmark_labels(records, read_benchmark_csv(benchmark_path))
    output_dir = run_dir / "error_experiments"
    n_j_for_experiments = effective_n_j(DATASET_DIR, records)
    type_error_rows = run_type_error_experiment(
        records=records,
        targets=TARGETS,
        alpha_values=alpha_grid(ALPHA_MIN, ALPHA_MAX, ALPHA_STEP),
        n_m_values=N_M_GRID,
        n_j=n_j_for_experiments,
        repeats=ERROR_EXPERIMENT_REPEATS,
        zeta=ZETA,
        seed=MONTE_CARLO_SEED,
        ridge_penalty=PPI_RIDGE_PENALTY,
        checkpoint_path=output_dir / "type_i_type_ii_checkpoint.csv",
        progress_interval=ERROR_EXPERIMENT_PROGRESS_INTERVAL,
    )
    write_error_experiment_artifacts(
        output_dir=output_dir,
        type_error_rows=type_error_rows,
        main_n_m=N_M,
        main_n_j=n_j_for_experiments,
    )
    print(f"Paper-style error experiments saved under {output_dir}")


def main() -> None:
    """Ask which run action to take, then execute the selected resumable workflow step."""
    print("\nVLM certification workflow")
    print("1. Continue local unfinished chunks")
    print("2. Start / run selected chunks from manifest")
    print("3. Build final results from completed chunks")
    print("4. Start a new run")
    choice = input("Choose action [1-4]: ").strip()
    if choice == "1":
        continue_local_chunks()
    elif choice == "2":
        run_selected_chunks()
    elif choice == "3":
        build_final_results()
    elif choice == "4":
        run_dir = start_new_run()
        answer = input("Run selected chunks now? [y/N]: ").strip().lower()
        if answer == "y":
            chunks = choose_chunks(run_dir)
            run_chunks(run_dir, chunks)
    else:
        raise SystemExit("Invalid action.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
