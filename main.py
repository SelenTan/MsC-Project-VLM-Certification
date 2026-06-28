#!/usr/bin/env python3
"""One-command workflow for AI checker validation and certification calculation."""

from __future__ import annotations

import sys
import traceback
import json
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
    write_csv,
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
    run_calibration_stability,
    run_type_error_experiment,
    write_error_experiment_artifacts,
)
from model_server import ManagedServer, auto_select_gpus, auto_select_gpus_by_balanced_memory, start_vllm_server
from workflow_state import (
    append_judge_row,
    append_target_row,
    chunk_indexes,
    collect_chunk_results,
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
DATASET_DIR = "dataset"
RUNS_DIR = "runs"
ANNOTATION_GUIDE_PATH = "ANNOTATION_GUIDE.md"
BENCHMARK_LABELS_DIR = "benchmark_labels"
BENCHMARK_REQUIRED_DATASETS = ("Large Dataset",)
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
CHECKER_GPU_MEMORY_UTILIZATION = 0.7
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
N_M = 50
N_M_GRID = (25, 50, 100)
N_J = 1000
N_J_BY_DATASET = {
    "Medium Dataset": 500,
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
CHECKER_MAX_TOKENS = 180
CHECKER_PROGRESS_INTERVAL = 25
CHECKER_TEMPERATURE = 0.0
CHECKER_TIMEOUT_SECONDS = 180
OVERWRITE_JUDGE_LABELS = False


def current_config(run_name: str) -> dict[str, Any]:
    """Collect every user-tunable workflow parameter into the saved run configuration."""
    return {
        "run_name": run_name,
        "dataset_dir": DATASET_DIR,
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
    print("Starting local target VLM server if needed...")
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
    )


def target_responses_needed() -> bool:
    if not RUN_TARGET_VLM_FIRST:
        return False
    if TARGET_OVERWRITE_RESPONSES:
        return True
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    return any(not record.get("target_model_response") for record in records)


def maybe_start_checker_server(run_dir: Path) -> ManagedServer:
    """Start the local checker server used to label target responses as success or failure."""
    if not AUTO_DEPLOY_CHECKER:
        return ManagedServer(process=None, log_path=None)
    cuda_visible_devices = CHECKER_CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = auto_select_gpus(CHECKER_NUM_GPUS, CHECKER_MIN_FREE_MEMORY_MB)
    print("Starting local checker server if needed...")
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
    print("\nSelected human gold pool. Fill human_label in the generated CSV label sheet.")
    for record in human_gold_records:
        print(
            f"- {record['item_id']} | {record['target']} | "
            f"{record['model_input_image_path']} | {PROJECT_ROOT / record['qa_json_path']}"
        )
    print("\nUse human_label = 0 for success and human_label = 1 for failure.")
    print("For failures, fill human_failure_reason if possible.")


def archive_run_artifacts(
    run_dir: Path,
    reliability_rows: list[dict[str, Any]],
    monte_carlo_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
) -> None:
    """Write every reusable run artifact without mutating the working dataset."""
    export_run_artifacts(
        project_root=PROJECT_ROOT,
        dataset_dir=DATASET_DIR,
        run_dir=run_dir,
        reliability_rows=reliability_rows,
        monte_carlo_rows=monte_carlo_rows,
        certificate_rows=certificate_rows,
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
            and (path / "manifest.jsonl").exists()
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


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{label}{suffix}: ").strip()
    return answer or (default or "")


def prompt_int(label: str, default: int) -> int:
    answer = input(f"{label} [{default}]: ").strip()
    return int(answer) if answer else default


def start_new_run() -> Path:
    """Create a manifest-first run and save the fixed human-gold sample for final certification."""
    global DATASET_DIR

    DATASET_DIR = select_dataset_dir(PROJECT_ROOT, DATASET_DIR)
    run_name = prompt_text("Run name", RUN_NAME or default_run_name(TARGET_MODEL_NAME))
    chunk_size = prompt_int("Chunk size in QA items", DEFAULT_CHUNK_SIZE)
    run_dir = create_run_dir(PROJECT_ROOT, RUNS_DIR, run_name)

    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    manifest_rows = group_records_for_chunks(records, chunk_size)
    human_gold_records = sample_human_gold_pool(
        records=records,
        targets=TARGETS,
        per_target=HUMAN_GOLD_PER_TARGET,
        seed=HUMAN_GOLD_SAMPLE_SEED,
        balance_by_image_type=BALANCE_HUMAN_GOLD_BY_CATEGORY,
    )

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


def chunk_needs_target_server(records: list[dict[str, Any]], run_dir: Path, chunks: list[int]) -> bool:
    for chunk_index in chunks:
        completed = successful_target_rows(target_results_path(run_dir, chunk_index))
        for record in records_for_chunks(records, load_manifest(run_dir), [chunk_index]):
            if record_key(record) not in completed and not record.get("target_model_response"):
                return True
    return False


def chunk_needs_checker_server(run_dir: Path, manifest_rows: list[dict[str, Any]], chunks: list[int]) -> bool:
    for chunk_index in chunks:
        completed = successful_judge_rows(judge_results_path(run_dir, chunk_index))
        keys = {
            row["record_key"]
            for row in manifest_rows
            if int(row["chunk_index"]) == chunk_index
        }
        if not keys <= set(completed):
            return True
    return False


def run_chunks(run_dir: Path, chunks: list[int]) -> None:
    """Run target responses and checker labels for selected chunks with per-item checkpointing."""
    global DATASET_DIR

    config = load_run_config(run_dir)
    DATASET_DIR = config["dataset_dir"]
    if load_run_status(run_dir).get("status") == "completed":
        raise SystemExit("This run is already completed. Start a new run for a new experiment.")
    manifest_rows = load_manifest(run_dir)
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    selected_records = records_for_chunks(records, manifest_rows, chunks)
    if not selected_records:
        raise SystemExit("Selected chunks contain no records.")

    write_run_status(run_dir, "running", active_chunks=chunks, dataset_dir=DATASET_DIR)
    save_local_chunks(run_dir, chunks)
    require_local_endpoint(CHECKER_INTERNAL_ENDPOINT, ALLOW_NON_LOCAL_CHECKER)

    target_server = ManagedServer(process=None, log_path=None)
    checker_server = ManagedServer(process=None, log_path=None)
    try:
        if RUN_TARGET_VLM_FIRST and chunk_needs_target_server(records, run_dir, chunks):
            target_server = maybe_start_target_server(run_dir)
        for chunk_index in chunks:
            chunk_records = records_for_chunks(records, manifest_rows, [chunk_index])
            checkpoint_path = target_results_path(run_dir, chunk_index)
            completed = successful_target_rows(checkpoint_path)
            run_target_vlm_records(
                target_args_for_config(config),
                chunk_records,
                completed,
                lambda row, path=checkpoint_path: append_target_row(path, row),
            )

        if target_server.started_by_script:
            target_server.stop()
            target_server = ManagedServer(process=None, log_path=None)

        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
        selected_records = records_for_chunks(records, manifest_rows, chunks)
        validate_target_responses(selected_records)

        if chunk_needs_checker_server(run_dir, manifest_rows, chunks):
            checker_server = maybe_start_checker_server(run_dir)
        annotation_guide = load_annotation_guide(PROJECT_ROOT / ANNOTATION_GUIDE_PATH)
        for chunk_index in chunks:
            chunk_records = records_for_chunks(records, manifest_rows, [chunk_index])
            checkpoint_path = judge_results_path(run_dir, chunk_index)
            completed = successful_judge_rows(checkpoint_path)
            judge_records(
                chunk_records,
                annotation_guide,
                stage=f"chunk-{chunk_index:03d}",
                checkpoint_path=checkpoint_path,
                persist_to_dataset=True,
                completed_rows=completed,
            )
        write_run_status(run_dir, "running", active_chunks=chunks, dataset_dir=DATASET_DIR)
        show_run_progress(run_dir)
    except Exception:
        error_path = run_dir / "logs" / "workflow_error.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Workflow failed. Full traceback saved to {error_path}", file=sys.stderr, flush=True)
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
    print(
        f"\nRun has {summary['total_chunks']} chunks, "
        f"{summary['completed_chunks']} completed, {summary['judge_done']}/{summary['total_records']} judged."
    )
    answer = input("Chunks to run, e.g. 0,3-5: ").strip()
    return parse_chunk_selection(answer, available)


def continue_local_chunks() -> None:
    run_dir = choose_run_dir()
    chunks = load_local_chunks(run_dir)
    if not chunks:
        raise SystemExit("This machine has no saved local chunk assignment. Choose selected chunks first.")
    print(f"Continuing local chunks: {', '.join(str(index) for index in chunks)}")
    run_chunks(run_dir, chunks)


def run_selected_chunks() -> None:
    run_dir = choose_run_dir()
    chunks = choose_chunks(run_dir)
    run_chunks(run_dir, chunks)


def show_run_progress(run_dir: Path | None = None) -> None:
    run_dir = run_dir or choose_run_dir()
    manifest_rows = load_manifest(run_dir)
    summary = progress_summary(run_dir, manifest_rows)
    status = load_run_status(run_dir).get("status", "unknown")
    print(f"\nRun: {run_dir.name} [{status}]")
    print(f"Chunks: {summary['completed_chunks']}/{summary['total_chunks']} completed")
    print(f"Target responses: {summary['target_done']}/{summary['total_records']}")
    print(f"Checker labels: {summary['judge_done']}/{summary['total_records']} ({summary['completion_percent']}%)")


def ensure_benchmark_label_csv(dataset_dir: str, target_model_name: str) -> Path:
    """Create or refresh the response-specific benchmark CSV while preserving valid manual labels."""
    benchmark_path = default_benchmark_path(PROJECT_ROOT, dataset_dir, target_model_name)
    fresh_rows = load_benchmark_rows(PROJECT_ROOT, dataset_dir, target_model_name)
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

    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    materialize_target_rows(records, target_rows)
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    materialize_judge_rows(records, judge_rows)
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)

    benchmark_path = ensure_benchmark_label_csv(DATASET_DIR, target_model_name)
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
            print(f"Benchmark labels are incomplete, falling back to run human sheet: {benchmark_path}")
        else:
            human_gold_records = human_labelled_records
            evaluation_records = human_labelled_records
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
            print("Fill it, then run this menu item again to build final results.")
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

    gold_checker_rows = [judge_rows[record_key(record)] for record in human_gold_records]
    evaluation_checker_rows = [judge_rows[record_key(record)] for record in evaluation_records]
    write_csv(run_dir / "judge_labels_gold.csv", gold_checker_rows)
    write_csv(run_dir / "judge_labels_evaluation.csv", evaluation_checker_rows)

    reliability_rows = [estimate_reliability(human_gold_records, target) for target in TARGETS]
    unreliable = [row for row in reliability_rows if not row["reliable"]]
    if unreliable:
        print("\nChecker unreliable. Certification will be summarized without Monte Carlo repeats.")
        for row in unreliable:
            print(f"- {row['target']}: {row['unreliable_reason']}")
        monte_carlo_rows: list[dict[str, Any]] = []
    else:
        print(f"Running {B} Monte Carlo repeats per target...", flush=True)
        monte_carlo_rows = run_monte_carlo(
            human_gold_records=human_gold_records,
            evaluation_records=evaluation_records,
            targets=TARGETS,
            n_m=N_M,
            n_j=configured_n_j(DATASET_DIR),
            repeats=B,
            zeta=ZETA,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            alpha_step=ALPHA_STEP,
            seed=MONTE_CARLO_SEED,
        )
    certificate_rows = summarize_certificates(monte_carlo_rows, reliability_rows, TARGETS)
    write_csv(run_dir / "reliability_by_target.csv", reliability_rows)
    archive_run_artifacts(
        run_dir=run_dir,
        reliability_rows=reliability_rows,
        monte_carlo_rows=monte_carlo_rows,
        certificate_rows=certificate_rows,
    )
    write_run_status(run_dir, "completed", dataset_dir=DATASET_DIR)
    if use_run_human_sheet:
        print("Paper-style experiments skipped: full response-specific benchmark labels are not available.")
    else:
        try:
            run_paper_style_experiments(run_dir)
        except ValueError as exc:
            print(f"Paper-style experiments skipped: {exc}")
    print(f"\nRun complete. Artifacts saved under {run_dir}")


def run_paper_style_experiments(run_dir: Path | None = None) -> None:
    """Run paper-style Direct/Noisy/Oracle/PPI Type I/II experiments from benchmark labels."""
    global DATASET_DIR

    if run_dir is None:
        run_dir = choose_run_dir()
    config = load_run_config(run_dir)
    DATASET_DIR = config["dataset_dir"]
    target_model_name = config.get("target_model_name", TARGET_MODEL_NAME)
    manifest_rows = load_manifest(run_dir)
    benchmark_path = default_benchmark_path(PROJECT_ROOT, DATASET_DIR, target_model_name)
    if not benchmark_path.exists():
        raise ValueError(f"benchmark labels not found: {benchmark_path}")

    judge_rows = collect_chunk_results(
        run_dir,
        manifest_rows,
        "judge_labels.jsonl",
        value_field="judge_label",
        error_field="checker_error",
    )
    missing_judge = missing_keys(manifest_rows, judge_rows)
    if missing_judge:
        raise ValueError(f"{len(missing_judge)} judge labels are missing")

    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    materialize_judge_rows(records, judge_rows)
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    records = apply_benchmark_labels(records, read_benchmark_csv(benchmark_path))
    output_dir = run_dir / "error_experiments"
    type_error_rows = run_type_error_experiment(
        records=records,
        targets=TARGETS,
        alpha_values=alpha_grid(ALPHA_MIN, ALPHA_MAX, ALPHA_STEP),
        n_m_values=N_M_GRID,
        n_j=configured_n_j(DATASET_DIR),
        repeats=ERROR_EXPERIMENT_REPEATS,
        zeta=ZETA,
        seed=MONTE_CARLO_SEED,
        ridge_penalty=PPI_RIDGE_PENALTY,
        checkpoint_path=output_dir / "type_i_type_ii_checkpoint.csv",
        progress_interval=ERROR_EXPERIMENT_PROGRESS_INTERVAL,
    )
    calibration_rows = run_calibration_stability(
        records=records,
        targets=TARGETS,
        n_m_values=N_M_GRID,
        repeats=ERROR_EXPERIMENT_REPEATS,
        seed=MONTE_CARLO_SEED,
    )
    write_error_experiment_artifacts(
        output_dir=output_dir,
        type_error_rows=type_error_rows,
        calibration_rows=calibration_rows,
        main_n_m=N_M,
        main_n_j=configured_n_j(DATASET_DIR),
    )
    print(f"Paper-style error experiments saved under {output_dir}")


def main() -> None:
    """Ask which run action to take, then execute the selected resumable workflow step."""
    print("\nVLM certification workflow")
    print("1. Continue local unfinished chunks")
    print("2. Start / run selected chunks from manifest")
    print("3. Show run progress")
    print("4. Build final results from completed chunks")
    print("5. Start a new run")
    choice = input("Choose action [1-5]: ").strip()
    if choice == "1":
        continue_local_chunks()
    elif choice == "2":
        run_selected_chunks()
    elif choice == "3":
        show_run_progress()
    elif choice == "4":
        build_final_results()
    elif choice == "5":
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
