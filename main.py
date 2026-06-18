#!/usr/bin/env python3
"""One-command workflow for AI checker validation and certification calculation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "vlm_evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "vlm_testing"))

from ai_checker import call_checker, load_annotation_guide, require_local_endpoint
from artifacts import (
    apply_checker_results,
    apply_human_label_sheet,
    create_run_dir,
    default_run_name,
    export_run_artifacts,
    load_records,
    merge_checker_rows,
    record_key,
    sample_human_gold_pool,
    validate_target_responses,
    write_human_label_sheet,
    write_csv,
    write_json,
)
from certification import estimate_reliability, run_monte_carlo, summarize_certificates
from dataset_selection import select_dataset_dir
from model_server import ManagedServer, auto_select_gpus, auto_select_gpus_by_balanced_memory, start_vllm_server
from reset_dataset import reset_dataset_fields
from run_target_vlm import run_dataset as run_target_vlm_dataset


# Paths
DATASET_DIR = "dataset"
RUNS_DIR = "runs"
ANNOTATION_GUIDE_PATH = "ANNOTATION_GUIDE.md"
RUN_NAME = None

# Target VLM response generation
RUN_TARGET_VLM_FIRST = True
TARGET_CATEGORIES = None
TARGET_CUDA_VISIBLE_DEVICES = None
TARGET_GPU_MEMORY_UTILIZATION = 0.75
TARGET_LIMIT = None
TARGET_LIMIT_MM_PER_PROMPT = '{"image":2,"video":0}'
TARGET_MAX_GPUS = 2
TARGET_MAX_MODEL_LEN = 8192
TARGET_MAX_TOKENS = 160
TARGET_MIN_FREE_MEMORY_MB = 40000
TARGET_MM_ENCODER_TP_MODE = None
TARGET_MODEL_ID = "mistralai/Pixtral-12B-2409"
TARGET_MODEL_NAME = "Pixtral-12B-2409"
TARGET_OVERWRITE_RESPONSES = False
TARGET_PORT = 8000
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
CHECKER_GPU_MEMORY_UTILIZATION = 0.90
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
# Human gold pool size per target; Monte Carlo samples N_M from this pool each repeat.
HUMAN_GOLD_PER_TARGET = 30
N_M = 15
N_J = 120

# Random seeds
HUMAN_GOLD_SAMPLE_SEED = 42
MONTE_CARLO_SEED = 20260608

# Certification calculation
B = 1000
ZETA = 0.05
ALPHA_MIN = 0.01
ALPHA_MAX = 0.80
ALPHA_STEP = 0.01

# Checker generation
CHECKER_MAX_TOKENS = 180
CHECKER_TEMPERATURE = 0.0
CHECKER_TIMEOUT_SECONDS = 180
OVERWRITE_JUDGE_LABELS = False

# Run behavior
RESET_DATASET_AT_START = True


def current_config(run_name: str) -> dict[str, Any]:
    """Collect every user-tunable workflow parameter into the saved run configuration."""
    return {
        "run_name": run_name,
        "dataset_dir": DATASET_DIR,
        "runs_dir": RUNS_DIR,
        "annotation_guide_path": ANNOTATION_GUIDE_PATH,
        "run_target_vlm_first": RUN_TARGET_VLM_FIRST,
        "target_model_id": TARGET_MODEL_ID,
        "target_model_name": TARGET_MODEL_NAME,
        "target_port": TARGET_PORT,
        "target_categories": TARGET_CATEGORIES,
        "target_limit": TARGET_LIMIT,
        "target_max_tokens": TARGET_MAX_TOKENS,
        "target_temperature": TARGET_TEMPERATURE,
        "target_timeout_seconds": TARGET_TIMEOUT_SECONDS,
        "target_overwrite_responses": TARGET_OVERWRITE_RESPONSES,
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
        "human_gold_sample_seed": HUMAN_GOLD_SAMPLE_SEED,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "repeats": B,
        "zeta": ZETA,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "alpha_step": ALPHA_STEP,
        "checker_max_tokens": CHECKER_MAX_TOKENS,
        "checker_temperature": CHECKER_TEMPERATURE,
        "checker_timeout_seconds": CHECKER_TIMEOUT_SECONDS,
        "overwrite_judge_labels": OVERWRITE_JUDGE_LABELS,
        "reset_dataset_at_start": RESET_DATASET_AT_START,
        "balance_human_gold_by_category": BALANCE_HUMAN_GOLD_BY_CATEGORY,
    }


def target_internal_endpoint() -> str:
    return f"http://localhost:{TARGET_PORT}/v1/chat/completions"


def maybe_start_target_server(run_dir: Path) -> ManagedServer:
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
        model=TARGET_MODEL_ID,
        served_model_name=TARGET_MODEL_ID,
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


def run_target_responses() -> int:
    """Generate target_model_response values before checker validation and certification."""
    if not RUN_TARGET_VLM_FIRST:
        return 0
    args = SimpleNamespace(
        dataset_dir=DATASET_DIR,
        port=TARGET_PORT,
        model=TARGET_MODEL_ID,
        api_key_env="LOCAL_VLM_API_KEY",
        categories=TARGET_CATEGORIES,
        targets=TARGETS,
        limit=TARGET_LIMIT,
        max_tokens=TARGET_MAX_TOKENS,
        temperature=TARGET_TEMPERATURE,
        timeout=TARGET_TIMEOUT_SECONDS,
        overwrite=TARGET_OVERWRITE_RESPONSES,
        dry_run=False,
    )
    return run_target_vlm_dataset(args)


def target_responses_needed() -> bool:
    if not RUN_TARGET_VLM_FIRST:
        return False
    if TARGET_OVERWRITE_RESPONSES:
        return True
    records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
    return any(not record.get("target_model_response") for record in records)


def maybe_start_checker_server(run_dir: Path) -> ManagedServer:
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


def judge_records(records: list[dict[str, Any]], annotation_guide: str, stage: str) -> list[dict[str, Any]]:
    """Call the local checker for selected records and attach stable run metadata to each result."""
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if record.get("judge_label") is not None and not OVERWRITE_JUDGE_LABELS:
            rows.append(
                {
                    "record_key": record_key(record),
                    "stage": stage,
                    "item_id": record["item_id"],
                    "image_id": record["image_id"],
                    "image_type": record["image_type"],
                    "target": record["target"],
                    "qa_json_path": record["qa_json_path"],
                    "judge_label": record["judge_label"],
                    "judge_failure_reason": record.get("failure_reason"),
                }
            )
            continue
        print(f"Checker {stage} {index}/{len(records)}: {record['item_id']} [{record['target']}]")
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
        result["record_key"] = record_key(record)
        result["stage"] = stage
        rows.append(result)
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
    run_name: str,
    config: dict[str, Any],
    human_gold_records: list[dict[str, Any]],
    gold_checker_rows: list[dict[str, Any]],
    evaluation_checker_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    monte_carlo_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
) -> None:
    """Write every reusable run artifact without mutating the working dataset."""
    export_run_artifacts(
        project_root=PROJECT_ROOT,
        dataset_dir=DATASET_DIR,
        run_dir=run_dir,
        run_name=run_name,
        target_model=TARGET_MODEL_NAME,
        checker_model=CHECKER_MODEL_NAME,
        config=config,
        human_gold_records=human_gold_records,
        gold_checker_rows=gold_checker_rows,
        evaluation_checker_rows=evaluation_checker_rows,
        reliability_rows=reliability_rows,
        monte_carlo_rows=monte_carlo_rows,
        certificate_rows=certificate_rows,
        targets=TARGETS,
    )


def reset_dataset_at_start() -> None:
    """Clear mutable working-dataset fields before any new model responses are generated."""
    if not RESET_DATASET_AT_START:
        return
    reset_dataset_fields(PROJECT_ROOT, DATASET_DIR)


def main() -> None:
    """Run checker validation, pause for human labels, calculate certification, archive, and reset."""
    global DATASET_DIR

    DATASET_DIR = select_dataset_dir(PROJECT_ROOT, DATASET_DIR)
    print(f"Using dataset directory: {DATASET_DIR}")

    run_name = RUN_NAME or default_run_name(TARGET_MODEL_NAME)
    run_dir = create_run_dir(PROJECT_ROOT, RUNS_DIR, run_name)
    run_name = run_dir.name
    config = current_config(run_name)
    write_json(run_dir / "run_config.json", config)
    reset_dataset_at_start()

    require_local_endpoint(CHECKER_INTERNAL_ENDPOINT, ALLOW_NON_LOCAL_CHECKER)
    target_server = maybe_start_target_server(run_dir)
    checker_server = ManagedServer(process=None, log_path=None)
    try:
        processed_target_items = run_target_responses()
        print(f"Target VLM processed {processed_target_items} item(s).")
        if target_server.started_by_script:
            target_server.stop()
            target_server = ManagedServer(process=None, log_path=None)
        checker_server = maybe_start_checker_server(run_dir)
        annotation_guide = load_annotation_guide(PROJECT_ROOT / ANNOTATION_GUIDE_PATH)

        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
        validate_target_responses(records)

        human_gold_records = sample_human_gold_pool(
            records=records,
            targets=TARGETS,
            per_target=HUMAN_GOLD_PER_TARGET,
            seed=HUMAN_GOLD_SAMPLE_SEED,
            balance_by_image_type=BALANCE_HUMAN_GOLD_BY_CATEGORY,
        )

        gold_checker_rows = judge_records(human_gold_records, annotation_guide, stage="gold")
        write_csv(run_dir / "judge_labels_gold.csv", gold_checker_rows)

        human_label_sheet_path = run_dir / "human_label_tasks.csv"
        write_human_label_sheet(human_label_sheet_path, human_gold_records)
        print_human_label_instructions(human_gold_records)
        print(f"\nHuman label sheet: {human_label_sheet_path}")
        answer = input("\nType yes after human labels are completed in the CSV: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Stopped before certification because human labels were not confirmed.")

        human_gold_records = apply_human_label_sheet(human_label_sheet_path, human_gold_records)
        records_by_key = {record_key(record): record for record in human_gold_records}
        apply_checker_results(records_by_key, gold_checker_rows)
        human_gold_records = merge_checker_rows(human_gold_records, gold_checker_rows)

        reliability_rows = [estimate_reliability(human_gold_records, target) for target in TARGETS]
        unreliable = [row for row in reliability_rows if not row["reliable"]]
        if unreliable:
            print("\nChecker unreliable. Exiting before evaluation-pool judging.")
            for row in unreliable:
                print(f"- {row['target']}: {row['unreliable_reason']}")
            archive_run_artifacts(
                run_dir=run_dir,
                run_name=run_name,
                config=config,
                human_gold_records=human_gold_records,
                gold_checker_rows=gold_checker_rows,
                evaluation_checker_rows=[],
                reliability_rows=reliability_rows,
                monte_carlo_rows=[],
                certificate_rows=[],
            )
            raise SystemExit(1)

        gold_keys = {record_key(record) for record in human_gold_records}
        evaluation_records = [record for record in records if record_key(record) not in gold_keys]
        evaluation_checker_rows = judge_records(evaluation_records, annotation_guide, stage="evaluation")
        evaluation_by_key = {record_key(record): record for record in evaluation_records}
        apply_checker_results(evaluation_by_key, evaluation_checker_rows)
        evaluation_records = merge_checker_rows(evaluation_records, evaluation_checker_rows)

        monte_carlo_rows = run_monte_carlo(
            human_gold_records=human_gold_records,
            evaluation_records=evaluation_records,
            targets=TARGETS,
            n_m=N_M,
            n_j=N_J,
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
            run_name=run_name,
            config=config,
            human_gold_records=human_gold_records,
            gold_checker_rows=gold_checker_rows,
            evaluation_checker_rows=evaluation_checker_rows,
            reliability_rows=reliability_rows,
            monte_carlo_rows=monte_carlo_rows,
            certificate_rows=certificate_rows,
        )
        print(f"\nRun complete. Artifacts saved under {run_dir}")
    finally:
        if checker_server.started_by_script:
            checker_server.stop()
        if target_server.started_by_script:
            target_server.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
