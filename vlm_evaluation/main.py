#!/usr/bin/env python3
"""One-command workflow for AI checker validation and certification calculation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ai_checker import call_checker, load_annotation_guide, require_local_endpoint
from artifacts import (
    apply_checker_results,
    create_run_dir,
    default_run_name,
    export_run_artifacts,
    load_records,
    merge_checker_rows,
    record_key,
    sample_human_gold_pool,
    validate_human_labels,
    validate_target_responses,
    write_csv,
    write_json,
)
from certification import estimate_reliability, run_monte_carlo, summarize_certificates
from model_server import ManagedServer, start_vllm_server
from reset_dataset import reset_dataset_fields


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = "dataset"
RUNS_DIR = "runs"
ANNOTATION_GUIDE_PATH = "ANNOTATION_GUIDE.md"
RUN_NAME = None

# Models and automatic local checker deployment
TARGET_MODEL_NAME = "Qwen2.5-VL-72B-Instruct"
CHECKER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CHECKER_PORT = 8001
CHECKER_INTERNAL_ENDPOINT = f"http://localhost:{CHECKER_PORT}/v1/chat/completions"
CHECKER_API_KEY_ENV = "LOCAL_CHECKER_API_KEY"
ALLOW_NON_LOCAL_CHECKER = False
AUTO_DEPLOY_CHECKER = True
CHECKER_CUDA_VISIBLE_DEVICES = "0"
CHECKER_TENSOR_PARALLEL_SIZE = 1
CHECKER_MAX_MODEL_LEN = 32768
CHECKER_GPU_MEMORY_UTILIZATION = 0.90
CHECKER_LIMIT_MM_PER_PROMPT = None
CHECKER_MM_ENCODER_TP_MODE = None
CHECKER_VLLM_EXTRA_ARGS: tuple[str, ...] = ()
CHECKER_SERVER_WAIT_TIMEOUT_SECONDS = 1800

# Dataset and target selection
TARGETS = ("visual_factuality", "robustness", "refusal_behavior")
HUMAN_GOLD_PER_TARGET = 5
N_M = 5
N_J = 20

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
RESET_DATASET_AFTER_ARCHIVE = True


def current_config(run_name: str) -> dict[str, Any]:
    """Collect every user-tunable workflow parameter into the saved run configuration."""
    return {
        "run_name": run_name,
        "dataset_dir": DATASET_DIR,
        "runs_dir": RUNS_DIR,
        "annotation_guide_path": ANNOTATION_GUIDE_PATH,
        "target_model_name": TARGET_MODEL_NAME,
        "checker_model_name": CHECKER_MODEL_NAME,
        "checker_port": CHECKER_PORT,
        "checker_internal_endpoint": CHECKER_INTERNAL_ENDPOINT,
        "checker_api_key_env": CHECKER_API_KEY_ENV,
        "allow_non_local_checker": ALLOW_NON_LOCAL_CHECKER,
        "auto_deploy_checker": AUTO_DEPLOY_CHECKER,
        "checker_cuda_visible_devices": CHECKER_CUDA_VISIBLE_DEVICES,
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
        "reset_dataset_after_archive": RESET_DATASET_AFTER_ARCHIVE,
    }


def maybe_start_checker_server(run_dir: Path) -> ManagedServer:
    if not AUTO_DEPLOY_CHECKER:
        return ManagedServer(process=None, log_path=None)
    print("Starting local checker server if needed...")
    return start_vllm_server(
        endpoint=CHECKER_INTERNAL_ENDPOINT,
        model=CHECKER_MODEL_NAME,
        served_model_name=CHECKER_MODEL_NAME,
        log_path=run_dir / "logs" / "checker_vllm.log",
        cuda_visible_devices=CHECKER_CUDA_VISIBLE_DEVICES,
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
    print("\nSelected human gold pool. Fill human_label in these QA JSON items:")
    for record in human_gold_records:
        print(
            f"- {record['item_id']} | {record['target']} | "
            f"{PROJECT_ROOT / record['qa_json_path']}"
        )
    print("\nUse human_label = 0 for success and human_label = 1 for failure.")
    print("For failures, fill failure_reason if possible. Do not edit judge_label.")


def archive_and_maybe_reset(
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
    """Write every reusable run artifact, then reset only mutable fields in the working dataset."""
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
    if RESET_DATASET_AFTER_ARCHIVE:
        reset_log = reset_dataset_fields(PROJECT_ROOT, DATASET_DIR)
        write_json(run_dir / "reset_log.json", reset_log)


def main() -> None:
    """Run checker validation, pause for human labels, calculate certification, archive, and reset."""
    run_name = RUN_NAME or default_run_name(TARGET_MODEL_NAME)
    run_dir = create_run_dir(PROJECT_ROOT, RUNS_DIR, run_name)
    config = current_config(run_name)
    write_json(run_dir / "run_config.json", config)

    require_local_endpoint(CHECKER_INTERNAL_ENDPOINT, ALLOW_NON_LOCAL_CHECKER)
    checker_server = maybe_start_checker_server(run_dir)
    try:
        annotation_guide = load_annotation_guide(PROJECT_ROOT / ANNOTATION_GUIDE_PATH)

        records = load_records(PROJECT_ROOT, DATASET_DIR, TARGETS)
        validate_target_responses(records)

        human_gold_records = sample_human_gold_pool(
            records=records,
            targets=TARGETS,
            per_target=HUMAN_GOLD_PER_TARGET,
            seed=HUMAN_GOLD_SAMPLE_SEED,
        )
        write_csv(run_dir / "human_gold_pool_selected.csv", human_gold_records)

        gold_checker_rows = judge_records(human_gold_records, annotation_guide, stage="gold")
        write_csv(run_dir / "judge_labels_gold_pending.csv", gold_checker_rows)

        print_human_label_instructions(human_gold_records)
        answer = input("\nType yes after human labels are completed in the QA JSON files: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Stopped before certification because human labels were not confirmed.")

        human_gold_records = validate_human_labels(human_gold_records)
        records_by_key = {record_key(record): record for record in human_gold_records}
        apply_checker_results(records_by_key, gold_checker_rows)
        human_gold_records = merge_checker_rows(human_gold_records, gold_checker_rows)

        reliability_rows = [estimate_reliability(human_gold_records, target) for target in TARGETS]
        write_csv(run_dir / "reliability_by_target_pending.csv", reliability_rows)
        unreliable = [row for row in reliability_rows if not row["reliable"]]
        if unreliable:
            print("\nChecker unreliable. Exiting before evaluation-pool judging.")
            for row in unreliable:
                print(f"- {row['target']}: {row['unreliable_reason']}")
            archive_and_maybe_reset(
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
        archive_and_maybe_reset(
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
