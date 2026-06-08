#!/usr/bin/env python3
"""Dataset record utilities, run artifacts, CSV exports, and certificate charts."""

from __future__ import annotations

import csv
import json
import random
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Iterable


VARIABLE_FIELDS = ("target_model_response", "human_label", "judge_label", "failure_reason")


def project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(" ", "_")


def default_run_name(target_model_name: str) -> str:
    return f"{date.today().isoformat()}_{short_model_name(target_model_name)}"


def create_run_dir(project_root: Path, runs_dir: str, run_name: str) -> Path:
    runs_path = project_path(project_root, runs_dir)
    run_dir = runs_path / run_name
    suffix = 2
    while run_dir.exists():
        run_dir = runs_path / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def load_records(project_root: Path, dataset_dir: str, targets: Iterable[str]) -> list[dict[str, Any]]:
    """Flatten image-level QA JSON files into one record per target evaluation item."""
    dataset_path = project_path(project_root, dataset_dir)
    selected_targets = set(targets)
    records: list[dict[str, Any]] = []

    for qa_path in sorted(dataset_path.glob("*/qa/*.json")):
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        for item_index, item in enumerate(data.get("items", [])):
            if item.get("target") not in selected_targets:
                continue
            record = {
                "qa_json_path": qa_path.relative_to(project_root).as_posix(),
                "qa_json_abs_path": str(qa_path),
                "item_index": item_index,
                "image_id": item.get("image_id") or data.get("image_id"),
                "image_type": item.get("image_type") or data.get("image_type"),
                "source_dataset": data.get("source_dataset"),
                "hf_dataset_id": data.get("hf_dataset_id"),
                "item_id": item.get("item_id"),
                "target": item.get("target"),
                "prompt": item.get("prompt"),
                "expected_evidence": item.get("expected_evidence"),
                "expected_answer_or_behavior": item.get("expected_answer_or_behavior"),
                "source_index": item.get("source_index"),
                "hf_id": item.get("hf_id"),
                "image_path": item.get("image_path") or data.get("image_path"),
                "variant_image_paths": json.dumps(item.get("variant_image_paths") or [], ensure_ascii=False),
                "target_model_response": item.get("target_model_response"),
                "human_label": item.get("human_label"),
                "judge_label": item.get("judge_label"),
                "failure_reason": item.get("failure_reason"),
                "notes": item.get("notes"),
            }
            records.append(record)

    return records


def record_key(record: dict[str, Any]) -> str:
    return f"{record['qa_json_path']}::{record['item_index']}::{record['item_id']}"


def validate_target_responses(records: list[dict[str, Any]]) -> None:
    missing = [record for record in records if not record.get("target_model_response")]
    if missing:
        examples = ", ".join(record["item_id"] for record in missing[:10])
        raise ValueError(
            f"{len(missing)} selected items do not have target_model_response. "
            f"Run the target VLM first. Examples: {examples}"
        )


def sample_human_gold_pool(
    records: list[dict[str, Any]],
    targets: tuple[str, ...],
    per_target: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for target in targets:
        target_records = [record for record in records if record["target"] == target]
        if len(target_records) < per_target:
            raise ValueError(f"Target {target} has {len(target_records)} records, need {per_target}.")
        selected.extend(rng.sample(target_records, per_target))
    return selected


def read_item(record: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(Path(record["qa_json_abs_path"]).read_text(encoding="utf-8"))
    return data["items"][record["item_index"]]


def update_item_fields(record: dict[str, Any], fields: dict[str, Any]) -> None:
    qa_path = Path(record["qa_json_abs_path"])
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    item = data["items"][record["item_index"]]
    for key, value in fields.items():
        item[key] = value
    qa_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_checker_results(records_by_key: dict[str, dict[str, Any]], checker_rows: list[dict[str, Any]]) -> None:
    for row in checker_rows:
        record = records_by_key[row["record_key"]]
        update_item_fields(
            record,
            {
                "judge_label": row["judge_label"],
                "failure_reason": row["judge_failure_reason"]
                if row["judge_label"] == 1 and read_item(record).get("human_label") is None
                else read_item(record).get("failure_reason"),
            },
        )


def validate_human_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        item = read_item(record)
        human_label = item.get("human_label")
        if human_label not in (0, 1):
            missing.append(record["item_id"])
            continue
        new_record = dict(record)
        for field in VARIABLE_FIELDS:
            new_record[field] = item.get(field)
        updated.append(new_record)

    if missing:
        raise ValueError(f"Human labels missing or invalid for: {', '.join(missing)}")
    return updated


def merge_checker_rows(records: list[dict[str, Any]], checker_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checker_by_key = {row["record_key"]: row for row in checker_rows}
    merged: list[dict[str, Any]] = []
    for record in records:
        row = checker_by_key[record_key(record)]
        new_record = dict(record)
        new_record["judge_label"] = row["judge_label"]
        new_record["judge_failure_reason"] = row["judge_failure_reason"]
        merged.append(new_record)
    return merged


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_category_csvs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_category_dir = run_dir / "responses_by_category"
    categories = sorted({row["image_type"] for row in rows})
    for category in categories:
        write_csv(by_category_dir / f"{category}.csv", [row for row in rows if row["image_type"] == category])


def copy_dataset_snapshot(project_root: Path, dataset_dir: str, run_dir: Path) -> None:
    source = project_path(project_root, dataset_dir)
    destination = run_dir / "dataset_snapshot"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def add_run_metadata(rows: list[dict[str, Any]], run_name: str, target_model: str, checker_model: str) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row.update({"run_name": run_name, "target_model": target_model, "checker_model": checker_model})
        enriched.append(new_row)
    return enriched


def make_charts(run_dir: Path, reliability_rows: list[dict[str, Any]], certificate_rows: list[dict[str, Any]], monte_carlo_rows: list[dict[str, Any]]) -> None:
    """Create the core certificate charts from saved reliability and Monte Carlo outputs."""
    import matplotlib.pyplot as plt

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    targets = [row["target"] for row in reliability_rows]
    x_positions = range(len(targets))
    tpr_values = [row["TPR"] if row["TPR"] is not None else 0 for row in reliability_rows]
    fpr_values = [row["FPR"] if row["FPR"] is not None else 0 for row in reliability_rows]

    plt.figure(figsize=(8, 4))
    plt.bar([x - 0.2 for x in x_positions], tpr_values, width=0.4, label="TPR")
    plt.bar([x + 0.2 for x in x_positions], fpr_values, width=0.4, label="FPR")
    plt.xticks(list(x_positions), targets, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("AI Checker Reliability by Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "tpr_fpr_by_target.png", dpi=160)
    plt.close()

    alpha_values = [row["alpha_median"] if row["alpha_median"] is not None else 0 for row in certificate_rows]
    plt.figure(figsize=(8, 4))
    plt.bar([row["target"] for row in certificate_rows], alpha_values)
    plt.ylim(0, 1)
    plt.ylabel("Median certifiable alpha")
    plt.title("Certificate Alpha by Target")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "certifiable_alpha_by_target.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    for target in targets:
        alphas = [
            row["certifiable_alpha"]
            for row in monte_carlo_rows
            if row["target"] == target and row.get("certifiable_alpha") is not None
        ]
        if alphas:
            plt.hist(alphas, bins=20, alpha=0.45, label=target)
    plt.xlabel("Certifiable alpha")
    plt.ylabel("Repeat count")
    plt.title("Monte Carlo Certifiable Alpha Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "monte_carlo_alpha_distribution.png", dpi=160)
    plt.close()


def export_run_artifacts(
    project_root: Path,
    dataset_dir: str,
    run_dir: Path,
    run_name: str,
    target_model: str,
    checker_model: str,
    config: dict[str, Any],
    human_gold_records: list[dict[str, Any]],
    gold_checker_rows: list[dict[str, Any]],
    evaluation_checker_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    monte_carlo_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
    targets: tuple[str, ...],
) -> None:
    """Persist the completed run in CSV, JSON, JSONL, dataset snapshot, and chart form."""
    write_json(run_dir / "run_config.json", config)
    write_csv(run_dir / "human_gold_pool.csv", add_run_metadata(human_gold_records, run_name, target_model, checker_model))
    write_jsonl(run_dir / "human_gold_pool.jsonl", human_gold_records)
    write_csv(run_dir / "judge_labels_gold.csv", gold_checker_rows)
    write_csv(run_dir / "judge_labels_evaluation.csv", evaluation_checker_rows)
    write_csv(run_dir / "reliability_by_target.csv", reliability_rows)
    write_csv(run_dir / "monte_carlo_repeats.csv", monte_carlo_rows)
    write_csv(run_dir / "certificate_summary.csv", certificate_rows)
    write_json(run_dir / "certificate_summary.json", certificate_rows)

    all_records = load_records(project_root, dataset_dir, targets)
    all_records = add_run_metadata(all_records, run_name, target_model, checker_model)
    write_csv(run_dir / "all_items.csv", all_records)
    write_category_csvs(run_dir, all_records)
    copy_dataset_snapshot(project_root, dataset_dir, run_dir)
    make_charts(run_dir, reliability_rows, certificate_rows, monte_carlo_rows)
