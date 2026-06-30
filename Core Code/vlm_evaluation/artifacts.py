#!/usr/bin/env python3
"""Dataset record utilities, run artifacts, CSV exports, and certificate charts."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dataset_selection import logical_dataset_path, resolve_dataset_path
from normalization import build_checker_reference_fields


VARIABLE_FIELDS = ("target_model_response", "human_label", "judge_label", "failure_reason")


def project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(" ", "_")


def default_run_name(target_model_name: str) -> str:
    return f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{short_model_name(target_model_name)}"


def create_run_dir(project_root: Path, runs_dir: str, run_name: str) -> Path:
    runs_path = project_path(project_root, runs_dir)
    run_dir = runs_path / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def load_records(
    project_root: Path,
    dataset_dir: str,
    targets: Iterable[str],
    dataset_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten image-level QA JSON files into one record per target evaluation item."""
    dataset_path = resolve_dataset_path(project_root, dataset_dir, dataset_paths)
    selected_targets = set(targets)
    records: list[dict[str, Any]] = []

    for qa_path in sorted(dataset_path.glob("*/qa/*.json")):
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        visual_item = next((source_item for source_item in items if source_item.get("target") == "visual_factuality"), {})
        for item_index, item in enumerate(items):
            if item.get("target") not in selected_targets:
                continue
            image_path = item.get("image_path") or data.get("image_path")
            variant_paths = item.get("variant_image_paths") or []
            model_input_image_path = (
                variant_paths[0]
                if item.get("target") == "robustness" and variant_paths
                else image_path
            )
            checker_fields = {
                **build_checker_reference_fields(item),
                **{
                    key: item[key]
                    for key in ("checker_reference_answer", "checker_reference_evidence", "checker_reference_values")
                    if key in item
                },
            }
            qa_json_path = logical_dataset_path(dataset_dir, dataset_path, qa_path)
            record = {
                "dataset_dir": dataset_dir,
                "qa_json_path": qa_json_path,
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
                "checker_reference_answer": checker_fields["checker_reference_answer"],
                "checker_reference_evidence": checker_fields["checker_reference_evidence"],
                "checker_reference_values": checker_fields["checker_reference_values"],
                "source_index": item.get("source_index"),
                "hf_id": item.get("hf_id"),
                "image_path": image_path,
                "model_input_image_path": model_input_image_path,
                "variant_image_paths": json.dumps(variant_paths, ensure_ascii=False),
                "original_expected_answer_or_behavior": visual_item.get("expected_answer_or_behavior"),
                "original_target_model_response": visual_item.get("target_model_response"),
                "target_model_response": item.get("target_model_response"),
                "human_label": item.get("human_label"),
                "judge_label": item.get("judge_label"),
                "failure_reason": item.get("failure_reason"),
                "notes": item.get("notes"),
            }
            records.append(record)

    return records


def write_human_label_sheet(path: Path, records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "human_label": "",
                "human_failure_reason": "",
                "target": record["target"],
                "model_input_image_path": record["model_input_image_path"],
                "prompt": record["prompt"],
                "expected_evidence": record["expected_evidence"],
                "expected_answer_or_behavior": record["expected_answer_or_behavior"],
                "target_model_response": record["target_model_response"],
                "notes": record["notes"],
                "record_key": record_key(record),
            }
        )
    write_csv(path, rows)


def apply_human_label_sheet(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Human label sheet not found: {path}")

    records_by_key = {record_key(record): record for record in records}
    labelled_keys: set[str] = set()
    missing: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.get("record_key", "")
            if key not in records_by_key:
                continue
            label_text = (row.get("human_label") or "").strip()
            if label_text not in {"0", "1"}:
                missing.append(records_by_key[key]["item_id"])
                continue
            human_label = int(label_text)
            reason = (row.get("human_failure_reason") or "").strip() or None
            update_item_fields(
                records_by_key[key],
                {
                    "human_label": human_label,
                    "failure_reason": reason if human_label == 1 else None,
                },
            )
            labelled_keys.add(key)

    for key, record in records_by_key.items():
        if key not in labelled_keys:
            missing.append(record["item_id"])

    if missing:
        raise ValueError(f"Human labels missing or invalid in label sheet for: {', '.join(missing)}")

    return validate_human_labels(records)


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
    balance_by_image_type: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for target in targets:
        target_records = [record for record in records if record["target"] == target]
        if len(target_records) < per_target:
            raise ValueError(f"Target {target} has {len(target_records)} records, need {per_target}.")
        if balance_by_image_type:
            selected.extend(sample_balanced_by_image_type(target_records, per_target, rng))
        else:
            selected.extend(rng.sample(target_records, per_target))
    return selected


def sample_balanced_by_image_type(records: list[dict[str, Any]], sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample records round-robin across image_type groups for broader gold-pool coverage."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["image_type"], []).append(record)
    for group_records in groups.values():
        rng.shuffle(group_records)

    image_types = sorted(groups)
    rng.shuffle(image_types)
    selected: list[dict[str, Any]] = []
    while len(selected) < sample_size and image_types:
        next_image_types: list[str] = []
        for image_type in image_types:
            group = groups[image_type]
            if group:
                selected.append(group.pop())
            if group:
                next_image_types.append(image_type)
            if len(selected) == sample_size:
                break
        image_types = next_image_types

    if len(selected) != sample_size:
        raise ValueError(f"Could only sample {len(selected)} records, need {sample_size}.")
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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


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


def copy_dataset_snapshot(
    project_root: Path,
    dataset_dir: str,
    run_dir: Path,
    dataset_paths: dict[str, str] | None = None,
) -> None:
    """Archive QA JSON state only; image files are immutable dataset inputs and are not copied."""
    source = resolve_dataset_path(project_root, dataset_dir, dataset_paths)
    destination = run_dir / "dataset_snapshot"
    for qa_path in sorted(source.glob("*/qa/*.json")):
        relative_path = qa_path.relative_to(source)
        target_path = destination / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(qa_path, target_path)


def make_charts(run_dir: Path, reliability_rows: list[dict[str, Any]], certificate_rows: list[dict[str, Any]], monte_carlo_rows: list[dict[str, Any]]) -> None:
    """Create the core certificate charts from saved reliability and Monte Carlo outputs."""
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    matplotlib_cache = Path(os.environ.get("MPLCONFIGDIR", Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chart generation requires Matplotlib. Install project dependencies with "
            "'python -m pip install -r requirement.txt'."
        ) from exc

    targets = [row["target"] for row in reliability_rows]
    x_positions = list(range(len(targets)))
    tpr_values = [row["TPR"] if row["TPR"] is not None else 0 for row in reliability_rows]
    fpr_values = [row["FPR"] if row["FPR"] is not None else 0 for row in reliability_rows]
    reliability_labels = [
        f"{row['target']}\nM1={row['n_M1']}, M0={row['n_M0']}" for row in reliability_rows
    ]

    plt.figure(figsize=(8, 4))
    plt.bar([x - 0.2 for x in x_positions], tpr_values, width=0.4, label="TPR")
    plt.bar([x + 0.2 for x in x_positions], fpr_values, width=0.4, label="FPR")
    plt.xticks(x_positions, reliability_labels)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("AI Checker Reliability by Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "tpr_fpr_by_target.png", dpi=160)
    plt.close()

    alpha_values = [row["alpha_median"] if row["alpha_median"] is not None else 0 for row in certificate_rows]
    if certificate_rows:
        certificate_labels = [
            f"{row['target']}\ncert={row['certified_rate']:.2f}"
            if row.get("certified_rate") is not None
            else row["target"]
            for row in certificate_rows
        ]
        plt.figure(figsize=(8, 4))
        plt.bar(certificate_labels, alpha_values)
        plt.ylim(0, 1)
        plt.ylabel("Median certifiable alpha")
        plt.title("Certificate Alpha by Target")
        plt.tight_layout()
        plt.savefig(charts_dir / "certifiable_alpha_by_target.png", dpi=160)
        plt.close()

    histogram_written = False
    plt.figure(figsize=(8, 4))
    for target in targets:
        alphas = [
            row["certifiable_alpha"]
            for row in monte_carlo_rows
            if row["target"] == target and row.get("certifiable_alpha") is not None
        ]
        if alphas:
            plt.hist(alphas, bins=20, alpha=0.45, label=target)
            histogram_written = True
    if histogram_written:
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
    reliability_rows: list[dict[str, Any]],
    monte_carlo_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
    dataset_paths: dict[str, str] | None = None,
) -> None:
    """Persist the completed run summary and QA JSON snapshot."""
    print("Writing certificate summary...", flush=True)
    write_csv(run_dir / "certificate_summary.csv", certificate_rows)

    print("Copying QA dataset snapshot...", flush=True)
    copy_dataset_snapshot(project_root, dataset_dir, run_dir, dataset_paths)
