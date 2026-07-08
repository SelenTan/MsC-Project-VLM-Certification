#!/usr/bin/env python3
"""Build a project-level human label benchmark CSV from image-level QA JSON files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from artifacts import record_key, short_model_name
from dataset_selection import logical_dataset_path, resolve_dataset_path


IMAGE_TYPES = ("charts", "documents", "forms", "receipts", "screenshots")
TARGETS = ("visual_factuality", "robustness", "refusal_behavior")
TARGET_ORDER = {target: index for index, target in enumerate(TARGETS)}
CSV_COLUMNS = (
    "record_key",
    "qa_json_path",
    "item_index",
    "image_id",
    "image_type",
    "target",
    "item_id",
    "model_input_image_path",
    "prompt",
    "expected_evidence",
    "expected_answer_or_behavior",
    "checker_reference_answer",
    "checker_reference_evidence",
    "target_model_name",
    "target_response_hash",
    "target_model_response",
    "human_label",
    "human_failure_reason",
    "label_status",
)


def project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def item_human_label(item: dict[str, Any]) -> str:
    label = item.get("human_label")
    if label in (0, 1):
        return str(label)
    return ""


def item_human_failure_reason(item: dict[str, Any]) -> str:
    if item.get("human_label") == 1:
        return csv_value(item.get("failure_reason"))
    return ""


def hash_target_response(response_text: str) -> str:
    if not response_text:
        return ""
    return hashlib.sha256(response_text.encode("utf-8")).hexdigest()


def item_label_status(item: dict[str, Any]) -> str:
    if not csv_value(item.get("target_model_response")):
        return "waiting_for_response"
    return "complete" if item.get("human_label") in (0, 1) else "pending"


def model_input_image_path(data: dict[str, Any], item: dict[str, Any]) -> str:
    image_path = item.get("image_path") or data.get("image_path")
    variant_paths = item.get("variant_image_paths") or []
    if item.get("target") == "robustness" and variant_paths:
        return str(variant_paths[0])
    return str(image_path)


def validate_item(category: str, data: dict[str, Any], item: dict[str, Any], qa_path: Path, item_index: int) -> None:
    target = item.get("target")
    if category not in IMAGE_TYPES:
        raise ValueError(f"Unsupported image_type directory {category!r} in {qa_path}")
    if target not in TARGETS:
        raise ValueError(f"Unsupported target {target!r} in {qa_path} item {item_index}")
    if data.get("image_type") not in (None, category) or item.get("image_type") not in (None, category):
        raise ValueError(f"image_type mismatch for {qa_path} item {item_index}")
    if not item.get("item_id"):
        raise ValueError(f"Missing item_id in {qa_path} item {item_index}")


def build_row(
    project_root: Path,
    dataset_dir: str,
    dataset_root: Path,
    qa_path: Path,
    item_index: int,
    data: dict[str, Any],
    item: dict[str, Any],
    target_model_name: str = "",
) -> dict[str, str]:
    """Create one benchmark CSV row for a single QA item without mutating the QA JSON."""
    category = qa_path.relative_to(dataset_root).parts[0]
    validate_item(category, data, item, qa_path, item_index)
    image_id = str(item.get("image_id") or data.get("image_id") or qa_path.stem)
    qa_json_path = logical_dataset_path(dataset_dir, dataset_root, qa_path)
    target_response = csv_value(item.get("target_model_response"))
    record = {
        "qa_json_path": qa_json_path,
        "item_index": item_index,
        "item_id": item["item_id"],
    }
    row = {
        "record_key": record_key(record),
        "qa_json_path": qa_json_path,
        "item_index": str(item_index),
        "image_id": image_id,
        "image_type": category,
        "target": str(item["target"]),
        "item_id": str(item["item_id"]),
        "model_input_image_path": model_input_image_path(data, item),
        "prompt": csv_value(item.get("prompt")),
        "expected_evidence": csv_value(item.get("expected_evidence")),
        "expected_answer_or_behavior": csv_value(item.get("expected_answer_or_behavior")),
        "checker_reference_answer": csv_value(item.get("checker_reference_answer")),
        "checker_reference_evidence": csv_value(item.get("checker_reference_evidence")),
        "target_model_name": target_model_name,
        "target_response_hash": hash_target_response(target_response),
        "target_model_response": target_response,
        "human_label": item_human_label(item),
        "human_failure_reason": item_human_failure_reason(item),
        "label_status": item_label_status(item),
    }
    return row


def benchmark_filename(dataset_dir: str, response_fingerprint: str = "") -> str:
    prefix = dataset_dir.replace(" Dataset", "").replace(" ", "_").lower()
    if response_fingerprint:
        return f"{prefix}_{response_fingerprint}_human_labels.csv"
    return f"{prefix}_human_labels.csv"


def response_fingerprint(rows: list[dict[str, str]]) -> str:
    """Identify one full benchmark response set from item keys and target response hashes."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["record_key"]):
        digest.update(row["record_key"].encode("utf-8"))
        digest.update(b"\0")
        digest.update((row.get("target_response_hash") or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def load_benchmark_rows(
    project_root: Path,
    dataset_dir: str = "Large Dataset",
    target_model_name: str = "",
    dataset_paths: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Scan all QA JSON files in one dataset and return sorted benchmark label rows."""
    dataset_path = resolve_dataset_path(project_root, dataset_dir, dataset_paths)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    rows: list[dict[str, str]] = []
    for qa_path in sorted(dataset_path.glob("*/qa/*.json")):
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        for item_index, item in enumerate(data.get("items", [])):
            rows.append(
                build_row(
                    project_root,
                    dataset_dir,
                    dataset_path,
                    qa_path,
                    item_index,
                    data,
                    item,
                    target_model_name=target_model_name,
                )
            )

    rows.sort(key=lambda row: (row["image_type"], row["image_id"], TARGET_ORDER[row["target"]]))
    return rows


def write_benchmark_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_large_human_label_benchmark(
    project_root: Path,
    dataset_dir: str = "Large Dataset",
    output_path: Optional[Path] = None,
    target_model_name: str = "",
    dataset_paths: dict[str, str] | None = None,
) -> Path:
    rows = load_benchmark_rows(project_root, dataset_dir, target_model_name, dataset_paths)
    if output_path is None:
        output_path = default_benchmark_path(
            project_root,
            dataset_dir,
            target_model_name,
            response_fingerprint(rows),
        )
    write_benchmark_csv(output_path, rows)
    return output_path


def default_benchmark_path(
    project_root: Path,
    dataset_dir: str,
    target_model_name: str = "",
    response_set_id: str = "",
) -> Path:
    if target_model_name:
        return project_root / "benchmark_labels" / short_model_name(target_model_name) / benchmark_filename(dataset_dir, response_set_id)
    return project_root / "benchmark_labels" / benchmark_filename(dataset_dir, response_set_id)


def infer_label_status(row: dict[str, str]) -> str:
    if not (row.get("target_model_response") or "").strip():
        return "waiting_for_response"
    if (row.get("human_label") or "").strip() in {"0", "1"}:
        return "complete"
    return "pending"


def read_benchmark_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark label CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    missing_columns = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or ())]
    allowed_missing = {"label_status", "target_model_name", "target_response_hash"}
    unexpected_missing = [column for column in missing_columns if column not in allowed_missing]
    if unexpected_missing:
        raise ValueError(f"Benchmark label CSV is missing columns: {unexpected_missing}")
    for row in rows:
        row.setdefault("target_model_name", "")
        if not row.get("target_response_hash"):
            row["target_response_hash"] = hash_target_response(row.get("target_model_response") or "")
        if not row.get("label_status"):
            row["label_status"] = infer_label_status(row)
    return rows


def validate_completed_benchmark_rows(rows: list[dict[str, str]]) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        label = (row.get("human_label") or "").strip()
        if label == "":
            missing.append(row.get("record_key", ""))
            continue
        if label not in {"0", "1"}:
            invalid.append(row.get("record_key", ""))
            continue
        if label == "1" and not (row.get("human_failure_reason") or "").strip():
            invalid.append(row.get("record_key", ""))
            continue
        if not (row.get("target_model_response") or "").strip():
            invalid.append(row.get("record_key", ""))
            continue
        if not (row.get("target_response_hash") or "").strip():
            invalid.append(row.get("record_key", ""))
    if missing or invalid:
        message = (
            f"Benchmark labels incomplete: {len(missing)} missing labels, "
            f"{len(invalid)} invalid rows."
        )
        examples = [key for key in (missing + invalid)[:10] if key]
        if examples:
            message += f" Examples: {', '.join(examples)}"
        raise ValueError(message)


def merge_existing_benchmark_labels(
    fresh_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Preserve completed manual labels only when they still match the current response hash."""
    existing_by_key = {row.get("record_key", ""): row for row in existing_rows}
    merged: list[dict[str, str]] = []
    for fresh_row in fresh_rows:
        row = dict(fresh_row)
        existing = existing_by_key.get(row["record_key"])
        if (
            existing
            and existing.get("target_response_hash") == row.get("target_response_hash")
            and (existing.get("human_label") or "").strip() in {"0", "1"}
        ):
            label = (existing.get("human_label") or "").strip()
            row["human_label"] = label
            row["human_failure_reason"] = existing.get("human_failure_reason") or ""
            row["label_status"] = "complete"
        merged.append(row)
    return merged


def apply_benchmark_labels(
    records: list[dict[str, Any]],
    benchmark_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Return records with human_label fields populated from the benchmark CSV."""
    validate_completed_benchmark_rows(benchmark_rows)
    rows_by_key = {row["record_key"]: row for row in benchmark_rows}
    labelled: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for record in records:
        key = record_key(record)
        row = rows_by_key.get(key)
        if row is None:
            missing.append(key)
            continue
        response_text = csv_value(record.get("target_model_response"))
        if hash_target_response(response_text) != (row.get("target_response_hash") or ""):
            mismatched.append(key)
            continue
        updated = dict(record)
        updated["human_label"] = int(row["human_label"])
        updated["failure_reason"] = row.get("human_failure_reason") or None
        labelled.append(updated)
    if missing:
        examples = ", ".join(missing[:10])
        raise ValueError(f"Benchmark CSV does not cover {len(missing)} records. Examples: {examples}")
    if mismatched:
        examples = ", ".join(mismatched[:10])
        raise ValueError(
            f"Benchmark CSV response hash does not match {len(mismatched)} current records. "
            f"Regenerate target responses or relabel those rows. Examples: {examples}"
        )
    return labelled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Large Dataset human label benchmark CSV.")
    parser.add_argument("--dataset-dir", default="Large Dataset", help="Dataset directory relative to project root.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path relative to project root.",
    )
    parser.add_argument("--target-model-name", default="", help="Target model name used for generation and model-specific CSV output.")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    rows = load_benchmark_rows(project_root, args.dataset_dir, args.target_model_name)
    output_path = (
        project_path(project_root, args.output)
        if args.output
        else default_benchmark_path(project_root, args.dataset_dir, args.target_model_name, response_fingerprint(rows))
    )
    write_benchmark_csv(output_path, rows)
    print(f"Wrote {len(rows)} benchmark rows to {output_path}")


if __name__ == "__main__":
    main()
