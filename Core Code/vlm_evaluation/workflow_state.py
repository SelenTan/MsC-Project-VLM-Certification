#!/usr/bin/env python3
"""Run manifest, chunk assignment, checkpoint, and progress helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from artifacts import append_jsonl, record_key, write_json


RUN_CONFIG_FILE = "run_config.json"
MANIFEST_FILE = "chunks/manifest.jsonl"
LEGACY_MANIFEST_FILE = "manifest.jsonl"
RUN_STATUS_FILE = "run_status.json"
LOCAL_STATE_FILE = "local_state.json"
HUMAN_GOLD_KEYS_FILE = "human_gold_keys.json"
TARGET_RESULTS_FILE = "target_responses.jsonl"
JUDGE_RESULTS_FILE = "judge_labels.jsonl"


def chunk_name(chunk_index: int) -> str:
    return f"chunk-{chunk_index:03d}"


def chunk_dir(run_dir: Path, chunk_index: int) -> Path:
    return run_dir / "chunks" / chunk_name(chunk_index)


def target_results_path(run_dir: Path, chunk_index: int) -> Path:
    return chunk_dir(run_dir, chunk_index) / TARGET_RESULTS_FILE


def judge_results_path(run_dir: Path, chunk_index: int) -> Path:
    return chunk_dir(run_dir, chunk_index) / JUDGE_RESULTS_FILE


def delete_chunk_checkpoints(run_dir: Path) -> int:
    """Remove per-chunk target/checker checkpoints after final QA snapshots are archived."""
    deleted = 0
    for filename in (TARGET_RESULTS_FILE, JUDGE_RESULTS_FILE):
        for path in (run_dir / "chunks").glob(f"*/{filename}"):
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_number}") from exc
    return rows


def group_records_for_chunks(records: list[dict[str, Any]], chunk_size: int) -> list[dict[str, Any]]:
    """Build a fixed manifest while keeping all QA items from the same file together."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["qa_json_path"], []).append(record)

    rows: list[dict[str, Any]] = []
    chunk_index = 0
    records_in_chunk = 0
    for qa_json_path in sorted(grouped):
        group = sorted(grouped[qa_json_path], key=lambda record: record["item_index"])
        if records_in_chunk and records_in_chunk + len(group) > chunk_size:
            chunk_index += 1
            records_in_chunk = 0
        for record in group:
            rows.append(
                {
                    "record_key": record_key(record),
                    "chunk_index": chunk_index,
                    "qa_json_path": record["qa_json_path"],
                    "item_index": record["item_index"],
                    "item_id": record["item_id"],
                    "image_id": record["image_id"],
                    "image_type": record["image_type"],
                    "target": record["target"],
                }
            )
        records_in_chunk += len(group)
    return rows


def write_manifest(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = run_dir / MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / MANIFEST_FILE
    if not path.exists():
        path = run_dir / LEGACY_MANIFEST_FILE
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"Manifest not found or empty: {path}")
    return rows


def read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_CONFIG_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_run_config(run_dir: Path, updates: dict[str, Any]) -> None:
    config = read_run_config(run_dir)
    config.update(updates)
    write_json(run_dir / RUN_CONFIG_FILE, config)


def write_run_status(run_dir: Path, status: str, **extra: Any) -> None:
    update_run_config(run_dir, {"status": status, **extra})


def load_run_status(run_dir: Path) -> dict[str, Any]:
    config = read_run_config(run_dir)
    if "status" in config:
        return config
    path = run_dir / RUN_STATUS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "unknown"}


def write_human_gold_keys(run_dir: Path, keys: Iterable[str]) -> None:
    update_run_config(run_dir, {"human_gold_record_keys": list(keys)})


def load_human_gold_keys(run_dir: Path) -> set[str]:
    config = read_run_config(run_dir)
    if "human_gold_record_keys" in config:
        return set(config.get("human_gold_record_keys", []))
    path = run_dir / HUMAN_GOLD_KEYS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Human gold key file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("record_keys", []))


def save_local_chunks(run_dir: Path, chunk_indexes: Iterable[int]) -> None:
    indexes = sorted(set(chunk_indexes))
    update_run_config(run_dir, {"assigned_chunks": indexes})


def load_local_chunks(run_dir: Path) -> list[int]:
    config = read_run_config(run_dir)
    if "assigned_chunks" in config:
        return [int(index) for index in config.get("assigned_chunks", [])]
    path = run_dir / LOCAL_STATE_FILE
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(index) for index in data.get("assigned_chunks", [])]


def parse_chunk_selection(text: str, available_chunks: Iterable[int]) -> list[int]:
    available = set(available_chunks)
    selected: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid chunk range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(selected - available)
    if invalid:
        raise ValueError(f"Chunk(s) not in manifest: {invalid}")
    if not selected:
        raise ValueError("No chunks selected.")
    return sorted(selected)


def chunk_indexes(manifest_rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["chunk_index"]) for row in manifest_rows})


def records_for_chunks(
    records: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    selected_chunks: Iterable[int],
) -> list[dict[str, Any]]:
    selected = set(selected_chunks)
    keys = {
        row["record_key"]
        for row in manifest_rows
        if int(row["chunk_index"]) in selected
    }
    return [record for record in records if record_key(record) in keys]


def successful_target_rows(path: Path) -> dict[str, dict[str, Any]]:
    return successful_rows_by_key(
        read_jsonl(path),
        value_field="target_model_response",
        error_field="target_error",
    )


def successful_judge_rows(path: Path) -> dict[str, dict[str, Any]]:
    return successful_rows_by_key(
        read_jsonl(path),
        value_field="judge_label",
        error_field="checker_error",
    )


def successful_rows_by_key(
    rows: list[dict[str, Any]],
    value_field: str,
    error_field: str,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("record_key")
        if not key or row.get(error_field):
            continue
        if row.get(value_field) is None:
            continue
        completed[key] = row
    return completed


def append_target_row(path: Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row)


def append_judge_row(path: Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row)


def collect_chunk_results(
    run_dir: Path,
    manifest_rows: list[dict[str, Any]],
    result_name: str,
    value_field: str,
    error_field: str,
) -> dict[str, dict[str, Any]]:
    """Collect completed rows and fail if the same record has conflicting results."""
    chunk_to_keys: dict[int, set[str]] = {}
    for row in manifest_rows:
        chunk_to_keys.setdefault(int(row["chunk_index"]), set()).add(row["record_key"])

    merged: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for chunk_index in sorted(chunk_to_keys):
        rows = read_jsonl(chunk_dir(run_dir, chunk_index) / result_name)
        for row in rows:
            key = row.get("record_key")
            if not key or row.get(error_field) or row.get(value_field) is None:
                continue
            if key not in chunk_to_keys[chunk_index]:
                raise ValueError(f"Result for {key} appears in the wrong chunk {chunk_name(chunk_index)}.")
            existing = merged.get(key)
            if existing is not None and existing.get(value_field) != row.get(value_field):
                conflicts.append(key)
            merged[key] = row
    if conflicts:
        examples = ", ".join(conflicts[:10])
        raise ValueError(f"Conflicting {result_name} rows for {len(conflicts)} records: {examples}")
    return merged


def progress_summary(run_dir: Path, manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize run-level and chunk-level checkpoint progress."""
    total = len(manifest_rows)
    total_chunks = len(chunk_indexes(manifest_rows))
    chunk_rows = chunk_progress_rows(run_dir, manifest_rows)
    completed_chunks = sum(1 for row in chunk_rows if row["complete"])
    target_done = sum(row["target_done"] for row in chunk_rows)
    judge_done = sum(row["judge_done"] for row in chunk_rows)
    return {
        "total_records": total,
        "total_chunks": total_chunks,
        "target_done": target_done,
        "judge_done": judge_done,
        "completed_chunks": completed_chunks,
        "completion_percent": round((judge_done / total * 100) if total else 0.0, 2),
        "chunks": chunk_rows,
    }


def chunk_progress_rows(run_dir: Path, manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one status row per chunk using target and checker checkpoints."""
    rows: list[dict[str, Any]] = []
    for chunk_index in chunk_indexes(manifest_rows):
        keys = {row["record_key"] for row in manifest_rows if int(row["chunk_index"]) == chunk_index}
        target_rows = successful_target_rows(target_results_path(run_dir, chunk_index))
        judge_rows = successful_judge_rows(judge_results_path(run_dir, chunk_index))
        target_done = len(keys & set(target_rows))
        judge_done = len(keys & set(judge_rows))
        total = len(keys)
        if judge_done == total:
            stage = "complete"
        elif target_done < total:
            stage = "target VLM"
        else:
            stage = "checker VLM"
        rows.append(
            {
                "chunk_index": chunk_index,
                "chunk_name": chunk_name(chunk_index),
                "total": total,
                "target_done": target_done,
                "judge_done": judge_done,
                "stage": stage,
                "complete": judge_done == total,
            }
        )
    return rows


def collect_existing_successes(
    run_dir: Path,
    manifest_rows: list[dict[str, Any]],
    result_name: str,
    value_field: str,
    error_field: str,
) -> dict[str, dict[str, Any]]:
    expected_keys = {row["record_key"] for row in manifest_rows}
    merged: dict[str, dict[str, Any]] = {}
    for chunk_index in chunk_indexes(manifest_rows):
        rows = read_jsonl(chunk_dir(run_dir, chunk_index) / result_name)
        for key, row in successful_rows_by_key(rows, value_field, error_field).items():
            if key in expected_keys:
                merged[key] = row
    return merged


def missing_keys(manifest_rows: list[dict[str, Any]], completed_rows: dict[str, dict[str, Any]]) -> list[str]:
    completed = set(completed_rows)
    return [row["record_key"] for row in manifest_rows if row["record_key"] not in completed]
