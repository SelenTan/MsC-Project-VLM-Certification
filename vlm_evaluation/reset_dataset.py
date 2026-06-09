#!/usr/bin/env python3
"""Reset mutable evaluation fields in the working dataset QA JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESET_FIELDS = ("target_model_response", "human_label", "judge_label", "failure_reason")


def reset_dataset_fields(project_root: Path, dataset_dir: str) -> dict[str, Any]:
    """Set only the mutable evaluation fields under dataset/*/qa/*.json back to null."""
    dataset_path = project_root / dataset_dir
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    files_changed = 0
    items_changed = 0
    reset_paths: list[str] = []

    for qa_path in sorted(dataset_path.glob("*/qa/*.json")):
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        changed = False
        for item in data.get("items", []):
            item_changed = False
            for field in RESET_FIELDS:
                if item.get(field) is not None:
                    item[field] = None
                    item_changed = True
            if item_changed:
                items_changed += 1
                changed = True
        if changed:
            qa_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            files_changed += 1
            reset_paths.append(qa_path.relative_to(project_root).as_posix())

    return {
        "dataset_dir": str(dataset_path),
        "reset_fields": list(RESET_FIELDS),
        "files_changed": files_changed,
        "items_changed": items_changed,
        "reset_paths": reset_paths,
    }


def main() -> None:
    """Reset mutable evaluation fields in the project dataset and print a short summary."""
    project_root = Path(__file__).resolve().parents[1]
    summary = reset_dataset_fields(project_root, "dataset")
    print("Dataset reset complete.")
    print(f"Dataset: {summary['dataset_dir']}")
    print(f"Fields: {', '.join(RESET_FIELDS)}")
    print(f"Files changed: {summary['files_changed']}")
    print(f"Items changed: {summary['items_changed']}")


if __name__ == "__main__":
    main()
