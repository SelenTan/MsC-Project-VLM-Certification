#!/usr/bin/env python3
"""Write normalized checker reference fields into dataset QA JSON files."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from normalization import build_checker_reference_fields


def normalize_dataset_references(project_root: Path, dataset_dir: str = "dataset") -> dict[str, Any]:
    """Populate checker_reference_* fields for every item under dataset/*/qa/*.json."""
    dataset_path = project_root / dataset_dir
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    files_changed = 0
    items_changed = 0
    for qa_path in sorted(dataset_path.glob("*/qa/*.json")):
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        changed = False
        for item in data.get("items", []):
            fields = build_checker_reference_fields(item)
            if any(item.get(key) != value for key, value in fields.items()):
                item.update(fields)
                changed = True
                items_changed += 1
        if changed:
            qa_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            files_changed += 1

    return {
        "dataset_dir": str(dataset_path),
        "files_changed": files_changed,
        "items_changed": items_changed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize checker reference fields in QA JSON files.")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset directory relative to the project root.")
    return parser.parse_args()


def main() -> None:
    """Normalize the requested project dataset in place and print a short summary."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary = normalize_dataset_references(project_root, args.dataset_dir)
    print("Dataset checker references normalized.")
    print(f"Dataset: {summary['dataset_dir']}")
    print(f"Files changed: {summary['files_changed']}")
    print(f"Items changed: {summary['items_changed']}")


if __name__ == "__main__":
    main()
