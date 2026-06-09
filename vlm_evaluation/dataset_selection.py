#!/usr/bin/env python3
"""Interactive dataset directory selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DATASET_CHOICES = ("dataset", "Large Dataset")


def select_dataset_dir(project_root: Path, configured_dataset_dir: Optional[str] = None) -> str:
    """Choose the dataset directory at startup when both project datasets are available."""
    available = [dataset_dir for dataset_dir in DATASET_CHOICES if (project_root / dataset_dir).exists()]

    if configured_dataset_dir and configured_dataset_dir not in DATASET_CHOICES:
        return configured_dataset_dir
    if configured_dataset_dir and len(available) < 2:
        return configured_dataset_dir
    if len(available) < 2:
        return configured_dataset_dir or (available[0] if available else DATASET_CHOICES[0])

    print("\nBoth dataset directories are present. Choose which dataset to run:")
    for index, dataset_dir in enumerate(DATASET_CHOICES, start=1):
        qa_count = len(list((project_root / dataset_dir).glob("*/qa/*.json")))
        print(f"{index}. {dataset_dir} ({qa_count} QA files)")

    while True:
        answer = input("Dataset choice [1=dataset, 2=Large Dataset]: ").strip().lower()
        if answer in {"1", "dataset", "d"}:
            return "dataset"
        if answer in {"2", "large dataset", "large", "l"}:
            return "Large Dataset"
        print("Please enter 1 for dataset or 2 for Large Dataset.")
