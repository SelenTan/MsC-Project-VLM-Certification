#!/usr/bin/env python3
"""Interactive dataset directory selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DATASET_CHOICES = ("dataset", "Medium Dataset", "Large Dataset")


def select_dataset_dir(project_root: Path, configured_dataset_dir: Optional[str] = None) -> str:
    """Choose the dataset directory at startup when multiple project datasets are available."""
    available = [dataset_dir for dataset_dir in DATASET_CHOICES if (project_root / dataset_dir).exists()]

    if configured_dataset_dir and configured_dataset_dir not in DATASET_CHOICES:
        return configured_dataset_dir
    if configured_dataset_dir and (project_root / configured_dataset_dir).exists() and len(available) < 2:
        return configured_dataset_dir
    if len(available) < 2:
        return configured_dataset_dir or (available[0] if available else DATASET_CHOICES[0])

    print("\nMultiple dataset directories are present. Choose which dataset to run:")
    for index, dataset_dir in enumerate(available, start=1):
        qa_count = len(list((project_root / dataset_dir).glob("*/qa/*.json")))
        print(f"{index}. {dataset_dir} ({qa_count} QA files)")

    while True:
        answer = input(f"Dataset choice [1-{len(available)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(available):
            return available[int(answer) - 1]
        for dataset_dir in available:
            if answer.casefold() == dataset_dir.casefold():
                return dataset_dir
        print(f"Please enter a number from 1 to {len(available)}.")
