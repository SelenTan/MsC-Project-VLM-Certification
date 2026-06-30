#!/usr/bin/env python3
"""Interactive dataset directory selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DATASET_CHOICES = ("Large Dataset",)
DEFAULT_DATASET_PATHS = {
    "Large Dataset": str(Path.home() / "datasets" / "Large_Dataset"),
}


def resolve_dataset_path(
    project_root: Path,
    dataset_dir: str,
    dataset_paths: Optional[dict[str, str]] = None,
) -> Path:
    """Resolve a logical dataset name, such as Large Dataset, to its real directory."""
    paths = dataset_paths or DEFAULT_DATASET_PATHS
    path_text = paths.get(dataset_dir, dataset_dir)
    path = Path(path_text).expanduser()
    if path.is_absolute():
        project_fallback = project_root / dataset_dir
        if not path.exists() and project_fallback.exists():
            return project_fallback
        return path
    return project_root / path


def qa_file_count(
    project_root: Path,
    dataset_dir: str,
    dataset_paths: Optional[dict[str, str]] = None,
) -> int:
    return len(list(resolve_dataset_path(project_root, dataset_dir, dataset_paths).glob("*/qa/*.json")))


def logical_dataset_path(dataset_dir: str, dataset_root: Path, path: Path) -> str:
    return (Path(dataset_dir) / path.relative_to(dataset_root)).as_posix()


def resolve_dataset_reference(
    project_root: Path,
    dataset_dir: str,
    path_text: str,
    dataset_paths: Optional[dict[str, str]] = None,
) -> Path:
    """Resolve image paths stored in QA JSON, including logical and external dataset prefixes."""
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    dataset_root = resolve_dataset_path(project_root, dataset_dir, dataset_paths)
    parts = path.parts
    aliases = {dataset_dir, Path(dataset_dir).name, dataset_root.name, dataset_root.name.replace("_", " ")}
    if parts and parts[0] in aliases:
        return dataset_root.joinpath(*parts[1:])
    project_candidate = project_root / path
    if project_candidate.exists():
        return project_candidate
    return dataset_root / path


def select_dataset_dir(
    project_root: Path,
    configured_dataset_dir: Optional[str] = None,
    dataset_paths: Optional[dict[str, str]] = None,
) -> str:
    """Return the configured large dataset when it contains QA JSON files."""
    dataset_dir = "Large Dataset"
    if (
        resolve_dataset_path(project_root, dataset_dir, dataset_paths).exists()
        and qa_file_count(project_root, dataset_dir, dataset_paths) > 0
    ):
        return dataset_dir
    raise SystemExit("No valid dataset found. Expected Large Dataset with */qa/*.json files.")
