#!/usr/bin/env python3
"""Generate robustness image variants and link them in QA JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ("screenshots", "charts", "documents", "receipts", "forms")
PERTURBATION_BY_LAST_DIGIT = {
    0: "blur",
    1: "compression",
    2: "crop",
    3: "low_contrast",
    4: "rotate_15",
    5: "blur",
    6: "compression",
    7: "crop",
    8: "low_contrast",
    9: "rotate_15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one robustness variant for each dataset image."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset directory.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        choices=DEFAULT_CATEGORIES,
        help="Image categories to process.",
    )
    return parser.parse_args()


def project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def numeric_suffix(image_id: str) -> int:
    match = re.search(r"(\d+)$", image_id)
    if not match:
        raise ValueError(f"Image id has no numeric suffix: {image_id}")
    return int(match.group(1))


def perturbation_for_image(image_id: str) -> str:
    return PERTURBATION_BY_LAST_DIGIT[numeric_suffix(image_id) % 10]


def apply_blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=2.0))


def apply_compression(image: Image.Image) -> Image.Image:
    return image


def apply_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    left = max(0, int(width * 0.05))
    top = max(0, int(height * 0.05))
    right = min(width, int(width * 0.95))
    bottom = min(height, int(height * 0.95))
    return image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)


def apply_low_contrast(image: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(0.45)


def apply_rotate_15(image: Image.Image) -> Image.Image:
    return image.rotate(15, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))


PERTURBATION_FUNCS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "blur": apply_blur,
    "compression": apply_compression,
    "crop": apply_crop,
    "low_contrast": apply_low_contrast,
    "rotate_15": apply_rotate_15,
}


def save_variant(image: Image.Image, perturbation: str, output_path: Path) -> None:
    if perturbation == "compression":
        image.save(output_path, format="JPEG", quality=35, optimize=True)
        return
    image.save(output_path, format="JPEG", quality=95)


def generate_variant(source_path: Path, output_path: Path, perturbation: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        variant = PERTURBATION_FUNCS[perturbation](rgb_image)
        save_variant(variant, perturbation, output_path)


def update_qa_variant_path(qa_path: Path, variant_path: Path, perturbation: str) -> None:
    """Attach the generated variant to the image's robustness item only."""
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    relative_variant_path = relative_to_project(variant_path)
    found_robustness_item = False

    for item in data.get("items", []):
        if item.get("target") == "robustness":
            item["variant_image_paths"] = [relative_variant_path]
            item["notes"] = f"Robustness variant: {perturbation}."
            found_robustness_item = True
        else:
            item["variant_image_paths"] = []

    if not found_robustness_item:
        raise ValueError(f"No robustness item found in {qa_path}")

    qa_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_category(dataset_dir: Path, category: str) -> int:
    """Generate one deterministic robustness variant for every image in one category."""
    image_dir = dataset_dir / category / "images"
    qa_dir = dataset_dir / category / "qa"
    robust_dir = dataset_dir / category / "robust"
    processed = 0

    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = image_path.stem
        qa_path = qa_dir / f"{image_id}.json"
        if not qa_path.exists():
            raise FileNotFoundError(f"Missing QA JSON for {image_id}: {qa_path}")

        perturbation = perturbation_for_image(image_id)
        variant_path = robust_dir / f"{image_id}_{perturbation}.jpg"
        generate_variant(image_path, variant_path, perturbation)
        update_qa_variant_path(qa_path, variant_path, perturbation)
        processed += 1

    return processed


def main() -> None:
    args = parse_args()
    dataset_dir = project_path(args.dataset_dir)

    total = 0
    for category in args.categories:
        processed = process_category(dataset_dir, category)
        total += processed
        print(f"{category}: generated {processed} robustness variants")

    print(f"Generated {total} robustness variants under {dataset_dir}")


if __name__ == "__main__":
    main()
