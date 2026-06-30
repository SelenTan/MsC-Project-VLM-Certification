from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vlm_evaluation"))

from dataset_selection import resolve_dataset_reference, select_dataset_dir  # noqa: E402
from reset_dataset import reset_dataset_fields  # noqa: E402


def write_qa_file(project_root: Path, dataset_dir: str) -> None:
    path = project_root / dataset_dir / "charts" / "qa" / "charts_0000.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"items": []}', encoding="utf-8")


class DatasetSelectionTest(unittest.TestCase):
    def test_empty_legacy_dataset_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "dataset").mkdir()
            write_qa_file(project_root, "Large Dataset")

            selected = select_dataset_dir(project_root, "dataset")

            self.assertEqual(selected, "Large Dataset")

    def test_multiple_valid_datasets_prompt_for_choice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            write_qa_file(project_root, "Medium Dataset")
            write_qa_file(project_root, "Large Dataset")

            with patch("builtins.input", return_value="2"):
                selected = select_dataset_dir(project_root, "Large Dataset")

            self.assertEqual(selected, "Large Dataset")

    def test_large_dataset_can_live_outside_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            external_root = Path(temp_dir) / "datasets" / "Large_Dataset"
            project_root.mkdir()
            write_qa_file(external_root.parent, "Large_Dataset")
            dataset_paths = {
                "Medium Dataset": "Medium Dataset",
                "Large Dataset": str(external_root),
            }

            selected = select_dataset_dir(project_root, "Large Dataset", dataset_paths)
            image_path = resolve_dataset_reference(
                project_root,
                selected,
                "Large Dataset/charts/images/charts_0000.jpg",
                dataset_paths,
            )

            self.assertEqual(selected, "Large Dataset")
            self.assertEqual(image_path, external_root / "charts" / "images" / "charts_0000.jpg")

    def test_reset_dataset_fields_uses_external_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            external_root = Path(temp_dir) / "datasets" / "Large_Dataset"
            qa_path = external_root / "charts" / "qa" / "charts_0000.json"
            qa_path.parent.mkdir(parents=True, exist_ok=True)
            qa_path.write_text(
                '{"items": [{"target_model_response": "x", "human_label": 1, "judge_label": 0, "failure_reason": "bad"}]}',
                encoding="utf-8",
            )
            project_root.mkdir()
            dataset_paths = {"Large Dataset": str(external_root)}

            summary = reset_dataset_fields(project_root, "Large Dataset", dataset_paths)

            self.assertEqual(summary["items_changed"], 1)
            self.assertIn("Large Dataset/charts/qa/charts_0000.json", summary["reset_paths"])
            self.assertIn('"target_model_response": null', qa_path.read_text(encoding="utf-8"))

    def test_no_valid_dataset_exits_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(SystemExit, "No valid dataset found"):
                select_dataset_dir(Path(temp_dir), "dataset")


if __name__ == "__main__":
    unittest.main()
