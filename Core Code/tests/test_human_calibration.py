from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Core Code" / "vlm_evaluation"))

from artifacts import (
    expand_human_label_sheet_for_missing_failures,
    human_label_sheet_row,
    selected_benchmark_human_gold_records,
    targets_without_human_failures,
)


def make_record(target: str, index: int, human_label: int | None = None) -> dict:
    return {
        "item_index": index,
        "item_id": f"{target}_{index:04d}",
        "image_id": f"image_{index:04d}",
        "image_type": "charts",
        "target": target,
        "qa_json_path": f"/tmp/{target}.json",
        "model_input_image_path": f"/tmp/{target}.jpg",
        "prompt": "prompt",
        "expected_evidence": "evidence",
        "expected_answer_or_behavior": "answer",
        "target_model_response": "response",
        "human_label": human_label,
        "notes": "",
    }


class HumanCalibrationTest(unittest.TestCase):
    def test_missing_failure_expansion_appends_to_existing_sheet(self) -> None:
        records = [make_record("refusal_behavior", index) for index in range(120)]
        labelled_records = [make_record("refusal_behavior", index, 0) for index in range(100)]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "human_label_tasks.csv"
            rows = [human_label_sheet_row(record) for record in labelled_records]
            for row in rows:
                row["human_label"] = "0"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            missing_targets = targets_without_human_failures(labelled_records, ("refusal_behavior",))
            added_records = expand_human_label_sheet_for_missing_failures(
                path,
                records,
                labelled_records,
                missing_targets,
                expansion_per_target=50,
                max_per_target=1000,
            )

            with path.open(encoding="utf-8", newline="") as file:
                written_rows = list(csv.DictReader(file))

        self.assertEqual(missing_targets, ["refusal_behavior"])
        self.assertEqual(len(added_records), 20)
        self.assertEqual(len(written_rows), 120)
        self.assertTrue(all(row["human_label"] == "0" for row in written_rows[:100]))
        self.assertTrue(all(row["human_label"] == "" for row in written_rows[100:]))

    def test_full_benchmark_selection_includes_failures_when_available(self) -> None:
        records = [
            make_record("refusal_behavior", index, 1 if index < 3 else 0)
            for index in range(120)
        ]

        selected_records = selected_benchmark_human_gold_records(
            records,
            ("refusal_behavior",),
            per_target=100,
        )
        selected_refusal = [record for record in selected_records if record["target"] == "refusal_behavior"]

        self.assertEqual(len(selected_refusal), 100)
        self.assertGreater(sum(record["human_label"] == 1 for record in selected_refusal), 0)


if __name__ == "__main__":
    unittest.main()
