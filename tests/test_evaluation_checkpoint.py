from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vlm_evaluation"))

import main
from ai_checker import CheckerError, build_checker_payload, parse_checker_json
from artifacts import write_human_label_sheet
from normalization import normalize_dataset_references, normalize_reference_answer


class EvaluationCheckpointTest(unittest.TestCase):
    def test_archive_wrapper_matches_export_interface(self) -> None:
        with patch.object(main, "export_run_artifacts") as export_mock:
            main.archive_run_artifacts(
                run_dir=Path("run"),
                run_name="run",
                config={},
                human_gold_records=[],
                evaluation_checker_rows=[],
                reliability_rows=[],
                monte_carlo_rows=[],
                certificate_rows=[],
            )

        export_mock.assert_called_once()
        self.assertNotIn("gold_checker_rows", export_mock.call_args.kwargs)

    def test_malformed_json_is_reported_as_checker_error(self) -> None:
        with self.assertRaisesRegex(CheckerError, "valid JSON"):
            parse_checker_json("prefix {not-json} suffix")

    def test_checker_payload_requires_schema_constrained_json(self) -> None:
        record = {
            "target": "robustness",
            "image_type": "documents",
            "prompt": "Which node is the root?",
            "checker_reference_evidence": "Root node label",
            "checker_reference_answer": (
                "laboratory research division; LABORATORY RESEARCH DIVISION"
            ),
            "expected_evidence": "Root node label",
            "expected_answer_or_behavior": "Laboratory Research Division",
            "target_model_response": 'The root is "Laboratory Research Division."',
            "notes": "",
            "original_expected_answer_or_behavior": "LABORATORY RESEARCH DIVISION",
            "original_target_model_response": "LABORATORY RESEARCH DIVISION",
            "variant_image_paths": [],
        }

        payload = build_checker_payload("checker", "guide", record, 100, 0.0)

        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["judge_label"]["enum"], [0, 1])
        self.assertFalse(schema["additionalProperties"])
        user_prompt = payload["messages"][1]["content"]
        self.assertIn(
            "checker_reference_answer: laboratory research division",
            user_prompt,
        )
        self.assertNotIn("raw_expected_answer_or_behavior", user_prompt)

    def test_reference_variants_are_deduplicated_case_insensitively(self) -> None:
        normalized = normalize_reference_answer(
            "laboratory research division; LABORATORY RESEARCH DIVISION"
        )

        self.assertEqual(normalized, "laboratory research division")

    def test_dataset_normalization_preserves_run_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            qa_dir = project_root / "Medium Dataset" / "documents" / "qa"
            qa_dir.mkdir(parents=True)
            qa_path = qa_dir / "sample.json"
            qa_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "expected_answer_or_behavior": (
                                    "laboratory research division; LABORATORY RESEARCH DIVISION"
                                ),
                                "expected_evidence": "  root   node label  ",
                                "target_model_response": "response",
                                "human_label": 0,
                                "judge_label": 1,
                                "failure_reason": "reason",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = normalize_dataset_references(project_root, "Medium Dataset")

            item = json.loads(qa_path.read_text(encoding="utf-8"))["items"][0]
            self.assertEqual(summary["files_changed"], 1)
            self.assertEqual(item["checker_reference_answer"], "laboratory research division")
            self.assertEqual(item["checker_reference_evidence"], "root node label")
            self.assertEqual(item["target_model_response"], "response")
            self.assertEqual(item["human_label"], 0)
            self.assertEqual(item["judge_label"], 1)
            self.assertEqual(item["failure_reason"], "reason")

    def test_human_label_sheet_is_blank_for_each_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "human_labels.csv"
            record = self.make_record(output_path, 0, "item-1")
            record.update(
                {
                    "human_label": 1,
                    "failure_reason": "old model failure",
                    "model_input_image_path": "image.png",
                    "prompt": "Which node is the root?",
                    "expected_evidence": "Root node label",
                    "expected_answer_or_behavior": "Laboratory Research Division",
                    "target_model_response": "Another division",
                    "notes": "",
                }
            )

            write_human_label_sheet(output_path, [record])

            with output_path.open(encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["human_label"], "")
            self.assertEqual(row["human_failure_reason"], "")

    def make_record(self, qa_path: Path, item_index: int, item_id: str) -> dict[str, object]:
        return {
            "qa_json_abs_path": str(qa_path),
            "qa_json_path": f"dataset/charts/qa/{qa_path.name}",
            "item_index": item_index,
            "item_id": item_id,
            "image_id": "image-1",
            "image_type": "charts",
            "target": "visual_factuality",
            "judge_label": None,
            "failure_reason": None,
        }

    def checker_result(self, item_id: str, label: int) -> dict[str, object]:
        return {
            "item_id": item_id,
            "image_id": "image-1",
            "image_type": "charts",
            "target": "visual_factuality",
            "qa_json_path": "dataset/charts/qa/sample.json",
            "judge_label": label,
            "judge_failure_reason": None if label == 0 else "wrong answer",
            "checker_raw_response": json.dumps({"judge_label": label, "failure_reason": None}),
            "checker_error": None,
        }

    def test_checkpoint_persists_success_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            qa_path = temp_path / "sample.json"
            qa_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {"item_id": "item-1", "judge_label": None, "failure_reason": None},
                            {"item_id": "item-2", "judge_label": None, "failure_reason": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            records = [
                self.make_record(qa_path, 0, "item-1"),
                self.make_record(qa_path, 1, "item-2"),
            ]
            checkpoint_path = temp_path / "checkpoint.jsonl"

            with patch.object(
                main,
                "call_checker",
                side_effect=[
                    self.checker_result("item-1", 0),
                    CheckerError("invalid JSON", raw_response="not-json"),
                ],
            ):
                with self.assertRaisesRegex(CheckerError, "item-2"):
                    main.judge_records(
                        records,
                        "guide",
                        "evaluation",
                        checkpoint_path=checkpoint_path,
                        persist_to_dataset=True,
                    )

            saved_items = json.loads(qa_path.read_text(encoding="utf-8"))["items"]
            self.assertEqual(saved_items[0]["judge_label"], 0)
            self.assertIsNone(saved_items[1]["judge_label"])
            checkpoint_rows = [
                json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(checkpoint_rows), 2)
            self.assertEqual(checkpoint_rows[-1]["checker_raw_response"], "not-json")
            self.assertEqual(checkpoint_rows[-1]["checker_error"], "invalid JSON")


if __name__ == "__main__":
    unittest.main()
