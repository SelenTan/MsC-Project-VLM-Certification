from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vlm_evaluation"))

import main
import artifacts
import model_server
import run_target_vlm
import workflow_state
from ai_checker import CheckerError, build_checker_payload, parse_checker_json
from artifacts import write_human_label_sheet
from normalization import normalize_dataset_references, normalize_reference_answer


class EvaluationCheckpointTest(unittest.TestCase):
    def test_archive_wrapper_matches_export_interface(self) -> None:
        with patch.object(main, "export_run_artifacts") as export_mock:
            main.archive_run_artifacts(
                run_dir=Path("run"),
                reliability_rows=[],
                monte_carlo_rows=[],
                certificate_rows=[],
            )

        export_mock.assert_called_once()
        self.assertNotIn("gold_checker_rows", export_mock.call_args.kwargs)

    def test_archive_does_not_recreate_redundant_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            run_dir = project_root / "runs" / "test-run"
            run_dir.mkdir(parents=True)
            reliability_rows = [
                {
                    "target": "visual_factuality",
                    "TPR": 1.0,
                    "FPR": 0.0,
                    "n_M1": 1,
                    "n_M0": 1,
                }
            ]

            artifacts.export_run_artifacts(
                project_root=project_root,
                dataset_dir="dataset",
                run_dir=run_dir,
                reliability_rows=reliability_rows,
                monte_carlo_rows=[{"target": "visual_factuality"}],
                certificate_rows=[],
            )

            csv_names = {path.name for path in run_dir.glob("*.csv")}
            self.assertEqual(
                csv_names,
                {"certificate_summary.csv"},
            )
            self.assertFalse((run_dir / "charts").exists())

    def test_archive_omits_monte_carlo_file_when_no_repeats_ran(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            run_dir = project_root / "runs" / "test-run"
            run_dir.mkdir(parents=True)

            artifacts.export_run_artifacts(
                project_root=project_root,
                dataset_dir="dataset",
                run_dir=run_dir,
                reliability_rows=[],
                monte_carlo_rows=[],
                certificate_rows=[],
            )

            self.assertFalse((run_dir / "monte_carlo_repeats.csv").exists())

    def test_archive_skips_snapshot_copy_when_source_is_existing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            run_dir = project_root / "runs" / "test-run"
            snapshot_qa_dir = run_dir / "dataset_snapshot" / "charts" / "qa"
            snapshot_qa_dir.mkdir(parents=True)
            qa_path = snapshot_qa_dir / "charts_0000.json"
            qa_path.write_text('{"items": []}', encoding="utf-8")

            artifacts.export_run_artifacts(
                project_root=project_root,
                dataset_dir="Large Dataset",
                run_dir=run_dir,
                reliability_rows=[],
                monte_carlo_rows=[],
                certificate_rows=[],
                dataset_paths={"Large Dataset": str(run_dir / "dataset_snapshot")},
            )

            self.assertTrue(qa_path.exists())

    def test_target_response_is_checkpointed_once_per_item(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            qa_dir = dataset_dir / "charts" / "qa"
            image_dir = dataset_dir / "charts" / "images"
            qa_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.jpg"
            image_path.write_bytes(b"image")
            qa_path = qa_dir / "sample.json"
            qa_path.write_text(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "items": [
                            {
                                "item_id": "item-1",
                                "target": "visual_factuality",
                                "prompt": "What is shown?",
                                "target_model_response": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                dataset_dir=str(dataset_dir),
                port=8000,
                api_key_env="UNSET_TEST_KEY",
                categories=None,
                targets=("visual_factuality",),
                limit=None,
                model="test-model",
                max_tokens=10,
                temperature=0.0,
                timeout=1,
                overwrite=False,
                dry_run=False,
            )

            real_write_json = run_target_vlm.write_json
            with (
                patch.object(run_target_vlm, "build_payload", return_value={}),
                patch.object(run_target_vlm, "call_vlm", return_value="answer"),
                patch.object(run_target_vlm, "write_json", wraps=real_write_json) as write_mock,
            ):
                processed = run_target_vlm.run_dataset(args)

            self.assertEqual(processed, 1)
            self.assertEqual(write_mock.call_count, 1)

    def test_target_record_resolves_external_large_dataset_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            external_root = Path(temp_dir) / "datasets" / "Large_Dataset"
            image_path = external_root / "charts" / "images" / "charts_0000.jpg"
            qa_path = external_root / "charts" / "qa" / "charts_0000.json"
            image_path.parent.mkdir(parents=True)
            qa_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            qa_path.write_text(
                json.dumps(
                    {
                        "image_path": "Large Dataset/charts/images/charts_0000.jpg",
                        "items": [
                            {
                                "item_id": "charts_0000_001",
                                "target": "visual_factuality",
                                "prompt": "What is shown?",
                                "target_model_response": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            record = {
                "dataset_dir": "Large Dataset",
                "qa_json_path": "Large Dataset/charts/qa/charts_0000.json",
                "qa_json_abs_path": str(qa_path),
                "item_index": 0,
                "item_id": "charts_0000_001",
                "image_id": "charts_0000",
                "image_type": "charts",
                "target": "visual_factuality",
                "prompt": "What is shown?",
                "image_path": "Large Dataset/charts/images/charts_0000.jpg",
                "variant_image_paths": "[]",
                "target_model_response": None,
            }
            args = SimpleNamespace(
                dataset_dir="Large Dataset",
                dataset_paths={"Large Dataset": str(external_root)},
                port=8000,
                api_key_env="UNSET_TEST_KEY",
                model="test-model",
                max_tokens=10,
                temperature=0.0,
                timeout=1,
                overwrite=False,
                progress_interval=1,
                dry_run=False,
            )
            rows = []

            with (
                patch.object(run_target_vlm, "build_payload", return_value={}),
                patch.object(run_target_vlm, "call_vlm", return_value="answer"),
            ):
                processed = run_target_vlm.run_records(args, [record], {}, rows.append)

            self.assertEqual(processed, 1)
            self.assertEqual(rows[0]["target_model_response"], "answer")
            saved = json.loads(qa_path.read_text(encoding="utf-8"))
            self.assertIsNone(saved["items"][0]["target_model_response"])

    def test_server_rejects_an_existing_endpoint_with_the_wrong_model(self) -> None:
        with patch.object(model_server, "endpoint_model_ids", return_value={"wrong-model"}):
            with self.assertRaisesRegex(model_server.ModelServerError, "not required model"):
                model_server.start_vllm_server(
                    endpoint="http://localhost:8000/v1/chat/completions",
                    model="required-model",
                    served_model_name="required-model",
                    log_path=Path("unused.log"),
                    cuda_visible_devices="0",
                    tensor_parallel_size=1,
                    max_model_len=None,
                    gpu_memory_utilization=None,
                    limit_mm_per_prompt=None,
                    mm_encoder_tp_mode=None,
                    extra_args=(),
                    wait_timeout_seconds=1,
                )

    def test_server_process_is_stopped_when_startup_fails(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(model_server, "endpoint_model_ids", return_value=None),
                patch.object(model_server.shutil, "which", return_value="/bin/vllm"),
                patch.object(model_server.subprocess, "Popen", return_value=process) as popen_mock,
                patch.object(model_server, "wait_for_endpoint", side_effect=RuntimeError("startup failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    model_server.start_vllm_server(
                        endpoint="http://localhost:8000/v1/chat/completions",
                        model="required-model",
                        served_model_name="required-model",
                        log_path=Path(temp_dir) / "server.log",
                        cuda_visible_devices="0",
                        tensor_parallel_size=1,
                        max_model_len=None,
                        gpu_memory_utilization=None,
                        limit_mm_per_prompt=None,
                        mm_encoder_tp_mode=None,
                        extra_args=(),
                        wait_timeout_seconds=1,
                    )
                popen_kwargs = popen_mock.call_args.kwargs
                self.assertEqual(popen_kwargs["env"]["HF_HUB_DISABLE_XET"], "1")

        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=30)

    def test_malformed_json_is_reported_as_checker_error(self) -> None:
        with self.assertRaisesRegex(CheckerError, "valid JSON"):
            parse_checker_json("prefix {not-json} suffix")

    def test_truncated_checker_reason_is_recovered_when_label_is_clear(self) -> None:
        result = parse_checker_json(
            '{\n'
            '  "judge_label": 1,\n'
            '  "failure_reason": "material inconsistency: response was more precise and called "\n'
        )

        self.assertEqual(result["judge_label"], 1)
        self.assertIn("material inconsistency", result["judge_failure_reason"])

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
            qa_dir = project_root / "Large Dataset" / "documents" / "qa"
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

            summary = normalize_dataset_references(project_root, "Large Dataset")

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

    def test_manifest_keeps_one_qa_file_in_the_same_chunk(self) -> None:
        records = [
            {
                **self.make_record(Path("a.json"), index, f"item-a-{index}"),
                "qa_json_path": "dataset/charts/qa/a.json",
                "item_index": index,
            }
            for index in range(3)
        ]
        records.extend(
            {
                **self.make_record(Path("b.json"), index, f"item-b-{index}"),
                "qa_json_path": "dataset/charts/qa/b.json",
                "item_index": index,
            }
            for index in range(3)
        )

        manifest_rows = workflow_state.group_records_for_chunks(records, chunk_size=4)

        chunks_by_path: dict[str, set[int]] = {}
        for row in manifest_rows:
            chunks_by_path.setdefault(row["qa_json_path"], set()).add(row["chunk_index"])
        self.assertEqual(chunks_by_path["dataset/charts/qa/a.json"], {0})
        self.assertEqual(chunks_by_path["dataset/charts/qa/b.json"], {1})

    def test_chunk_selection_accepts_ranges_and_rejects_unknown_chunks(self) -> None:
        selected = workflow_state.parse_chunk_selection("0, 2-4", available_chunks=range(6))

        self.assertEqual(selected, [0, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "not in manifest"):
            workflow_state.parse_chunk_selection("0,9", available_chunks=range(3))

    def test_collect_chunk_results_reports_missing_and_uses_latest_successful_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest_rows = [
                {"record_key": "key-1", "chunk_index": 0},
                {"record_key": "key-2", "chunk_index": 0},
            ]
            workflow_state.append_target_row(
                workflow_state.target_results_path(run_dir, 0),
                {
                    "record_key": "key-1",
                    "target_model_response": "answer",
                    "target_error": None,
                },
            )

            collected = workflow_state.collect_chunk_results(
                run_dir,
                manifest_rows,
                workflow_state.TARGET_RESULTS_FILE,
                value_field="target_model_response",
                error_field="target_error",
            )

            self.assertEqual(set(collected), {"key-1"})
            self.assertEqual(workflow_state.missing_keys(manifest_rows, collected), ["key-2"])

            workflow_state.append_target_row(
                workflow_state.target_results_path(run_dir, 0),
                {
                    "record_key": "key-1",
                    "target_model_response": "different answer",
                    "target_error": None,
                },
            )
            collected = workflow_state.collect_chunk_results(
                run_dir,
                manifest_rows,
                workflow_state.TARGET_RESULTS_FILE,
                value_field="target_model_response",
                error_field="target_error",
            )
            self.assertEqual(collected["key-1"]["target_model_response"], "different answer")

    def test_run_metadata_is_consolidated_into_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "run_config.json").write_text('{"run_name": "test"}', encoding="utf-8")
            manifest_rows = [{"record_key": "key-1", "chunk_index": 0}]

            workflow_state.write_manifest(run_dir, manifest_rows)
            workflow_state.write_run_status(run_dir, "created", dataset_dir="Large Dataset")
            workflow_state.write_human_gold_keys(run_dir, ["key-1"])
            workflow_state.save_local_chunks(run_dir, [0])

            self.assertTrue((run_dir / "chunks" / "manifest.jsonl").exists())
            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertFalse((run_dir / "run_status.json").exists())
            self.assertFalse((run_dir / "human_gold_keys.json").exists())
            self.assertFalse((run_dir / "local_state.json").exists())
            self.assertEqual(workflow_state.load_manifest(run_dir), manifest_rows)
            self.assertEqual(workflow_state.load_run_status(run_dir)["status"], "created")
            self.assertEqual(workflow_state.load_human_gold_keys(run_dir), {"key-1"})
            self.assertEqual(workflow_state.load_local_chunks(run_dir), [0])

    def test_completed_run_without_manifest_is_recoverable_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            run_dir = project_root / "runs" / "test-run"
            snapshot_qa_dir = run_dir / "dataset_snapshot" / "charts" / "qa"
            snapshot_qa_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_name": "test-run",
                        "dataset_dir": "Large Dataset",
                        "chunk_size": 1000,
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            (snapshot_qa_dir / "charts_0000.json").write_text(
                json.dumps(
                    {
                        "image_id": "charts_0000",
                        "image_type": "charts",
                        "image_path": "Large Dataset/charts/images/charts_0000.jpg",
                        "items": [
                            {
                                "item_id": "charts_0000_001",
                                "image_id": "charts_0000",
                                "image_type": "charts",
                                "target": "visual_factuality",
                                "prompt": "What is shown?",
                                "expected_evidence": "visible",
                                "expected_answer_or_behavior": "42",
                                "target_model_response": "42",
                                "judge_label": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(main, "PROJECT_ROOT", project_root), patch.object(main, "RUNS_DIR", "runs"):
                self.assertEqual(main.run_dirs(), [run_dir])
                main.ensure_run_manifest(run_dir)

            self.assertFalse((run_dir / "chunks").exists())
            self.assertTrue((run_dir / "manifest.jsonl").exists())
            self.assertEqual(workflow_state.load_manifest(run_dir)[0]["item_id"], "charts_0000_001")

    def test_effective_n_j_is_capped_by_evaluation_pool(self) -> None:
        records = (
            [{"target": "visual_factuality"} for _ in range(50)]
            + [{"target": "robustness"} for _ in range(60)]
            + [{"target": "refusal_behavior"} for _ in range(70)]
        )

        self.assertEqual(main.effective_n_j(records), 50)

    def test_progress_summary_reports_chunk_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest_rows = [
                {"record_key": "key-1", "chunk_index": 0},
                {"record_key": "key-2", "chunk_index": 0},
                {"record_key": "key-3", "chunk_index": 1},
            ]
            workflow_state.write_manifest(run_dir, manifest_rows)
            workflow_state.append_target_row(
                workflow_state.target_results_path(run_dir, 0),
                {"record_key": "key-1", "target_model_response": "a"},
            )
            workflow_state.append_target_row(
                workflow_state.target_results_path(run_dir, 0),
                {"record_key": "key-2", "target_model_response": "b"},
            )
            workflow_state.append_judge_row(
                workflow_state.judge_results_path(run_dir, 0),
                {"record_key": "key-1", "judge_label": 0},
            )

            summary = workflow_state.progress_summary(run_dir, manifest_rows)

            self.assertEqual(summary["target_done"], 2)
            self.assertEqual(summary["judge_done"], 1)
            self.assertEqual(summary["chunks"][0]["stage"], "checker VLM")
            self.assertEqual(summary["chunks"][1]["stage"], "target VLM")

    def test_delete_chunk_checkpoints_removes_target_and_judge_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            target_path = workflow_state.target_results_path(run_dir, 0)
            judge_path = workflow_state.judge_results_path(run_dir, 0)
            target_path.parent.mkdir(parents=True)
            target_path.write_text('{"record_key": "key-1"}\n', encoding="utf-8")
            judge_path.write_text('{"record_key": "key-1"}\n', encoding="utf-8")
            manifest_path = run_dir / workflow_state.MANIFEST_FILE
            manifest_path.write_text('{"record_key": "key-1", "chunk_index": 0}\n', encoding="utf-8")
            archived_manifest_path = run_dir / workflow_state.LEGACY_MANIFEST_FILE

            deleted = workflow_state.delete_chunk_checkpoints(run_dir)

            self.assertEqual(deleted, 2)
            self.assertFalse(target_path.exists())
            self.assertFalse(judge_path.exists())
            self.assertFalse(manifest_path.exists())
            self.assertFalse((run_dir / "chunks").exists())
            self.assertTrue(archived_manifest_path.exists())
            self.assertEqual(workflow_state.load_manifest(run_dir)[0]["record_key"], "key-1")


if __name__ == "__main__":
    unittest.main()
